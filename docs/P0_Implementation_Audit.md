# P0 Implementation Audit — MAG_baseline 仓库审查结论

> 审查日期：2026-09-01。依据 `docs/BiAxis_MAG_P0_Implementation_Plan.md` §22 Prompt 1 的要求输出。
> 结论先行：**P0-A 可以以"只新增文件 + 两处默认关闭的 additive 改动"实现，不需要重写 NC/LP 协议。**

---

## 1. Model 接口契约（factory / forward / inference）

- `src/models/factory.py`：`importlib.import_module(f"src.models.{cfg.model.name}")` → 模块必须导出 `class Model(cfg, data_info)`。**新模型文件命名为 `biaxis_p0.py` 即可被 `model=biaxis_p0` 动态加载。**
- `Model.forward(x, edge_index)` 必须返回 5 元组 `(z, None, None, aux_loss, aux_info)`。
  - `z`：`(num_nodes_in_batch, out_dim)`；`aux_loss`：标量 tensor（requires_grad）；`aux_info`：标量 dict。
- `out_dim` 属性被两处直接消费：
  - NC：`nn.Linear(model.out_dim, num_classes)`（`src/tasks/nc.py:76`）
  - LP：`LinkPredictor(in_dim=model.out_dim, ...)`（`src/tasks/lp.py:272-277`）
- `inference(x, edge_index, device, batch_size)`：仅当 `task.inference_mode=layerwise` 时被调用，必须返回 CPU 上的全图 `z`（`src/tasks/inference.py:35-36`）。`full` 模式直接调 `model(x, edge_index)`（在 GPU 上）。

## 2. data.x 的模态顺序：[text, image]（确认）

- MAGB loader：`x = torch.cat([text_feat, image_feat], dim=1)`（`src/data/loaders.py:104`）。
- MM-Graph loader：`x_t = x[:, :text_dim]`, `x_i = x[:, text_dim:text_dim+visual_dim]`（`loaders.py:85-86`）。
- 两者一致：**`x_t` 在前，`x_i` 在后**（configs 中亦有注释确认）。
- `data_info` 提供 `text_dim` / `visual_dim`（分别取 `data.x_t.shape[1]` / `data.x_i.shape[1]`，`nc.py:72-73`、`lp.py:268-269`）。模型内切分：`x_t = x[:, :text_dim]`, `x_v = x[:, text_dim:text_dim+visual_dim]`，并 assert `x.size(-1) >= text_dim + visual_dim`。
- 各数据集维度：Movies text=512/img=768；Toys 512/768；Grocery **256**/768；Reddit-S **100**/768；ele-fashion 512/512；sports/cloth 512/512。

## 3. NC NeighborLoader 下 forward 收到什么

- `model.name != "mlp"` 且 `full_graph_training=false` → `NeighborLoader`（`nc.py:93-99`）：
  - `batch.x`：子图节点特征 `(batch.num_nodes, input_dim)`（seed 节点在前 `[:batch.batch_size]`，邻居在后）；
  - `batch.edge_index`：子图边（局部索引）；`batch.n_id` 局部→全局映射（若 model 有 `_batch_n_id` 属性会预先写入，`nc.py:168-169`）。
  - runner 只用 `z[: batch.batch_size]`（`nc.py:171`）。
- `full_graph_training=true`：`model(x_all, edge_index_all)` 全图 GPU forward。
- mlp 路径：`model(x_batch, None)` —— edge_index 可能为 None。
- **对 biaxis_p0 的含义**：factorizer 是 per-node 的，NeighborLoader 只会让 aux loss 在"子图节点"上计算（多算邻居节点、少覆盖全图），语义无碍；forward 必须容忍 `edge_index=None`。

## 4. LP LinkNeighborLoader 下 forward 收到什么

- `LinkNeighborLoader` 采样子图；**当前 batch 的 positive label edges 已从 message graph 中剔除**（`_exclude_positive_label_edges_from_message_graph`，`lp.py:375-380`）后传给 `model(batch.x, message_edge_index)`。
- `batch.edge_label_index` 是 batch 内局部索引，runner 用 `z[src]`、`z[dst]` 局部取值（`lp.py:382-384`）。
- LP 的 `data.edge_index` 由 loader 构造成 **train-edge-only graph**（`loaders.py:151-157`，undirected，无 self-loop）——P0-D LP 探针的 propagation 图用它即可天然防泄漏。

## 5. aux_loss / aux_info 的消费方式

- `loss = criterion(logits, labels) + cfg.task.loss.aux_weight * aux_loss`（`nc.py:151`、`lp.py:392`），两个 task config 默认 `aux_weight: 1.0` → **采用计划 §4 方案 A（模型内部设 lambda），无需改 task config。**
- ⚠️ **关键发现**：`aux_info` 只有 `AUX_INFO_KEYS` 白名单（`src/tasks/common.py:9-18`）里的 key 会被汇总/打印；`p0_*` key 不在其中，runner 会静默忽略。
  - **需要的改动 1（additive）**：把 `p0_common_loss/p0_orth_loss/p0_recon_loss/p0_common_sim/p0_private_sim/p0_c_norm/p0_pt_norm/p0_pv_norm/p0_cp_overlap_t/p0_cp_overlap_v` 追加进 `AUX_INFO_KEYS`。现有模型不产出这些 key，行为零变化。

## 6. inference_mode 对新模型的约束

- `full`：全图 GPU forward。Movies/Toys/Grocery/Reddit-S（~10K–130K 节点）无压力；**ele-fashion（~775K×1024）勉强，sports-copurchase（~2.56M）与 cloth-copurchase（~2.2M）会 OOM**。
- `layerwise`：调 `model.inference(...)`。biaxis_p0 是 topology-free，`inference()` 实现为"分块 per-node forward（忽略 edge_index）"即可，零误差、无图依赖。
- **推荐**：小图可用 full，大图（ele-fashion / sports / cloth）强制 `task.inference_mode=layerwise`。

## 7. 最小侵入改动方案（总）

1. **新增文件（不触碰任何现有模块）**
   - `src/models/biaxis_components.py`（ModalityProjector / SemanticFactorizer / ReconstructionHead）
   - `src/models/biaxis_p0.py`（`Model`，兼容 5 元组接口 + `encode_factors()` + chunked `inference()`）
   - `configs/model/biaxis_p0.yaml`
   - `src/utils/biaxis_p0_diagnostics.py`（P0-B/C/D 工具，复用 `canonicalize_edges`）
   - `scripts/run_p0_*.py`（per-dataset 训练+诊断流水线、批量脚本）
   - `tests/test_biaxis_p0.py`
2. **两处 additive 改动（默认关闭，不影响任何现有结果）**
   - `src/tasks/common.py`：`AUX_INFO_KEYS` 追加 p0 key（见 §5）。
   - `src/tasks/nc.py` / `lp.py`：支持可选 `task.save_ckpt_path`（默认 null）——训练结束、best checkpoint 恢复后落盘 `{model_state, head_state, seed, dims}`，供 P0-D 探针离线复用 `encode_factors()`。**不保存时行为与现在逐字节一致。**

## 8. sampling / global-node-id 风险

- **factorizer 本身零风险**：per-node、无 id embedding、不用 `_batch_n_id`。
- **诊断阶段风险**：factor 在全图节点集上计算（global id），`data.edge_index` 也是 global id → 天然对齐；无向图统计用现成的 `canonicalize_edges()`（`graph_utils.py:39-45`，去双向、去自环、去重）。
- **LP 防泄漏**：propagation 图只用 `data.edge_index`（train-only）；探针训练只用 `edge_split.train`；valid/test 用 `edge_split.valid/test` 现有 negative sets。诊断代码加显式 assert：propagation adjacency 与 valid/test positive 边交集为空。
- P0-C 大图边采样：固定 `torch.Generator(seed)`，`max_edges=500k`。
- NC 的 `data.edge_index`（MAGB/MM-Graph）无 self-loop（config `add_self_loops: false`）→ **fixed GCN propagation 必须自行加 self-loop**：`D^{-1/2}(A+I)D^{-1/2}`。

## 9. 可直接复用的现有组件

- `src.models.predictor.LinkPredictor`（LP 探针，计划 §12 要求同结构）
- `src.tasks.lp` 的 `_build_forbidden_edge_keys` / `_build_epoch_train_labels`（filtered negative 协议，探针训练负采样与其一致）
- `src.tasks.nc` 的 `_evaluate_split` 风格（探针 eval 协议一致）
- `src.data.loaders.load_mag_data` + Hydra compose（诊断脚本重载同一数据/配置）
- `src.data.graph_utils.canonicalize_edges`、`edge_dict_to_index`

## 10. 环境

- 2× RTX 3090 (24GB)，`conda activate yhf_env`。GPU 0/1 基本空闲。
- git：detached HEAD，有未提交修改（mgat.yaml、task configs）+ 未跟踪 docs/——**P0 只新增文件，不动这些**。

## 11. P0-A 实现要点备忘（下一步）

- `hidden_dim=256, factor_dim=128, out_dim=256`；LayerNorm（batch 大小无关，eval 无统计依赖）。
- 投影：`Linear(in,256)→LN→GELU→Dropout→Linear(256,256)`，text/visual 不共享。
- `E_C` 共享：MLP 256→128→128；`E_t^P`、`E_v^P` 独立同构；`c=(c_t+c_v)/2`。
- Loss（模型内 lambda，runner `aux_weight=1.0`）：
  - `L_common = 1 − mean cos(c_t, c_v)`（先 `F.normalize`）
  - `L_orth = ‖Cov(C_t,P_t)‖²_F/d² + ‖Cov(C_v,P_v)‖²_F/d²`（batch 中心化 cross-covariance；batch 过小（<16）fallback 到 cosine-overlap）
  - `L_rec = MSE(D_t[c_t‖p_t], h_t) + MSE(D_v[c_v‖p_v], h_v)`（不 detach，同时训练 decoder 与投影）
- `encode_factors(x, edge_index=None)`：`@torch.no_grad` 返回 `{c, c_t, c_v, p_t, p_v, z_local}`，输出与 edge_index 完全无关（有测试保证）。
- forward 忽略 edge_index；训练态返回 aux losses + `p0_*` aux_info；eval 态由 runner 走全图/layerwise。
