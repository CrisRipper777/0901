# 0901 协议改造：RPTA 对齐 + DGF/DMGC 接入（2026-09-02）

> 参考：`../RPTA/docs/experiment_protocol.md`、`lp_protocol.md`、`LP_SPEED_OPTIMIZATION.md`、
> `baseline_adaptations.md`；模型架构参考 `../OpenMAG/src/model/DGF.py` / `DMGC.py`。

## 1. 为什么之前慢

| | 旧协议 | 新协议（RPTA 式） |
|---|---|---|
| NC 训练 | NeighborLoader 15× 子图，每 batch 只为 ~1/16 的计算量出 loss | **全图训练**：每 epoch 对完整传导图一次前向，CE 只作用 train 节点 |
| NC 评估 | 每 epoch 独立全图推理（重复前向） | 每 epoch 一次 eval 前向（复用已上卡的全图张量），val-Accuracy 早停，test 只在 checkpoint 固定后测一次 |
| LP 训练 | 1-hop 单向子图，每 epoch 重建 Data | **[5,5] 2-hop 双向子图** + v7 加速（4 workers / prefetch 2 / 8 CPU threads / Data 跨 epoch 复用） |
| LP 评估 | 每 epoch 全量评估、平均 tie 排名 | 每 2 epoch 评估、**pessimistic tie**（`neg >= pos` 算落后，OpenMAG 同款）、eval batch 512 |

实测 Movies-NC：GCN 整轮 **22.6s**（旧协议同类 run ~30min 量级）；RPTA 文档参考值 21–23s。

## 2. NC 协议（`src/tasks/nc.py` + `configs/task/nc.yaml`）

- `task.training_mode: full_graph`（默认；`sampled` 为巨图 books-nc 的显存逃生门）
- 优化器 AdamW（RPTA 默认），**无梯度裁剪**（`task.grad_clip: null`）、无 LR scheduler
- 300 epochs / patience 30 / `early_stop_min_epoch: 1`；checkpoint 指标 val Accuracy
- 每 epoch 一个全图前向出 train loss；eval 用独立 eval-mode 前向（dropout off）
- 模型级官方超参：`cfg.model.lr` / `cfg.model.weight_decay` 优先于 task 级
- 保留：`inference_mode: layerwise` 逃生门、`save_ckpt_path`、aux_info 统计、5 元组 forward 契约

## 3. LP 协议（`src/tasks/lp.py` + `configs/task/lp.yaml`）

- LinkNeighborLoader `[5,5]`、`subgraph_type: bidirectional`、batch 2048、1:1 filtered
  negatives（每 epoch 重采样、对全部已知正 pair 过滤）、逐 batch 监督边遮蔽（双方向）
- message graph = train-only 对称图；valid/test 固定 150 negatives 原样使用
- BCE；Adam lr 1e-3 / wd 1e-5；150 epochs **不提前终止**（`patience: null`，仍按 val MRR 存最优 checkpoint）；test 只测一次
- 每 2 epoch 全量 validation 评估（preload 到 GPU，RPTA 实测 0.4–0.6s，非瓶颈）
- **`decoder.proj_dim: 128`**：所有 encoder 输出投影到共享 128 维再进同一 Hadamard-MLP scorer（RPTA build_encoder 同款，保证模型间 scorer 容量一致）
- v7 系统加速：`loader_num_workers: 4`、`loader_prefetch_factor: 2`、`torch_threads: 8`（main.py 应用）、pyg Data 每 run 只建一次

## 4. 报告口径

- `src/utils/summary.py`：std 改为 **population（ddof=0）**，对齐 RPTA/OpenMAG 汇总方式
- 指标：NC = Accuracy + Macro-F1；LP = MRR + Hits@1/3/10；per-seed + mean±std

## 5. 官方 preset（RPTA baseline_adaptations.md / lp_protocol.md §3）

| 模型 | hidden | layers/steps | dropout | lr | wd | 说明 |
|---|---:|---:|---:|---:|---:|---|
| MLP | 128 | 2 | 0.5 | 1e-3 | 1e-4 | norm: none |
| GCN | 256 | 3 | 0.2 | 5e-3 | 1e-5 | ReLU+dropout，无 norm（OpenMAG 同款） |
| GraphSAGE | 256 | 3 | 0.02 | 5e-3 | 1e-5 | 无 norm |
| MMGCN | 128 | 2 | 0.5 | 5e-3 | 1e-5 | **mean (t+v)/2 融合保持不变**；LP 侧 dropout 0.0（run_all_lp.py 传覆盖） |
| MGAT | 128 | 1 | 0.2 | 5e-3 | 1e-5 | **mean (t+v)/2 保持不变**；LP 侧 dropout 0.0 |
| DGF | 64 | 10 filter steps | 0.0 | 5e-3 | 1e-5 | 新增 |
| DMGC | 128 | 1 | 0.5 | 5e-3 | 1e-5 | 新增（OpenMAG nc.yaml preset） |
| DiP | 256/256 | 3 hops | 0.1 | 1e-3 | 1e-5 | 保持原结构（full_graph_training 由 `requires_full_graph_training` 强制） |

## 6. 新增模型（监督适配版）

- `src/models/dgf.py`：保留 OpenMAG DGF 滤波核心（模态投影 + L2 norm + mean 融合 +
  特征域对称 softmax shift operator + 节点域截断 Neumann 级数）；**去掉**聚类三损失
  （跨模态 NCE / 随机游走对比 / K-means community）与 vision/text head，统一 CE。
  自环在归一化邻接内显式加入（OpenMAG NC loader `self_loop: True` 语义）。
- `src/models/dmgc.py`：保留双频滤波（共享 GraphEncoder 分别作用于低通归一化邻接与高通
  Laplacian）+ 每模态 sigmoid 低/高通融合 + 学习式跨模态注意力融合（L2 norm）；**去掉**
  InfoNCE 三损失。OpenMAG 的稠密 [N,N] 算子改为**数学等价的稀疏 matmul**（巨图可跑）。
  自环显式加入。
- 两模型均满足 5 元组 forward 契约 + chunked `inference()`；单元测试覆盖
  shape / 图依赖 / train-eval 一致 / 梯度 / inference-forward 等价（tests/test_dgf_dmgc.py）。

## 7. 与 RPTA 的已知差异（记录在案）

- **特征归一化**：RPTA 对每模态做 `auto_train_center_l2`（CLIP 数据解析为 node LayerNorm）；
  0901 特征保持原样（数据层未动，不改变既有 baseline 特征口径）。
- DiP-LP 官方结构（q_dim 512 / pseudo 128/32）未纳入默认 LP 批跑（0901 dip.yaml 为 NC
  preset），需要时单独跑。
- 逐 batch 监督边遮蔽用 searchsorted 实现（与 RPTA v7 的布尔查找表同语义，规模小）。

## 8. 冒烟验证（Movies，5 epochs 级）

- NC：GCN test 54.24/F1 45.39（22.6s）、DGF 48.31/F1 27.15（23.2s）、DMGC 51.96/F1 28.49
  （22.5s）、biaxis_p0 全图训练兼容 ✓
- LP：GCN MRR 39.79、DGF 45.62、DMGC 15.33（3–5 epochs 协议路径验证，非正式数字）
- 测试：59 passed
