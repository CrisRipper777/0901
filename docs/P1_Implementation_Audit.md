# P1 Implementation Audit — Bi-Axis MAG 仓库审查结论

> 审查日期：2026-09-02。依据 `docs/BiAxis_MAG_P1_Implementation_and_Experiment_Plan.md` §35 Prompt 1 的要求输出。
> 结论先行：**P1 可以以"新增文件 + 零修改 frozen 文件"实现**。P1 模型继承 `biaxis_p0.Model`，只覆盖 `forward` / `inference` 并新增 graph-side 模块；四 variant 由一个 model config switch 实现；不需要触碰 `nc.py` / `nc.yaml` / `loaders.py` / `splits.py` / dataset configs / 现有 baseline configs。

---

## 1. P1 为什么必须重写 inference

- P0 `inference()`（`biaxis_p0.py:172-190`）是 **per-node chunked forward，显式忽略 edge_index**（factorizer 是 per-node 的，chunking 零误差）。
- P1 的 graph module 需要**完整邻域**：chunk 掉节点会让每个节点的 message 只看到 chunk 内邻居，结果错误。
- 好消息：NC 协议 `task.inference_mode=full` 时 runner 在 eval/test 阶段直接调 `model(x_all, edge_index_all)`（`inference.py:38-43`），**根本不会调用 `inference()`**；只有 `layerwise` 模式才走 `model.inference(...)`。
- 结论：P1 重写 `inference(x, edge_index, device, batch_size)` 为**单次 full-graph forward**（`x`/`edge_index` 上 device，返回 CPU `z`）。5 个 NC 数据集（最大 ele-fashion 775K 节点）全图可在 24GB 卡内完成。`encode_factors()` 保持继承（topology-free，供对照检查）。

## 2. 如何继承/复用 P0 而不复制逻辑

```python
# src/models/biaxis_p1.py
from .biaxis_p0 import Model as P0Model

class Model(P0Model):
    def __init__(self, cfg, data_info):
        super().__init__(cfg, data_info)   # factorizer / recon heads / fusion / lambda_* / out_dim
        # ... 只新增 P1 graph-side 模块
```

- P0 `__init__` 只读 `cfg.model` 的 P0 keys（`hidden_dim/factor_dim/dropout/activation/norm/lambda_*`），P1 config（`configs/model/biaxis_p1.yaml`）**完整复制这些 keys 并保持 frozen 数值**，另加 `p1:` 块。P1 的 `__init__` 只读 `cfg.model.p1.*`。
- 复用的成员：`self.factorizer`（`SemanticFactorizer`，M1 **架构/目标不变但 jointly optimized**，非 weight-frozen——参数照常进 optimizer 并接收梯度）、`self.recon_text_head/recon_visual_head`、`self.fusion`（P0 融合，F1 路径复用）、`self._encode/_split_modalities/_compute_aux`、`self.out_dim=256`、`self.encode_factors`（诊断对照用）。
- P1 覆盖：`forward()`、`inference()`。aux loss 在 graph module **之前**从 pre-graph factors 计算（与 P0 完全相同的数值路径），P0 三项损失保持不变。
- `self.requires_full_graph_training = True`（对齐 `dip.py:303` 的既有模式）：即使 `task.training_mode` 被误设为 sampled，也强制 full-graph——P1 的 relation 分解依赖完整 topology。这是 model 属性，不是协议修改。

## 3. Topology signature 缓存位置

- 缓存放在 **Model 内部 buffer**（plan §10 推荐）：

```python
self.register_buffer("_cached_raw_struct_signature", torch.empty(0), persistent=False)
```

- 缓存内容：z-scored `s̄ ∈ R^{N×3}`（`u0, u1, u2` 归一化后）。`s̄` 是 `A` 的确定性函数且无参数，不会产生 autograd 图；显式 `detach()`。
- `persistent=False` → 不进 state_dict、不占 checkpoint；换图（不同 dataset/seed）时失效。
- 失效判定：`edge_index.data_ptr() == cached_ptr and num_nodes == cached_n`。NC 训练/eval 全程复用同一个 hosted `edge_index_all` tensor（`nc.py:126`），指针稳定；测试里新建 tensor 会触发重算，但 `s̄=f(A)` 确定性保证数值一致。
- **泄漏边界**：`s̄` 只来自 `edge_index`；`MLP_S / MLP_E / prototypes` 是唯一学习它的参数。cache 输入绝不依赖 `x` / factors（单元测试强制验证：改 x 不动 edge_index → r 完全一致）。

## 4. Relation 加权聚合的内存安全实现（不生成 [N,N]/[K,N,N]/[E,K,d]）

- 只保存 `edge_index [2,E]` 与 `r [E,K]`。
- 每条边 `(src→dst)`，消息沿 src→dst：
  - 入度 `d = bincount(dst, minlength=N)`（无自环图，入度=出度=度）。
  - relation mass：`m_k = index_add(zeros(N), 0, dst, r[:,k])`
  - availability：`a_k = m_k / (d + eps)`（非孤立节点 `Σ_k a = 1`，因 `Σ_k r = 1`）
  - weighted mean：`g̃_k = index_add(zeros(N,dim), 0, dst, r[:,k,None] * F[src])`，`g_k = g̃_k / (m_k + eps)[:,None]`
- **§9.2 优化**：`F_cat = [C‖Pt‖Pv] ∈ R^{N×3d_f}`，每 relation 只做一次聚合再 split 回三个 factor message。
- **edge chunking**（`p1.edge_chunk_size=500000`）：把 E 切成块做 index_add，避免 materialize `[E, 384]` 的 `r*F[src]` 大 tensor。K=4 时每块 [500K, 384] ≈ 768MB 瞬时量，ele-fashion 安全。
- 复杂度：O(K·E·d_f) 时间，峰值显存 O(chunk·d_f + N·K) 数量级。
- **重要简化（写入设计）**：`ḡ_i^f = Σ_k a_ik g_ik^f` 恰好等于 **plain neighbor mean**（因 `Σ_k r_jik = 1`，与 r 无关）：

```
Σ_k m_ik g_ik^f = Σ_k Σ_j r_jik f_j = Σ_j (Σ_k r_jik) f_j = Σ_j f_j
```

  所以 budget 输入 `[f_i ‖ ḡ_i^f]` 只需**一次普通邻居均值聚合**（等价于 K=1 路径），不随 K 增加计算；selector/update 才走 per-relation 聚合。K=1 时 `ḡ = g`，两条路径数值一致。

## 5. 四 variant 如何用一个 model config switch 实现

- 统一模型 `biaxis_p1.py`；内部统一 `[N, F, d_f]` 表示，`F = 3`（factor-aware）或 `1`（factor-blind），`K = 1 | 4`。
- `model.p1.factor_aware=false model.p1.num_relations=1` → F0R0；`true/1` → F1R0；`false/4` → F0R1；`true/4` → F1R1。
- **F=1 路径**：仍先跑完整 P0 factorizer（M1 架构/目标不变、jointly optimized、aux loss 照常），再 `q = Proj_q(self.fusion([C‖Pt‖Pv])) ∈ R^{d_f}`（plan §15 定义），graph module 只作用于 q；输出 `z = fusion_q(q') ∈ R^{256}`（`fusion_q` 结构对齐 P0 fusion：Linear→LN→GELU→Dropout）。**F=1 的 graph module 看不到 factor identity**——这就是 Factor OFF 的定义（不是重训无解耦 encoder）。
- **F=3 路径**：graph module 作用于 `[C, Pt, Pv]`，输出 `z = self.fusion([C'‖Pt'‖Pv'])`。
- **K=1 fast path**：`r = ones(E,1)`（不跑 prototype softmax），`alpha = ones(N,F,1)`（不跑 selector），聚合退化为普通邻居均值，budget 照常。plan §22 要求。
- F0/F1 参数差仅 `Proj_q + fusion_q ≈ 131K`，可接受（plan 接受同一 factorizer 下的实现差异；记录进 Params 列）。

## 6. Semantic leakage 风险点清单（实现时逐条防）

1. Relation 模块（signature / edge token / prototypes）**只接收 edge_index 与 num_nodes**，签名上不接受 x。
2. 缓存 `s̄` detach；`MLP_S/MLP_E/ρ` 的梯度只来自 graph-module 输出，不来自任何 semantic 输入。
3. Budget/Selector 输入 `[f_i ‖ g_ik^f ‖ f_i⊙g_ik^f ‖ a_ik]` 是 coupling 层的**合法**输入（factor 在此处才进入 relation 消费侧）。
4. F=1 路径不构造 3-factor selector（测试断言只产生一个 graph state）。
5. `compute_p1_diagnostics` 用 `@torch.no_grad()`，不修改模型状态。
6. 测试 `test_relation_topology_only`：同一 edge_index、不同 x → r 逐位一致。

## 7. 显存风险点

| 对象 | 规模（ele-fashion, 775K 节点） | 估计 |
|---|---|---|
| `x`（hosted by nc.py） | 775K×1024×4B | 3.2 GB |
| `F_cat` | 775K×384×4B | 1.2 GB |
| per-relation `g̃`（chunked 聚合） | chunk 500K×384×4B | 0.77 GB/chunk |
| `g [N,3,128]`、`alpha [N,3,4]`、`r [E,4]` | 小 | <0.5 GB |
| autograd 中间量（训练） | ≈ 激活总和 | 估 6-10 GB |

训练峰值估 ~12-15 GB（单卡 24GB OK）。Movies/Toys/Grocery/Reddit-S 规模小一个量级，无压力。
- **GPU 调度**：GPU0 当前有用户自跑的 DiP sports LP（8.7GB，未结束）；ele-fashion 任务只放 GPU1。其余小图两卡随意。
- 若 ele-fashion 仍紧张：`p1.edge_chunk_size` 下调即可，不牺牲正确性。

## 8. Frozen 接口清单（P1 全程禁止修改）

`src/tasks/nc.py`、`configs/task/nc.yaml`、`src/data/loaders.py`、`src/data/splits.py`、`configs/dataset/*.yaml`、现有 baseline model configs、`src/models/biaxis_p0.py`、`src/models/biaxis_components.py`、P0 diagnostics 统计定义（`src/utils/biaxis_p0_diagnostics.py` / `biaxis_p0_probes.py`）。

- `src/tasks/common.py` 的 `AUX_INFO_KEYS` **不加 p1 键**——P1 机制诊断走离线 checkpoint 分析（`compute_p1_diagnostics`），训练期只在 smoke 日志里观察，避免任何 frozen 文件改动。
- 复用既有 additive 接口：`task.save_ckpt_path`（P0 已加，默认 null）保存 best checkpoint，`analyze_p1_checkpoint.py` 离线加载。

## 9. 新增文件清单（对齐 plan §4）

```text
src/models/biaxis_p1_components.py   M2 relation 分解 + M3 budget/selector
src/models/biaxis_p1.py              主模型（继承 P0 Model）
configs/model/biaxis_p1.yaml         默认配置（P0 keys 数值不变 + p1 块）
tests/test_biaxis_p1.py              单元测试（plan §24 清单）
scripts/run_p1_screen.py             screen 批跑（resume/skip、双卡）
scripts/run_p1_confirm.py            confirm 批跑（screen GO 后）
scripts/analyze_p1_checkpoint.py     单 checkpoint 机制诊断
scripts/summarize_p1.py              汇总 CSVs + 报告
```

## 10. 环境确认

- `conda run -n yhf_env`（torch 2.4.0+cu121 / PyG 2.7.0 / hydra 1.3.2）；裸 `python` 是另一 env（torch 2.5.1），**所有实验命令必须走 yhf_env**。
- GPU0 有用户 DiP sports LP（pid 729539）在跑；GPU1 空闲。
- 数据流确认：MAGB/MM-Graph NC 的 `edge_index` 均已 `to_undirected` 且 **无自环**（config `add_self_loops: false`），`x=[x_t‖x_v]` 顺序稳定（`loaders.py`）。relation 分解按无向双向前提实现（reverse-edge 一致性测试覆盖）。

## 11. 实现顺序（下一步）

Prompt 2 → M2 组件 + relation 测试；Prompt 3 → M3 + 主模型 + 四 variant 测试；Prompt 4 → diagnostics + 脚本；Prompt 5 → smoke；Prompt 6 → screen。每步先全量 tests 再进入下一步。
