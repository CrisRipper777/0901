# P3 Implementation Audit（2026-09-03）

> 对应计划 `BiAxis_MAG_P3_Implementation_and_Experiment_Plan_v2.md` §32 Prompt 1。
> 结论先行：**P3 只需 6 个新文件 + 零 frozen 文件改动**，最小侵入继承 P2 成立。

---

## 1. deterministic mode 的位置与 P3 如何保证它始终 false

**位置**（`src/models/biaxis_p2.py:98-104`，P2 frozen）：

- `self.p2_deterministic = bool(p2.get("deterministic", False))` 从 `model.p2.deterministic` 读取；
- 为 true 时：`os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ...)` + `torch.use_deterministic_algorithms(True, warn_only=True)`；
- 并在 `_get_raw_signature` / `_graph_update` 内切换到 CSR SpMM / Hillis-Steele 确定性聚合路径（慢 5–10×）。

**P3 三道防线保证 false**：

1. `configs/model/biaxis_p3.yaml` 显式写 `p2.deterministic: false`（Hydra 默认即读它）；
2. P3 模型 `__init__` 硬断言：`assert not self.p2_deterministic`（继承 P2 初始化后检查），任何 override 都会直接失败而不是静默慢跑；
3. P3 driver（`run_p3_operator_screen.py`）不传任何 `deterministic` 相关 override，且不包含 `--deterministic` 开关。

P2 仓库中的 deterministic 路径**保留不动**（作为未来复现/排查工具），P3 从不启用。

---

## 2. P3 最小侵入继承 P2 的方式

```
biaxis_p3.Model(biaxis_p2.Model)
```

- `__init__`：先 `super().__init__(cfg, data_info)`（P1→P2 全部初始化照旧），断言 `p2.mode == "null_softmax"`、`not self.p2_deterministic`，然后只**新增** `self.operator = FullResidualFactorRelationOperator(...)`；
- `_graph_update`：整体覆盖（P2 逻辑照抄到 message-passing 段），唯一改动是 `W0(g_mix)` → `operator(g_perm, gamma_graph, W0)`；
- 新增 `compute_p3_diagnostics`：调用父类 `compute_p2_diagnostics` 后补 operator 诊断（两遍 `_graph_update`，raw signature 有缓存，代价可忽略）；
- P1 的 `forward`/`inference` 不依赖 `_graph_update` 返回 dict 的其它内容（只用 `f_tilde`），P3 的返回 dict 是 P2 的超集，安全。

已验证（读代码）：`biaxis_p1.forward` 只硬依赖 `graph_out["f_tilde"]`；`_graph_regularization` 权重全为 0 且只读 `r/alpha/beta`（P3 照常返回）。**P2 文件零改动。**

---

## 3. P2 可复用逻辑 vs 必须在 relation sum 前改写的部分

**完全复用（P3 `_graph_update` 照抄 P2 的段落）**：

- `_decompose_relations`（M2 冻结：TopologyDiffusionSignature / EdgeStructuralToken / RelationPrototypes）
- `relation_weighted_mean` → `g_perm [N,F,K,d]`
- `transport_scorer` + `build_augmented_scores` + `null_score`
- `build_reference_capacity`（仅 NS 模式不参与，保留调用以维持数值路径一致）
- `compute_node_relation_confidence`
- `null_augmented_softmax`（P3 强制 mode=null_softmax）
- isolated-node fast path（全 Local）

**必须改写的部分（只有一处）**：

P2 先 `g_mix = Σ_k Γ g` 再 `m = W0(g_mix)`；P3 必须在 relation sum **之前**按 (f,k) cell 应用 `T_fk`。实现按计划 §9 分项累加：

```text
m = W0(Σ_k Γ g)                        shared term（聚合先行，与 P2 同 op 顺序）
  + Σ_f A_f(Σ_k Γ_ifk g_ik^f)          factor term（聚合先行）
  + Σ_k Γ_ifk B_k g_ik^f               relation term（K 循环，transient [N,F,d]）
  + Σ_{f,k} Γ_ifk C_fk g_ik^f          pair term（12 cell 循环，transient [N,d]）
```

Local 列（Γ_if0）与 P2 一致：只经过 `f_block` 残差连接（`f_tilde = LayerNorm(f_block + m)`），不经过任何 graph operator。

---

## 4. P3 config 如何显式固定 p2.mode=null_softmax

- `configs/model/biaxis_p3.yaml`：`p2.mode: null_softmax`、`p2.deterministic: false`、`p2.epsilon: 0.2`；
- 模型断言兜底：`assert self.p2_mode == "null_softmax"`（任何 override 到其它 mode 直接 AssertionError）。

---

## 5. 五 operator modes 而不构造 [N,F,K,d,d]

参数张量本身很小：`A [F,d,d]`、`B [K,d,d]`、`C [F,K,d,d]`（d=128 时 C 仅 12×128×128=196K 参数）。**禁止的是 node 维度的瞬时/常驻张量**：

- shared / factor term：聚合先行（`Σ_k Γ g` 后一次 matmul），transient `[N,F,d]`（与 P2 相同）；
- relation term：K=4 次 `[N,F,d] → [N,F,d]`，每次立即与 Γ 相乘累加；
- pair term：F×K=12 次 `[N,d] → [N,d]`，每次立即累加进 `m[:,f]`。

成本实测预估：单次 `[N,128]×[128,128]` GEMM 在 ele-fashion（N≈775K）上 ~1–2ms（25.4 GFLOP，20 TFLOPS 有效算力），20 个 matmul 合计 **~30ms/epoch**，相对 P2 的 0.77s/epoch 可忽略。显存无新增节点级张量。

---

## 6. zero residual 如何保证等价 P2

- `A/B/C` 全部 `torch.zeros` 初始化（zero-init 纪律，计划 §7）；
- 数学上 `T_fk = W0` → `m = Σ_k Γ_ifk W0 g_ik^f = W0(Σ_k Γ g)`；
- **数值上**：shared term 在 P3 中与 P2 使用完全相同的 op 顺序（`(Γ.unsqueeze(-1)*g_perm).sum(2)` → reshape `[N*F,d]` → `graph_w0` → reshape），零残差项乘出来是精确 0.0，加法恒等 → O0 与 P2 同权重输出 **bitwise 一致**；
- 单元测试：`test_zero_residual_equivalence`（5 modes 零残差 vs 直接 W0 路径）+ `test_p3_shared_matches_p2_model`（同 seed 权重下 P3 shared 与 P2 null_softmax 模型输出一致）。

---

## 7. 为什么 O0 必须作为 P3 内部 3-seed control 重新跑

1. P2 screen 的 null_softmax 只有 seed 42（confirm 里 NS 也只有部分 runs），而 P3-A 全部 5 variants 是 3 seeds——control 必须同 seed 可比才能做 paired-seed delta；
2. P3 的 `_graph_update` 是**新代码路径**（即使 shared term op 顺序与 P2 相同，整体 kernel 调用序列/autograd 图已变），默认原子聚合本身有 1e-6 级非确定性，0.1pp 级比较必须用 P3 内部同实现的 control；
3. 混合 P2 历史值与 P3 新值做 paired delta 会在 δ 里混入实现差异噪声。

---

## 8. P3-A 哪些比较存在参数量/可辨识性 confound

| 比较 | 额外残差参数（d=128） | confound |
|---|---|---|
| OF − O0 | 3d² ≈ 49K | factor main effect 与参数量的混合 |
| OR − O0 | 4d² ≈ 66K | relation main effect 与参数量的混合 |
| OADD − O0 | 7d² ≈ 115K | 同上 |
| **OFR − OADD** | 12d² ≈ 196K | **capacity confound + 不可辨识性** |

- `W0 + A_f + B_k + C_fk` 中 C 可以吸收 A/B 的表达（分解在 loss 下不可辨识）→ OFR>OADD 只能证明"pair-specific capacity 有价值"，**不能**单独作为 interaction 因果证据；
- 因果证据留到 P3-B：LR-ADD vs LR-INT（参数完全匹配，唯一差别是 `a_f·b_k` 项）；
- 报告 §8 已明确：OFR 是 upper-bound probe。

---

## 9. Batch runner 如何支持 5×5×3 并 resume/skip

复用 P2 driver 骨架（`run_p2_screen.py` 的 `_WeightedSemaphore` / `_poll_peak_mem` / `_parse_train_log` / `_parse_best_val_f1`）：

- jobs = 5 datasets × 5 variants × 3 seeds，排序后按数据集 pin GPU（ele-fashion 权重=全部 slots，独占整卡；小数据集权重 1 两两打包）；
- `summary.json` 存在且无 `--force` → SKIP（resume 幂等）；`--epochs N` 仅 smoke 用；
- 每个 job：`src.main`（训练，`model=biaxis_p3` + `model.p3.operator_mode=<mode>`）→ `analyze_p3_checkpoint.py`（best-ckpt diagnostics）→ `summary.json`；
- 输出根：`outputs/p3/operator/<Dataset>/<mode>/seed_<s>/`；
- 统一 override：`p2.mode=null_softmax, p2.deterministic=false` 由 config 默认值提供（driver 不传，模型断言兜底）。

---

## 10. 最小侵入 implementation audit（文件清单）

**新增（8 个）**：

```text
src/models/biaxis_p3_components.py    # FullResidualFactorRelationOperator（D1）
src/models/biaxis_p3.py               # Model(P2Model)，override _graph_update（D2）
configs/model/biaxis_p3.yaml          # 计划 §30 默认 config
tests/test_biaxis_p3.py               # 计划 §29 全部测试
scripts/analyze_p3_checkpoint.py      # best-ckpt P3 diagnostics（P2 版复制改造）
scripts/run_p3_operator_screen.py     # P3-A 75-run driver
scripts/summarize_p3.py               # mean±std + paired-seed delta + report
scripts/run_p3_lowrank_screen.py      # P3-B（P3-A GO 后）
```

**零改动（frozen）**：`biaxis_p0.py` / `biaxis_p1.py` / `biaxis_p1_components.py` / `biaxis_p2.py` / `biaxis_p2_components.py` / `src/tasks/nc.py` / `configs/task/nc.yaml` / 全部 dataset configs / P1-P2 scripts。

**可复用基础设施（已核实）**：

- NC runner：`results.json` 含 `val_acc/test_acc/test_macro_f1 {mean,std}`；Val Macro-F1 从 train.log `Val Acc X | Val F1 Y` 最佳 acc 行解析（P2 driver 同款）；
- `analyze_p2_checkpoint.py` 的 cfg 解析/ckpt 加载骨架直接复制改 `model=biaxis_p3`；
- 模型 config 的 `lr/weight_decay` 会被 nc.py 读取为 override（`cfg.model.get("lr", cfg.task.lr)`），P3 yaml 沿用 P2 的 `lr: 0.001, weight_decay: 0.0001`。

---

## 附：P3-D1/D2 关键实现约定

1. operator 组件不持有 W0 —— `forward(g_perm, gamma_graph, w0)` 接收模型继承的 `graph_w0`（单一 W0，状态字典保持 `graph_w0.weight`，与 P2 诊断/命名兼容）；
2. 所有残差 `bias=False`、zero init；regularizer 钩子存在但 config 默认 0（计划 §7）；
3. `_graph_update` 返回 dict = P2 超集（+`g_perm`，供诊断复用，避免二次聚合）；
4. P3 model 不删任何 P2 组件（transport_scorer/null_score 照用），`del` 只发生在 P2 内部（P1 gate）；
5. 训练路径默认原子聚合（快），`p2.deterministic=false` 硬断言。
