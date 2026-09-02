# P2 Implementation Audit — Bi-Axis MAG 仓库审查结论

> 审查日期:2026-09-02。依据 `docs/BiAxis_MAG_P2_Implementation_and_Experiment_Plan.md` §42 Prompt 1 的要求输出。
> 结论先行:**P2 可以零修改 frozen 文件实现**:`biaxis_p2.py` subclass P1Model 并只 override `_graph_update`,新增 transport 数学层文件;P1 全部文件不动,P1 results 作为 frozen baseline。

## 1. subclass + override `_graph_update` 的安全性(§24 问题 1)

- P1 `forward()`(`biaxis_p1.py:289`)通过 `self._graph_update(f_block, edge_index, num_nodes)` **动态 dispatch** → subclass override 自动生效;`inference()` 与 `encode_factors()` 继承即可(full-graph exact,topology-free 属性不变)。
- P1 `forward` 对 `graph_out` 的**硬依赖只有 `["f_tilde"]`**(§25 问题 4);`["r"]["alpha"]["beta"]` 仅在正则分支被读,而该分支由 `relation_balance_weight or alpha_entropy_weight or budget_reg_weight` 控制,P2 config 全为 0 → 永不执行。P2 `_graph_update` 仍按 §25 返回全字段保持兼容。
- P1 `compute_p1_diagnostics` 也调用 `_graph_update` 并读 r/beta/alpha——P2 用自己的 `compute_p2_diagnostics`,不复用该函数,无冲突。

## 2. 删除 gate 参数的安全性(§24 问题 2)

- `graph_budget` / `factor_selector` 仅在 P1 `__init__` 与 `_graph_update` 中被引用(后者被 override)→ `del self.graph_budget; del self.factor_selector` 安全,不产生 unused trainable params。
- 额外删除 `proj_q` / `fusion_q`(仅 F0 路径使用;P2 `assert factor_aware=True` 使该路径不可达,误配时 AttributeError 即 fail-fast)。
- 继承仍保留(不删):`use_graph_budget/budget_shared` 等普通属性(P2 不读,无害)、`_graph_regularization`(永不触发)、`_get_raw_signature` 缓存、`_compute_aux`、`fusion`、`graph_w0`(唯一 graph operator,§25 问题 16)、`graph_norm`。

## 3. 可原样复用的 P1 函数(§24 问题 3)

`_decompose_relations`(r, availability, deg)、`_encode`/`_split_modalities`/`_compute_aux`、`fusion`、`graph_w0`、`graph_norm`、`_get_raw_signature` 缓存、`inference`/`encode_factors`、components 的 `relation_weighted_mean`/`compute_degree`/`relation_mass`。

## 4. Gamma 显存(§24 问题 6)

`Gamma [N,3,5]` float32:ele-fashion 775K 节点 → **46.5MB**,可忽略;`g_mix [N,3,128]` 1.2GB 与 P1 的 `g_f` 同量级。transport 成本 O(N·F·(K+1)·T)=775K×3×5×10 ≈ 1.2e8 次 log-sum-exp,每 epoch 远小于 edge aggregation。

## 5. 泄漏与 gaming 防线(§24 问题 7/8/9)

| 风险 | 防线 |
|---|---|
| semantic → relation decomposition | relation 模块只吃 edge_index(继承 P1,已测) |
| capacity prior gaming(relation 模块迁就 ν) | `nu = build_reference(availability.detach())`(§10) |
| relation 模块人为 uniform r 以弱化约束 | `q_rel = q_rel.detach()`(§17);r 仍通过 message path 受 task gradient 更新 |
| scorer 混入 capacity | scorer 输入只有 [f‖g_k‖f⊙g_k],**不含 availability**(§6) |
| isolated node NaN | degree==0 fast path:γ=全 Local,f_tilde=f,不跑 transport(§21) |

## 6. 数值稳定性

- `logK = S/ε`,先按 row 减去 max(S)(行常数被 hard row-marginal 的 u 吸收,plan 不变)→ 防 exp 溢出(测试 #12)。
- log-domain Sinkhorn 全用 logsumexp;`theta = τ/(τ+ε)`;τ=0 → θ=0 → 精确退化为 NullSoftmax(测试 #4)。
- 最终重算 `log_u` 保证行和精确(=1,残差 <1e-5,§20)。

## 7. 最小侵入 implementation plan

1. `src/models/biaxis_p2_components.py`(纯数学层:FactorRelationScore、null score、augmented scores、reference capacity、relation confidence、NullSoftmax、SemiRelaxedTransport)
2. `src/models/biaxis_p2.py`(subclass + 三 mode)
3. `configs/model/biaxis_p2.yaml`
4. `tests/test_biaxis_p2.py`(§31 清单)
5. `scripts/{analyze_p2_checkpoint,run_p2_screen,run_p2_confirm,summarize_p2}.py`(镜像 P1,复用 driver 模式)
6. Smoke → Screen 15 runs → Confirm(按 §38 情况分流)

P2 禁改清单(§42):nc.py/nc.yaml/loaders/splits/dataset configs/biaxis_p0*/biaxis_p1*/K=4/W0——全部保持零改动。
