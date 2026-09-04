# Performance_R2_0_Audit — Repository Audit + Common Utilities（2026-09-04）

> 计划 `docs/BiAxis_Performance_R2_0_Architecture_Falsification_Plan.md` §3/§9 Prompt 1。
> 只做审计 + 公共工具 + 复现，不执行 R2-0A/B/C，不训练模型，不读取 Test。

## 1. 结论：Audit PASS ✅

| 检查项 | 结果 |
|---|---|
| 旧 OFR checkpoints 加载 | 9/9（M/T/G × 42/43/44）加载成功 |
| current main eval 与 R0 数学一致 | 126 行 full 复现 + 3 行 smoke **全部 diff = 0.0（位级精确）** |
| `extract_forward()` 中间量 | 全部在预期位置（见 §3） |
| `neighbor_mean()` message direction | src→dst，与模型聚合一致（见 §5） |
| no-Test discipline | 显式切断 + 源码级测试（见 §8） |
| 单元测试 | `tests/test_perf_r20_utils.py` 19/19 PASS；全库 309 PASS |

复现证据：`outputs/perf_r20/audit/reproduction.csv`（129 行：3 smoke + 126 full，FAIL=0，PASS_TOL=0）。

### Smoke（Movies seed42，计划 §3.1）

| representation | R2-0 复现值 | R0 存档值 | abs diff |
|---|---:|---:|---:|
| Probe([C\|Pt\|Pv]) | 0.4943011397720456 | 0.4943011397720456 | 0.0 |
| Probe(z_local) | 0.5065986802639472 | 0.5065986802639472 | 0.0 |
| Probe(z_final) | 0.528494301139772 | 0.528494301139772 | 0.0 |

### Full（M/T/G × 42/43/44，126 行）

- 9 lifecycles × 13 个 R0 表示（h_t … z_final）+ 9 个 head z_final Val-Acc
  （对照 `perf_r0/audit/checkpoint_audit.csv` 的 checkpoint_val_acc）。
- **全部 diff = 0.0**，判定 PASS（阈值 1e-5，软阈值 1e-4 未动用）。
- 说明：R0 运行与本次运行同环境（yhf_env、sklearn 1.8.0、同 GPU 原子路径），
  特征逐位一致，未触发 R15-0 报告的 ~2e-6 GPU 原子噪声地板。

---

## 2. 十个审计问题（计划 §9）逐项回答

### 1. 是否可以直接复用 R0 的 15 个 OFR checkpoints？

**是。** R0-D0 审计（`outputs/perf_r0/audit/R0_AUDIT.md`）已 15/15 PASS（字段完整、
Val-Acc 位级复现）。R2-0 第一轮只使用其中 9 个：

```
outputs/p3/operator/<dataset>/OFR/seed_<seed>/model.pt   (seed ∈ {42,43,44})
```

`model.pt` 含 `model_state` / `head_state` / `data_info` / `seed` / `task`。
注意 `OFR/seed_45|46` 也存在（P3-A borderline 补种子产物），**R2-0 不使用**（计划 §1.4）。

### 2. 当前 main 下 old checkpoint + eval 是否与 R0 数学一致？

**是。** 自 R0 以来唯一的 config 变化是 `p3.memory_checkpoint: true`（2026-09-04 显存补丁，
`biaxis_p3.yaml` 与 `biaxis_final.yaml` 均已含）。该路径只在
`torch.is_grad_enabled() and self.training` 下激活（`biaxis_p3.py:_graph_update`），
R0/R2-0 的提取走 `@torch.no_grad()` + `model.eval()`，**永远不进 checkpoint 分支**。
`biaxis_final` 是 `biaxis_p3` 的薄 alias（config 钉住 null_softmax + full_interaction），
R2-0 沿用 R0 的 `model=biaxis_p3` + `operator_mode=full_interaction` compose 覆盖，
与训练时完全一致。经验证据：126+3 行复现全部 diff=0.0。

### 3. extract_forward 是否已提供所需中间量？

**是**（`src/analysis/perf_r0_utils.py:80-122`），一次 full-graph forward 返回：

| 量 | shape | 用途 |
|---|---|---|
| `factors.h_t / h_v` | [N, 256] | R2-0C modal interaction |
| `factors.c_t / c_v` | [N, 128] | （R2-0 未用，审计完整性） |
| `factors.c / p_t / p_v`（即 C/Pt/Pv） | [N, 128] | A/B/C 全部 |
| `z_local / z_final` | [N, 256] | C 的 local 对照 |
| `f_block` | [N, 3, 128] | factor block |
| `f_tilde` | [N, 3, 128] | C'/Pt'/Pv' |
| `graph_out.g_perm` | [N, 3, 4, 128] | A2 / B3-B4 current contexts |
| `graph_out.r / availability / gamma / beta / alpha` | — | 诊断 |
| `scores.s_rel / s_aug` | — | 诊断 |
| `deg` | [N] | Splus、isolated 判断 |
| `edge_index` | [2, E] | 全部图聚合 |

### 4. g_perm 的 factor order 和 relation order 是什么？

- **factor order = [C, Pt, Pv]**：`torch.stack([c, p_t, p_v], dim=1)` 的堆叠序
  （`biaxis_p1.py:280-283`），`g_perm = [N, F=3, K=4, d=128]`。
- **relation order = prototype 索引 R1..R4**：任意初始化顺序，**跨 seed 不可对齐**
  （R0 已记录 Hungarian 稳定性结论）。R2-0A 的 3×3 矩阵按位置取 cell
  （factor 按 C/Pt/Pv 位置，source factor 按 g_perm[:, a] 位置），
  不赋予 R1..R4 任何固定语义。

### 5. neighbor_mean message direction 是否为 src→dst？是否与模型聚合一致？

**是。** `neighbor_mean`（`biaxis_p1_components.py:233-254`）：
`acc.index_add_(0, dst, features[src])`，除以 in-degree `bincount(dst)` ——
即 `D^{-1} A F`，与 `relation_weighted_mean`、`_graph_update`、R0 的 hop diffusion、
以及 R2-0B 的 `P F` 定义全部同向（src→dst，message direction）。
`weighted_neighbor_mean`（新）按同一方向实现，并有手算方向测试
（`test_weighted_mean_src_dst_direction_hand_computed`）与
「全 1 权重 == neighbor_mean」位级测试。

### 6. isolated node 当前怎样处理？

- **模型内**：P3 `_graph_update` isolated fast path（`deg<=0` → γ 全押 Local 列，
  f_tilde = graph_norm(f_block)）。Toys 有 10 个、ele-fashion 有 3 个 isolated 节点。
- **neighbor_mean / weighted_neighbor_mean**：`acc / (deg + eps)` 或
  `acc / (wsum + eps)` → isolated 为**有限零向量**（测试覆盖）。
- **availability**：`mass / (deg + eps)` → isolated 行和为 0（不为 1；R0 的
  row-sum sanity 只在非 isolated 节点上做，沿用）。
- **Splus**：isolated 的 raw 行全零 → 列 z-score 后为常数行 −μ/σ → row L2 归一化后
  有限单位范数（测试覆盖）；与 isolated 相连的边 w_sim/w_diff 有限。

### 7. Grocery / ele-fashion safe chunking 应怎样设置？

第一轮数据集全部很小（round 1 无显存压力）：

| dataset | N | E | isolated |
|---|---:|---:|---:|
| Movies | 16,672 | 160,802 | 0 |
| Toys | 20,695 | 113,402 | 10 |
| Grocery | 17,074 | 142,262 | 0 |
| （ele-fashion，第二轮） | 97,766 | 399,172 | 3 |
| （Reddit-S，第二轮） | 15,894 | 283,080 | 0 |

所有 E < `edge_chunk_size = 500000` → 单 chunk 路径；chunk 路径仅作安全兜底。
峰值瞬时 ≤ [500K, 128] × 4B = 256MB。ele-fashion 的 g_perm = [97,766, 3, 4, 128]
≈ 0.6GB（checkpoint data_info 实测 num_nodes=97,766 —— R0 审计文档 §8 的
「775K」系笔误，实际为该数据集真实节点数；仍遵循 R0 审计的纪律：禁止常驻
E×K×d）。`weighted_neighbor_mean` / `raw_splus` 均按 chunk 循环实现（计划 §5.8 内存纪律）。

### 8. 新 R2-0 scripts 如何显式禁止 test_idx / Test？

双重防线：
1. **运行时切断**：`perf_r20_utils.load_setup` = R0 `load_setup` + `guard_no_test`，
   加载后立即 `data.test_idx = None`（并先断言数据层确实提供了 test_idx——
   防止「空 split 假象」把没读到测试集当成纪律）。之后任何代码路径都拿不到 test。
2. **源码级测试**：`test_test_idx_only_referenced_inside_guard` 扫描
   `perf_r20_utils.py`，断言 `test_idx` 只出现在 `guard_no_test` 函数体内。
   `ridge_probe`（R0 原版）只读 train_idx / val_idx / y。

### 9. Fixed Ridge 是否完全复用 R0 协议？

**是。** 未重写：`r20.ridge_probe is r0.ridge_probe`（身份测试）。
协议 = `StandardScaler` fit TRAIN → `RidgeClassifier(alpha=1.0)` fit TRAIN →
predict VAL → val_acc + val_macro_f1。无任何超参搜索。

### 10. 是否修改了任何 frozen model files？

**没有。** 本轮只新增 3 个文件（`src/analysis/perf_r20_utils.py`、
`tests/test_perf_r20_utils.py`、`scripts/perf_r20_audit.py`）。
`src/models/biaxis_p0.py / p1*.py / p2*.py / p3*.py / biaxis_final.py` 零改动。

---

## 3. 新增公共工具 `src/analysis/perf_r20_utils.py`（计划 §9 清单）

| 计划条目 | 实现 | 测试 |
|---|---|---|
| 1. frozen setup wrapper | `load_setup`（复用 R0 + no-test 切断）、`guard_no_test` | guard ×3 + 源码扫描 |
| 2. factor aliases | `factor_tensor(fex, name)`、`factor_block(fex)`、`FACTOR_NAMES` | 别名 + KeyError |
| 3. weighted_neighbor_mean | `g_i = Σ_j w_ji F_j / (Σ_j w_ji + eps)`，src→dst、chunk-safe、isolated→0、无 [N,N] | 全 1==neighbor_mean（位级）、chunk/full 等价（位级）、手算方向、isolated/空图、权重长度校验 |
| 4. topology-only Splus | `raw_splus`（u0..u3、μ_d/σ_d、μ_gap/σ_gap）+ `compute_splus`（列 z-score → 行 L2） | 手算小图、确定性、finite/unit-norm、isolated finite、**源码级 topology-only 断言**（函数体不含 features/labels/factors/logits） |
| 5. context concat helper | `context_concat(parts)` | shape + 内容 |
| 6. CSV helper | 复用 `perf_r0_utils.write_csv` | 由 R0 覆盖 |
| 7. no-test guard | `guard_no_test` | 见上 |

R2-0A/B 所需的 `neighbor_mean`（plain 1-hop）与 `g_perm` 直接复用 R0 路径，
无需在 r20 中重复实现。

---

## 4. 面向 A/B/C 的复用要点（审计确认）

- **A1**：`N^a = neighbor_mean(edge_index, F^a, N, edge_chunk_size=500000)`；
  `X_{a→b} = [F^b | N^a]`，全部 9 cell 严格 2·d_f = 256。
- **A2**：`G^a = g_perm[:, a].reshape(N, K·d)`（K·d = 512）；`X_{a→b}^rel` 严格 640。
- **A3**：all-source plain = [F^b | N^C | N^Pt | N^Pv]（512 维）；rel（2048 维）；
  仅作 upper bound。
- **B2**：`G1 = P F = neighbor_mean(F)`；`G2 = P G1 = neighbor_mean(G1)`（不物化 2-hop 边表）；
  `Gsim/Gdiff = weighted_neighbor_mean(edge_index, w, F, N)`，w 由 Splus 余弦构造
  （`w_sim = (1+c_ji)/2 + 1e-8`，无阈值/温度/可学参数/label）。
- **B3/B4**：`X_current^f = [F | g_perm[:, f].reshape(N, K·d)]` 与
  `X_basis^f = [F | G1 | G2 | Gsim | Gdiff]` 严格同为 5·d_f = 640；joint 严格同为 15·d_f = 1920。
- **C1/C2**：h_t/h_v/C/Pt/Pv/z_local 全部在 `extract_forward` 单次 forward 内可得。
- **C4**：shuffle permutation `seed = 20260904` 固定，只打乱 interaction-only 列，
  [C|Pt|Pv] 不动，只用一个 permutation。

---

## 5. 待人工审查清单（计划 §14）

1. ✅ frozen checkpoint protocol —— 9/9 加载 + 129/129 位级复现；
2. ✅ reproduction 是否复现 R0 —— 全部 diff=0.0；
3. ✅ `g_perm` factor indexing —— [C, Pt, Pv] × R1..R4（位置语义）；
4. ✅ message direction —— src→dst，与模型聚合一致；
5. ✅ `Splus` 是否严格 topology-only —— 手算 + 源码级测试；
6. ✅ `weighted_neighbor_mean` full/chunk 数学等价 —— CPU 位级等价；
7. ✅ isolated node finite zero —— 4 个测试覆盖；
8. ✅ Test leakage —— guard 切断 + 源码扫描 + ridge_probe 只读 train/val。

**审查通过后进入 Prompt 2（R2-0A Cross-Factor Transfer）。**
