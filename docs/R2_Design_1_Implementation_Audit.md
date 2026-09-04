# R2-Design-1 Implementation Audit（D1-0 / Prompt 1A）

**日期:** 2026-09-04
**范围:** 计划 `docs/BiAxis_R2_Design_1_Implementation_Validation_Plan.md` §37-A 的五项确认
**结论:** 五项全部 PASS — R2 可直接继承 P0 factorizer + aux，neighbor_mean 可直接复用，R2 不依赖 P1/P2/P3，A0 行为不变，无需改 registry。

---

## A1. R2 可以直接继承/复用 P0 factorizer 和 `_compute_aux` — PASS

`src/models/biaxis_p0.py` 的 `Model` 提供全部所需：

| 成员 | 说明 | R2 用法 |
|---|---|---|
| `factorizer` (`SemanticFactorizer`) | 输出 `{h_t,h_v,c_t,c_v,c,p_t,p_v}`，其中 `c=0.5*(c_t+c_v)` | R2-B0/F 直接用 `c`；R2-S/J 用 `c_t,c_v` 重建 adaptive common |
| `_compute_aux(factors)` | `L_common/L_orth/L_recon`，**只作用于** `c_t,c_v,p_t,p_v` | 原样继承（计划 §8：aux 只作用于 base decomposition，不对 refined factors 加新 loss） |
| `recon_text_head / recon_visual_head` | 重建头 | 原样继承 |
| `fusion` | `Linear(3d,hidden) → LN → GELU → Dropout` | 计划 §15 要求继续使用现有 fusion — 直接复用 |
| `_encode` / `_split_modalities` | 输入 `[x_t|x_v]` 拆解 | 直接复用 |

R2 继承方式与 P1 完全一致（`class Model(P0Model)`，override `forward`），
这是仓库内已审计的既有模式（`biaxis_p1.py:47`）。

## A2. A0（biaxis_final）P0 超参原样复制 — PASS

`configs/model/biaxis_final.yaml` 实测值，四个 R2 YAML 逐项复制：

```yaml
hidden_dim: 256        factor_dim: 128       dropout: 0.2
activation: gelu       norm: layernorm
lambda_common: 0.02    lambda_orth: 0.01     lambda_recon: 0.3
orth_fallback_batch: 16
full_graph_training: true
lr: 0.001              weight_decay: 0.0001
```

与计划 §16 预算（hidden=256 / factor=128 / dropout=0.2）一致。

## A3. `neighbor_mean` 可直接复用 — PASS

`biaxis_p1_components.neighbor_mean`（`biaxis_p1_components.py:233`）是
audit 过的 plain 1-hop 聚合（src→dst、isolated→0、chunk 支持）。R0 审计已
位级复现（129/129 PASS, max diff 5.96e-8），R2-0 的
`weighted_neighbor_mean`（全 1 权重）与它 bitwise 相等。

R2 只 `from .biaxis_p1_components import neighbor_mean` — **不实例化任何 P1 模块**，
故不构成对 P1 的依赖（文件依赖为 import 级，无构造/无权重）。

## A4. R2 不依赖 P1/P2/P3 — PASS

- R2 继承链：`biaxis_r2.Model → biaxis_p0.Model`，唯一跨文件 import 是
  `biaxis_components`（P0 依赖）+ `biaxis_p1_components.neighbor_mean`（纯函数）。
- 不 import / 不实例化 `biaxis_p1 / p2 / p3` 的任何类。
- 旧复杂关系链（`[log d, P log d, P² log d] → K prototypes → Γ → T_fk`）在 R2 中**完全不存在**，
  符合计划 §1.3 REMOVE。

## A5. Existing A0 行为不变 — PASS

- 本阶段不修改任何既有文件：`biaxis_p0/p1/p2/p3/final.py`、`biaxis_final.yaml`
  全部只读。
- `src/models/factory.py` 是动态 import（`src.models.{cfg.model.name}`）——**无需改 registry**。
  四个 YAML 的 `name:` 字段都写 `biaxis_r2`（同一实现，计划 §36「四个 YAML 都指向同一个
  model implementation」），Hydra config 组按文件名 `biaxis_r2_b0|f|s|j` 选择，
  model 模块按 `name:` 选择。
- A0 regression smoke 通过（tests/test_biaxis_r2.py::test_a0_regression_smoke_unchanged，
  构造完整 biaxis_final cfg 跑 forward）。

## A6. 训练协议兼容性（nc.py）— PASS

| 需求 | nc.py 机制 | R2 实现 |
|---|---|---|
| full-graph NC | `requires_full_graph_training=True` → `training_mode=full_graph` | 设置该属性（config `full_graph_training: true`） |
| 接口契约 | `forward → (z,None,None,aux_loss,aux_info)`；`inference → CPU z`；`out_dim` | 与 P1 相同：inference = 一次精确 full-graph forward |
| Val-only 协议 | `task.evaluate_test=false`（R1.5 硬化） | driver 强制传该 override；模型内无任何 test 逻辑（unit test #15 守护） |
| §34 训练信息 | `task.history_path` CSV（per-epoch train/val/aux/lr/patience） | driver 解析 best_epoch/stop_epoch/train_acc@best 等 |
| best checkpoint | `task.save_ckpt_path` | driver 保存 model.pt，analyzer 读取做机制诊断 |
| AdamW lr/wd | `model.lr / model.weight_decay` 优先于 task 级 | YAML 中写 A0 的 1e-3 / 1e-4 |

## A7. 实现决策记录（与计划的逐条对应）

### A7.1 Message LayerNorm 必须 `bias=False`（计划 §4.4 的推论）

计划要求 isolated node：`N^b=0 ⇒ LayerNorm(0)=0 ⇒ F'=F` 并通过 unit test。
标准 `LayerNorm` 有 `LN(0)=β≠0`，违反恒等式。故 `msg_norm_base` 与
`msg_norm_func` 均用 `nn.LayerNorm(d, bias=False)`（保留可学 scale γ，去 bias）。
这是为满足计划显式恒等约束的必要修正，在文档中披露。

（语义 trunk 的 LayerNorm 无此恒等要求，用标准默认。）

### A7.2 功能 cell 内存纪律（计划 §17）

scorer 输入 `u_ab=[F_b,N_a,F_b⊙N_a,|F_b−N_a|,e_src[a],e_tgt[b]] ∈ R^{4d+16}`。
实现按 `for target b: for source a:` 在线构造，每个 cell 的 `[N,528]` 张量用后即释放；
**从不** materialize `[N,3,3,4d]` 或 `[N,9,d]`。峰值瞬态 = 1 个 `[N,528]` + 3 个 `[N,d]`（V_a(N^a)）。
类型 embedding 按 cell 索引 expand（td=8，可忽略）。

### A7.3 参数预算核算（计划 §16）

| 模块 | 参数量 |
|---|---:|
| P0 backbone（factorizer+recon+fusion，不变） | ≈ 985k |
| B0：3 source transforms (3×128², bias=False) + 3 msg LN(γ only) + 3 raw_rho_base | ≈ 49.5k |
| Semantic Refiner：common gate (512→64→2) 33k + interaction trunk (768→128) 98.7k + 3 heads (128×128, bias=False) 49.2k | ≈ 181k |
| Functional：scorer (528→64→1) 33.9k + type embs 48 + 3 msg LN 384 + 3 rho_func | ≈ 34.4k |
| **R2-B0 合计** | **≈ 1.03M**（A0 = 1.40M） |
| **R2-J 合计** | **≈ 1.25M**（仍低于 A0；远低于 DiP 8M） |

说明：Semantic Refiner ≈181k 略超计划「~150k」目标 — 超出部分全部来自计划 §7
明确规定的 `Linear(6d,d)` interaction trunk（98k，不可削减除非改公式）。
计划 §16 的硬约束是「R2 total 不应接近 DiP 8M 级别」— 满足（1.25M < A0 的 1.40M）。
每个 variant 的实测 parameter count 会写入 run summary 与报告（计划 §16 要求）。

### A7.4 初始化纪律（计划 §6/§7/§13/§14）

| 模块 | 初始化 | Step-0 行为 |
|---|---|---|
| common gate 末层 | weight=0, bias=0 | w_t=w_v=0.5 严格等于 current average |
| semantic 3 heads | 全 0 | Δ=0 严格恒等 |
| raw_rho_base | 0 | ρ_base=σ(0)=0.5 |
| functional scorer 末层 | weight~N(0,1e-3), bias=0 | g≈0.5；配合 ρ_func=0.01 → 新路径 ≈0.005×message |
| rho_func | 0.01（直接 LayerScale，无 sigmoid） | — |

### A7.5 变量隔离纪律（计划 §3/§37-C）

- `semantic_refiner.enabled=false` → 不实例化 `adaptive_common / semantic_residual`
  （无闲置参数，沿用 R1「只构造 variant 使用的模块」纪律，unit test #4 守护）。
- `functional_transfer.enabled=false` → 不实例化 `func_scorer / src_type_emb /
  tgt_type_emb / msg_norm_func / rho_func`，forward 不进入功能路径（unit test #5 守护）。
- B0 diagonal 路径在全部 4 个 variant 中恒存在（计划 §13「R2-F = B0 + minimal functional residual」）。

### A7.6 Gate 序约定

gate/contribution 矩阵维度约定：**row = source factor a，col = target factor b**，
顺序 `[C, Pt, Pv]`（与 R0/R2-0 审计的 factor stack 序一致）。diagnostics JSON
显式携带 row/col 标签，不依赖隐式序。

## A8. 已实现文件清单（对应计划 §36）

```text
src/models/biaxis_r2_components.py   AdaptiveCommonGate / SemanticInteractionResidual / FunctionalScorer
src/models/biaxis_r2.py              Model（P0 继承 + B0 path + 可选 S/F 路径 + 诊断）
configs/model/biaxis_r2_b0.yaml      语义 OFF / 功能 OFF
configs/model/biaxis_r2_f.yaml       语义 OFF / 功能 ON
configs/model/biaxis_r2_s.yaml       语义 ON  / 功能 OFF
configs/model/biaxis_r2_j.yaml       语义 ON  / 功能 ON
tests/test_biaxis_r2_components.py   组件级 15 项
tests/test_biaxis_r2.py              模型级（计划 §37-I 的 15 项全覆盖）
scripts/analyze_perf_r2_checkpoint.py best-checkpoint 机制诊断（只读，no-test）
scripts/run_perf_r2d1.py             训练 driver（GPU 轮转 / 权重信号量 / 结果汇总）
scripts/summarize_perf_r2d1.py       汇总 + GO/NO-GO 判定 + 报告生成
src/analysis/perf_r2_utils.py        resolve_cfg / load_r2_setup / assert_no_test_access
```

未修改任何既有文件（§36 约束满足）。

## A9. 下一步

D1-0 完成条件（计划 §37-J）：synthetic + Movies full 单 forward/backward 资源 smoke，
输出 `outputs/perf_r2d1/audit/R2D1_AUDIT.md` 后停止。正式训练（D1-1）在 audit 通过后进行。
