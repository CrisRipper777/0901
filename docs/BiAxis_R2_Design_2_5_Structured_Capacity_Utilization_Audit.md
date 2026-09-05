# Bi-Axis R2-Design-2.5
## Representation-to-Utility, Structured Capacity & Utilization Audit

**Protocol**: all formal experiments use seeds 42/43/44; Val-only; No Test.  
**Resource policy**: no hard lightweight constraint. Models may use materially more capacity if every added block has a defined function and a parameter-matched control.

## Core question

Repeated diagnostics show useful information can be decoded, but first-pass trainable modules often learn the expected direction while producing ~0 downstream gain. This stage tests whether the main bottleneck is:

1. structured-capacity deficiency;
2. premature compression;
3. gradient starvation / strong-path dominance;
4. train-objective vs validation-generalization mismatch;
5. generic capacity rather than mechanism-specific value.

The causal chain to audit is:

`Information -> Transformation -> Preservation -> Readout -> Optimization -> Task gain`

## Frozen facts

- Current scale M1 performs early mixing:
  `Hmix = H1 + alpha*(H2-H1) -> shared V_f -> LN -> residual -> fusion`.
- Pt fixed-alpha curves showed a useful region around alpha=0.5-0.75, while SGD learns only ~0.04-0.07.
- Current interaction adapters compress 9 source-target cells into 3 target corrections by source mean, then feed the old fusion.
- Therefore existing NO-GO results do not yet rule out dedicated hop/cell transforms plus delayed fusion.

---

# Stage map

1. **D2.5-0** Audit & infrastructure  
2. **D2.5-A** Formal alpha objective landscape, all 5 datasets x 3 seeds  
3. **D2.5-B** Layer-wise utility transmission audit  
4. **D2.5-C** Structured-capacity model matrix  
5. **D2.5-D** Optimization-accessibility interventions  
6. **D2.5-E** Mature hop-token attention, conditional  
7. **D2.5-F** Final synthesis

Primary datasets: Movies, Toys, Grocery.  
Guards: ele-fashion, Reddit-S.

---

# D2.5-A — Formal alpha objective landscape

Use B0 checkpoints. Fix `alpha_C=alpha_Pv=0`. Evaluate:

`alpha_Pt in {0, 0.25, 0.5, 0.75, 1.0}`

Run all five datasets and seeds 42/43/44. For every dataset/seed/alpha:

- parent frozen;
- fresh linear classifier;
- exact same classifier initialization across alpha values;
- 300 epochs, patience 30, best Val Accuracy.

Record:

- best Train CE / Train Accuracy;
- best Val CE / Val Accuracy / Macro-F1;
- best epoch.

At alpha = 0, 0.25, 0.5, with the trained head fixed, compute diagnostic-only:

`d TrainCE / d alpha_Pt` and `d ValCE / d alpha_Pt`.

Never use Val gradients for updating parameters.

Interpretation:

- Train and Val both favor large alpha but SGD remains near 0 -> **Optimization Accessibility Failure**.
- Train favors alpha~0 but Val favors 0.5-0.75 -> **Objective-Generalization Mismatch**.
- 3-seed fixed-alpha value disappears -> downgrade the Pt-2hop hypothesis.

---

# D2.5-B — Layer-wise utility transmission

For each dataset x seed, trace Pt H1/H2 information through the actual B0 computation.

## S0 raw
Compare probes on `[Pt | H1]` vs `[Pt | H2]`.

## S1 source transform
Compare `[Pt | V_Pt(H1)]` vs `[Pt | V_Pt(H2)]`.

## S2 after LayerNorm
Compare `[Pt | LN(V(H1))]` vs `[Pt | LN(V(H2))]`.

## S3 after factor residual
Compare:
`Pt + rho*LN(V(H1))`
vs
`Pt + rho*LN(V(H2))`.

## S4 after fusion
Keep C/Pv on H1 and replace only Pt graph context with H2. Evaluate:

- fixed parent classifier;
- retrained same-init linear classifier.

At every stage use matched `StandardScaler + RidgeClassifier(alpha=1.0)` on TRAIN -> VAL and report:

- Accuracy / Macro-F1;
- H2-H1 utility delta;
- retention ratio relative to raw S0;
- cosine;
- CKA;
- relative norm;
- effective rank.

The report must identify the first stage where utility materially collapses.

### Secondary interaction transmission audit
On Movies/Toys/Grocery x 3 seeds, track the strongest PRODDIFF signal through:

`raw 9 cells -> source mean -> factor add -> fusion`

Report probe utility, rank and pairwise cosine at each stage.

---

# D2.5-C — Structured-capacity model matrix

All variants run Movies/Toys/Grocery x seeds 42/43/44.

Use a common staged training schedule:

- epoch 1-20: freeze P0 factorizer;
- epoch 21+: unfreeze P0 with LR 1e-4;
- graph/readout/fusion/classifier LR 1e-3;
- AdamW, wd=1e-4;
- same warmup10+cosine schedule for every variant;
- 300 epochs, patience30, best Val Accuracy.

## C0 EARLY_MIX
Current M1 control:
`T_f((1-alpha)H1 + alpha H2)`.

## C1 SEP_SUM
Independent hop transforms:
`e1_f=T_f1(H1_f)`
`e2_f=T_f2(H2_f)`
`m_f=e1_f + beta_f*e2_f`

Each expert should be a real 2-layer MLP:
`Linear(d,2d) -> LN -> GELU -> Linear(2d,d)`.

Do not zero-initialize the entire H2 expert. Initialize beta around 0.1 and use a normal expert initialization so the branch receives gradients.

## C2 SEP_CONCAT — main candidate
Independent hop transforms, no early sum:

`e1_f=T_f1(H1_f)`
`e2_f=T_f2(H2_f)`
`q_f=[F_f | e1_f | e2_f]`

Factor readout:
`Linear(3d,2d) -> LN -> GELU -> Dropout -> Linear(2d,d)`

Then:
`Fout_f = F_f + R_f(q_f)`

Use a stronger 2-layer factor fusion instead of the current one-layer fusion.

## C3 INCEPTION_012
Maintain independent H0/H1/H2 experts:

`e0_f=T_f0(H0_f)`
`e1_f=T_f1(H1_f)`
`e2_f=T_f2(H2_f)`

Late concatenate and read out per factor. Never average hops before their transforms.

## C4 CAP_H1_DUP — parameter-matched mechanism control
Match C2/C3 structure and parameter count as closely as possible, but replace H2 with an independent second H1 branch:

`e1a_f=T_f1a(H1_f)`
`e1b_f=T_f1b(H1_f)`

Same concatenate/readout/fusion.

This controls for “two branches / more parameters”.

## C5 WIDE_B0 — generic-capacity control
Use only H1, but make source transforms and fusion deep/wide enough to match C2/C3 parameter count within ±5% where feasible.

## C6 DEEP_FUSION_ONLY
Keep B0 graph path exactly unchanged and only replace the current one-layer fusion with a stronger residual 2-3 layer MLP.

This tests whether downstream fusion/readout is the bottleneck.

### Main verdicts

Mechanism GO:
- candidate - B0 >= +0.50pp macro on M/T/G;
- >=2/3 dataset means positive;
- positive datasets >=2/3 seeds positive;
- Macro-F1 safe.

Final GO:
- candidate - A0 >= +0.30pp macro with the same stability constraints.

Mechanism-specific evidence:
- candidate - CAP_H1_DUP >= +0.20pp, or
- candidate - WIDE_B0 >= +0.20pp.

Otherwise label the gain **GENERIC CAPACITY GAIN**, not multiscale gain.

Any candidate within +0.20pp of Final GO or better must run ele-fashion/Reddit-S x seeds 42/43/44.

---

# D2.5-D — Optimization accessibility

Only run for at most two top candidates when representation evidence remains positive but training usage is weak.

## D1 expert-specific LR
Only if the landscape shows Train CE also favors stronger H2. Compare H2-expert LR:

`1e-3, 5e-3, 1e-2`

All 3 seeds.

## D2 deep supervision
Compare:
`lambda=0` vs `lambda=0.1`

Attach lightweight auxiliary training heads to hop/factor experts so secondary experts remain task-discriminative. Auxiliary heads are removed at inference.

## D3 path/expert dropout
Only if diagnostics support H1 dominance. During training, drop H1 expert with p=0.2; use all experts at inference.

Do not stack D1+D2+D3 at once. Run as single-factor interventions.

Record per expert:
- gradient norm;
- parameter update ratio;
- output norm;
- classifier sensitivity;
- H2-off ablation;
- train/val CE;
- Accuracy/F1.

Verdict:
- Gradient Starvation: SUPPORTED / NOT SUPPORTED.
- Objective mismatch: SUPPORTED / NOT SUPPORTED.

---

# D2.5-E — Mature factor-hop token attention, conditional

Only enter if C2/C3 already beats WIDE_B0 or CAP_H1_DUP by >=+0.20pp but late concat may still limit utilization.

For every factor, independently transform H0/H1/H2 to three hop tokens. Use:

- embed_dim=d;
- 4 attention heads;
- 2 Pre-LN Transformer-style blocks;
- FFN width=4d;
- dropout=0.1.

Use the ego/H0 token as query/summary, then produce the factor representation and perform factor fusion.

Strict capacity control:
same attention architecture, but all three tokens are independent H1 transforms.

Run M/T/G x 3 seeds, then guards x 3 seeds if promising.

Report:
- vs A0/B0;
- vs H1-attention capacity control;
- attention weights;
- H2-off ablation;
- expert rank/cosine.

Attention is entered only because independent experts are already useful, not because attention is assumed to be superior.

---

# Causal usage diagnostics for every top candidate

At the best checkpoint evaluate without retraining:

- FULL;
- H2-OFF;
- H1-OFF;
- optionally H0-OFF for C3/E.

A claimed multiscale gain requires a measurable H2-off drop. If H2-off≈0, do not attribute gains to multi-hop information.

Also report:
- expert effective rank;
- pairwise expert cosine;
- CKA;
- readout weight norms / attention weights;
- gradient sensitivity.

---

# Interpretation matrix

## Case 1
C2/C3 > A0 and > WIDE_B0/CAP_H1_DUP:
**Premature compression / insufficient specialized transforms was a real bottleneck.**

Next route: Ownership-Aware Multi-Scale Expert Fusion.

## Case 2
WIDE_B0 ~= C2/C3 and both improve:
**Generic backbone capacity was limiting.**

First build a stronger backbone, then revisit mechanism claims.

## Case 3
Representation utility remains, main CE underuses it, deep supervision helps:
**Gradient starvation / strong-path dominance is supported.**

Use expert-preserving supervision as a training strategy.

## Case 4
Separate transforms + late fusion + capacity controls + supervision all fail:
**Post-aggregation utilization is no longer the leading bottleneck.**

Then formally switch to:
**Semantic-Ownership-Aware Neighbor Utility Learning**, i.e. learn which neighbor is useful for which factor *before aggregation*.

---

# Prompt 1 — Audit & infrastructure

```text
进入 R2-Design-2.5：Representation-to-Utility, Structured Capacity & Utilization Audit。

背景：
过去多个 frozen probe 稳定发现 cross-factor interaction 和 Pt 2-hop 信息，但简单 trainable realization 经常“方向学对、最终增益≈0”。R2D2.0.5 又显示固定 alpha_Pt≈0.5~0.75 有明确 Val value，而 SGD alpha 只学到≈0.04~0.07。

本阶段要检验：
1 structured capacity deficiency
2 premature compression
3 gradient starvation
4 train-vs-val objective mismatch
5 generic capacity effect

正式实验全部 seeds42/43/44，Val only，No Test。

本 Prompt 只做审计和实现，不跑正式实验。

审查当前 B0 source transforms / LayerNorm / rho / P0 fusion / M1 EarlyMix / PRODDIFF source-mean compression。

实现：
src/models/biaxis_r2_capacity.py
src/models/biaxis_r2_capacity_components.py

modes:
EARLY_MIX
SEP_SUM
SEP_CONCAT
INCEPTION_012
CAP_H1_DUP
WIDE_B0
DEEP_FUSION

必须支持提取：
H0/H1/H2
hop expert outputs
before/after LN
before/after residual
pre/post fusion
per-expert ablation

实现 scripts:
perf_r2d25_landscape.py
perf_r2d25_transmission.py
perf_r2d25_capacity_train.py
perf_r2d25_optimization.py
summarize_perf_r2d25.py

测试至少覆盖：
EARLY_MIX alpha0复现B0；
SEP experts参数独立；
CAP_H1_DUP绝不访问H2；
INCEPTION分别访问H0/H1/H2；
H2-off只关闭H2；
SEP_CONCAT/INCEPTION无pre-transform hop mean；
classifier init bitwise replay；
No Test；
参数统计正确；
诊断finite。

输出 outputs/perf_r2d25/audit/R2D25_AUDIT.md
然后停止。
```

# Prompt 2 — Formal alpha landscape

```text
执行 R2D25-A。

Datasets:
Movies/Toys/Grocery/ele-fashion/Reddit-S
Seeds42/43/44

alpha_C=alpha_Pv=0
alpha_Pt in {0,0.25,0.5,0.75,1.0}

每 dataset/seed 复用对应B0 checkpoint和同一个 classifier init。
Parent frozen，每个alpha训练fresh linear classifier。
300 epochs, patience30, best ValAcc。
No Test。

输出：
Train CE/Acc
Val CE/Acc/Macro-F1
best epoch

在alpha=0/0.25/0.5处，固定trained head，计算diagnostic-only dTrainCE/dalpha 和 dValCE/dalpha；禁止用Val gradient更新模型。

生成 alpha_landscape.csv、alpha_gradients.csv、R2D25_LANDSCAPE_REPORT.md。

报告只能判为：
Optimization Accessibility Failure
Objective-Generalization Mismatch
3-seed headroom unstable
或 mixed。
停止。
```

# Prompt 3 — Layer-wise transmission

```text
执行 R2D25-B，全部5数据集×seeds42/43/44，No Test。

针对Pt H1/H2依次测：
S0 raw
S1 V(H)
S2 LN(V(H))
S3 factor residual
S4 fusion z

每阶段用相同 StandardScaler+Ridge(alpha=1)，TRAIN fit / VAL eval。
S4同时做 fixed parent head 和 retrained same-init linear head。

输出：
Acc/F1
H2-H1 delta
retention ratio
cosine
CKA
relative norm
effective rank

M/T/G额外追踪PRODDIFF：
raw9cells -> source mean -> factor add -> fusion，
输出probe utility/rank/cosine。

生成 scale_transmission.csv、interaction_transmission.csv、R2D25_TRANSMISSION_REPORT.md。
明确指出utility首次明显流失的位置。
停止。
```

# Prompt 4 — Structured capacity matrix

```text
执行 R2D25-C。

Models:
EARLY_MIX
SEP_SUM
SEP_CONCAT
INCEPTION_012
CAP_H1_DUP
WIDE_B0
DEEP_FUSION

Movies/Toys/Grocery × seeds42/43/44。
No Test。

统一训练：
epoch1-20 P0 frozen
epoch21+ P0 unfreeze lr=1e-4
其他模块lr=1e-3
AdamW wd1e-4
统一warmup10+cosine
300 epochs
patience30
best ValAcc

C2/C3/C4/C5参数量尽量匹配±5%；不能匹配则报告精确差值。

输出：
Acc/Macro-F1/per-class
params/memory/runtime/best epoch

比较：
candidate-B0
candidate-A0
candidate-CAP_H1_DUP
candidate-WIDE_B0

Mechanism GO:
vs B0 >=+0.50pp macro + 稳定 + F1 safe

Final GO:
vs A0 >=+0.30pp macro + 稳定

Mechanism-specific:
vs capacity control >=+0.20pp

所有接近或达到Final GO者，再跑ele-fashion/Reddit-S×42/43/44。

生成 capacity_results.csv、capacity_resources.csv、capacity_mechanism.csv、R2D25_CAPACITY_REPORT.md。
不要调参救单个variant。
停止。
```

# Prompt 5 — Optimization accessibility

```text
根据Landscape+Capacity，只选最多2个“representation有价值但训练使用不足”的candidate。

M/T/G×42/43/44。

如果Train CE也偏好H2：
单因素比较H2 expert LR = 1e-3 / 5e-3 / 1e-2。

如果存在strong-path dominance：
单因素比较 Deep Supervision lambda=0 vs 0.1。

若诊断进一步支持H1 dominance：
单因素比较 Path Dropout p=0 vs 0.2。

禁止一次叠加多个intervention。

记录：
grad norm
update ratio
expert norm
H2-off ablation
train/val CE
Acc/F1

输出 optimization_results.csv、optimization_gradients.csv、R2D25_OPTIMIZATION_REPORT.md。
给出 Gradient Starvation SUPPORTED/NOT SUPPORTED 与 Objective mismatch结论。
停止。
```

# Prompt 6 — Hop-token attention, conditional

```text
只有SEP_CONCAT/INCEPTION已证明比WIDE_B0或CAP_H1_DUP至少+0.20pp，但late readout可能仍限制时执行。

每factor独立变换H0/H1/H2成3 hop tokens。
使用2个Pre-LN Transformer block：
embed_dim=d
heads=4
FFN=4d
dropout=.1
ego/H0 token作为query/summary。

严格capacity control：
同attention架构，但3 tokens全部来自independent H1 transforms。

M/T/G×42/43/44。
若有希望，再guards×42/43/44。

报告vs A0/B0/control、attention weights、H2-off、rank/cosine。
输出 R2D25_HOP_ATTN_REPORT.md。
停止。
```

# Prompt 7 — Final synthesis

```text
R2-Design-2.5所有允许实验完成。
不要新实验，不要Test。

读取Landscape、Transmission、Capacity、Optimization、HopAttention(if entered)。

生成：
R2D25_MASTER_TABLE.csv
R2D25_HYPOTHESIS_LEDGER.csv
R2D25_FINAL_DIAGNOSIS.md

必须回答：
1 fixed Pt-H2 value是否3-seed稳定？
2 Train CE与Val optimum是否一致？
3 是否gradient starvation？
4 H2 utility在哪层首次丢失？
5 shared transform是否瓶颈？
6 LayerNorm是否瓶颈？
7 residual是否瓶颈？
8 fusion/readout是否瓶颈？
9 independent hop transforms是否提升？
10 late fusion是否提升？
11 是否超过H1-duplicate control？
12 是否超过Wide-B0？
13 增益来自机制还是generic capacity？
14 Deep Supervision/path dropout是否改善？
15 H2-off是否证明模型真的使用H2？
16 interaction premature compression是否成立？
17 下一步：
A Ownership-Aware Multi-Scale Expert Fusion
B Stronger Generic Backbone
C Optimization-aware Expert Training
D Semantic-Ownership-Aware Neighbor Utility Learning

给出 R2-Design-2.5 PASS/PARTIAL/NO-GO。
不要设计最终论文模型，等待人工审查。
```

## 完成后返给我的材料

- `outputs/perf_r2d25/audit/`
- `outputs/perf_r2d25/landscape/`
- `outputs/perf_r2d25/transmission/`
- `outputs/perf_r2d25/capacity/`
- `outputs/perf_r2d25/optimization/`（若进入）
- `outputs/perf_r2d25/hop_attention/`（若进入）
- `outputs/perf_r2d25/summary/`
- `R2D25_FINAL_DIAGNOSIS.md`
- `R2D25_MASTER_TABLE.csv`
- `R2D25_HYPOTHESIS_LEDGER.csv`
- 最新 GitHub commit

## 最终纪律

只有同时满足：

1. candidate > A0；
2. candidate > parameter-matched H1 capacity control；
3. H2-off造成可测性能下降；
4. 3-seed稳定；
5. guards安全；

才能把 factor-specific multi-scale 升级为正式方法主线。
