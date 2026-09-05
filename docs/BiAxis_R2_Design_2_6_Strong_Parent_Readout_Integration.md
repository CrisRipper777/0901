# Bi-Axis R2-Design-2.6
## Strong-Parent Readout Integration Plan

**Repository:** `CrisRipper777/0901`  
**Previous stage:** `R2-Design-2.5 = PARTIAL-POSITIVE MECHANISTIC DISCOVERY` after manual review  
**Current main question:**  
\[
\boxed{
\textbf{Can already-validated graph evidence be integrated into a strong parent
without paying an architecture-replacement tax or collapsing at readout?}
}
\]

**Protocol:** all formal experiments use seeds `42/43/44`; Val-only; **No Test**.  
**Resource policy:** no lightweight constraint. Capacity may grow materially when justified.  
**Performance parent:** A0 / `biaxis_final` formal reference.  
**Diagnostic parent:** B0 remains available for secondary transfer checks.

---

# 0. Why R2-Design-2.6 exists

R2D2.5 changed the diagnosis.

The important findings are no longer:

```text
“H2 does not work”
“interaction does not work”
“capacity is too small”
```

The stronger evidence is:

1. Pt-H2 utility survives:
   \[
   H_2
   \rightarrow V_f
   \rightarrow LN
   \rightarrow factor\ residual
   \]
   but largely collapses at the old final factor fusion/readout.

2. PRODDIFF interaction utility also survives source-cell aggregation and
   collapses primarily downstream near final fusion.

3. Generic width is not sufficient:
   `WIDE_B0` is worse than structured multi-hop variants.

4. Structured multi-hop computation has mechanism-specific value:
   - `SEP_CONCAT / INCEPTION_012` beat parameter-matched H1-only / Wide-B0 controls;
   - Hop-token attention beats parameter-identical H1-token attention strongly.

5. Deep supervision is a stable training lever.

6. The strongest structured candidates still fail to beat A0 because the
   candidate architecture replaces too much of the already-strong parent
   computation.

Therefore the next stage should **not** immediately redesign aggregation.

The next stage is:

\[
\boxed{
\textbf{Preserve the strong A0 base path}
+
\textbf{add validated side evidence}
+
\textbf{use a non-collapsing hierarchical readout}
+
\textbf{train the side evidence explicitly}
}
\]

---

# 1. Core hypotheses

## H1 — Architecture tax hypothesis

Previous candidates often replaced the original graph/readout path.

A mechanism can have positive relative value while the whole candidate remains below A0 because:

\[
\boxed{
\text{mechanism gain} < \text{replacement tax}
}
\]

R2D2.6 must preserve:

\[
z_{\text{A0}}
\]

as an explicit direct path.

---

## H2 — Readout bottleneck hypothesis

The validated graph evidence is not destroyed primarily by graph propagation;
it is lost when semantic factors are compressed into a single old fusion output.

Therefore:

\[
\boxed{
\text{late evidence preservation + hierarchical readout}
}
\]

should outperform old early/uniform compression.

---

## H3 — Structured evidence hypothesis

H0/H1/H2 provide useful content beyond capacity alone.

A candidate must beat an **architecture-identical H1-only control** before
claiming multi-hop value.

---

## H4 — Expert supervision hypothesis

Auxiliary supervision on local expert outputs prevents useful secondary
evidence from becoming weak or poorly trained.

Deep supervision is treated as a validated training tool, not the core novelty.

---

# 2. Stage overview

```text
D2.6-0  Audit + strong-parent extraction infrastructure
        ↓
D2.6-A  Strong-parent replay + no-compression readout diagnosis
        ↓
D2.6-B  Strong-parent integration matrix
        ↓
D2.6-C  Causal evidence-usage audit
        ↓
D2.6-D  Controlled parent adaptation
        ↓
D2.6-E  Deep-supervision confirmation / readout ablations
        ↓
D2.6-F  Optional interaction-token extension
        ↓
D2.6-G  Final synthesis
```

All formal stages use three seeds.

---

# 3. Strong parent definition

## Parent-P: A0

A0 is the only primary parent for R2D2.6.

Requirements:

```text
load the formal / accepted same-code-path A0 checkpoint
preserve all trained A0 weights
extract the full A0 output z_base
extract pre-graph semantic factors C/Pt/Pv
do not alter A0 forward unless a later schedule explicitly unfreezes it
```

The model must expose:

\[
F^0=[C,P_t,P_v]
\]

and:

\[
z_{\rm base}=A0(x,G).
\]

---

## Parent-C: B0

B0 is not used to select the final architecture.

It is used only after a top A0 candidate appears, to ask:

> Is the integration mechanism parent-specific or transferable?

---

# 4. A0 baseline replay discipline

For every:

```text
dataset × seed
```

create one saved classifier initialization.

Every variant for that dataset/seed must reuse it.

Two baselines are always reported:

### A0-MATCHED

Frozen A0 representation:

\[
z_{\rm base}
\]

plus a fresh matched classifier trained under the same protocol as the candidate.

### A0-FORMAL

The previously frozen formal A0 Val Accuracy / Macro-F1.

Scientific attribution uses:

\[
Candidate-A0_{\rm matched}.
\]

Final model quality additionally reports:

\[
Candidate-A0_{\rm formal}.
\]

---

# 5. Side evidence source

The side branch must **not** replace A0.

From the A0 semantic ownership factors:

\[
H_0^f=F^f
\]

\[
H_1^f=PH_0^f
\]

\[
H_2^f=P^2H_0^f.
\]

for:

\[
f\in\{C,P_t,P_v\}.
\]

No new relation prototype.

No high-pass.

No edge router in this stage.

---

# 6. Hop-specific experts

Each factor-hop pair has its own transform:

\[
e_k^f=E_{f,k}(H_k^f).
\]

Recommended expert:

```text
Linear(d, 2d)
LayerNorm
GELU
Dropout(0.1)
Linear(2d, d)
```

Independent parameters for:

```text
C-H0/C-H1/C-H2
Pt-H0/Pt-H1/Pt-H2
Pv-H0/Pv-H1/Pv-H2
```

No shared hop transform.

This is the structured capacity that D2.5 suggested was useful.

---

# 7. H1-only mechanism controls

Every hop-based architecture must have an architecture-identical control:

```text
H1_CONTROL
```

where all three token positions use independent transforms of H1:

\[
e_{1a}^f=E_{f,1a}(H_1^f)
\]

\[
e_{1b}^f=E_{f,1b}(H_1^f)
\]

\[
e_{1c}^f=E_{f,1c}(H_1^f).
\]

Same:

```text
parameter count
attention/readout architecture
training schedule
deep supervision
classifier init
```

Difference:

```text
candidate uses H0/H1/H2 evidence
control uses H1/H1/H1 evidence
```

This is mandatory.

---

# 8. D2.6-A — Strong-parent no-compression diagnosis

Before designing another compressed fusion, directly expose graph evidence to
the task head.

This is a **diagnostic**, not necessarily the final representation.

---

# 9. A0-BASE

\[
z=z_{\rm base}.
\]

Fresh linear classifier.

---

# 10. A-NC-HOP: No-Compression Hop Evidence

Use the nine independently transformed tokens:

\[
\mathcal E=
[
e_0^C,e_1^C,e_2^C,
e_0^{Pt},e_1^{Pt},e_2^{Pt},
e_0^{Pv},e_1^{Pv},e_2^{Pv}
].
\]

Representation:

\[
\boxed{
z_{\rm NC}
=
[
z_{\rm base}
\mid
\mathcal E
]
}
\]

No factor averaging.

No 9-token compression.

No projection back to 256.

Train only:

```text
hop experts
classifier
aux expert heads
```

A0 frozen.

---

# 11. A-NC-H1 — parameter-identical no-compression control

Same dimension and same architecture, but nine tokens are independent H1
transforms.

Compare:

\[
A\text{-NC-HOP}
-
A\text{-NC-H1}.
\]

This directly answers:

\[
\boxed{
\text{Does multi-hop content survive if we simply stop compressing it?}
}
\]

---

# 12. D2.6-A protocol

Run:

```text
Movies/Toys/Grocery/ele-fashion/Reddit-S
seeds42/43/44
```

A0 frozen.

Default deep supervision:

\[
\lambda_{aux}=0.1.
\]

Use identical classifier initialization for:

```text
A0-BASE
A-NC-HOP
A-NC-H1
```

Training:

```text
AdamW
experts/head lr=1e-3
wd=1e-4
warmup10 + cosine
300 epochs
patience30
best ValAcc
```

---

# 13. D2.6-A verdict

### NO-COMPRESSION CONTENT SUPPORT

On M/T/G:

\[
A\text{-NC-HOP}
-
A\text{-NC-H1}
\ge +0.30pp
\]

Accuracy macro, with:

```text
>=2/3 datasets positive
positive datasets >=2/3 seeds positive
Macro-F1 macro >= +0.20pp
```

If this is satisfied:

\[
\boxed{
\textbf{multi-hop content survives when compression is removed}
}
\]

is formally supported.

If both NC-HOP and NC-H1 improve similarly:

```text
generic expanded readout capacity
```

not multi-hop.

---

# 14. D2.6-B — Strong-parent integration matrix

This is the core stage.

All variants retain:

\[
\boxed{z_{\rm base}}
\]

as a direct skip path.

The new branch may only add information.

It must never replace the parent representation.

---

# 15. B0 — A0_BASE

Frozen A0 + matched fresh linear classifier.

---

# 16. B1 — BASE + FACTOR-HOP-CONCAT

For each factor, use factor-local hop attention:

\[
s_f
=
HopAttn_f(
e_0^f,e_1^f,e_2^f
).
\]

Then preserve the factor summaries without compression:

\[
\boxed{
z_{\rm FHC}
=
[
z_{\rm base}
\mid s_C
\mid s_{Pt}
\mid s_{Pv}
]
}
\]

Output dimension:

\[
h+3d.
\]

The downstream classifier consumes the expanded representation directly.

No projection to \(h\).

This is a representation-level late-concat candidate.

---

# 17. B1-control — BASE + H1-FACTOR-CONCAT

Exact same architecture.

Use H1/H1/H1 tokens.

Mandatory parameter-matched control.

---

# 18. B2 — BASE + RESIDUAL-SIDE-FUSION

Factor-local hop attention first:

\[
s_C,s_{Pt},s_{Pv}.
\]

Side readout:

\[
u=
ResidualFusion(
[s_C|s_{Pt}|s_{Pv}]
)
\in\mathbb R^h.
\]

Final:

\[
\boxed{
z_{\rm RSF}
=
z_{\rm base}
+
R_{\rm side}(u)
}
\]

`R_side` is a residual projection:

```text
Linear(h,h)
LayerNorm
GELU
Linear(h,h)
```

Use a small nonzero final-layer initialization, not exact zero.

Recommended final-layer std:

```text
1e-3
```

Do not use a scalar gate.

Deep supervision ensures the side experts do not starve.

---

# 19. B2-control — H1-RESIDUAL-SIDE-FUSION

Exact same architecture with H1/H1/H1 side tokens.

---

# 20. B3 — BASE-ANCHORED HIERARCHICAL ATTENTION

This is the most mature readout candidate.

Step 1 — within each factor:

\[
s_f=HopAttn_f(e_0^f,e_1^f,e_2^f).
\]

Step 2 — project factor summaries to hidden dimension:

\[
q_f=W_f s_f\in\mathbb R^h.
\]

Create four tokens:

\[
[
z_{\rm base},
q_C,
q_{Pt},
q_{Pv}
].
\]

Use 2 Pre-LN Transformer blocks:

```text
embed_dim = h
heads = 4
FFN = 4h
dropout = 0.1
```

The base token is the anchor / summary token.

Final:

\[
\boxed{
z_{\rm HIER}
=
z_{\rm base}
+
W_o(
T_{\rm final}[0]-z_{\rm base}
)
}
\]

where \(W_o\) is small-init residual output projection.

This ensures a direct strong-parent path.

---

# 21. B3-control — H1-HIERARCHICAL ATTENTION

Same architecture and parameters.

All factor hop tokens are independent H1 transforms.

---

# 22. B4 — BASE + RESIDUAL-READOUT-ONLY CONTROL

No new graph evidence.

Input only:

\[
z_{\rm base}
\]

through a parameter-matched residual MLP.

Purpose:

\[
\boxed{
\text{Is any gain just because the final readout is deeper?}
}
\]

Match B2/B3 added parameter count as closely as possible.

---

# 23. Default deep supervision

For B1/B2/B3 and their H1 controls:

attach auxiliary task heads to each factor-hop expert output.

Main loss:

\[
L=
L_{task}
+
0.1L_{expert}.
\]

Auxiliary heads are removed at inference.

All compared controls use the same supervision.

---

# 24. D2.6-B execution

Run **all variants**:

```text
A0_BASE
FHC_HOP
FHC_H1
RSF_HOP
RSF_H1
HIER_HOP
HIER_H1
READOUT_ONLY
```

on:

```text
Movies
Toys
Grocery
ele-fashion
Reddit-S
```

and:

```text
seeds42/43/44.
```

Total formal coverage:

\[
8\times5\times3=120
\]

runs.

No single-seed screen.

---

# 25. Frozen-parent training

In D2.6-B:

```text
A0 fully frozen
side experts train
side readout train
aux heads train
classifier train
```

Do not unfreeze A0 yet.

This is deliberate:

\[
\boxed{
\text{first prove the side evidence adds value without changing the parent}
}
\]

---

# 26. Main performance metrics

For every run report:

```text
Val Accuracy
Val Macro-F1
per-class F1
best epoch
population std ddof=0
params
peak memory
runtime
```

All deltas are paired by dataset/seed.

---

# 27. Mechanism-specific verdict

For each HOP candidate:

\[
\Delta_{\rm content}
=
Candidate_{HOP}
-
Candidate_{H1}.
\]

### CONTENT GO

M/T/G macro:

\[
\Delta Acc_{\rm content}\ge+0.30pp
\]

and:

\[
\Delta F1_{\rm content}\ge+0.20pp
\]

with:

```text
>=2/3 datasets positive
positive dataset >=2/3 seeds positive
```

This is the minimum evidence required to say H0/H1/H2 content matters.

---

# 28. Strong-parent integration verdict

For each HOP candidate:

\[
\Delta_{\rm parent}
=
Candidate_{HOP}
-
A0_{\rm matched}.
\]

### INTEGRATION GO

M/T/G macro:

\[
\Delta Acc_{\rm parent}\ge+0.30pp
\]

and:

\[
\Delta F1_{\rm parent}\ge+0.20pp.
\]

Additionally:

```text
>=2/3 target datasets mean positive
positive dataset >=2/3 seeds positive
```

Guards:

```text
ele-fashion mean Acc >= A0_matched -0.20pp
Reddit-S mean Acc >= A0_matched -0.20pp

each guard Macro-F1 >= A0_matched -0.50pp
```

---

# 29. Final-quality verdict vs formal A0

Additionally compare to formal A0.

### FINAL GO

M/T/G macro:

\[
Candidate-A0_{\rm formal}\ge+0.20pp
\]

Accuracy and:

\[
\Delta MacroF1\ge+0.20pp.
\]

A candidate can be:

```text
mechanistically valid
but not yet final-quality
```

if it beats the matched control but not formal A0.

Do not conflate these.

---

# 30. D2.6-C — Causal evidence-usage audit

For every top HOP candidate, at the trained best checkpoint, evaluate without retraining:

### FULL

normal model.

### H2-ZERO

set H2 expert/token output to zero.

### H2->H1

replace the H2 token input with H1 while keeping the H2 expert slot active.

This is more distribution-preserving than zero ablation.

### H2-SHUFFLE

fixed node permutation:

```text
seed=20260904
```

replace \(H_2(i)\) by \(H_2(\pi(i))\).

### PT-H2-OFF

only remove/replace the Pt-H2 branch.

### C-H2-OFF / PV-H2-OFF

factor-specific H2 contribution.

---

# 31. Causal-use interpretation

Strong H2 usage requires more than zero-ablation.

At least two must hold:

\[
FULL-(H2\to H1)\ge+0.20pp
\]

\[
FULL-H2_{\rm shuffle}\ge+0.20pp
\]

\[
FULL-PtH2_{\rm off}\ge+0.20pp
\]

on M/T/G macro.

If H2-zero is large but H2->H1 and shuffle are tiny:

```text
branch-presence dependence
```

rather than specific H2 information.

---

# 32. Base-preservation diagnostics

For frozen-A0 variants:

side-off must recover:

\[
\boxed{z_{\rm base}}
\]

exactly or within floating-point tolerance.

At best checkpoint report:

```text
CKA(z_final, z_base)
mean cosine
relative L2
side/base norm ratio
```

A candidate that gains performance while preserving a clear strong-base path is preferred.

---

# 33. Readout-utilization diagnostics

For B1/B2/B3 record:

### Factor contribution

ablate each:

```text
C side summary
Pt side summary
Pv side summary
```

### Hop contribution

H0/H1/H2.

### Attention

For B3:

```text
hop-attention matrices
cross-factor attention matrices
base-token attention
```

Attention nonuniformity alone is NOT success evidence.

### Gradient sensitivity

At best checkpoint:

\[
\|\partial z/\partial s_C\|,
\|\partial z/\partial s_{Pt}\|,
\|\partial z/\partial s_{Pv}\|.
\]

---

# 34. D2.6-D — Controlled parent adaptation

Only the top two frozen-A0 candidates may enter.

Use exact same:

```text
A0 checkpoint
side branch init
classifier init
seed
```

Compare three schedules.

---

# 35. S0 — FROZEN

The D2.6-B setting.

---

# 36. S1 — READOUT-ADAPT

Epoch 1–30:

```text
A0 frozen
side branch + classifier train
```

Epoch 31+:

```text
unfreeze A0 final fusion / final graph-readout layers only
P0 factorizer frozen
early A0 graph modules frozen
```

Parent LR:

```text
1e-4
```

Side LR:

```text
1e-3
```

---

# 37. S2 — GRAPH+READOUT-ADAPT

Epoch 1–30 frozen parent.

Epoch 31+:

```text
unfreeze A0 graph transformation/readout blocks
keep P0 semantic ownership factorizer frozen
```

Parent LR:

```text
1e-4
```

Side LR:

```text
1e-3
```

---

# 38. Optional S3 — P0 low-LR adaptation

Only if:

```text
S2 > S0 by >= +0.20pp
```

and ownership health remains stable.

Then:

```text
P0 factorizer lr=1e-5
```

after epoch 60.

This is optional.

Do not enter automatically.

---

# 39. Parent drift diagnostics

For S1/S2/S3 report:

```text
CKA current-parent-z vs frozen-A0-z
relative L2
P0 ownership similarities
C-Pt/C-Pv/Pt-Pv overlap
```

The goal is:

\[
\boxed{
\text{adapt enough to exploit side evidence,
without destroying the strong semantic anchor}
}
\]

---

# 40. D2.6-E — Deep-supervision confirmation

For the final top candidate only:

compare full 3-seed:

```text
lambda_aux = 0
lambda_aux = 0.1
```

on M/T/G.

If the candidate is close to final GO, also guards.

Do not sweep more lambda values in this stage.

Deep supervision is retained only if:

\[
+0.20pp
\]

macro on either Acc or F1 with no safety harm.

---

# 41. D2.6-F — Optional interaction-token extension

Do **not** enter unless a hop/readout candidate already achieves:

```text
CONTENT GO
AND
INTEGRATION GO
```

This extension asks whether the same evidence-preserving readout can also use the
validated PRODDIFF interaction cells.

---

# 42. Interaction token representation

Do not source-mean before the readout.

Take the strongest 9 source-target cell outputs:

\[
\Delta^{a\to b}.
\]

Project each to token space.

Either:

```text
9 interaction tokens
```

or three target groups with internal attention.

Fuse them only inside the already successful hierarchical readout.

Compare:

```text
HOP-only
HOP + INTERACTION
```

with the same parent.

No separate relation router.

This stage is optional and should not distract from proving the hop/readout route.

---

# 43. D2.6 final route matrix

## Route R1 — Strong-Parent Multi-Hop Readout

Enter if:

```text
CONTENT GO
INTEGRATION GO
FINAL GO or near-FINAL
H2-specific usage confirmed
```

Then R2D3 can consolidate the paper model around:

\[
\boxed{
\textbf{Semantic Ownership}
\times
\textbf{Evidence-Preserving Multi-Scale Readout}
}
\]

---

## Route R2 — Readout-only bottleneck

If:

```text
READOUT_ONLY improves strongly
but HOP-H1 content difference is weak
```

then the real lesson is:

```text
base fusion/readout was weak
```

not multi-hop.

Build a stronger generic semantic-graph backbone before claiming a new mechanism.

---

## Route R3 — Structured content is real but A0 integration still fails

If:

```text
HOP > H1-control strongly
H2-specific usage strong
but HOP <= A0
even with strong-parent residual integration
```

then:

\[
\boxed{
\textbf{post-aggregation evidence is real but not sufficient for final gain}
}
\]

At that point formally enter:

\[
\boxed{
\textbf{Semantic-Ownership-Aware Neighbor Utility Learning}
}
\]

before aggregation.

This is the clean entry condition for Route E.

---

## Route R4 — Interaction extension

Only after R1.

Not before.

---

# 44. Hypothesis ledger statuses

Use only:

```text
SUPPORTED
STRONGLY_SUPPORTED
CONDITIONAL
WEAK
CLOSED
OPEN
```

At minimum track:

```text
A0 strong-parent preservation
architecture replacement tax
fusion/readout bottleneck
generic capacity bottleneck
structured hop-specific evidence
H2-specific causal usage
factor-local hop attention
cross-factor hierarchical readout
expanded no-compression representation
residual side integration
deep supervision
parent adaptation
neighbor utility learning
interaction-token extension
```

---

# 45. Required outputs

```text
outputs/perf_r2d26/
  audit/
  no_compression/
  integration/
  causal_usage/
  parent_adapt/
  deep_supervision/
  interaction_optional/
  summary/
```

Core files:

```text
R2D26_AUDIT.md

no_compression_results.csv
R2D26_NO_COMPRESSION_REPORT.md

integration_results.csv
integration_controls.csv
integration_resources.csv
integration_attention.csv
R2D26_INTEGRATION_REPORT.md

causal_usage.csv
base_preservation.csv
R2D26_CAUSAL_REPORT.md

parent_adapt_results.csv
parent_drift.csv
R2D26_PARENT_ADAPT_REPORT.md

deep_supervision_results.csv
R2D26_DEEP_SUP_REPORT.md

R2D26_MASTER_TABLE.csv
R2D26_HYPOTHESIS_LEDGER.csv
R2D26_FINAL_DIAGNOSIS.md
```

---

# 46. Prompt 1 — Audit + infrastructure

```text
进入 R2-Design-2.6：Strong-Parent Readout Integration。

上一阶段人工审查结论：

1. A0 继续作为 Performance Parent。
2. D2.5 的 graph evidence 在 V/LN/residual 中基本保留，
   首个主要 collapse 出现在 final factor fusion/readout。
3. Generic width 不是答案：WIDE_B0 很差。
4. Structured H0/H1/H2 content 是真实的：
   structured candidates > H1-only parameter controls；
   hop_attention > parameter-identical h1_attention。
5. deep supervision 是稳定训练 lever。
6. D2.5 structured candidates绝对性能不够，部分原因可能是
   candidate替换了已经很强的 parent computation，产生 architecture tax。
7. 本阶段不能直接跳 Neighbor Utility；
   必须先测试：
   strong A0 base path + side evidence + non-collapsing readout。

所有正式实验 seeds42/43/44。
Val only。
No Test。

本 Prompt 只实现/audit，不跑正式实验。

请先审查当前 main 分支最新实现和 formal A0 checkpoint/source。
必须建立：

A. StrongParentAdapter：
   - load A0
   - expose pre-graph factors C/Pt/Pv
   - expose z_base
   - frozen parent forward
   - side-off exact z_base reproduction

B. Hop evidence：
   H0=F
   H1=P F
   H2=P^2 F
   independent E_{f,k} experts

C. H1-only exact architecture controls：
   3 independent H1 transforms replacing H0/H1/H2.

D. Readouts：
   1. no_compression_concat
   2. factor_hop_concat
   3. residual_side_fusion
   4. base_anchored_hier_attention
   5. readout_only_control

E. Default expert deep supervision lambda=0.1，
   inference removes aux heads.

F. causal overrides：
   H2-zero
   H2->H1
   H2-shuffle(seed20260904)
   factor-specific H2-off

G. matched classifier init replay。

Suggested files：

src/models/biaxis_r2_strong_parent.py
src/models/biaxis_r2_strong_parent_components.py
src/analysis/perf_r2d26_utils.py

scripts/
 perf_r2d26_no_compression.py
 perf_r2d26_integration.py
 perf_r2d26_causal.py
 perf_r2d26_parent_adapt.py
 perf_r2d26_deepsup.py
 summarize_perf_r2d26.py

Tests at minimum：

1. frozen A0 full path reproduction;
2. side-off exact z_base;
3. H1 control never accesses H0/H2 as semantic hop evidence;
4. HOP and H1-control parameter parity for each architecture;
5. H2->H1 only changes the intended token;
6. shuffle deterministic;
7. factor-specific ablation correct;
8. aux heads absent from inference output;
9. no Test access;
10. all diagnostics finite;
11. exact classifier init replay;
12. A0 weights unchanged in frozen mode.

输出：
outputs/perf_r2d26/audit/R2D26_AUDIT.md

完成后停止。
```

---

# 47. Prompt 2 — No-compression strong-parent diagnosis

```text
执行 D2.6-A。

Datasets：
Movies/Toys/Grocery/ele-fashion/Reddit-S
Seeds：
42/43/44

Val only。
No Test。

Variants：

A0_BASE：
frozen A0 z_base + fresh linear classifier。

NC_HOP：
[z_base | 9 independent H0/H1/H2 expert tokens]
不压回256维。
直接训练 matched classifier。
A0 frozen。
expert deep supervision lambda=0.1。

NC_H1：
完全相同 architecture/dimension/params，
但9 tokens全部来自independent H1 transforms。

同 dataset/seed：
A0_BASE/NC_HOP/NC_H1
必须复用 exact classifier init policy。

训练：
experts/head lr1e-3
wd1e-4
warmup10+cosine
300ep
patience30
best ValAcc

输出：
Acc
Macro-F1
per-class F1
params
memory
runtime

关键 paired delta：

NC_HOP - NC_H1
NC_HOP - A0_MATCHED
NC_HOP - A0_FORMAL

CONTENT SUPPORT：
M/T/G
Acc >= +0.30pp vs NC_H1
F1 >= +0.20pp
>=2/3 dataset mean positive
positive dataset >=2/3 seeds positive。

若 NC_HOP 与 NC_H1 同幅提升：
标 GENERIC EXPANDED READOUT。

输出：
outputs/perf_r2d26/no_compression/
 no_compression_results.csv
 R2D26_NO_COMPRESSION_REPORT.md

不要据此直接设计新模型。
停止。
```

---

# 48. Prompt 3 — Strong-parent integration matrix

```text
执行 D2.6-B。

全部：
Movies/Toys/Grocery/ele-fashion/Reddit-S
× seeds42/43/44。
Val only。
No Test。

A0 全冻结。

Variants：

1 A0_BASE

2 FHC_HOP
   factor-local hop attention over H0/H1/H2
   z=[z_base|s_C|s_Pt|s_Pv]
   不压回h

3 FHC_H1
   exact architecture control
   tokens=independent H1/H1/H1

4 RSF_HOP
   factor-local hop attention
   side=ResidualFusion([s_C|s_Pt|s_Pv])
   z_final=z_base+R_side(side)
   no scalar gate
   final residual projection small nonzero init std1e-3

5 RSF_H1
   exact H1-only control

6 HIER_HOP
   factor-local hop attention
   project s_C/s_Pt/s_Pv to h
   tokens=[z_base,q_C,q_Pt,q_Pv]
   2 Pre-LN Transformer blocks
   h dim, 4 heads, FFN4h, dropout.1
   z_final=z_base+small-init residual of final base token

7 HIER_H1
   exact H1-only control

8 READOUT_ONLY
   no new graph evidence
   parameter-matched residual MLP on z_base

所有 HOP/H1 variants：
expert deep supervision lambda=.1。

训练：
A0 frozen
side/readout/head train
lr1e-3
wd1e-4
warmup10+cosine
300/patience30
best ValAcc

输出：
integration_results.csv
integration_controls.csv
integration_resources.csv
integration_attention.csv

必须比较：

HOP - corresponding H1 control
HOP - A0_MATCHED
HOP - A0_FORMAL
HOP - READOUT_ONLY

CONTENT GO：
M/T/G macro:
Acc HOP-H1 >=+0.30pp
F1 HOP-H1 >=+0.20pp
稳定性满足计划。

INTEGRATION GO：
M/T/G:
Acc HOP-A0_MATCHED >=+0.30pp
F1 >=+0.20pp
稳定。

FINAL GO：
M/T/G:
Acc HOP-A0_FORMAL >=+0.20pp
F1 >=+0.20pp
guards safe。

Guards：
ele/Reddit
Acc >= A0_MATCHED-0.20pp
F1 >= A0_MATCHED-0.50pp。

输出：
R2D26_INTEGRATION_REPORT.md

不要自动进入 parent unfreeze。
停止。
```

---

# 49. Prompt 4 — Causal evidence-usage audit

```text
读取 D2.6-B。

最多选择 top3 HOP candidates：
优先 CONTENT GO，
其次 A0_MATCHED增益，
再看F1 safety/stability。

不重新训练。

在 best checkpoint 进行：

FULL
H2_ZERO
H2_TO_H1
H2_SHUFFLE(seed=20260904)
PT_H2_OFF
C_H2_OFF
PV_H2_OFF

全部5数据集×3 seeds。

输出：

paired Acc/F1 drop
factor-specific H2 contribution
side/base norm ratio
CKA(z_final,z_base)
relative L2
mean cosine
readout/attention statistics
gradient sensitivity to factor summaries

Strong H2-specific usage：
至少两项在M/T/G macro >=+0.20pp：

FULL-H2_TO_H1
FULL-H2_SHUFFLE
FULL-PT_H2_OFF

如果只有 H2_ZERO 很大：
判 branch-presence dependency，
不能判 H2-specific utility。

输出：
outputs/perf_r2d26/causal_usage/
 causal_usage.csv
 base_preservation.csv
 R2D26_CAUSAL_REPORT.md

完成后停止。
```

---

# 50. Prompt 5 — Controlled parent adaptation

```text
只选择 D2.6-B/C top2 candidate。

对每 dataset/seed 保存并复用 exact：
A0 checkpoint
side init
classifier init

比较：

S0 FROZEN
A0全冻结。

S1 READOUT_ADAPT
epoch1-30 frozen
epoch31+ unfreeze A0 final fusion/readout only
parent lr1e-4
side lr1e-3
P0 frozen。

S2 GRAPH_READOUT_ADAPT
epoch1-30 frozen
epoch31+ unfreeze A0 graph transformation/readout blocks
P0 semantic factorizer frozen
parent lr1e-4
side lr1e-3。

第一轮不是seed42 screen：
直接 Movies/Toys/Grocery × seeds42/43/44。

如果任一 schedule 相比 S0 >=+0.20pp macro且安全，
再跑 ele-fashion/Reddit-S × 3 seeds。

记录：

Acc/F1
parent representation drift
CKA to frozen A0
P0 ownership health
parameter update ratio
best epoch
gradient norms

S3 P0 low-LR adaptation：
只有 S2>S0 >=+0.20pp 且 ownership stable 才允许。
epoch60+ P0 lr1e-5。

输出：
outputs/perf_r2d26/parent_adapt/
 parent_adapt_results.csv
 parent_drift.csv
 R2D26_PARENT_ADAPT_REPORT.md

完成后停止。
```

---

# 51. Prompt 6 — Deep-supervision confirmation

```text
选择最终 top1 candidate/schedule。

正式比较：
deep supervision lambda=0
vs
lambda=0.1

Movies/Toys/Grocery × seeds42/43/44。

若 candidate 接近/达到 FINAL GO，
再运行 guards × 3 seeds。

所有其它 init/schedule严格相同。

输出：
Acc/F1
per-class F1
expert output norm
expert gradient norm
H2_TO_H1 drop
H2_SHUFFLE drop

保留 deep supervision 的条件：
macro Acc 或 F1 >=+0.20pp
且没有 safety degradation。

输出：
outputs/perf_r2d26/deep_supervision/
 deep_supervision_results.csv
 R2D26_DEEP_SUP_REPORT.md

停止。
```

---

# 52. Prompt 7 — Optional interaction-token extension

```text
只有当前 top hop/readout model 同时达到：

CONTENT GO
AND
INTEGRATION GO

才执行。

不要新建 relation router。

复用同一个 strong-parent readout，
增加 PRODDIFF 9 source-target cell tokens。

不要先 source-mean。

训练：
HOP_ONLY
vs
HOP_PLUS_INTERACTION

Movies/Toys/Grocery × seeds42/43/44。

如果 HOP_PLUS_INTERACTION >= HOP_ONLY +0.20pp
且稳定/F1安全，
再跑 guards。

必须做 fixed permutation mismatch：
破坏 source-target correspondence。

输出：
interaction_extension_results.csv
interaction_mismatch.csv
R2D26_INTERACTION_EXTENSION_REPORT.md

否则 interaction继续保持 secondary。
停止。
```

---

# 53. Prompt 8 — Final synthesis

```text
R2-Design-2.6 所有允许阶段完成。

不要新实验。
不要 Test。

读取：
audit
no_compression
integration
causal_usage
parent_adapt
deep_supervision
interaction_optional(if entered)

生成：

outputs/perf_r2d26/summary/
 R2D26_MASTER_TABLE.csv
 R2D26_HYPOTHESIS_LEDGER.csv
 R2D26_FINAL_DIAGNOSIS.md

必须回答：

1. strong A0 direct path是否消除了architecture replacement tax？
2. no-compression HOP是否显著优于H1-only control？
3. structured hop content在A0 parent上是否仍成立？
4. FHC/RSF/HIER哪个readout最好？
5. READOUT_ONLY是否已经解释主要增益？
6. residual/hierarchical readout是否真正恢复D2.5丢失的utility？
7. candidate是否超过A0_MATCHED？
8. candidate是否超过A0_FORMAL？
9. H2_TO_H1 / H2_SHUFFLE是否证明specific H2 content？
10. Pt-H2是否仍是最重要factor-hop？
11. deep supervision是否稳定有益？
12. parent unfreeze是否改善还是破坏strong parent？
13. P0 ownership是否保持？
14. generic readout capacity与structured evidence如何区分？
15. 是否值得正式进入 R2-Design-3 模型整合？
16. 或者是否已经满足进入 Semantic-Ownership-Aware Neighbor Utility Learning 的条件？

最终 route：

R1 Strong-Parent Multi-Hop Readout
R2 Stronger Generic Readout
R3 Neighbor Utility Learning
R4 Optional Hop+Interaction Hybrid

给出：
R2-Design-2.6 = PASS / PARTIAL / NO-GO

不要设计最终paper模型。
等待人工/ChatGPT审查。
```

---

# 54. Completion package

完成后返回：

```text
outputs/perf_r2d26/audit/
outputs/perf_r2d26/no_compression/
outputs/perf_r2d26/integration/
outputs/perf_r2d26/causal_usage/
outputs/perf_r2d26/parent_adapt/
outputs/perf_r2d26/deep_supervision/
outputs/perf_r2d26/interaction_optional/   # if entered
outputs/perf_r2d26/summary/

R2D26_FINAL_DIAGNOSIS.md
R2D26_MASTER_TABLE.csv
R2D26_HYPOTHESIS_LEDGER.csv

no_compression_results.csv
integration_results.csv
integration_controls.csv
causal_usage.csv
base_preservation.csv
parent_adapt_results.csv
parent_drift.csv
deep_supervision_results.csv

latest GitHub commit
```

---

# 55. Most important discipline

This stage must not repeat the old pattern:

```text
discover useful evidence
→ replace the entire parent
→ candidate falls below A0
→ conclude evidence is useless
```

Instead:

\[
\boxed{
\textbf{Keep the strong parent intact and make new evidence prove incremental value.}
}
\]

A new mechanism is accepted only if it satisfies **both**:

\[
\boxed{
\textbf{mechanism-specific gain over an architecture-identical H1 control}
}
\]

and:

\[
\boxed{
\textbf{incremental gain over the strong A0 parent}
}
\]

with three-seed stability and guard safety.

Only if this fails after non-collapsing readout should the research formally
move upstream to pre-aggregation neighbor utility learning.
