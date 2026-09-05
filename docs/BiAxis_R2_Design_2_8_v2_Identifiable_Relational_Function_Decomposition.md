# Bi-Axis R2-Design-2.8 v2
## Relational Function Decomposition with Identifiability Controls

**Repository:** `CrisRipper777/0901`  
**Previous stage:** `R2-Design-2.7 = PARTIAL-POSITIVE / causal attribution requires repair`  
**Protocol:** all formal experiments use **seeds 42/43/44**; Val-only; **No Test**.  
**Primary strong parent:** accepted A0 / `biaxis_final` checkpoint.  
**Resource policy:** do not enforce a lightweight-model constraint. Added capacity is allowed when it corresponds to a clearly defined relational function and is accompanied by matched controls.

---

# 0. Why this v2 replaces the previous R2D2.8 plan

The previous R2D2.8 decomposition was directionally correct:

\[
m_i^b
=
\sum_a
r_i^{ab}
\sum_{j\in N(i)}
\pi_{ji}^{ab}
\mathcal O_{ji}^{ab}(F_j^a).
\]

However, this form still permits substantial **functional leakage**:

- \(r\) and \(\mathcal O\) can both change message magnitude;
- \(\pi\) changes both neighbor identity and the norm of the weighted sum;
- a source-channel mixer can absorb part of \(r\);
- if previously validated components are jointly retrained while testing a new component, attribution becomes ambiguous.

Therefore v2 introduces explicit **identifiability controls**, staged freezing, and a normalized operator diagnostic.

The goal is not mathematical identifiability in the strict statistical sense. The goal is:

\[
\boxed{
\textbf{functional identifiability through intervention and controlled freedom}
}
\]

so that every claimed relation function corresponds to a specific experimentally isolated degree of freedom.

---

# 1. Unified relational-function formulation

Use the diagnostic formulation:

\[
\boxed{
m_i^b
=
\sum_a
\lambda_i^{ab}
\, r_i^{ab}
\sum_{j\in N(i)}
\pi_{ji}^{ab}
\,\widehat{\mathcal O}_{ji}^{ab}(U_aF_j^a)
}
\]

with:

\[
\sum_a\lambda_i^{ab}=1,
\qquad
\sum_{j\in N(i)}\pi_{ji}^{ab}=1.
\]

The four functions are:

| object | shape | semantic role |
|---|---:|---|
| \(r_i^{ab}\) | scalar in \([0,1]\) | **Exposure:** how much graph information from source factor \(a\) enters target factor \(b\) |
| \(\pi_{ji}^{ab}\) | scalar per edge, simplex over \(N(i)\) | **Composition:** which neighbors contribute |
| \(\widehat{\mathcal O}_{ji}^{ab}\) | \(d\to d\) operator | **Operator:** how the incoming feature content is transformed |
| \(\lambda_i^{ab}\) | scalar, simplex over source factors \(a\) | **Channel integration:** how source-factor transfer channels are combined |

The base semantic content is:

\[
v_j^a = U_aF_j^a,
\]

where \(U_a\) is a source-factor transform shared across target factor \(b\).

---

# 2. These quantities are NOT free per-node/per-edge parameters

The trainable parameters are the parameters of shared predictor networks.

For example:

\[
r_i^{ab}
=
\sigma(g_r(F_i^b,e_a,e_b;\theta_r)).
\]

\[
s_{ji}^{ab}
=
g_\pi(F_i^b,F_j^a,e_a,e_b;\theta_\pi),
\qquad
\pi_{ji}^{ab}
=
Softmax_{j\in N(i)}(s_{ji}^{ab}).
\]

\[
\mathcal O_{ji}^{ab}
=
g_O(F_i^b,F_j^a,e_a,e_b;\theta_O).
\]

\[
\lambda_i^{ab}
=
Softmax_a(g_\lambda(F_i^b,e_a;\theta_\lambda)).
\]

Never create a free learnable table indexed directly by:

```text
node id
edge id
```

for \(r,\pi,\lambda\), or operator routing coefficients.

---

# 3. Core identifiability rules

## Rule I — Exposure is the only explicit graph-amplitude variable during exposure tests

When testing \(r\):

```text
pi = uniform
operator = source-linear U_a
lambda = 1/3
```

Only \(r\) may vary.

---

## Rule II — Composition uses real-neighbor softmax only

When testing \(\pi\):

\[
\sum_{j\in N(i)}\pi_{ji}=1.
\]

There is no null token inside composition softmax.

The chosen exposure \(r\) is kept identical across all composition variants.

---

## Rule III — Channel integration is normalized over source channels

For learned channel mixing:

\[
\lambda_i^{ab}
=
Softmax_a(q_i^{ab}).
\]

Thus:

\[
\sum_a\lambda_i^{ab}=1.
\]

This prevents the channel mixer from becoming an unconstrained second exposure gate.

---

## Rule IV — Operator diagnostic must separate “content” from “magnitude”

For the primary operator attribution experiment, the transformed message must be norm-matched to the base source payload.

Let:

\[
\tilde v_{ji}^{ab}
=
\mathcal O_{ji}^{ab}(v_j^a).
\]

Define:

\[
\boxed{
\widehat v_{ji}^{ab}
=
\frac{\tilde v_{ji}^{ab}}
{\|\tilde v_{ji}^{ab}\|_2+\epsilon}
\cdot
\|v_j^a\|_2.
}
\]

Use \(\widehat v\) in the primary operator diagnostic.

This makes the operator mainly responsible for **feature direction/content**, while \(r\) remains the explicit graph-amplitude variable.

If a norm-preserving operator is GO, an unrestricted version may be tested later as a secondary performance variant.

---

## Rule V — Stage-wise freezing for attribution

When testing a new relational function, previously selected functions are frozen in the main attribution experiment.

Example:

```text
Exposure stage:
train r only (+ U_a/head as defined)

Composition stage:
load E*;
freeze exposure predictor;
train composition predictor

Channel stage:
load E* + C*;
freeze exposure/composition;
train channel mixer

Operator stage:
load E* + C* + M*;
freeze exposure/composition/channel;
train operator
```

After a mechanism receives GO, a separate integrated co-training confirmation may be run.

Do not use joint co-training as the first evidence for a mechanism.

---

# 4. Main stage map

```text
D2.8-0  Repair causal machinery + exact old-model factorization
        ↓
D2.8-A  Re-evaluate neighbor identity with repaired interventions
        ↓
D2.8-B  Exposure decomposition
        ↓
D2.8-C  Composition decomposition with exposure frozen
        ↓
D2.8-D  Source-channel integration with E*/C* frozen
        ↓
D2.8-E  Functional operators with E*/C*/M* frozen
        ↓
D2.8-F  Controlled co-training + factorial complementarity
        ↓
D2.8-G  Strong-parent confirmation + guards
        ↓
D2.8-H  Final synthesis
```

---

# 5. D2.8-0 — Repair D2.7 causal attribution

## 5.1 Correct within-target shuffle

Do not use a float composite key.

Implement:

1. integer sort/group by `dst`;
2. exact integer segment boundaries;
3. deterministic local `randperm(deg_i)` for each target with degree > 1;
4. permute scores only inside the same target segment;
5. restore original edge order.

Seed:

```text
20260904
```

### Mandatory tests

For every target:

```text
sorted histogram before == sorted histogram after
```

No score crosses target groups.

Among edges whose target degree > 1:

```text
changed-score fraction >= 80%
```

Among targets with degree > 1:

```text
non-identity permutation fraction >= 95%
```

A no-op implementation must fail the test.

---

## 5.2 Correct per-target ranking interventions

Implement:

```text
REMOVE_TOP_PER_TARGET_10/25/50
REMOVE_RANDOM_PER_TARGET_10/25/50
REMOVE_BOTTOM_PER_TARGET_10/25/50
KEEP_TOP_PER_TARGET_25/50
```

Selection must happen independently inside each target's real neighborhood.

Random removes the same count as top/bottom for that target.

Null/exposure is kept unchanged unless explicitly testing exposure.

---

## 5.3 Exact decomposition of old PAIR_EDGE

For the old null-augmented model:

\[
Z_i^{ab}=\sum_j e^{s_{ji}^{ab}}
\]

\[
r_i^{ab}
=
\frac{Z_i^{ab}}
{e^{s_{\emptyset,i}^{ab}}+Z_i^{ab}}
\]

\[
\pi_{ji}^{ab}
=
\frac{e^{s_{ji}^{ab}}}{Z_i^{ab}}
\]

and:

\[
\alpha_{ji}^{ab}=r_i^{ab}\pi_{ji}^{ab}.
\]

Implement:

```text
COUPLED_EQUIV
```

that explicitly computes \(r\) and \(\pi\) but reproduces old PAIR_EDGE exactly.

Require:

\[
\max|m_{\rm old}-m_{\rm factorized}|<10^{-6}.
\]

---

# 6. D2.8-A — Repaired neighbor-identity audit

No retraining initially.

Load existing best:

```text
PAIR_EDGE
TARGET_FACTOR_ONLY
```

for all five datasets × seeds 42/43/44.

Evaluate:

```text
FULL
WITHIN_TARGET_SHUFFLE_FIXED
REMOVE_TOP_PER_TARGET_10/25/50
REMOVE_RANDOM_PER_TARGET_10/25/50
REMOVE_BOTTOM_PER_TARGET_10/25/50
KEEP_TOP_PER_TARGET_25/50
```

### Identity verdict

**SUPPORTED**

if either:

\[
FULL-SHUFFLE\ge+0.30pp
\]

Accuracy macro on Movies/Toys/Grocery,

or:

\[
DROP_{top}-DROP_{random}\ge+0.20pp
\]

with Macro-F1 nonnegative and >=2/3 target dataset means positive.

**WEAK**

if consistent +0.10–0.30pp.

**NOT SUPPORTED**

only if corrected interventions remain approximately zero.

Do not carry forward the old `shuffle=0` conclusion.

---

# 7. D2.8-B — Exposure decomposition

Fix:

```text
composition = uniform real-neighbor mean
operator = U_a
channel lambda = 1/3
```

Only exposure changes.

---

## E0 — FIXED_FULL

\[
r=1.
\]

---

## E1 — NODE_EXPOSURE

One:

\[
r_i.
\]

Shared across all source/target factors.

---

## E2 — TARGET_FACTOR_EXPOSURE

Three:

\[
r_i^b.
\]

---

## E3 — SOURCE_FACTOR_EXPOSURE

Three:

\[
r_i^a.
\]

---

## E4 — PAIR_EXPOSURE

Nine:

\[
r_i^{ab}
=
\sigma(g_r(F_i^b,e_a,e_b)).
\]

This is the critical test of whether factor-pair granularity belongs to **graph exposure**, even if pair-specific neighbor ranking does not.

---

## Exposure implementation

All \(r\) values are predictor outputs, not free tables.

Recommended predictor:

```text
input:
[F_i^b, factor embeddings]

Linear(d+2t, d)
LayerNorm
GELU
Linear(d,1)
sigmoid
```

Use matched width/capacity controls when comparing simpler granularities.

---

## Exposure formal matrix

Run all five datasets × 3 seeds:

```text
A0_MATCHED
E0
E1
E2
E3
E4
TARGET_NULL_ONLY_D27
PAIR_EDGE_D27
```

### Exposure GO

Best exposure vs E0:

\[
\Delta Acc\ge+0.30pp
\]

and:

\[
\Delta F1\ge+0.20pp.
\]

### Pair-exposure specificity

E4 must beat strongest E1/E2/E3 by:

\[
+0.20pp
\]

Accuracy with nonnegative F1,

or:

\[
+0.30pp
\]

F1 with nonnegative Accuracy.

If E2≈E4, prefer target-factor exposure.

---

## Exposure diagnostics

Report:

```text
mean/std r
r quantiles
frac r<0.1
frac r>0.9
degree correlation
per factor/pair r matrix
diagonal exposure
off-diagonal exposure
TRAIN-label-only class diagnostics
```

Do not regularize \(r\) toward sparsity in the first experiment.

---

# 8. D2.8-C — Composition decomposition

Select best scientifically supported exposure formulation \(E^*\).

For the primary composition attribution:

```text
load E*
freeze all exposure parameters
```

Composition is over real neighbors only.

---

## C0 — UNIFORM_COMP

\[
\pi_{ji}=1/|N(i)|.
\]

---

## C1 — GENERIC_COMP

One edge distribution shared across semantic factors.

---

## C2 — TARGET_FACTOR_COMP

Three:

\[
\pi_{ji}^b.
\]

This is high-priority because D2.7 target-factor-only was close to full pair performance.

---

## C3 — SOURCE_FACTOR_COMP

Three:

\[
\pi_{ji}^a.
\]

---

## C4 — PAIR_COMP

Nine:

\[
\pi_{ji}^{ab}.
\]

---

## Composition normalization

\[
\pi_{ji}
=
Softmax_{j\in N(i)}(s_{ji}).
\]

No null score.

Exposure \(r\) is multiplied afterwards.

---

## Composition GO

Best learned composition vs C0:

\[
+0.20pp
\]

Accuracy or:

\[
+0.30pp
\]

Macro-F1,

with the other metric nonnegative.

### Pair-composition specificity

C4 must beat strongest C1/C2/C3 by:

\[
+0.20pp
\]

Accuracy with nonnegative F1.

---

## Composition causal confirmation

For best C:

```text
corrected within-target shuffle
per-target top/random/bottom removal
source-node shuffle
factor-id shuffle
```

Composition is not accepted solely because edge scores are nonuniform.

---

# 9. D2.8-D — Source-channel integration

Select \(E^*,C^*\).

For primary attribution:

```text
freeze exposure
freeze composition
operator = U_a
```

Now preserve the three source-factor transfer channels:

\[
m_i^{C\to b},
m_i^{Pt\to b},
m_i^{Pv\to b}.
\]

---

## M0 — SOURCE_MEAN

\[
\lambda^{ab}=1/3.
\]

---

## M1 — SOURCE_SOFTMAX_MIX

\[
\lambda_i^{ab}
=
Softmax_a(g_\lambda(F_i^b,e_a)).
\]

Then:

\[
m_i^b=\sum_a\lambda_i^{ab}m_i^{ab}.
\]

Because \(\lambda\) is a simplex, it cannot arbitrarily amplify total graph magnitude.

---

## M2 — SOURCE_CONCAT_MLP

Input:

\[
[m^{C\to b}|m^{Pt\to b}|m^{Pv\to b}].
\]

Readout:

```text
Linear(3d,2d)
LayerNorm
GELU
Dropout(.1)
Linear(2d,d)
```

---

## M3 — TARGET_QUERY_SOURCE_ATTN

Three source-channel tokens.

Query = \(F_i^b\).

Recommended:

```text
2 Pre-LN blocks
4 heads
FFN 4d
dropout .1
```

---

## Matched controls

For M2/M3:

```text
MEAN_DUP
```

Feed the same mean message to all three input slots/tokens.

Thus M2/M3 > MEAN_DUP demonstrates benefit from preserving source-channel identity rather than generic readout capacity.

---

## Channel GO

Best channel vs M0:

\[
+0.20pp
\]

on Accuracy or F1,

and:

\[
Best-MEAN\_DUP\ge+0.20pp
\]

on Accuracy or F1,

with the complementary metric nonnegative.

---

# 10. D2.8-E — Functional operator capacity

Select \(E^*,C^*,M^*\).

For the primary operator attribution:

```text
freeze exposure
freeze composition
freeze channel integration
```

Only operator freedom changes.

---

## O0 — SOURCE_LINEAR

\[
v_j^a=U_aF_j^a.
\]

No target/edge-conditioned operator.

---

## O1 — STATIC_PAIR_RESIDUAL

\[
\tilde v
=
v+\Delta T_{ab}(v).
\]

Use small/zero-init residual output so step 0 equals O0.

Apply **NormMatch** for the primary diagnostic:

\[
\widehat v
=
\frac{\tilde v}{\|\tilde v\|+\epsilon}
\|v\|.
\]

---

## O2 — TARGET-FiLM

\[
[\Delta\gamma_i^{ab},\beta_i^{ab}]
=
\phi(F_i^b,e_a,e_b)
\]

\[
\tilde v
=
(1+\Delta\gamma_i^{ab})\odot v_j^a
+
\beta_i^{ab}.
\]

Final projection zero-init.

Primary diagnostic uses NormMatch(\(\tilde v,v\)).

This tests **target-conditioned content transformation**.

---

## O3 — EDGE-FiLM

\[
[\Delta\gamma_{ji}^{ab},\beta_{ji}^{ab}]
=
\phi(
F_i^b,
F_j^a,
F_i^b\odot F_j^a,
|F_i^b-F_j^a|,
e_a,e_b
).
\]

Chunked implementation.

Final projection zero-init.

Primary diagnostic uses NormMatch.

This tests whether operator semantics must depend on the specific source-target edge, not only the target state.

---

## O4 — DYNAMIC_BASIS

Use \(K=4\) residual basis operators:

\[
B_k(v).
\]

Router:

\[
q_{ji,k}^{ab}
=
Softmax_k(\rho(F_i^b,F_j^a,e_a,e_b)).
\]

\[
\tilde v
=
v+\sum_kq_{ji,k}^{ab}B_k(v).
\]

Small-init residual bases.

Primary diagnostic uses NormMatch.

---

## Mandatory operator controls

### O4_UNIFORM

Same bases:

\[
q_k=1/K.
\]

### O4_TARGET

Same bases, router depends only on target factor/node.

### O4_EDGE_SHUFFLE

At evaluation, shuffle router assignments within target neighborhoods while leaving exposure/composition fixed.

---

## Operator GO

Best dynamic operator vs O0:

\[
+0.30pp
\]

Accuracy or:

\[
+0.40pp
\]

F1,

with the other metric nonnegative.

### Dynamic specificity

Best dynamic vs O1/static or matched capacity control:

\[
+0.20pp
\]

on at least one primary metric.

### Edge-conditioned value

O3 or edge-router O4 must beat target-conditioned counterpart by:

\[
+0.20pp
\]

before claiming edge-specific operator semantics.

---

## Operator diagnostics

For FiLM:

```text
delta-gamma mean/std
beta mean/std
feature-wise variance
pair divergence
target dependence
edge dependence
```

For basis operators:

```text
router entropy
effective expert count
basis output pairwise cosine
basis output effective rank
router JSD across factor pairs
router dependence on target/source
```

A nonuniform router with nearly identical basis outputs does not count as functional specialization.

---

# 11. Secondary unrestricted-operator test

Only if a norm-preserving operator receives GO or strong WEAK evidence.

Then run one additional comparison:

```text
NORM_PRESERVING
vs
UNRESTRICTED
```

with identical architecture.

If unrestricted improves further, report it as a performance enhancement.

Do not use unrestricted results as the primary evidence that “operator” rather than “exposure amplitude” matters.

---

# 12. D2.8-F — Controlled co-training and factorial complementarity

After B–E, choose only components that received SUPPORTED/STRONGLY_SUPPORTED status.

For each supported component, first run:

```text
frozen-previous-functions
```

attribution result.

Then run a separate integrated version where the supported components are jointly fine-tuned.

This distinguishes:

```text
mechanism existence
```

from:

```text
co-adaptation benefit
```

---

## Factorial matrix

Let:

```text
E* = best supported exposure, else E0
C* = best supported composition, else C0
M* = best supported mixer, else M0
O* = best supported operator, else O0
```

Run:

```text
F0 E0+C0+M0+O0
F1 E*+C0+M0+O0
F2 E0+C*+M0+O0
F3 E0+C0+M*+O0
F4 E0+C0+M0+O*
F5 E*+C*+M0+O0
F6 E*+C0+M0+O*
F7 E*+C0+M*+O0
F8 E0+C*+M0+O*
F9 E*+C*+M*+O*
```

Movies/Toys/Grocery × seeds42/43/44.

Promising combinations then run guards.

---

## Synergy attribution

\[
Synergy(E,C)=F5-\max(F1,F2)
\]

\[
Synergy(E,O)=F6-\max(F1,F4)
\]

\[
Synergy(E,M)=F7-\max(F1,F3)
\]

\[
Synergy(C,O)=F8-\max(F2,F4)
\]

\[
Synergy_{full}
=
F9-\max(F5,F6,F7,F8).
\]

Do not call F9 a four-module success unless the corresponding components were independently supported.

---

# 13. D2.8-G — Final strong-parent confirmation

Run the final scientifically supported candidate on:

```text
Movies
Toys
Grocery
ele-fashion
Reddit-S
× seeds42/43/44
```

Compare:

```text
A0_MATCHED
A0_FORMAL
PAIR_EDGE_D27
TARGET_FACTOR_ONLY_D27
best exposure-only model
```

Metrics:

```text
Accuracy
Macro-F1
per-class F1
mean±std ddof=0
positive seed count
params
peak GPU memory
runtime
best epoch
```

### Incremental GO

\[
Candidate-A0_{\rm matched}
\ge+0.40pp
\]

Accuracy and:

\[
\Delta F1\ge+0.30pp.
\]

### Formal GO

\[
Candidate-A0_{\rm formal}
\ge+0.20pp
\]

Accuracy with nonnegative F1.

### Guards

```text
ele-fashion Acc >= A0_FORMAL -0.20pp
Reddit-S Acc >= A0_FORMAL -0.20pp
Macro-F1 >= A0_FORMAL -0.50pp
```

No Test.

---

# 14. Route outcomes

## RFE-1 — Relational Exposure

If exposure is strong while composition/operator/channel are weak:

\[
\boxed{
\textbf{Semantic Ownership}
\times
\textbf{Relational Exposure}
}
\]

---

## RFE-2 — Exposure + Composition

If corrected neighbor identity/composition has independent value:

\[
\boxed{
\textbf{Semantic Ownership}
\times
\textbf{Adaptive Structural Composition}
}
\]

Use the simplest supported composition granularity.

---

## RFE-3 — Functional Operator Routing

Only if a dynamic operator beats:

```text
O0
static-pair operator
matched capacity control
```

and operator causal interventions are positive.

Then:

\[
\boxed{
\textbf{Semantic Ownership}
\times
\textbf{Functional Operator Routing}
}
\]

is justified.

---

## RFE-4 — Exposure + Operator

If exposure and dynamic operator are both supported but composition is weak:

\[
\boxed{
\textbf{relations are characterized primarily by
how much and how information is transferred,
rather than by selecting individual neighbors.}
}
\]

This is a particularly interesting final route.

---

## RFE-5 — Source-channel-preserving transfer

If source-channel preservation beats both mean and MEAN_DUP control:

\[
\boxed{
\textbf{premature source-factor transfer collapse is confirmed.}
}
\]

---

## RFE-6 — Reassessment

If no decomposed function survives matched causal controls:

```text
do not add more routing complexity
```

and reassess the second-axis hypothesis itself.

---

# 15. Revised execution prompts

## Prompt 1 — Implement v2 identifiability controls

```text
进入 R2-Design-2.8 v2：
Relational Function Decomposition with Identifiability Controls。

重要修订：

统一诊断公式：

m_i^b =
sum_a lambda_i^{ab} r_i^{ab}
      sum_j pi_ji^{ab}
            O_hat_ji^{ab}(U_a F_j^a)

其中：

r scalar in [0,1] = graph exposure
pi real-neighbor simplex = composition
O_hat d->d = content operator
lambda source-factor simplex = channel integration

这些都不是free node/edge parameters，
而是shared learnable functions的dynamic outputs。

新增五条identifiability纪律：

1 Exposure test:
  pi uniform
  O=U_a
  lambda=1/3

2 Composition:
  softmax只在real neighbors上；
  no null inside pi；
  load/freeze best exposure E*

3 Channel:
  lambda必须softmax over source a；
  load/freeze E*+C*

4 Operator:
  load/freeze E*+C*+M*
  primary diagnostic使用NormMatch：
  O_hat(v)=O(v)/||O(v)|| * ||v||
  将operator content effect与exposure amplitude尽量分离

5 joint co-training只在单机制GO后作为secondary confirmation，
不能作为机制首次证据。

同时完成：
- 修复within-target shuffle
- per-target top/random/bottom removal
- PAIR_EDGE -> r*pi exact COUPLED_EQUIV

严格tests：
shuffle changed fraction >=80%
nonidentity targets>=95%
coupled message max diff<1e-6
no Test
A0 untouched

更新R2D28 implementation/docs，使后续所有scripts遵守v2。

输出：
outputs/perf_r2d28/audit/R2D28_AUDIT_V2.md

不要跑正式实验。
停止。
```

---

## Prompt 2 — Repair causal attribution

```text
执行 D2.8-A。

使用已有：
PAIR_EDGE
TARGET_FACTOR_ONLY
best checkpoints。

5 datasets × seeds42/43/44。
No retraining。
No Test。

运行：
FULL
WITHIN_TARGET_SHUFFLE_FIXED
REMOVE_TOP_PER_TARGET_10/25/50
REMOVE_RANDOM_PER_TARGET_10/25/50
REMOVE_BOTTOM_PER_TARGET_10/25/50
KEEP_TOP_PER_TARGET_25/50

输出：
Acc/F1 drops
fraction_score_changed
fraction_nonidentity_targets

判：
IDENTITY SUPPORTED / WEAK / NOT SUPPORTED。

不要引用旧shuffle=0作为证据。
停止。
```

---

## Prompt 3 — Exposure decomposition

```text
执行 D2.8-B。

固定：
pi uniform
O=U_a
lambda=1/3

比较：
E0 FIXED_FULL
E1 NODE_EXPOSURE
E2 TARGET_FACTOR_EXPOSURE
E3 SOURCE_FACTOR_EXPOSURE
E4 PAIR_EXPOSURE

r必须是shared predictor output，不是free table。

全部5 datasets × 42/43/44。
A0 frozen。
No Test。

同时报告：
A0_MATCHED
TARGET_NULL_ONLY_D27
PAIR_EDGE_D27

输出Acc/F1/per-class/resources及：
r mean/std/quantiles
diag/offdiag exposure
degree corr
TRAIN-label-only diagnostics。

Exposure GO和pair-specificity按计划阈值。

选出E*。
停止。
```

---

## Prompt 4 — Composition decomposition

```text
执行 D2.8-C。

load E*，
freeze所有exposure参数。

固定：
O=U_a
lambda=1/3

pi只在real neighbors上softmax。

比较：
C0 UNIFORM
C1 GENERIC
C2 TARGET_FACTOR
C3 SOURCE_FACTOR
C4 PAIR

全部5 datasets × 3 seeds。

简化scorer做parameter matching。

对best C额外做：
corrected within-target shuffle
per-target top/random/bottom
source shuffle
factor-id shuffle。

只有performance gain + correspondence causal evidence同时成立，
才接受composition。

选出C*，否则C*=C0。
停止。
```

---

## Prompt 5 — Source-channel integration

```text
执行 D2.8-D。

load E*+C*。
freeze exposure/composition。
O=U_a。

比较：

M0 mean
M1 source-softmax lambda
M2 concat-MLP
M3 target-query source attention

M1的lambda必须sum_a=1。

M2/M3必须有MEAN_DUP matched controls。

5 datasets × 3 seeds。

只有：
candidate>M0
AND
candidate>MEAN_DUP
才接受source-channel preservation。

选M*，否则M*=M0。
停止。
```

---

## Prompt 6 — Functional operators

```text
执行 D2.8-E。

load E*+C*+M*。
freeze exposure/composition/channel。

比较：

O0 SOURCE_LINEAR
O1 STATIC_PAIR_RESIDUAL
O2 TARGET_FILM
O3 EDGE_FILM
O4 DYNAMIC_BASIS K=4

所有primary operator attribution使用NormMatch：
transformed message的L2 norm匹配base U_a(F_j^a) norm。

O1/O2/O3/O4 step0必须接近/等于O0：
zero/small-init residual。

O4 controls：
UNIFORM_BASIS
TARGET_BASIS

正式M/T/G×3 seeds；
promising者再guards×3。

Operator causal：
FiLM neutralize
operator condition shuffle
router uniformization
within-target router assignment permutation

Operator GO必须同时满足：
performance
dynamic-vs-static specificity
functional output diversity
causal usage。

若norm-preserving operator GO，
再允许一次 unrestricted-vs-norm-preserving secondary test。

选O*，否则O*=O0。
停止。
```

---

## Prompt 7 — Controlled co-training + factorial

```text
执行 D2.8-F。

只把前面SUPPORTED的E*/C*/M*/O*带入。

先保留单机制frozen-attribution结论，
然后才允许joint fine-tuning作为secondary integration。

运行F0-F9 factorial matrix。

M/T/G×3 seeds。
promising者guards×3。

输出main effects与：
E×C
E×O
E×M
C×O
full synergy。

禁止因为Full最高就宣称所有模块有效。
停止。
```

---

## Prompt 8 — Final confirmation

```text
执行 D2.8-G。

选最终scientifically-supported candidate。

5 datasets ×42/43/44。
No Test。

比较：
A0_MATCHED
A0_FORMAL
PAIR_EDGE_D27
TARGET_FACTOR_ONLY_D27
best exposure-only model

输出：
Acc/F1/per-class
mean±std ddof0
positive seed count
params/memory/runtime。

给：
INCREMENTAL GO
FORMAL GO
guard verdict。

如果只matched GO不formal GO，
可按计划只做一次controlled parent adaptation。

停止。
```

---

## Prompt 9 — Final synthesis

```text
R2D2.8 v2所有允许阶段完成。

不要新实验。
不要Test。
不要设计paper final model。

必须回答：

1 repaired neighbor identity结果？
2 exposure是否独立成立？
3 exposure最简granularity？
4 composition是否提供exposure之外的价值？
5 composition最简granularity？
6 source-cell mean是否premature collapse？
7 channel improvement是否超过MEAN_DUP？
8 static operator是否有价值？
9 target-FiLM是否有价值？
10 edge-FiLM是否进一步有价值？
11 dynamic basis是否真正specialized且task-useful？
12 norm-preserving operator是否有效？
13 unrestricted gain是否只是amplitude freedom？
14 exposure×composition synergy？
15 exposure×operator synergy？
16 channel与其它机制synergy？
17 final candidate vs A0_MATCHED？
18 vs A0_FORMAL？
19 guards？
20 第二轴应定义为：
   RFE-1 Relational Exposure
   RFE-2 Exposure+Composition
   RFE-3 Functional Operator Routing
   RFE-4 Exposure+Operator
   RFE-5 Source-channel-preserving transfer
   RFE-6 Reassessment

生成：
R2D28_MASTER_TABLE.csv
R2D28_HYPOTHESIS_LEDGER.csv
R2D28_FINAL_DIAGNOSIS.md

给R2D2.8 = PASS/PARTIAL/NO-GO。
停止。
```

---

# 16. Completion package

Return:

```text
outputs/perf_r2d28/audit/
outputs/perf_r2d28/repair/
outputs/perf_r2d28/exposure/
outputs/perf_r2d28/composition/
outputs/perf_r2d28/channel/
outputs/perf_r2d28/operator/
outputs/perf_r2d28/factorial/
outputs/perf_r2d28/confirm/
outputs/perf_r2d28/summary/

R2D28_FINAL_DIAGNOSIS.md
R2D28_MASTER_TABLE.csv
R2D28_HYPOTHESIS_LEDGER.csv

repair_results.csv
repair_shuffle_validation.csv
repair_removal.csv
exposure_results.csv
exposure_stats.csv
composition_results.csv
composition_causal.csv
channel_results.csv
channel_ablation.csv
operator_results.csv
operator_controls.csv
operator_usage.csv
operator_causal.csv
factorial_results.csv
factorial_attribution.csv
confirm_results.csv
confirm_resources.csv

latest GitHub commit
```

---

# 17. Final discipline

The purpose of R2D2.8 is **not** to make four modules all look active.

It is to identify which relational freedom is actually necessary:

\[
\boxed{
\textbf{how much}
}
\]

\[
\boxed{
\textbf{which neighbor}
}
\]

\[
\boxed{
\textbf{how to transform}
}
\]

\[
\boxed{
\textbf{how source-factor channels are integrated}
}
\]

A function is accepted only if it survives:

```text
matched control
staged freezing
causal intervention
3-seed stability
strong-parent comparison
```

The final second axis must be named **after** these results, not before.
