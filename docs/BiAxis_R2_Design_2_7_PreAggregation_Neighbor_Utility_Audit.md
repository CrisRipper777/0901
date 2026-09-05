# Bi-Axis R2-Design-2.7
## Pre-Aggregation Semantic-Ownership Neighbor Utility Audit

**Repository:** `CrisRipper777/0901`  
**Previous stage:** `R2-Design-2.6 = NO-GO` for strong-parent post-aggregation integration  
**Current route:** `R3 — Semantic-Ownership-Aware Neighbor Utility Learning`  
**Protocol:** all formal experiments use seeds `42/43/44`; Val-only; **No Test**.  
**Resource policy:** do not optimize for minimal parameter count. Use the capacity required for a scientifically clean test, but always include parameter-/architecture-matched controls.

---

# 0. Stage purpose

The R2 series has now repeatedly established the following pattern:

\[
\boxed{
\text{post-aggregation graph evidence can be real and even causally used,
yet still provide little incremental value over A0}
}
\]

R2D2.6 further showed that this remains true even when:

- A0 is preserved as a direct strong-parent path;
- H0/H1/H2 have independent experts;
- no-compression concatenation is allowed;
- residual / hierarchical readouts are used;
- deep supervision is available;
- selected A0 graph/readout modules are allowed to adapt.

Therefore R2D2.7 moves **upstream of aggregation**.

The new scientific question is:

\[
\boxed{
\textbf{
Before neighbors are mixed together,
which neighbor carries useful evidence
for which semantic-ownership factor of the target node?
}
}
\]

The stage is **not** yet the final paper model.

It is an audit/falsification stage designed to answer:

1. Is there real edge-level neighbor-utility heterogeneity?
2. Does pre-aggregation selection beat uniform aggregation?
3. Is the useful ranking factor-to-factor dependent?
4. Is semantic ownership actually necessary, or would a generic edge scorer work equally well?
5. Is pre-aggregation scoring materially better than a parameter-matched post-aggregation gate?
6. Does learned edge utility contain information not already absorbed by A0?
7. If selection works, is factor-pair transformation additionally needed?

---

# 1. Collision guardrails

This stage must remain clearly separated from nearby 2026 MAG work.

## 1.1 Do not reproduce RoleMAG

Do **not** define neighbors as:

```text
shared
complementary
heterophilous
```

or use predefined role channels.

R2D2.7 does not assign semantic role labels.

It learns a continuous task-conditioned utility:

\[
u_{ji}^{a\rightarrow b}
\]

for a **source ownership factor** \(a\) transferring to a **target ownership factor** \(b\).

---

## 1.2 Do not reproduce TMTE

Do not globally reconstruct / evolve the topology as the core mechanism.

R2D2.7 uses the **observed graph as support**.

No edge addition in the main stage.

No learned global adjacency matrix.

No anchor-based topology reconstruction.

---

## 1.3 Do not reproduce CoMAG-style semantic reliability

Do not define edge quality as raw multimodal semantic consistency.

No:

```text
cos(text_i,text_j)+cos(image_i,image_j)
```

as the core scorer.

No semantic-neighbor augmentation.

A simple semantic-similarity scorer is allowed only as a **control baseline**.

---

# 2. Core formulation

For an observed directed edge:

\[
j\rightarrow i
\]

and semantic ownership factors:

\[
a,b\in\{C,P_t,P_v\},
\]

define a pre-aggregation utility score:

\[
s_{ji}^{a\rightarrow b}
=
\psi(
F_i^b,
F_j^a,
F_i^b\odot F_j^a,
|F_i^b-F_j^a|,
e_a,e_b
).
\]

Use one **shared scorer** \(\psi\) across all 9 factor pairs, with learned source/target factor embeddings.

Recommended scorer:

```text
input: [F_i^b, F_j^a, F_i^b*F_j^a, |F_i^b-F_j^a|, emb_a, emb_b]

Linear(4d+2t, 2d)
LayerNorm
GELU
Dropout(0.1)
Linear(2d, d)
GELU
Linear(d, 1)
```

Recommended:

```text
factor-type embedding t = 16
```

No edge label supervision.

The scorer learns only through node-task CE unless explicitly stated in diagnostics.

---

# 3. Null-augmented neighbor normalization

A normal neighbor softmax forces every target to consume graph information.

R2D2.7 must allow:

\[
\text{“none of these neighbors are useful”}.
\]

For every:

\[
(i,a,b),
\]

introduce a target-conditioned null score:

\[
s_{\varnothing,i}^{a\rightarrow b}
=
\phi(
F_i^b,e_a,e_b
).
\]

Normalize over:

\[
\{\varnothing\}\cup N(i).
\]

Thus:

\[
\alpha_{ji}^{a\rightarrow b}
=
\frac{
\exp(s_{ji}^{a\rightarrow b}/\tau)
}{
\exp(s_{\varnothing,i}^{a\rightarrow b}/\tau)
+
\sum_{\ell\in N(i)}
\exp(s_{\ell i}^{a\rightarrow b}/\tau)
}.
\]

and:

\[
\alpha_{\varnothing,i}^{a\rightarrow b}
=
1-\sum_j\alpha_{ji}^{a\rightarrow b}.
\]

Use:

```text
tau = 1.0
```

throughout D2.7.

No temperature sweep.

No top-k during training.

---

# 4. Selection-only message payload

The first formal mechanism must isolate **neighbor selection**.

Do not simultaneously introduce complex factor-pair message transforms.

Use source-factor transforms:

\[
U_a:\mathbb R^d\rightarrow\mathbb R^d.
\]

One transform for each source factor:

```text
U_C
U_Pt
U_Pv
```

shared across target factor \(b\).

Recommended:

```text
Linear(d,d,bias=False)
```

or, if stronger capacity is required for all controls:

```text
Linear(d,2d) -> LN -> GELU -> Linear(2d,d)
```

but the same payload transforms must be used in every selection-control variant.

For each pair:

\[
m_i^{a\rightarrow b}
=
\sum_{j\in N(i)}
\alpha_{ji}^{a\rightarrow b}U_a(F_j^a).
\]

Target-factor message:

\[
m_i^b
=
\frac13
\sum_{a\in\{C,P_t,P_v\}}
m_i^{a\rightarrow b}.
\]

No learned source-factor mixer in the first stage.

This prevents a later gate from explaining the result.

---

# 5. Strong-parent integration

A0 remains the primary parent.

The utility branch must not replace A0.

Extract:

\[
F=[C,P_t,P_v]
\]

from A0 pre-graph factors and:

\[
z_{\rm base}=A0(x,G).
\]

Primary side representation:

\[
\boxed{
z_{\rm util}
=
[z_{\rm base}\mid m^C\mid m^{Pt}\mid m^{Pv}]
}
\]

No projection back to 256 in the initial audit.

Use a matched fresh classifier.

This is deliberate:

\[
\boxed{
\text{test edge selection before introducing another readout bottleneck}
}
\]

---

# 6. Main stage map

```text
D2.7-0  Audit + collision contract + edge scorer infrastructure
        ↓
D2.7-A  Pre-aggregation utility model matrix
        ↓
D2.7-B  Edge-utility structure & causal ranking audit
        ↓
D2.7-C  PRE vs POST aggregation timing test
        ↓
D2.7-D  Ownership-specificity audit
        ↓
D2.7-E  Selection × factor-pair transformation decomposition
        ↓
D2.7-F  Optional noise / edge-corruption stress test
        ↓
D2.7-G  Final synthesis
```

---

# 7. Formal datasets

Primary:

```text
Movies
Toys
Grocery
```

Guards:

```text
ele-fashion
Reddit-S
```

All formal experiments:

```text
seeds 42/43/44
```

No seed-42-only scientific verdict.

---

# 8. D2.7-0 — Infrastructure audit

Implement a new family:

```text
src/models/biaxis_r2_neighbor_utility.py
src/models/biaxis_r2_neighbor_utility_components.py
src/analysis/perf_r2d27_utils.py
```

Scripts:

```text
scripts/perf_r2d27_matrix.py
scripts/perf_r2d27_edge_audit.py
scripts/perf_r2d27_prepost.py
scripts/perf_r2d27_ownership.py
scripts/perf_r2d27_transfer.py
scripts/perf_r2d27_noise.py
scripts/summarize_perf_r2d27.py
```

Output root:

```text
outputs/perf_r2d27/
```

---

# 9. Edge implementation discipline

All edge computations must support chunking.

Do not materialize:

\[
[E,3,3,d]
\]

for the full graph if memory is excessive.

Required:

```text
edge_chunk_size
```

and streaming/scatter aggregation.

For each edge chunk:

1. gather target factor block;
2. gather source factor block;
3. produce 9 utility scores;
4. normalize per target / factor-pair;
5. aggregate messages.

Because softmax needs target denominators, implement either:

- two-pass chunked log-sum-exp; or
- stable segment/scatter softmax.

No approximate neighbor sampling in D2.7 unless full graph is impossible.

---

# 10. Matrix variants

All variants below use the same:

```text
A0 parent
U_a payload transforms
side representation [z_base|mC|mPt|mPv]
classifier architecture
training schedule
```

Only the neighbor-utility scorer changes.

---

# 11. U0 — A0_BASE

Frozen A0:

\[
z=z_{\rm base}.
\]

Matched fresh classifier.

---

# 12. U1 — UNIFORM

No utility scorer.

For every observed neighbor:

\[
\alpha_{ji}^{a\rightarrow b}
=
1/|N(i)|.
\]

Null mass = 0 for non-isolated nodes.

Same payload transforms and side readout as all utility candidates.

Purpose:

\[
\boxed{
\text{Does merely adding another pre-aggregation side branch help?}
}
\]

---

# 13. U2 — TARGET_NULL_ONLY

No neighbor ranking.

For a given:

\[
(i,a,b)
\]

all neighbors share the same edge logit.

Only the null score may depend on the target.

Therefore among real neighbors:

\[
\alpha_{ji}^{a\rightarrow b}
\]

is uniform.

This model can learn:

\[
\text{how much graph to use}
\]

but not:

\[
\text{which neighbor to use}.
\]

This is the clean control for **neighbor selection**.

---

# 14. U3 — GENERIC_EDGE

One utility score per observed edge:

\[
s_{ji}
\]

shared across:

```text
C/Pt/Pv
source-target factor pairs
```

Input:

\[
[z^0_i,z^0_j,z^0_i\odot z^0_j,|z^0_i-z^0_j|]
\]

where \(z^0\) is the local ownership-concatenated representation projected to d or A0 local representation.

Widen the scorer as needed so total scorer params are within ±5% of PAIR_EDGE.

Apply the same edge weights to all factor messages.

This is a **generic task-aware edge attention control**.

---

# 15. U4 — DIAG_EDGE

Three factor-specific edge rankings:

\[
C\rightarrow C,\quad
P_t\rightarrow P_t,\quad
P_v\rightarrow P_v.
\]

No off-diagonal factor transfer.

This asks whether factor awareness is enough without factor-to-factor transfer.

---

# 16. U5 — PAIR_EDGE

The main candidate.

Nine utility distributions:

\[
a\rightarrow b,
\qquad
a,b\in\{C,P_t,P_v\}.
\]

Shared \(\psi\), factor-type embeddings.

This is the first formal test of:

\[
\boxed{
\textbf{Semantic Ownership}
\times
\textbf{Pre-Aggregation Functional Utility}
}
\]

---

# 17. U6 — SEMANTIC_SIM_CONTROL

No trainable edge scorer.

Use cosine similarity:

\[
s_{ji}^{a\rightarrow b}
=
\cos(F_i^b,F_j^a).
\]

Same null normalization / payload / readout.

Purpose:

\[
\boxed{
\text{Is a simple semantic-consistency heuristic already sufficient?}
}
\]

This is a control, never the proposed mechanism.

---

# 18. D2.7-A training

Run:

```text
U0 A0_BASE
U1 UNIFORM
U2 TARGET_NULL_ONLY
U3 GENERIC_EDGE
U4 DIAG_EDGE
U5 PAIR_EDGE
U6 SEMANTIC_SIM_CONTROL
```

on:

```text
Movies/Toys/Grocery/ele-fashion/Reddit-S
×
seeds42/43/44
```

A0 frozen.

Train:

```text
side payload transforms
utility scorer/null scorer if present
fresh classifier
```

Do not unfreeze A0 in D2.7-A.

Optimizer:

```text
AdamW
lr=1e-3
wd=1e-4
warmup10 + cosine
300 epochs
patience30
best Val Accuracy
grad clip 1.0
```

No auxiliary loss.

No edge supervision.

---

# 19. D2.7-A metrics

Report:

```text
Val Accuracy
Macro-F1
per-class F1
best epoch
params
peak memory
runtime
```

Paired deltas:

```text
PAIR_EDGE - UNIFORM
PAIR_EDGE - TARGET_NULL_ONLY
PAIR_EDGE - GENERIC_EDGE
PAIR_EDGE - DIAG_EDGE
PAIR_EDGE - SEMANTIC_SIM
PAIR_EDGE - A0_MATCHED
PAIR_EDGE - A0_FORMAL
```

---

# 20. D2.7-A main verdicts

## SELECTION GO

Pre-aggregation neighbor ranking is supported if:

\[
PAIR\_EDGE-TARGET\_NULL\_ONLY
\ge +0.30pp
\]

Accuracy macro on M/T/G and:

```text
>=2/3 datasets positive
positive datasets >=2/3 seeds positive
Macro-F1 delta >= +0.20pp
```

---

## A0 INCREMENTAL GO

\[
PAIR\_EDGE-A0_{\rm matched}
\ge +0.30pp
\]

Accuracy and:

\[
\Delta MacroF1\ge+0.20pp.
\]

Same stability rule.

---

## OWNERSHIP-PAIR PRELIMINARY SUPPORT

\[
PAIR\_EDGE-GENERIC\_EDGE
\ge +0.20pp
\]

Accuracy and Macro-F1 nonnegative,

or:

\[
\Delta MacroF1\ge+0.30pp
\]

with Accuracy nonnegative.

This is only preliminary; formal ownership specificity is D2.7-D.

---

# 21. Guard safety

For any candidate that beats A0_MATCHED on M/T/G by at least +0.20pp:

require:

```text
ele-fashion Acc >= A0_MATCHED -0.20pp
Reddit-S Acc >= A0_MATCHED -0.20pp

Macro-F1 >= A0_MATCHED -0.50pp
```

---

# 22. D2.7-B — Edge-utility structure audit

Run on:

```text
top PAIR_EDGE checkpoint
all 5 datasets
all 3 seeds
```

No retraining.

---

# 23. Neighbor-ranking statistics

For each target node and factor pair:

record:

```text
real-neighbor mass
null mass
entropy over real neighbors
normalized entropy
Gini coefficient
top-10% mass
top-25% mass
effective neighbor count = exp(entropy)
```

If PAIR_EDGE weights are nearly uniform everywhere:

\[
\boxed{
\text{pre-aggregation utility mechanism is not actually active}
}
\]

even if performance changes.

---

# 24. Factor-pair diversity

For the same target node compare neighbor distributions across:

\[
a\rightarrow b.
\]

Compute:

```text
Jensen-Shannon divergence
Spearman rank correlation
top-k overlap
```

Aggregate a 9×9 relation matrix.

Important comparisons:

```text
C->C vs Pt->Pt vs Pv->Pv
Pt->C vs Pt->Pt
Pv->C vs Pv->Pv
C->Pt vs Pv->Pt
```

If all pair rankings are effectively identical:

```text
factor-pair utility is not supported
```

even if generic edge selection is useful.

---

# 25. Correlation with simple heuristics

Measure correlation between learned utility and:

```text
cos(F_i^b,F_j^a)
source degree
target degree
A0 relation probability / Gamma if exactly accessible
```

Do not use these diagnostics to train the model.

If learned utility is almost perfectly explained by cosine similarity:

```text
semantic-ownership utility novelty is weak
```

---

# 26. Train-label-only homophily analysis

Diagnostic only.

Use only edges where:

```text
target node in TRAIN
source node in TRAIN
```

Compare utility distributions for:

```text
same-label edges
different-label edges
```

No validation labels are ever used here.

This answers whether the scorer simply learns class homophily.

Do not optimize against this metric.

---

# 27. Causal ranking test: remove edges by learned utility

At the best checkpoint, **no retraining**.

For the side utility branch only, evaluate:

### REMOVE-TOP

remove the highest-scored real neighbors:

```text
10%
25%
50%
```

### REMOVE-RANDOM

same number of edges, fixed permutation seed.

### REMOVE-BOTTOM

remove the lowest-scored neighbors.

### KEEP-TOP

keep only top:

```text
25%
50%
```

and renormalize.

Strong ranking evidence:

\[
Drop_{\rm top}
>
Drop_{\rm random}
>
Drop_{\rm bottom}
\]

and:

\[
Drop_{\rm top}-Drop_{\rm random}
\ge+0.20pp
\]

M/T/G macro for at least one removal rate.

---

# 28. Utility permutation controls

No retraining:

### WITHIN-TARGET SHUFFLE

For each target/factor-pair, permute learned edge weights across that target's neighbors.

Preserves:

```text
same target
same weight histogram
same null mass
```

but destroys score-to-neighbor correspondence.

### SOURCE-NODE SHUFFLE

fixed permutation of source factor features.

### FACTOR-ID SHUFFLE

permute source/target factor IDs at scoring time.

Strong correspondence evidence:

\[
FULL-SHUFFLE\ge+0.30pp
\]

M/T/G macro.

---

# 29. D2.7-C — PRE vs POST timing test

This stage directly tests whether the location of the computation matters.

Use the best PAIR_EDGE architecture as PRE.

Build a parameter-matched POST model.

---

# 30. POST_PAIR definition

First aggregate uniformly:

\[
N_i^a
=
\frac1{|N(i)|}
\sum_jF_j^a.
\]

Then compute:

\[
g_i^{a\rightarrow b}
=
\psi(
F_i^b,
N_i^a,
F_i^b\odot N_i^a,
|F_i^b-N_i^a|,
e_a,e_b
).
\]

Message:

\[
m_i^{a\rightarrow b}
=
\sigma(g_i^{a\rightarrow b})U_a(N_i^a).
\]

Same:

```text
9 factor pairs
scorer depth
payload transforms
side representation
classifier
```

Match total side params within ±5%.

This is a modern parameter-matched reconstruction of the old post-aggregation logic.

---

# 31. PRE vs POST formal experiment

Run:

```text
PRE_PAIR
POST_PAIR
TARGET_NULL_ONLY
```

on all 5 datasets × 3 seeds.

Use exact same A0 checkpoint and classifier init policy.

### PRE-AGGREGATION GO

\[
PRE\_PAIR-POST\_PAIR
\ge+0.30pp
\]

Accuracy macro on M/T/G,

and:

```text
Macro-F1 >= +0.20pp
>=2/3 dataset means positive
positive datasets >=2/3 seeds positive
guards safe
```

This is the cleanest evidence for moving computation before aggregation.

---

# 32. D2.7-D — Ownership-specificity audit

Only enter if generic pre-aggregation neighbor selection is at least WEAK:

```text
PAIR_EDGE or GENERIC_EDGE > TARGET_NULL_ONLY
by >= +0.15pp Acc macro
```

No need for full +0.30 GO to run this diagnostic.

---

# 33. O1 — OWNERSHIP_PAIR

The full U5 model.

---

# 34. O2 — NODE_SHARED

One edge rank per edge, shared across all factors.

Parameter matched.

This is generic GAT-like task-aware ranking.

---

# 35. O3 — FACTOR_DIAG

Three diagonal utilities only.

---

# 36. O4 — SOURCE_FACTOR_ONLY

Utility depends on:

\[
a
\]

but not target factor:

\[
u_{ji}^{a}.
\]

One neighbor ranking per source factor.

---

# 37. O5 — TARGET_FACTOR_ONLY

Utility depends on:

\[
b
\]

but not source factor:

\[
u_{ji}^{\rightarrow b}.
\]

One neighbor ranking per target factor.

---

# 38. Ownership-specificity verdict

Full factor-pair utility is supported only if:

\[
OWNERSHIP\_PAIR
\]

beats the strongest of:

```text
NODE_SHARED
FACTOR_DIAG
SOURCE_FACTOR_ONLY
TARGET_FACTOR_ONLY
```

by:

\[
\ge+0.20pp
\]

Accuracy macro M/T/G and F1 nonnegative,

with:

```text
>=2/3 datasets positive
```

Additionally, factor-pair weight distributions must show nontrivial divergence in D2.7-B.

If this fails but generic edge selection works:

\[
\boxed{
\textbf{neighbor utility is real,
but Semantic Ownership is not yet the reason}
}
\]

This is a critical novelty guard.

---

# 39. D2.7-E — Selection × transformation decomposition

Only enter if:

```text
PRE_PAIR selection GO
or
PRE_PAIR-A0_MATCHED >= +0.20pp
```

This stage asks:

> Is selecting the right neighbor sufficient, or do factor pairs also require different transfer functions?

---

# 40. T0 — SHARED_PAYLOAD + UNIFORM

Uniform neighbor aggregation.

Shared source-factor transforms \(U_a\).

Baseline.

---

# 41. T1 — SHARED_PAYLOAD + PRE_PAIR

Current PAIR_EDGE selection.

Tests selection only.

---

# 42. T2 — PAIR_TRANSFORM + UNIFORM

Uniform neighbor weights.

Nine pair-specific transforms:

\[
T_{ab}(F_j^a).
\]

Recommended:

```text
Linear(d,2d)
LN
GELU
Linear(2d,d)
```

for each pair, or a parameter-efficient residual decomposition if 9 full MLPs are too large.

This tests transformation without selection.

---

# 43. T3 — PAIR_TRANSFORM + PRE_PAIR

Both:

\[
\boxed{
\text{neighbor selection}
+
\text{factor-pair transfer}
}
\]

Message:

\[
m_i^b
=
\frac13
\sum_a
\sum_{j\in N(i)}
\alpha_{ji}^{a\rightarrow b}
T_{ab}(F_j^a).
\]

---

# 44. Parameter-capacity control for T3

Implement:

```text
PAIR_TRANSFORM_H1_UNIFORM_CONTROL
```

or a duplicated-transform capacity control with the same parameter count but uniform neighbor weighting.

The formal comparison must isolate the value of learned neighbor assignment.

---

# 45. D2.7-E verdict

### Selection contribution

\[
T1-T0.
\]

### Transform contribution

\[
T2-T0.
\]

### Interaction / complementarity

\[
T3-\max(T1,T2).
\]

A true Functional Relational Transfer signal requires:

\[
T3-\max(T1,T2)
\ge+0.20pp
\]

Accuracy or F1 macro with the other metric nonnegative.

Do not call the method “functional transfer” if T3 provides no additional value beyond selection alone.

---

# 46. D2.7-F — Optional random-edge stress test

Only for a candidate that achieves:

```text
A0 incremental GO
```

or is within +0.10pp of it.

Do not use this to rescue a weak candidate.

At evaluation only, add random edges equal to:

```text
10%
25%
```

of original edge count.

Do not retrain.

Compare:

```text
A0
UNIFORM side
PRE_PAIR
```

Metrics:

```text
performance degradation
mean utility assigned to injected edges
top-k occupancy of injected edges
```

Strong noise-rejection evidence:

\[
PRE\_PAIR
\]

assigns lower utility to injected edges and degrades less than UNIFORM.

---

# 47. Optional parent adaptation

Do not unfreeze A0 during D2.7-A–E.

If a PRE candidate achieves:

```text
PRE_PAIR - A0_MATCHED >= +0.20pp
```

but falls short of final GO,

one controlled adaptation may be run:

```text
epoch1-30 A0 frozen
epoch31+ unfreeze A0 final fusion + graph operator
parent lr=1e-4
utility branch lr=1e-3
P0 ownership factorizer frozen
```

All M/T/G × 3 seeds.

No broad adaptation matrix.

If it does not add >=+0.20pp, stop.

---

# 48. What counts as success?

R2D2.7 is not successful merely because:

```text
edge scores are nonuniform
attention maps look interpretable
PAIR_EDGE has more parameters
top-k edges look semantically reasonable
```

A strong positive stage requires multiple levels.

---

# 49. Level I — Neighbor-selection evidence

\[
PAIR\_EDGE>TARGET\_NULL\_ONLY.
\]

---

# 50. Level II — Pre-aggregation evidence

\[
PRE\_PAIR>POST\_PAIR.
\]

---

# 51. Level III — Ownership specificity

\[
PAIR\_EDGE>GENERIC/DIAG/SOURCE/TARGET controls.
\]

---

# 52. Level IV — Strong-parent increment

\[
PAIR\_EDGE>A0_{\rm matched}
\]

with guard safety.

---

# 53. Level V — Functional transfer

\[
Selection+PairTransform
>
SelectionOnly
\]

and:

\[
>
PairTransformOnly.
\]

Only Level V justifies returning to the original paper phrase:

\[
\boxed{
\textbf{Semantic Ownership}
\times
\textbf{Functional Relational Transfer}
}
\]

---

# 54. Route matrix

## Route N1 — Ownership-conditioned neighbor utility

Enter R2-Design-3 if:

```text
Level I PASS
Level II PASS
Level III PASS
Level IV PASS or near-PASS
```

Then formal model design can use factor-pair pre-aggregation utility.

---

## Route N2 — Generic edge utility

If:

```text
PRE > POST
PAIR ≈ GENERIC_EDGE
```

then the real phenomenon is generic task-aware edge selection.

Do not claim semantic-ownership novelty.

Reconsider whether the paper should center on a stronger generic graph learner.

---

## Route N3 — Factor awareness but not factor-pair transfer

If:

```text
DIAG / SOURCE / TARGET factor scorer ≈ PAIR
```

then use the simpler factor-conditioned formulation.

Do not keep 9 factor-pair cells without evidence.

---

## Route N4 — Selection works, pair transforms add value

If T3 gives complementary gain:

\[
\boxed{
\textbf{Semantic-Ownership-Aware Functional Edge Transfer}
}
\]

becomes the leading formal model direction.

---

## Route N5 — Pre-aggregation route NO-GO

If:

```text
PAIR_EDGE ≈ TARGET_NULL_ONLY
and
PRE_PAIR ≈ POST_PAIR
```

then neighbor identity is not the missing variable.

At that point the entire second-axis story needs reassessment rather than adding more routers.

---

# 55. Hypothesis ledger

Track at least:

```text
Observed topology as useful support
Uniform neighbor aggregation
Target-only graph-mass control
Generic edge utility
Factor-specific neighbor utility
Factor-pair neighbor utility
Pre-vs-post aggregation timing
Semantic similarity as sufficient edge score
Null neighbor option
Within-neighborhood utility heterogeneity
Factor-pair ranking diversity
H2/multihop as downstream symptom
Selection-only transfer
Pair-specific message transform
Selection × transform complementarity
A0 incremental utility
Noise-edge rejection
RoleMAG collision risk
TMTE topology-evolution collision risk
CoMAG reliability collision risk
```

Statuses:

```text
STRONGLY_SUPPORTED
SUPPORTED
WEAK
CONDITIONAL
OPEN
CLOSED
```

---

# 56. Required outputs

```text
outputs/perf_r2d27/
  audit/
  matrix/
  edge_audit/
  prepost/
  ownership/
  transfer/
  noise_optional/
  summary/
```

Core files:

```text
R2D27_AUDIT.md

matrix_results.csv
matrix_controls.csv
matrix_resources.csv
R2D27_MATRIX_REPORT.md

edge_score_stats.csv
edge_pair_diversity.csv
edge_heuristic_corr.csv
edge_homophily_train_only.csv
edge_causal_ranking.csv
edge_shuffle_controls.csv
R2D27_EDGE_AUDIT_REPORT.md

prepost_results.csv
R2D27_PREPOST_REPORT.md

ownership_results.csv
R2D27_OWNERSHIP_REPORT.md

transfer_results.csv
transfer_ablation.csv
R2D27_TRANSFER_REPORT.md

noise_results.csv
R2D27_NOISE_REPORT.md

R2D27_MASTER_TABLE.csv
R2D27_HYPOTHESIS_LEDGER.csv
R2D27_FINAL_DIAGNOSIS.md
```

---

# 57. Prompt 1 — Audit + infrastructure

```text
进入 R2-Design-2.7：
Pre-Aggregation Semantic-Ownership Neighbor Utility Audit。

上一阶段正式结论：

1. A0 strong parent保持。
2. 多种 post-aggregation H2/interaction evidence可以真实存在甚至被模型使用，
   但无法稳定形成A0上的增量。
3. strong-parent direct path、non-compressing readout、attention/residual readout、
   deep supervision、parent adaptation均已测试，不能解释最终NO-GO。
4. 下一未充分测试变量是：
   neighbor identity在aggregation之前是否应该被选择。
5. 本阶段不设计最终paper模型，先做edge-level utility falsification。

Collision guardrails：

- 不做RoleMAG式 shared/complementary/heterophilous role分类；
- 不做TMTE式全局topology evolution；
- 不做CoMAG式“raw multimodal semantic consistency=edge reliability”；
- observed graph只作为support；
- main candidate是 continuous factor-pair utility u_{ji}^{a->b}。

所有正式实验 seeds42/43/44。
Val only。
No Test。

本Prompt只实现和审计，不跑正式实验。

实现：

src/models/biaxis_r2_neighbor_utility.py
src/models/biaxis_r2_neighbor_utility_components.py
src/analysis/perf_r2d27_utils.py

核心：

F=[C,Pt,Pv] 来自 frozen A0 pre-graph ownership factors。

score:
s_{ji}^{a->b} =
MLP([
 F_i^b,
 F_j^a,
 F_i^b*F_j^a,
 abs(F_i^b-F_j^a),
 emb_a,
 emb_b
])

shared scorer across 9 pairs。
factor embedding dim=16。

null score:
target-conditioned per (i,a,b)。

neighbor+null softmax：
tau=1.0 fixed。

selection-only payload：
U_a source-factor transform，
shared across target b。

m_i^{a->b} =
sum_j alpha_{ji}^{a->b} U_a(F_j^a)

m_i^b =
mean_a m_i^{a->b}

side representation：
[z_base | m_C | m_Pt | m_Pv]
no projection to h。

实现 variants：

A0_BASE
UNIFORM
TARGET_NULL_ONLY
GENERIC_EDGE
DIAG_EDGE
PAIR_EDGE
SEMANTIC_SIM_CONTROL

此外实现未来需要但默认不跑：

POST_PAIR
SOURCE_FACTOR_ONLY
TARGET_FACTOR_ONLY
PAIR_TRANSFORM_UNIFORM
PAIR_TRANSFORM_PRE

必须支持：

within-target edge score shuffle
source node shuffle
factor-id shuffle
top/random/bottom edge removal
top-k keep
factor-pair score export
null mass export
edge score entropy/Gini/top-k mass
edge chunking
stable segment softmax

Tests：

1 A0 parent never updated in frozen mode；
2 UNIFORM mathematically uniform；
3 TARGET_NULL_ONLY real-neighbor ranking uniform；
4 PAIR_EDGE produces 9 pair scores；
5 GENERIC_EDGE produces one ranking shared across factors；
6 DIAG_EDGE off-diagonal messages exactly zero；
7 null+neighbor weights sum to1；
8 isolated nodes all-null and finite；
9 edge shuffle preserves per-target weight histogram；
10 H1/post model aggregates before scoring；
11 no Test access；
12 factor order C/Pt/Pv；
13 chunked vs unchunked numerical equivalence on toy graph；
14 no predefined RoleMAG role labels；
15 no edge addition/topology reconstruction。

输出：
outputs/perf_r2d27/audit/R2D27_AUDIT.md

完成后停止。
```

---

# 58. Prompt 2 — Pre-aggregation utility matrix

```text
执行 R2D2.7-A。

Variants：

A0_BASE
UNIFORM
TARGET_NULL_ONLY
GENERIC_EDGE
DIAG_EDGE
PAIR_EDGE
SEMANTIC_SIM_CONTROL

Datasets：
Movies
Toys
Grocery
ele-fashion
Reddit-S

Seeds：
42/43/44

A0 fully frozen。
Val only。
No Test。

所有 side variants 使用完全一致：

U_a payload transforms
[z_base|mC|mPt|mPv]
classifier family
optimizer
training schedule

训练：
side/scorer/head lr=1e-3
AdamW wd=1e-4
warmup10+cosine
300 epochs
patience30
best ValAcc
grad clip1.0

同 dataset/seed 必须复用 matched classifier init。

输出：

Val Acc
Macro-F1
per-class F1
params
runtime
peak memory
best epoch

paired deltas：

PAIR-UNIFORM
PAIR-TARGET_NULL
PAIR-GENERIC
PAIR-DIAG
PAIR-SEM_SIM
PAIR-A0_MATCHED
PAIR-A0_FORMAL

SELECTION GO：
M/T/G
PAIR-TARGET_NULL >= +0.30pp Acc
MacroF1 >= +0.20pp
>=2/3 dataset mean positive
positive dataset >=2/3 seeds positive。

A0 INCREMENTAL GO：
PAIR-A0_MATCHED >= +0.30pp Acc
F1 >= +0.20pp
same stability。

OWNERSHIP preliminary：
PAIR-GENERIC >= +0.20pp Acc with F1 nonnegative，
or F1 >=+0.30pp with Acc nonnegative。

Any candidate >= A0_MATCHED+0.20pp on M/T/G must satisfy guards：
ele/Reddit Acc >= -0.20pp
F1 >= -0.50pp。

输出：
outputs/perf_r2d27/matrix/
 matrix_results.csv
 matrix_controls.csv
 matrix_resources.csv
 R2D27_MATRIX_REPORT.md

不要调参救单个variant。
停止。
```

---

# 59. Prompt 3 — Edge-utility structure & causal ranking audit

```text
执行 R2D2.7-B。

使用 PAIR_EDGE best checkpoints。
全部5 datasets × seeds42/43/44。
No retraining。
No Test。

输出每 target/factor-pair：

real-neighbor mass
null mass
entropy
normalized entropy
Gini
top10% mass
top25% mass
effective neighbor count

Factor-pair diversity：
9×9 JSD
Spearman
top-k overlap

Simple heuristic correlations：
cos(F_i^b,F_j^a)
source degree
target degree
A0 Gamma/relation weight（仅若可精确提取，不近似）

Train-label-only homophily：
只分析 train->train edges。
比较 same-label / different-label utility。
禁止使用Val label做edge诊断。

Causal ranking：

REMOVE_TOP 10/25/50%
REMOVE_RANDOM same count fixed seed
REMOVE_BOTTOM 10/25/50%
KEEP_TOP 25/50%

renormalize remaining real+null weights。

Permutation：

within-target weight shuffle
source-node shuffle
factor-id shuffle

输出所有 Acc/F1 paired drops。

Strong ranking：
top-removal drop > random > bottom，
且 top-random >=+0.20pp macro 至少一个rate。

Strong correspondence：
FULL-within-target-shuffle >=+0.30pp M/T/G macro。

输出：

outputs/perf_r2d27/edge_audit/
 edge_score_stats.csv
 edge_pair_diversity.csv
 edge_heuristic_corr.csv
 edge_homophily_train_only.csv
 edge_causal_ranking.csv
 edge_shuffle_controls.csv
 R2D27_EDGE_AUDIT_REPORT.md

必须回答：
scorer是否真正non-uniform？
pair rankings是否真的different？
是否只是cosine similarity？
是否只是label homophily？
高utility edge是否具有causal importance？

停止。
```

---

# 60. Prompt 4 — PRE vs POST aggregation

```text
执行 R2D2.7-C。

实现并验证 parameter-matched POST_PAIR：

N_i^a = mean_j F_j^a

g_i^{a->b} =
MLP([
 F_i^b,
 N_i^a,
 F_i^b*N_i^a,
 abs(F_i^b-N_i^a),
 emb_a,
 emb_b
])

m_i^{a->b} =
sigmoid(g_i^{a->b}) U_a(N_i^a)

与 PRE_PAIR 使用：
same scorer depth
same pair embeddings
same U_a
same side output dimension
same classifier
same A0 parent
side params尽量±5%。

正式运行：

PRE_PAIR
POST_PAIR
TARGET_NULL_ONLY

全部5 datasets × seeds42/43/44。

PRE-AGGREGATION GO：

M/T/G
PRE-POST >=+0.30pp Acc
F1 >=+0.20pp
>=2/3 dataset means positive
positive dataset >=2/3 seeds positive
guards safe。

同时报告：
PRE-A0_MATCHED
POST-A0_MATCHED

输出：
outputs/perf_r2d27/prepost/
 prepost_results.csv
 R2D27_PREPOST_REPORT.md

若PRE≈POST：
不得继续声称“aggregation timing是核心”。
停止。
```

---

# 61. Prompt 5 — Ownership-specificity audit

```text
只有以下任一成立才执行：

PAIR_EDGE-TARGET_NULL >=+0.15pp Acc macro
或
GENERIC_EDGE-TARGET_NULL >=+0.15pp。

运行：

OWNERSHIP_PAIR
NODE_SHARED (= GENERIC_EDGE)
FACTOR_DIAG
SOURCE_FACTOR_ONLY
TARGET_FACTOR_ONLY

全部5 datasets × seeds42/43/44。

所有模型尽量parameter-matched；
readout/payload/A0完全相同。

输出：

PAIR - NODE_SHARED
PAIR - FACTOR_DIAG
PAIR - SOURCE_ONLY
PAIR - TARGET_ONLY

Formal ownership support：

PAIR beats strongest control
>=+0.20pp Acc M/T/G macro
F1 nonnegative
>=2/3 dataset mean positive

并结合D2.7-B pair-ranking divergence。

如果generic edge utility有效但PAIR无优势：
结论必须写：
“neighbor utility supported; semantic-ownership factor-pair specificity not supported.”

输出：
outputs/perf_r2d27/ownership/
 ownership_results.csv
 R2D27_OWNERSHIP_REPORT.md

停止。
```

---

# 62. Prompt 6 — Selection × factor-pair transformation

```text
只有以下任一成立才执行：

PRE_PAIR selection GO
或
PRE_PAIR-A0_MATCHED >=+0.20pp。

实现：

T0 SHARED_PAYLOAD_UNIFORM
T1 SHARED_PAYLOAD_PRE_PAIR
T2 PAIR_TRANSFORM_UNIFORM
T3 PAIR_TRANSFORM_PRE_PAIR

Pair transform：
每a->b独立：
Linear(d,2d)
LN
GELU
Linear(2d,d)

如果显存过高，可以使用 low-rank residual，
但T2/T3必须完全同构。

T3：
m_i^b =
mean_a sum_j alpha_{ji}^{a->b} T_ab(F_j^a)

所有其它readout/A0/head一致。

Movies/Toys/Grocery × seeds42/43/44。
达到/接近final GO再跑guards 3 seeds。

报告：

Selection = T1-T0
Transform = T2-T0
Complementarity = T3-max(T1,T2)

Functional Transfer support：

T3-max(T1,T2) >=+0.20pp
on Acc or F1 macro，
另一metric非负，
>=2/3 datasets positive。

还必须做：
T3 within-target score shuffle
以证明pair transform没有替代neighbor selection。

输出：
outputs/perf_r2d27/transfer/
 transfer_results.csv
 transfer_ablation.csv
 R2D27_TRANSFER_REPORT.md

停止。
```

---

# 63. Prompt 7 — Optional random-edge stress test

```text
只有top PRE candidate：

A0_MATCHED增益 >=+0.20pp

或距离A0 incremental GO <=0.10pp

才执行。

Evaluation-only：
随机加入原edge count的10%/25% edges。
固定random seed。
不重新训练。

比较：

A0
UNIFORM
top PRE utility model

输出：

clean vs noisy Acc/F1 drop
injected-edge mean utility
original-edge mean utility
injected edges entering top10/top25比例

若utility model真正学习neighbor usefulness，
应该：

assign lower utility to injected random edges
and degrade less than UNIFORM。

输出：
outputs/perf_r2d27/noise_optional/
 noise_results.csv
 R2D27_NOISE_REPORT.md

停止。
```

---

# 64. Prompt 8 — Final synthesis

```text
R2-Design-2.7 所有允许阶段完成。

不要新实验。
不要Test。
不要设计final paper model。

读取：

audit
matrix
edge_audit
prepost
ownership(if entered)
transfer(if entered)
noise(if entered)

生成：

outputs/perf_r2d27/summary/
 R2D27_MASTER_TABLE.csv
 R2D27_HYPOTHESIS_LEDGER.csv
 R2D27_FINAL_DIAGNOSIS.md

必须回答：

1. pre-aggregation learned neighbor ranking是否优于uniform/target-null？
2. utility scores是否真实non-uniform？
3. top-ranked neighbor causal removal是否比random removal更伤？
4. within-target shuffle是否破坏性能？
5. PRE是否显著优于POST？
6. learned utility是否只是semantic cosine？
7. learned utility是否只是train-label homophily？
8. generic edge utility是否已经足够？
9. C/Pt/Pv factor-specific utility是否成立？
10. full a->b pair utility是否优于diag/source-only/target-only？
11. pair ranking distributions是否真的不同？
12. PAIR_EDGE是否超过A0_MATCHED？
13. 是否超过A0_FORMAL？
14. null mass是否有意义，还是几乎0/1 collapse？
15. 哪些factor-pair最常保留/抑制neighbor？
16. selection-only是否足够？
17. pair-specific transform是否有独立价值？
18. selection × transform是否具有complementarity？
19. guards是否安全？
20. 下一阶段 route：

N1 Ownership-conditioned Neighbor Utility
N2 Generic Edge Utility
N3 Simpler Factor-aware Utility
N4 Semantic-Ownership Functional Edge Transfer
N5 Second-axis reassessment

最终给：
R2-Design-2.7 = PASS / PARTIAL / NO-GO

若N4成立，允许下一阶段R2-Design-3开始formal architecture design；
本阶段不要提前写paper贡献。
```

---

# 65. Completion package

完成后返回：

```text
outputs/perf_r2d27/audit/
outputs/perf_r2d27/matrix/
outputs/perf_r2d27/edge_audit/
outputs/perf_r2d27/prepost/
outputs/perf_r2d27/ownership/          # if entered
outputs/perf_r2d27/transfer/           # if entered
outputs/perf_r2d27/noise_optional/     # if entered
outputs/perf_r2d27/summary/

R2D27_FINAL_DIAGNOSIS.md
R2D27_MASTER_TABLE.csv
R2D27_HYPOTHESIS_LEDGER.csv

matrix_results.csv
matrix_controls.csv

edge_score_stats.csv
edge_pair_diversity.csv
edge_heuristic_corr.csv
edge_homophily_train_only.csv
edge_causal_ranking.csv
edge_shuffle_controls.csv

prepost_results.csv
ownership_results.csv
transfer_results.csv
transfer_ablation.csv
noise_results.csv

latest GitHub commit
```

---

# 66. Most important scientific discipline

Do not repeat the old pattern:

```text
nonuniform gate
→ call it relation learning
```

The stage only supports a new relational axis if all relevant causal levels line up:

\[
\boxed{
\textbf{Neighbor ranking is useful}
}
\]

\[
\boxed{
\textbf{ranking must happen before aggregation}
}
\]

\[
\boxed{
\textbf{ranking differs by semantic ownership factor}
}
\]

\[
\boxed{
\textbf{the difference creates incremental value over A0}
}
\]

and, if claimed:

\[
\boxed{
\textbf{factor-pair-specific transformation adds value beyond selection}
}
\]

Only then should the project return to the paper-level concept:

\[
\boxed{
\textbf{Semantic Ownership}
\times
\textbf{Functional Relational Transfer}
}
\]

with a genuinely task-relevant pre-aggregation meaning.
