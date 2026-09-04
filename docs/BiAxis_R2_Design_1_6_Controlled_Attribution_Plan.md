# Bi-Axis R2-Design-1.6
## Controlled Attribution, Dual-Parent Frozen Audits & Warm-Start Realization Plan

**Repository:** `CrisRipper777/0901`  
**Current code baseline:** after R2-Design-1.5 (`8efa306` reported by previous stage)  
**Previous stage:** `R2-Design-1.5 = PARTIAL DIAGNOSTIC SUCCESS` after manual review  
**Protocol:** Val-only until an explicit later decision; **no Test in this stage**  
**Primary metric:** Val Accuracy  
**Safety metric:** Val Macro-F1 + per-class F1  
**Main objective:** 用更严格的控制变量实验回答：R2-0 中发现的 interaction / propagation headroom，究竟能否在不破坏稳定 parent representation 的前提下转化为可训练机制。

---

# 0. 为什么需要 R2-Design-1.6

R2-Design-1.5 有三个确定结果：

1. **B0 的 seed42 优势未能 formal-confirm。**
   - M/T/G 3-seed macro vs A0 ≈ `-0.045pp`；
   - Movies +0.12pp；
   - Toys / Grocery 约 -0.13pp；
   - ele-fashion ≈ -0.30pp；
   - 因此 B0 不应被称为“更强最终 parent”。

2. **Scalar functional routing 可以关闭。**
   - trained F checkpoint 的 functional branch post-hoc forward effect ≈ 0；
   - off-diagonal contribution ≈ 0；
   - 说明：
     \[
     g^{a\to b}\in\mathbb R
     \]
     只决定“传多少”的 realization 没有兑现 R2-0C 的 interaction evidence。

3. **Semantic branch 与 shared backbone 存在强训练耦合风险。**
   - S checkpoint 中 branch-on vs branch-off 的局部 CE 梯度出现强负 cosine；
   - 但 post-hoc `S_full - S_both_off` 不能直接解释成 semantic refiner 的独立 causal gain；
   - 因为 independently-trained B0/S 并非严格 common-random-trajectory 对照。

因此本阶段不再继续：

```text
A0/B0 二选一
scalar gate
from-scratch joint stacking
```

而改成：

\[
\boxed{
\textbf{Dual-Parent + Frozen Attribution + Matched Initialization + Controlled Unfreezing}
}
\]

---

# 1. 本阶段的核心科学问题

R2-Design-1.6 只回答四个问题：

### Q1. Parent question
A0 与 B0 各自应该承担什么角色？

- A0：formal performance reference；
- B0：clean diagnostic scaffold；
- 不再要求 B0 必须优于 A0 才允许做 frozen diagnosis。

### Q2. Propagation question
Factor-specific：

\[
1\text{-hop},\ 2\text{-hop},\ high\text{-pass}
\]

是否在 **A0/B0 两个 parent** 上都存在稳定、可重复的 signal？

### Q3. Interaction question
R2-0C 的：

\[
F_i^b\odot N_i^a,
\qquad
|F_i^b-N_i^a|
\]

能否以 **vector-valued correction** 的方式，在 frozen parent 上带来真实增益？

### Q4. Optimization question
若 frozen branch 有价值：

> 为什么 from-scratch joint training 失败？

进一步比较：

```text
Frozen
Full-from-start
Gradual-unfreeze
```

并采用：

```text
同一 parent checkpoint
同一 adapter init
同一 classifier init
同一 seed
```

做真正的 matched-control。

---

# 2. 本阶段不做什么

本阶段禁止：

```text
Test
OT/UOT
new contrastive/synergy loss
PCGrad
MoE
Graph Transformer
RoleMAG-style predefined shared/complementary/heterophily experts
new K-prototype relation system
edge-level relation model
large hyperparameter sweep
3/4-hop expansion
PPR/GPR end-to-end model
feature-wise multi-head attention
```

这些都由本轮结果决定是否在 R2-Design-2 重新开放。

---

# 3. 总执行流程

```text
D1.6-0  Audit + metric backfill infrastructure
        ↓
D1.6-A  A0 Val Macro-F1 / per-class backfill + Parent Characterization
        ↓
D1.6-B  Dual-Parent Frozen Propagation Audit
        ↓
D1.6-C  Dual-Parent Frozen Interaction Adapter Screen
        ↓
D1.6-D  Semantic Residual-Only Warm-Start Screen
        ↓
D1.6-E  Controlled Optimization Schedule Study
        ↓
D1.6-F  Final Synthesis / R2-Design-2 Route Decision
```

说明：

- A/B/C 都允许执行，不再以 `B0 > A0` 作为 gate；
- D 与 E 有条件 gate；
- 全程 Val-only。

---

# 4. 两个 Parent 的正式角色

以后统一使用：

## Parent-P：A0 / `biaxis_final`

角色：

\[
\boxed{\textbf{Performance Parent}}
\]

用于回答：

> 新机制能否在当前 formal、较稳定的完整模型上继续增加价值？

---

## Parent-C：R2-B0

角色：

\[
\boxed{\textbf{Clean Diagnostic Parent}}
\]

用于回答：

> 当旧 K-relation / Gamma / OFR machinery 不存在时，新机制是否仍然有效？

注意：

B0 不再被描述为：

```text
better parent
stronger parent
```

而是：

```text
clean scaffold with approximately A0-level M/T/G performance
```

---

# 5. Dual-Parent 的解释纪律

任何新机制 \(M\) 都要分类：

### Parent-Robust

\[
M(A0)>A0
\]

且：

\[
M(B0)>B0.
\]

这是最强证据。

### A0-Dependent

只在 A0 上有效。

说明机制依赖 A0 既有 graph organization。

### B0-Dependent

只在 B0 上有效。

说明机制与 simple graph backbone 兼容，但被 A0 machinery 抑制/冗余。

### Parent-Insensitive NO-GO

两个 parent 都没有价值。

此时才真正考虑关闭该 realization。

---

# 6. 统一 Safety 规则

此前 R2D1.5 Final Diagnosis 对 Macro-F1 safety 的总结存在遗漏。

本阶段统一修正：

Primary：

```text
Val Accuracy
```

Safety：

```text
Val Macro-F1
per-class F1
confusion matrix
```

任何 candidate：

\[
\Delta MacroF1 < -0.50pp
\]

则标记：

```text
SAFETY WARNING
```

如果：

\[
\Delta Accuracy > 0
\]

但：

\[
\Delta MacroF1 < -1.00pp
\]

不能判为正式 GO，除非后续人工明确接受该 trade-off。

---

# 7. D1.6-0 — Audit & Infrastructure

目标：

1. 确认 A0 / B0 formal checkpoints；
2. 确认是否能恢复 A0 best-epoch Val Macro-F1；
3. 建立 dual-parent factor/state extractor；
4. 建立 frozen parent adapter training；
5. 建立 matched-init schedule runner；
6. 不改变任何旧模型行为。

建议新增：

```text
src/analysis/perf_r2d16_utils.py

src/models/biaxis_r2d16_adapters.py

scripts/
  perf_r2d16_a_parent_metrics.py
  perf_r2d16_b_propagation_dual_parent.py
  perf_r2d16_c_interaction_dual_parent.py
  perf_r2d16_d_semantic_residual.py
  perf_r2d16_e_schedule_control.py
  summarize_perf_r2d16.py

tests/
  test_perf_r2d16_utils.py
  test_biaxis_r2d16_adapters.py
```

输出：

```text
outputs/perf_r2d16/
  audit/
  parent_metrics/
  propagation/
  interaction/
  semantic/
  schedule/
  summary/
```

---

# 8. D1.6-A — A0 Metric Backfill

当前 A0 formal reference 只有 Val Accuracy。

本阶段首先补：

```text
Val Macro-F1
per-class F1
confusion matrix
```

对：

```text
Movies
Toys
Grocery
ele-fashion
Reddit-S

seeds 42/43/44
```

---

# 9. A0 Metric Backfill 的数据来源优先级

严格按以下顺序：

## Priority 1

如果 A0 history / summary 中已经保存：

```text
best epoch val_macro_f1
```

直接读取。

## Priority 2

如果 formal checkpoint 同时保存：

```text
model state
classifier state
```

重新进行 Val-only inference。

## Priority 3

如果 classifier state 不可恢复，但 best-epoch history 可恢复 Macro-F1：

使用 history。

## 禁止

```text
重新训练 A0 只为了补 F1
使用 Test
用另一 epoch 的 head 代替 best Val-Acc epoch head
```

若无法恢复 per-class F1，则明确标 `unavailable`，不要伪造。

---

# 10. Parent Characterization Table

输出：

```text
dataset
seed
A0 ValAcc
A0 MacroF1
B0 ValAcc
B0 MacroF1
B0-A0 Acc
B0-A0 F1
best_epoch
train-val gap
```

再汇总：

```text
mean±std
positive seed count
```

---

# 11. 可选低成本 A0 Graph-Control Diagnostic

此部分只在 audit 能保证“不改 trained weights + 明确数学语义”时执行。

目标：

解释 ele-fashion 中：

\[
A0>B0
\]

是否主要来自：

```text
local-vs-graph suppression
```

而不是：

```text
K relation specialization
```

允许的 counterfactual 必须来自当前 A0 原生计算图中可清晰 mask 的已有路径。

例如如果能严格定义：

```text
A0 full
graph residual OFF
relation selection neutralized
local-only
```

则做 Val-only post-hoc comparison。

禁止为了这个诊断重新发明一套近似 A0。

如果现代码不能做严格 counterfactual：

```text
SKIP
```

并在报告说明。

---

# 12. D1.6-B — Dual-Parent Frozen Propagation Audit

不训练新 GNN。

Parents：

```text
A0
B0
```

Datasets：

```text
Movies
Toys
Grocery
```

Seeds：

```text
42
43
44
```

guards：

```text
ele-fashion
Reddit-S
```

可作为 secondary frozen probe。

---

# 13. Parent-specific state extraction

对每一个 parent checkpoint，提取其当前训练后的 semantic ownership states：

\[
H_0^a
=
F^a,\qquad
a\in\{C,P_t,P_v\}.
\]

注意：

A0 和 B0 的 factorizer 参数经过各自 end-to-end 训练，因此：

\[
H_{0,A0}\neq H_{0,B0}.
\]

不要跨 parent 直接比较 feature coordinate。

只比较：

\[
\boxed{\text{within-parent delta}}
\]

---

# 14. Propagation basis

对每个 parent：

\[
H_1^a=P H_0^a
\]

\[
H_2^a=P H_1^a
\]

\[
H_{HP}^a=H_0^a-H_1^a.
\]

仍只做：

```text
1-hop
2-hop
high-pass
```

不扩 3-hop/PPR。

---

# 15. Per-factor matched probe

Fixed：

```text
StandardScaler
RidgeClassifier(alpha=1.0)
TRAIN fit
VAL eval
```

对每 factor：

\[
X_1^a=[H_0^a|H_1^a]
\]

\[
X_2^a=[H_0^a|H_2^a]
\]

\[
X_{HP}^a=[H_0^a|H_{HP}^a].
\]

定义：

\[
\Delta_{2-1}^a
=
Probe(X_2^a)-Probe(X_1^a)
\]

\[
\Delta_{HP-1}^a
=
Probe(X_{HP}^a)-Probe(X_1^a).
\]

---

# 16. Joint matched probe

\[
L=[C|P_t|P_v].
\]

构造：

\[
X_1^{joint}
=
[L|H_1^C|H_1^{Pt}|H_1^{Pv}]
\]

\[
X_2^{joint}
=
[L|H_2^C|H_2^{Pt}|H_2^{Pv}]
\]

\[
X_{HP}^{joint}
=
[L|H_{HP}^C|H_{HP}^{Pt}|H_{HP}^{Pv}].
\]

全部严格 same dimension。

---

# 17. Final residual probe

分别用：

\[
Z_{A0}
\]

和：

\[
Z_{B0}.
\]

构造：

\[
[Z|H_1^{all}]
\]

\[
[Z|H_2^{all}]
\]

\[
[Z|H_{HP}^{all}].
\]

定义：

\[
\Delta_{2-1}^{final}
\]

与：

\[
\Delta_{HP-1}^{final}.
\]

---

# 18. Cross-parent propagation verdict

### Cross-Parent 2-hop SUPPORT

至少一个 factor：

```text
>=2/3 target datasets
```

在 A0 与 B0 两个 parent 上都满足：

\[
mean(\Delta_{2-1}^a)\ge+0.20pp
\]

且 dataset 内：

```text
>=2/3 seeds positive
```

### Parent-Specific 2-hop SUPPORT

只在一个 parent 上满足上述规则。

### High-pass

使用同样规则。

### Final-residual stronger evidence

若：

\[
\Delta_{2-1}^{final}
\]

或：

\[
\Delta_{HP-1}^{final}
\]

M/T/G macro ≥ +0.20pp，且 2/3 dataset mean positive：

标记：

```text
FINAL-RESIDUAL SUPPORT
```

若只 per-factor 强、final weak：

```text
INDUCTIVE-BIAS SUPPORT ONLY
```

---

# 19. D1.6-C — Dual-Parent Frozen Interaction Adapter Screen

这是本阶段核心。

不再测试旧 scalar routing 作为候选。

只保留：

```text
HEAD
CONCAT-VECTOR
PRODDIFF-VECTOR
FiLM-VECTOR
```

---

# 20. Parents

对：

```text
A0
B0
```

分别独立运行。

第一轮：

```text
Movies
Toys
Grocery
seed42
```

---

# 21. Frozen Parent Protocol

对指定 parent：

```text
load best checkpoint
freeze ALL parent parameters
parent.eval()
```

训练：

```text
adapter
+
fresh classifier
```

需要对 adapter 反向传播，因此 frozen parent forward 不能 `torch.no_grad()` 包住需要 adapter gradient 的后半段。

但 parent parameters：

```text
requires_grad=False
```

---

# 22. Interaction source states

统一使用 parent 的 pre-graph semantic factors：

\[
F_i^b.
\]

统一使用 simple 1-hop source context：

\[
N_i^a=P F^a.
\]

这样：

A0/B0 上 adapter 的 scientific object 完全一致：

\[
\boxed{
\text{target semantic state}
\times
\text{source-factor neighborhood context}
}
\]

---

# 23. Adapter insertion

提取 parent 已完成 graph propagation 后、final fusion 前的 factor outputs：

\[
F_{parent,out}^b.
\]

Adapter 输出：

\[
\Delta_i^b.
\]

定义：

\[
\hat F_i^b
=
F_{parent,out}^b+\Delta_i^b.
\]

然后：

\[
\hat Z
=
Fusion_{parent}(
\hat F^C,\hat F^{Pt},\hat F^{Pv}
).
\]

parent fusion frozen。

---

# 24. HEAD Control

无 adapter：

\[
\hat F=F_{parent,out}.
\]

只训练 fresh classifier。

对于同一：

```text
parent
dataset
seed
```

所有 candidate 必须复用：

```text
same classifier initial state
same classifier optimizer settings
same adapter initialization seed
same early stop protocol
```

---

# 25. CONCAT-VECTOR

输入：

\[
u^{concat}_{ab}
=
[
F_b,N_a,e_a^{src},e_b^{tgt}
].
\]

共享：

```text
Linear(2d+2t, h)
GELU
Linear(h,d)
```

推荐：

```text
type_dim=8
hidden=128
```

最后层：

```text
zero-init
```

输出：

\[
\Delta_{ab}^{concat}\in\mathbb R^d.
\]

---

# 26. PRODDIFF-VECTOR

输入：

\[
u^{int}_{ab}
=
[
F_b\odot N_a,
|F_b-N_a|,
e_a^{src},
e_b^{tgt}
].
\]

网络结构与 CONCAT 完全一致：

```text
Linear(2d+2t,128)
GELU
Linear(128,d)
```

因此：

\[
\boxed{
\text{CONCAT vs PRODDIFF = parameter-matched}
}
\]

最后层 zero-init。

---

# 27. FiLM-VECTOR

输入：

\[
u^{film}_{ab}
=
[
F_b,N_a,
F_b\odot N_a,
|F_b-N_a|,
e_a,e_b
].
\]

网络：

```text
Linear(4d+2t,128)
GELU
Linear(128,2d)
```

输出：

\[
\Delta\gamma_{ab},
\beta_{ab}\in\mathbb R^d.
\]

独立 adapter source transform：

\[
v_a=U_aN_a.
\]

定义：

\[
\Delta_{ab}^{film}
=
\Delta\gamma_{ab}\odot v_a
+
\beta_{ab}.
\]

最后层 zero-init：

\[
\Delta\gamma=0,\quad\beta=0.
\]

注意：

这是：

\[
\boxed{\text{feature-wise zero-init correction}}
\]

不是替换 parent message。

---

# 28. Cell aggregation

三个 vector candidate：

\[
\Delta^b
=
\frac13
\sum_a
\Delta^{a\to b}.
\]

本轮：

```text
NO softmax
NO top-k
NO MoE
NO cell competition
```

先证明不同 cell output 是否真的有新信息。

---

# 29. Interaction mismatch control

best checkpoint 后：

固定：

```text
perm seed=20260904
```

只对 source context：

\[
N_i^a
\rightarrow
N_{\pi(i)}^a.
\]

target：

\[
F_i^b
\]

不动。

计算：

\[
Real-Mismatch.
\]

---

# 30. Message novelty diagnostics

对每个：

\[
a\to b
\]

输出：

### Norm

\[
E\|\Delta_{ab}\|.
\]

### Cosine to parent base target graph message

\[
\cos(\Delta_{ab},M_{parent}^{b})
\]

如果 A0 的 target graph message 无法定义成单一 tensor，则使用：

```text
parent factor update:
F_parent_out^b - F_pregraph^b
```

作为 reference。

### Orthogonal novelty

\[
Novelty
=
\frac{
\|\Delta-Proj_{M_{parent}}\Delta\|
}{
\|\Delta\|+\epsilon
}.
\]

### 9-cell specialization

```text
pairwise cosine 9x9
mean off-diagonal cosine
effective rank
cell norm 3x3
```

---

# 31. Interaction screen verdict — within parent

对每 parent：

\[
Gain(M,parent)
=
mean_{M/T/G}
[
Acc(M)-Acc(HEAD)
].
\]

### STRONG

\[
Gain\ge+0.50pp
\]

且：

```text
>=2/3 datasets positive
Real-Mismatch macro >= +0.20pp
no F1 safety warning
```

### GO

\[
Gain\ge+0.30pp
\]

且 2/3 positive，并满足至少一个：

\[
PRODDIFF-CONCAT\ge+0.15pp
\]

或：

\[
Real-Mismatch\ge+0.20pp.
\]

### WEAK

\[
+0.15\sim+0.30pp.
\]

### NO-GO

\[
<+0.15pp.
\]

---

# 32. Cross-parent interaction verdict

### PARENT-ROBUST GO

D3 或 D4：

```text
A0 parent GO
AND
B0 parent GO
```

### PARENT-SPECIFIC GO

只一个 parent GO。

### REALIZATION NO-GO

A0/B0 两 parent：

```text
D3 + D4 均 < +0.15pp
```

且 mismatch 不支持。

这时才真正考虑关闭：

```text
vector factor-context realization
```

---

# 33. Interaction confirmation gate

如果 D3 或 D4 在任一 parent 达到 GO：

先跑 guards：

```text
ele-fashion
Reddit-S
seed42
```

同 parent。

要求：

```text
Acc delta vs HEAD >= -0.20pp
Macro-F1 delta >= -0.50pp
```

guards safe 后：

```text
Movies/Toys/Grocery
seeds42/43/44
```

使用对应 seed 的 parent checkpoint。

---

# 34. D1.6-D — Semantic Residual-Only Warm-Start Screen

目的：

真正隔离：

\[
\boxed{\text{Factor Interaction Residual}}
\]

避免 Adaptive Common gate confound。

本阶段：

\[
C
=
\frac12(c_t+c_v)
\]

固定不变。

禁止 adaptive scalar common gate。

---

# 35. Semantic interaction block

Base factors：

\[
F^0=\{C,P_t,P_v\}.
\]

构造：

\[
I=
[
C\odot P_t,
C\odot P_v,
P_t\odot P_v,
|C-P_t|,
|C-P_v|,
|P_t-P_v|
].
\]

使用：

```text
Linear(6d, h)
LayerNorm
GELU
Linear(h, 3d)
```

推荐：

```text
h=128
```

最后层：

```text
zero-init
```

拆成：

\[
\Delta C,\Delta P_t,\Delta P_v.
\]

\[
F^*=F^0+\Delta F.
\]

---

# 36. Semantic residual 插入位置

这里和 interaction adapter 不同。

Semantic residual 必须发生：

\[
\boxed{\textbf{before graph propagation}}
\]

因为 R2-0C C2 已经说明 final-layer semantic concat residual 很弱。

因此：

```text
P0 ownership factors
→ semantic residual
→ frozen parent graph mechanism
→ frozen parent fusion
→ classifier
```

---

# 37. Dual-parent semantic feasibility

## B0

直接把 refined：

\[
F^*
\]

送入 frozen B0 graph path。

## A0

必须在 audit 中确认：

当前 A0 是否能在不重新训练 parent parameters 的情况下，
用 overridden/refined factors 继续走：

```text
P1/P2/P3 graph path
```

若可以：

```text
执行 A0 semantic residual screen
```

若因当前 forward 强耦合无法严格注入：

```text
A0 = NOT FEASIBLE
```

不得用近似 substitute。

---

# 38. Semantic frozen screen

先：

```text
Movies/Toys/Grocery seed42
```

Parents：

```text
A0 if feasible
B0
```

Train：

```text
semantic residual
fresh classifier
```

Parent：

```text
all frozen
eval mode
```

HEAD：

同 parent 不加 semantic residual，只训练同 init classifier。

---

# 39. Semantic screen verdict

每 parent：

\[
Gain_{sem}
=
mean_{M/T/G}(SEM-HEAD).
\]

### GO

\[
Gain_{sem}\ge+0.20pp
\]

且：

```text
>=2/3 datasets positive
no Macro-F1 safety warning
```

### STRONG

\[
\ge+0.50pp.
\]

若：

```text
B0 GO
A0 GO
```

标：

```text
PARENT-ROBUST SEMANTIC SUPPORT
```

若只一个：

```text
PARENT-SPECIFIC
```

---

# 40. Semantic diagnostics

输出：

### residual ratio

\[
\|\Delta F^b\|/\|F^b\|.
\]

### ownership health

因为 parent frozen，base P0 ownership 不变。

额外计算 refined factor：

```text
C-Pt cosine
C-Pv cosine
Pt-Pv cosine
```

只作 diagnostics。

### mismatch control

fixed perm：

```text
seed=20260904
```

把 interaction partner factor rows mismatch，例如保持 C_i，使用 \(P_{\pi(i)}\) 构造 interaction。

只在 best checkpoint eval，不训练。

真实 > mismatch 才支持 same-node factor correspondence。

---

# 41. D1.6-E — Controlled Optimization Schedule Study

只有以下任一机制通过 frozen GO 才执行：

```text
D3 PRODDIFF
D4 FiLM
Semantic residual
```

目的：

严格验证：

\[
\boxed{
\text{frozen branch value}
\rightarrow
\text{joint training compatibility}
}
\]

---

# 42. 选择候选

如果多个候选 GO：

优先顺序不是按最高 seed42，而是：

1. formal 3-seed 已确认的 candidate；
2. parent-robust；
3. Macro-F1 safe；
4. message novelty / correspondence evidence 更强。

最多选择：

```text
2 candidates
```

进入 schedule study。

---

# 43. Matched initialization discipline

对每：

```text
parent
dataset
seed
candidate
```

先保存：

```text
parent checkpoint state
adapter initial state
classifier initial state
optimizer-independent RNG seed
```

三种 schedule 必须：

```text
reload exactly same parent
reload exact same adapter init
reload exact same classifier init
set same random seed
```

不要重新 instantiate 后依赖“同 seed 应该一样”。

直接保存/load state_dict。

---

# 44. 三种 schedule

## E0 — FROZEN

```text
parent all frozen
adapter + classifier train
```

即已经验证过的 frozen candidate。

---

## E1 — FULL-FROM-START

从同一个 parent checkpoint 开始：

```text
parent all trainable
adapter trainable
classifier trainable
```

注意：

这不是 from-scratch model training。

它是：

\[
\boxed{\textbf{warm-start full fine-tune}}
\]

因此可以更干净地测试：

> unfreezing parent 是否破坏 adapter value？

---

## E2 — GRADUAL

Epoch：

```text
1-30:
  parent frozen
  adapter + classifier train

31+:
  unfreeze graph/fusion blocks
  keep P0 factorizer frozen
```

不要在本阶段继续 unfreeze P0 factorizer。

理由：

\[
\boxed{
\text{preserve semantic ownership anchor}
}
\]

同时允许 graph/fusion 适应新机制。

---

# 45. Common Randomness

三种 schedule forward graph architecture应保持一致。

要求：

```text
same dropout configuration
same train split
same data order
same classifier init
same adapter init
same parent checkpoint
same seed
```

如果某个 schedule 因 requires_grad 差异导致 RNG stream 不完全相同，应明确披露，但不要为了完全 bitwise deterministic 改动正式 benchmark protocol。

---

# 46. Gradient conflict trajectory

E1/E2：

在以下节点采样：

```text
before training / epoch1
epoch10
epoch30
first epoch after unfreeze
best checkpoint
```

对 shared trainable groups：

```text
graph path
fusion
```

计算：

### branch ON CE gradient

\[
g_{on}
\]

### same checkpoint branch OFF CE gradient

\[
g_{off}
\]

### branch-induced

\[
g_\Delta=g_{on}-g_{off}.
\]

输出：

\[
\|g_\Delta\|/\|g_{off}\|
\]

\[
\cos(g_{off},g_\Delta).
\]

这次是 **同一训练 trajectory / 同一 checkpoint** 的局部 counterfactual，比 D1.5 independently-trained B0/S 更干净。

---

# 47. Schedule verdict

Primary：

比较同一 candidate：

\[
FULL-FROZEN
\]

和：

\[
GRADUAL-FROZEN.
\]

### Strong optimization-coupling evidence

Frozen candidate GO，但：

\[
FULL-FROZEN\le-0.30pp
\]

且：

\[
GRADUAL-FROZEN\ge-0.10pp
\]

或 GRADUAL 进一步提升。

并伴随 FULL early gradient conflict 强于 GRADUAL。

### Joint-compatible

FULL：

\[
\ge FROZEN-0.10pp
\]

且 no F1 safety warning。

### Gradual-preferred

GRADUAL：

\[
FULL+0.20pp
\]

以上，且 2/3 target positive。

---

# 48. R2-Design-2 的 Route Matrix

最终只允许以下路线。

## Route A — Vector Functional Transfer

条件：

```text
D3 or D4 frozen GO
formal stable
semantic/propagation secondary
```

Design-2：

```text
Vector-valued functional semantic transfer
+
best schedule
```

---

## Route B — Semantic Interaction Refinement

条件：

```text
semantic residual frozen GO
interaction vector weak
```

Design-2：

```text
ownership-preserving semantic refinement
+
simple/frozen or gradual graph backbone
```

---

## Route C — Factor-Specific Multi-Scale

条件：

```text
2-hop / high-pass cross-parent support
并且 interaction/semantic weak
```

Design-2 才进入 end-to-end：

```text
factor-specific multi-scale propagation
```

---

## Route D — Hybrid

只有：

```text
至少两个机制已经独立 formal GO
```

才允许考虑组合。

不要直接 MoE。

组合前先证明：

```text
expert message novelty
low redundancy
complementary dataset/factor utility
```

---

## Route E — Task-Aware Relation Learning

如果：

```text
vector interaction NO-GO
semantic residual NO-GO
multi-scale weak
```

则进入：

\[
\boxed{
\textbf{Task-aware / semantic-aware Relation Learning Audit}
}
\]

重点重新调研/设计：

```text
RoleMAG
NRI/fNRI
IDGL
ACM/FAGCN
semantic-aware edge roles
feature-conditioned edge transformations
```

这时才进入 edge-level relation learning。

---

# 49. D1.6 最终 Hypothesis Ledger

最终状态只能使用：

```text
SUPPORTED
PARENT-ROBUST
PARENT-SPECIFIC
CONDITIONAL
WEAK
CLOSED
OPEN
```

至少包含：

```text
A0 performance parent
B0 diagnostic scaffold
current K-prototype relation
task-aware relation learning
scalar functional routing
PRODDIFF vector interaction
FiLM vector modulation
factor interaction semantic residual
adaptive scalar common
1-hop propagation
factor-specific 2-hop
high-pass/diversification
frozen training
full warm-start fine-tune
gradual unfreeze
MoE
edge-level relation learning
```

---

# 50. 输出目录

```text
outputs/perf_r2d16/
  audit/
  parent_metrics/
  propagation/
  interaction/
  interaction_confirm/
  semantic/
  semantic_confirm/
  schedule/
  summary/
```

核心交付：

```text
R2D16_AUDIT.md

parent_metrics.csv
parent_perclass.csv
R2D16_PARENT_REPORT.md

propagation_factor.csv
propagation_joint.csv
propagation_final.csv
R2D16_PROPAGATION_REPORT.md

interaction_results.csv
interaction_mismatch.csv
interaction_message_novelty.csv
interaction_expert_similarity.csv
interaction_effective_rank.csv
R2D16_INTERACTION_REPORT.md

semantic_results.csv
semantic_mismatch.csv
semantic_mechanism.csv
R2D16_SEMANTIC_REPORT.md

schedule_results.csv
schedule_gradient_trajectory.csv
schedule_mechanism.csv
R2D16_SCHEDULE_REPORT.md

R2D16_MASTER_TABLE.csv
R2D16_HYPOTHESIS_LEDGER.csv
R2D16_FINAL_DIAGNOSIS.md
```

---

# 51. Prompt 1 — D1.6-0 Audit + Infrastructure

```text
我们进入 Bi-Axis R2-Design-1.6：
Controlled Attribution, Dual-Parent Frozen Audits & Warm-Start Realization。

R2-Design-1.5 的人工复审结论：

1. A0 恢复为 formal performance reference；
2. B0 不再声称更强，但保留为 clean diagnostic scaffold；
3. 不再以 B0 > A0 作为 propagation / adapter 的执行 gate；
4. scalar functional routing CLOSED；
5. vector-valued interaction / FiLM OPEN；
6. adaptive scalar common CLOSED；
7. factor interaction residual OPEN；
8. from-scratch joint training HIGH-RISK；
9. 需要 dual-parent + frozen + matched-init + controlled-unfreeze。

本 Prompt 只做审计和基础设施。
不要正式训练。
不要 Test。

请审查：

- A0 formal checkpoint / history / summary 结构；
- B0 formal 42/43/44 checkpoints；
- biaxis_final forward / P1/P2/P3；
- biaxis_r2 B0；
- current R2D1.5 helpers/adapters；
- nc.py checkpoint 保存逻辑。

必须回答：

A. A0 Val Macro-F1 能否从：
   1) history
   2) saved classifier head
   3) summary
   恢复？
   按此优先级，不重新训练。

B. 如何对 A0 与 B0 都严格提取：
   - pre-graph semantic factors F=[C,Pt,Pv]
   - simple N=PF
   - parent graph-updated factor outputs before final fusion
   - z_final

C. A0 是否支持在不改 parent weights 的情况下：
   - factor output correction before fusion
   - semantic factor override before graph path
   如果某项不严格可行，报告 NOT FEASIBLE，不做近似替代。

D. 建立 dual-parent frozen adapter trainer：
   - parent requires_grad=False
   - parent eval
   - adapter + fresh classifier train
   - same saved classifier init across variants

E. 建立 matched schedule runner：
   - save/load exact parent state
   - exact adapter init
   - exact classifier init
   - same seed
   - FROZEN/FULL/GRADUAL

F. 修正 Macro-F1 safety：
   delta < -0.50pp = WARNING。

新增：

src/analysis/perf_r2d16_utils.py
src/models/biaxis_r2d16_adapters.py
scripts/perf_r2d16_*.py
tests/test_perf_r2d16_utils.py
tests/test_biaxis_r2d16_adapters.py

Unit tests 至少覆盖：

1. A0/B0 state extraction reproduces original forward;
2. HEAD frozen path reproduces parent z;
3. adapter zero-init -> exact parent factor output;
4. CONCAT/PRODDIFF parameter shapes and counts matched;
5. FiLM delta_gamma/beta zero-init;
6. fixed parent has zero parameter grads;
7. adapter gradients finite;
8. classifier init can be saved/reloaded bitwise;
9. matched schedule loads identical t=0 states;
10. propagation H1/H2/HP finite;
11. mismatch permutation deterministic;
12. no Test access.

输出：
outputs/perf_r2d16/audit/R2D16_AUDIT.md

完成后停止。
```

---

# 52. Prompt 2 — D1.6-A Parent Metric Backfill

```text
D1.6-0 PASS 后执行。

目标：
补齐 A0 Val Macro-F1 / per-class metrics，
并正式固定 A0/B0 双 parent 角色。

Datasets：
Movies/Toys/Grocery/ele-fashion/Reddit-S
seeds42/43/44

Val only。
禁止 Test。
禁止重新训练 A0/B0。

A0 metric source priority：

1. best-epoch history
2. saved model+classifier checkpoint Val inference
3. existing summary

不得用非 best-Val-Acc epoch 替代。

B0 使用已有 formal checkpoint/history。

输出：

outputs/perf_r2d16/parent_metrics/
  parent_metrics.csv
  parent_perclass.csv
  parent_confusions.json
  R2D16_PARENT_REPORT.md

必须包含：

A0/B0:
Val Acc
Macro-F1
per-class F1 if recoverable
best epoch
paired seed delta

重新报告：

M/T/G macro
guards
positive seed count

但不要重新做 A0/B0 winner gate。

最终只冻结：

A0 = Performance Parent
B0 = Clean Diagnostic Parent

如果 audit 证明可以对 A0 做严格 graph-control post-hoc masking，
可追加：
A0 full / local-only / graph-suppressed / relation-neutralized
但只能使用数学上严格定义的 existing path mask。
否则 SKIP。

完成后停止。
```

---

# 53. Prompt 3 — D1.6-B Dual-Parent Frozen Propagation Audit

```text
现在执行 dual-parent frozen propagation audit。

Parents：
A0
B0

Datasets：
Movies/Toys/Grocery
seeds42/43/44

Val only。
No Test。
不训练新 GNN。

对每 parent checkpoint 提取自身训练后的：

H0=[C,Pt,Pv]
H1=P H0
H2=P H1
HP=H0-H1

Fixed Ridge：
StandardScaler
RidgeClassifier(alpha=1.0)
TRAIN fit
VAL eval

Per-factor matched：

[H0|H1]
[H0|H2]
[H0|HP]

Joint matched：

[L|H1_all]
[L|H2_all]
[L|HP_all]

Final residual：

[z_parent|H1_all]
[z_parent|H2_all]
[z_parent|HP_all]

固定 permutation seed=20260904 做 H2/HP negative control。

输出：

outputs/perf_r2d16/propagation/
  propagation_factor.csv
  propagation_joint.csv
  propagation_final.csv
  propagation_shuffle.csv
  R2D16_PROPAGATION_REPORT.md

Verdict：

Cross-parent 2-hop SUPPORT：
至少一个 factor 在 A0/B0 两 parent，
>=2/3 target datasets mean(H2-H1)>=+0.20pp，
dataset 内 >=2/3 seeds positive。

Parent-specific：
只一个 parent 成立。

High-pass 同规则。

Final residual：
M/T/G macro >=+0.20pp 且 2/3 datasets positive
=> FINAL-RESIDUAL SUPPORT。

只 factor 强 final 弱：
=> INDUCTIVE-BIAS SUPPORT ONLY。

不要训练 propagation model。
完成后停止。
```

---

# 54. Prompt 4 — D1.6-C Dual-Parent Frozen Interaction Adapter Screen

```text
执行 dual-parent frozen interaction screen。

Parents：
A0
B0

Datasets：
Movies/Toys/Grocery
seed42

Val only。
No Test。

对每 parent/dataset：

load parent best checkpoint
freeze entire parent
parent.eval()

训练：
adapter + fresh classifier only。

同一个 parent/dataset：
HEAD / CONCAT / PRODDIFF / FiLM
必须复用 exact same classifier initial state。

Scientific states：

F_b = parent pre-graph semantic factor
N_a = P F_a

adapter correction 加在：
parent graph-updated factor outputs before frozen fusion。

Variants：

HEAD：
no adapter。

CONCAT：
input=[F_b,N_a,e_src,e_tgt]
Linear(2d+2t,128)
GELU
Linear(128,d)
last layer zero-init。

PRODDIFF：
input=[F_b*N_a,abs(F_b-N_a),e_src,e_tgt]
与 CONCAT 完全 parameter-matched。
last layer zero-init。

FiLM：
input=[F_b,N_a,F_b*N_a,abs(F_b-N_a),types]
Linear(...,128)
GELU
Linear(128,2d)
output delta_gamma,beta
delta_ab =
delta_gamma * U_a(N_a) + beta
last layer zero-init。

所有：
Delta_b=mean_a Delta_ab
Fhat_parent_out_b=F_parent_out_b+Delta_b
frozen parent fusion
fresh classifier。

Training：
AdamW lr1e-3 wd1e-4
300 epochs
patience30
best ValAcc
不扫参。

best checkpoint：
fixed permutation seed20260904
N_a -> N_perm_a
做 mismatch。

Diagnostics：

Acc
Macro-F1
per-class F1

D-HEAD
PRODDIFF-CONCAT
FiLM-CONCAT
Real-Mismatch

cell norm 3x3
cell vs parent-update cosine
orthogonal novelty
9x9 pairwise cosine
effective rank
residual ratio

Safety：
Macro-F1 delta<-0.50pp = WARNING。

Within-parent GO：
Gain>=+0.30pp
>=2/3 datasets positive
并且：
PRODDIFF-CONCAT>=+0.15pp
或 Real-Mismatch>=+0.20pp。

STRONG>=+0.50pp。

Cross-parent：
两个 parent GO = PARENT-ROBUST
仅一个 = PARENT-SPECIFIC。

如果 D3/D4 任一 parent GO：
不要自动 confirm；
报告后停止等待下一 Prompt。
```

---

# 55. Prompt 5 — D1.6-C2 Interaction Confirmation

```text
读取 D1.6-C。

只有 PRODDIFF 或 FiLM 在至少一个 parent 达到 GO 才执行。

若两个 candidate 都 GO：
都确认，不临时只选 seed42 更高者。

对每 GO 的：
(candidate,parent)

Step A Guards：

ele-fashion
Reddit-S
seed42

要求：
Acc vs HEAD >= -0.20pp
Macro-F1 vs HEAD >= -0.50pp

Step B Formal：

Movies/Toys/Grocery
seeds42/43/44

每 seed 使用对应 parent checkpoint。

比较：

candidate vs HEAD
candidate vs CONCAT
real vs mismatch

输出：
mean
population std ddof=0
positive seed count

Formal GO：

candidate-HEAD M/T/G macro >=+0.30pp
>=2/3 datasets positive
对应 dataset >=2/3 seeds positive
guards safe

输出：

outputs/perf_r2d16/interaction_confirm/
  interaction_confirm_results.csv
  interaction_confirm_mechanism.csv
  R2D16_INTERACTION_CONFIRM_REPORT.md

No Test。
完成后停止。
```

---

# 56. Prompt 6 — D1.6-D Semantic Residual-Only Warm-Start Screen

```text
现在单独测试 Semantic Factor Interaction Residual。

严格移除 Adaptive Common。
固定：

C = 0.5*(c_t+c_v)

不要训练 scalar common gate。

Parents：
B0 必做
A0 仅当 D1.6-0 audit 证明“factor override before frozen A0 graph path”严格可行时执行。

Datasets：
Movies/Toys/Grocery
seed42

Val only。
No Test。

加载 parent best checkpoint。
freeze entire parent。
parent.eval()。

Semantic residual：

F0=[C,Pt,Pv]

I=[
 C*Pt,
 C*Pv,
 Pt*Pv,
 abs(C-Pt),
 abs(C-Pv),
 abs(Pt-Pv)
]

Linear(6d,128)
LayerNorm
GELU
Linear(128,3d)

last layer zero-init。

split:
DeltaC,DeltaPt,DeltaPv

F*=F0+Delta

关键：
refinement 必须在 parent graph propagation 之前。

然后：
frozen parent graph path
frozen parent fusion
fresh classifier

HEAD：
同 parent，无 semantic residual，
使用 same exact classifier init。

Training：
adapter + classifier only
AdamW 1e-3 wd1e-4
300/patience30
best ValAcc

Best checkpoint 做 same-node mismatch：
固定 permutation 20260904，
破坏 factor interaction partner correspondence，
不训练。

输出：

outputs/perf_r2d16/semantic/
  semantic_results.csv
  semantic_mismatch.csv
  semantic_mechanism.csv
  semantic_perclass.csv
  R2D16_SEMANTIC_REPORT.md

Primary：

Gain_sem=SEM-HEAD

GO：
M/T/G macro>=+0.20pp
>=2/3 datasets positive
no F1 safety warning。

STRONG>=+0.50pp。

Diagnostics：
residual ratio per factor
real-mismatch
refined factor pair cosine
per-class F1

若 GO：
报告后停止，等待 schedule Prompt。
```

---

# 57. Prompt 7 — D1.6-E Controlled Schedule Study

```text
只对已经通过 frozen GO 的 candidate 执行。

Candidate 只能来自：
PRODDIFF
FiLM
SemanticResidual

最多选择 2 个。

Selection priority：
1. 3-seed formal GO
2. parent-robust
3. Macro-F1 safe
4. correspondence/message novelty stronger

对每 candidate/parent/dataset/seed：

保存并复用 exact：
parent checkpoint
adapter init state
classifier init state

比较三种 schedule：

E0 FROZEN：
parent frozen
adapter+classifier train。

E1 FULL-WARMSTART：
从同一个 parent checkpoint 出发，
parent ALL trainable from epoch1
adapter+classifier trainable。
注意不是 from-scratch。

E2 GRADUAL：
epoch1-30:
 parent frozen

epoch31+:
 unfreeze parent graph path + fusion
 keep P0 factorizer frozen

同一：
lr=1e-3
wd=1e-4
300 epochs
patience30
best ValAcc

第一轮：
Movies/Toys/Grocery seed42。

如果 schedule 差异值得确认，再补 43/44。

Gradient trajectory：

epoch1
epoch10
epoch30
first epoch after unfreeze
best checkpoint

对 trainable parent groups：
graph
fusion

计算 branch ON vs branch OFF：

g_on
g_off
g_delta=g_on-g_off

输出：
norm_delta/norm_off
cos(g_off,g_delta)

同时：
Acc
Macro-F1
per-class
adapter residual ratio
message novelty

Optimization coupling strong evidence：

frozen candidate GO，
FULL-FROZEN <= -0.30pp，
且 GRADUAL-FROZEN >= -0.10pp
或 GRADUAL > FULL by >=+0.20pp，
并且 FULL early gradient conflict 明显更强。

Joint-compatible：
FULL >= FROZEN-0.10pp 且 F1 safe。

输出：

outputs/perf_r2d16/schedule/
  schedule_results.csv
  schedule_gradient_trajectory.csv
  schedule_mechanism.csv
  R2D16_SCHEDULE_REPORT.md

完成后停止。
```

---

# 58. Prompt 8 — D1.6 Final Synthesis

```text
R2-Design-1.6 所有允许阶段已完成。

不要跑新实验。
不要 Test。
不要调参。
不要实现 R2-Design-2。

读取：

parent metrics
dual-parent propagation
interaction screen/confirm
semantic screen
schedule study

输出：

outputs/perf_r2d16/summary/
  R2D16_MASTER_TABLE.csv
  R2D16_HYPOTHESIS_LEDGER.csv
  R2D16_FINAL_DIAGNOSIS.md

必须回答：

1. A0/B0 的正式角色是否应保持：
   A0 performance / B0 diagnostic？
2. B0 与 A0 的 Macro-F1 是否存在此前遗漏的系统差异？
3. 2-hop 是否 cross-parent stable？
4. high-pass 是否 cross-parent stable？
5. PRODDIFF 是否在 A0/B0 上有真实 value？
6. PRODDIFF 是否超过 parameter-matched CONCAT？
7. FiLM 是否超过 PRODDIFF/CONCAT？
8. mismatch 是否证明 target-source correspondence？
9. message novelty/effective rank 是否证明 cell specialization？
10. semantic residual-only 是否有效？
11. semantic residual 是否 parent-robust？
12. frozen vs full vs gradual 哪个最优？
13. 是否真正证明 optimization coupling？
14. Macro-F1 safety 是否通过？
15. 下一阶段进入 Route A/B/C/D/E 哪一条？

Hypothesis ledger status 只能：

SUPPORTED
PARENT-ROBUST
PARENT-SPECIFIC
CONDITIONAL
WEAK
CLOSED
OPEN

最后给：

R2-Design-1.6:
PASS / PARTIAL / NO-GO

Recommended R2-Design-2 Route:
A Vector Functional
B Semantic Refinement
C Factor-Specific Multi-Scale
D Hybrid
E Task-Aware Relation Learning

不要设计最终模型。
等待 ChatGPT / 人工审查。
```

---

# 59. 完整执行后返给我的材料

请把以下全部返给我：

```text
outputs/perf_r2d16/audit/R2D16_AUDIT.md

outputs/perf_r2d16/parent_metrics/
outputs/perf_r2d16/propagation/
outputs/perf_r2d16/interaction/
outputs/perf_r2d16/interaction_confirm/   # 若进入
outputs/perf_r2d16/semantic/
outputs/perf_r2d16/schedule/              # 若进入
outputs/perf_r2d16/summary/

R2D16_FINAL_DIAGNOSIS.md
R2D16_MASTER_TABLE.csv
R2D16_HYPOTHESIS_LEDGER.csv

parent_metrics.csv
propagation_factor.csv
propagation_final.csv

interaction_results.csv
interaction_mismatch.csv
interaction_message_novelty.csv
interaction_expert_similarity.csv
interaction_effective_rank.csv

semantic_results.csv
semantic_mismatch.csv
semantic_mechanism.csv

schedule_results.csv
schedule_gradient_trajectory.csv

最新 GitHub commit
```

---

# 60. 本阶段最终纪律

本轮不再允许下列逻辑：

```text
一个 seed 高 → 立刻升级为核心模块
gate 非均匀 → 机制有效
branch-on > branch-off in co-adapted checkpoint → causal gain
B0 没赢 A0 → 不能做 frozen diagnosis
参数更多 → automatically better
```

真正进入 R2-Design-2 必须至少满足：

\[
\boxed{
\textbf{
stable performance
+
matched-control evidence
+
correspondence/functional novelty
+
optimization compatibility
}
}
\]

---

# 61. R2-Design-1.6 的最终科学问题

\[
\boxed{
\textbf{
Can the interaction or propagation signals identified in R2-0
be converted into reproducible vector-valued computation
on a frozen stable representation,
and can that computation survive controlled unfreezing?
}
}
\]

如果这个问题回答清楚，下一阶段才有资格正式进入 R2-Design-2。
