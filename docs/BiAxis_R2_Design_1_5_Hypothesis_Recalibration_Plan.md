# Bi-Axis R2-Design-1.5
## Hypothesis Recalibration, Propagation-Basis Audit & Interaction-Realization Diagnosis

**Repository:** `CrisRipper777/0901`  
**Previous stage:** `R2-Design-1 = NO-GO as implemented`  
**Current best simple candidate:** `R2-B0`（目前仅 seed42，尚未 formal-confirm）  
**Stage objective:** 不再继续堆叠模块，而是系统回答：  
1. B0 的优势是否稳定；  
2. R2-F / R2-S 为什么失败；  
3. factor-specific multi-hop / high-pass propagation 是否真的有价值；  
4. R2-0C 的 target-source interaction headroom 是“信息真实但表达方式错误”，还是“probe-only headroom”；  
5. 若 interaction 有价值，哪一种 **vector-valued realization** 最值得进入 R2-Design-2。

---

# 0. 本阶段对已有结论的重新冻结

R2-Design-1.5 开始前，先修正几个容易过度解释的结论。

## 0.1 不能关闭整个 Relation Learning

关闭的是：

\[
\boxed{
\textbf{Current topology-only prototype realization}
}
\]

即：

\[
[\log d,P\log d,P^2\log d]
\rightarrow
K=4
\rightarrow
\Gamma
\rightarrow
OFR.
\]

暂不关闭：

\[
\boxed{
\textbf{task-aware / semantic-aware relation learning}
}
\]

包括未来可能的：

- semantic-aware edge roles；
- feature-conditioned relation inference；
- heterophily/frequency-aware channels；
- target-conditioned edge/message transformation。

---

## 0.2 R2-F 证伪的是 scalar-gated realization，不是整个 Functional Transfer

R2-F 当前实际做的是：

\[
g_i^{a\to b}\in\mathbb R
\]

\[
m_i^{a\to b}
=
g_i^{a\to b}
V_a(N_i^a).
\]

target 只决定“传多少”，没有真正决定“传什么”。

因此当前冻结：

```text
Scalar-gated functional routing = CLOSED
Vector-valued functional interaction = OPEN
Feature-wise FiLM-style modulation = OPEN
Low-rank interaction = OPEN (conditional)
```

---

## 0.3 Semantic Refinement 也没有被完全证伪

R2-S 同时包含：

```text
Adaptive scalar common gate
+
Factor interaction residual
```

Movies 出现：

\[
w_t\approx0,\quad w_v\approx1.
\]

所以当前关闭：

```text
scalar convex adaptive common fusion
```

但不关闭：

```text
common interaction residual
factor interaction residual
ownership-preserving semantic refinement
```

---

## 0.4 Simple 1-hop strong ≠ multi-hop unnecessary

当前证据更适合写成：

```text
global multi-hop default = unsupported
factor-specific multi-hop = OPEN
high-pass / diversification = OPEN
```

尤其 R2-0B 中：

\[
P_t: G_2>G_1
\]

跨 Movies/Toys/Grocery 稳定存在。

---

# 1. R2-Design-1.5 的四条主线

整个阶段只围绕四条线展开：

\[
\boxed{\textbf{A. B0 Stability}}
\]

\[
\boxed{\textbf{B. Failure Decomposition}}
\]

\[
\boxed{\textbf{C. Propagation Basis}}
\]

\[
\boxed{\textbf{D. Interaction Realization}}
\]

不要在本阶段加入：

```text
OT/UOT
new contrastive loss
synergy loss
MoE training
Graph Transformer
RoleMAG-style predefined edge roles
new K-prototype relation system
large hyperparameter sweep
Test
```

MoE / attention / edge-role learning是否进入后续阶段，由本轮证据决定。

---

# 2. 总执行顺序

```text
D1.5-0  Audit + infrastructure
        ↓
D1.5-A  B0 formal confirmation
        ↓
D1.5-B  F/S best-checkpoint counterfactual + optimization diagnosis
        ↓
D1.5-C  Frozen-B0 propagation-basis audit
        ↓
        ├─ 若 multi-hop/high-pass GO → D1.5-C2 minimal end-to-end propagation variant
        ↓
D1.5-D  Frozen-B0 interaction-realization audit
        ↓
        ├─ 若 interaction adapter GO → D1.5-D2 3-seed confirmation
        ↓
D1.5-E  Final synthesis / route decision
```

所有阶段：

```text
Val only
No Test
```

---

# 3. D1.5-0 — Repository Audit & Analysis Infrastructure

目标：

1. 确认当前 main commit 与 R2-Design-1 报告一致；
2. 不改变 B0/F/S/J 已有模型行为；
3. 为 counterfactual / propagation / frozen-adapter 诊断建立独立分析工具；
4. 统一输出格式；
5. 建立 Macro-F1 safety 与 message/gradient diagnostics。

建议新增：

```text
src/analysis/
  perf_r2d15_utils.py

scripts/
  perf_r2d15_a_b0_confirm.py
  perf_r2d15_b_counterfactual.py
  perf_r2d15_c_propagation_basis.py
  perf_r2d15_c2_propagation_train.py
  perf_r2d15_d_interaction_adapter.py
  perf_r2d15_d2_interaction_confirm.py
  summarize_perf_r2d15.py

src/models/
  biaxis_r2d15_adapters.py

tests/
  test_perf_r2d15_utils.py
  test_biaxis_r2d15_adapters.py

outputs/perf_r2d15/
  audit/
  b0_confirm/
  counterfactual/
  propagation/
  propagation_train/
  interaction/
  interaction_confirm/
  summary/
```

禁止修改：

```text
biaxis_p0.py
biaxis_p1.py
biaxis_p2.py
biaxis_p3.py
biaxis_final.yaml
```

`biaxis_r2.py` 也尽量不改；counterfactual 优先通过 hooks / helper / eval flags 完成。

---

# 4. 新的统一诊断体系

以后不能只看 gate。

本阶段统一增加六类 diagnostics。

## 4.1 Optimization diagnostics

对 shared parameter groups：

```text
factorizer
source_transforms
fusion
classifier
```

计算：

### Full gradient

\[
g_{full}
\]

### Branch-off gradient

\[
g_{off}
\]

### Approximate branch-induced gradient

\[
g_{\Delta}
=
g_{full}-g_{off}.
\]

输出：

\[
\|g_{off}\|
\]

\[
\|g_\Delta\|
\]

\[
\frac{\|g_\Delta\|}{\|g_{off}\|+\epsilon}
\]

\[
\cos(g_{off},g_\Delta).
\]

Primary 用：

```text
CE(train nodes) only
```

Secondary 可同时记录：

```text
CE + P0 aux
```

但不要据此调 loss weight。

---

## 4.2 Parameter drift

对于 B0 与同 seed F/S best checkpoint 中的 shared groups：

\[
D(\theta)
=
\frac{
\|\theta_{variant}-\theta_{B0}\|_2
}{
\|\theta_{B0}\|_2+\epsilon
}.
\]

输出：

```text
factorizer drift
source_transform drift
fusion drift
```

---

## 4.3 Representation drift

在 Val nodes 上比较：

```text
B0 z_final
F with functional branch OFF
S with semantic branches OFF
```

计算：

```text
mean cosine similarity
linear CKA
mean L2 relative drift
```

---

## 4.4 Message novelty

以后任何新 message：

\[
m_{new}^{a\to b}
\]

都必须与 B0 target-base message：

\[
m_{base}^{b}
\]

比较：

\[
Cos_{new,base}^{a\to b}
\]

以及 orthogonal novelty：

\[
Novelty
=
\frac{
\|m_{new}-Proj_{m_{base}}(m_{new})\|
}{
\|m_{new}\|+\epsilon
}.
\]

还要计算 9 cell outputs 的：

```text
pairwise cosine matrix
effective rank
mean off-diagonal cosine
```

如果：

```text
gate 很不同
但 message novelty≈0
```

则不能宣称 expert specialization。

---

## 4.5 Counterfactual causality

best checkpoint 必须支持：

```text
full
branch off
diagonal only
off-diagonal only
source-row only
```

用同一 checkpoint、同一 classifier 做 forward counterfactual。

---

## 4.6 Task safety

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

规则：

\[
\Delta MacroF1<-0.50pp
\]

即使 Accuracy 正增益，也必须标记：

```text
SAFETY WARNING
```

---

# 5. D1.5-A — B0 Formal Confirmation

这是本阶段第一优先级。

当前 B0 只有：

```text
Movies/Toys/Grocery
seed42
```

不能直接升级为正式 parent。

---

## 5.1 数据集与 seeds

全部 NC datasets：

```text
Movies
Toys
Grocery
ele-fashion
Reddit-S
```

seeds：

```text
42
43
44
```

现有 seed42 B0 若 checkpoint/config/hash 一致：

```text
直接复用
```

只补缺失 runs。

A0：

```text
复用已有 formal 42/43/44 Val reference
```

禁止重新选超参。

---

## 5.2 输出

每 dataset：

```text
B0 mean±std
A0 mean±std
paired seed delta
positive seed count
Macro-F1 delta
best epoch distribution
train-val gap
params
peak memory
```

---

## 5.3 B0 Parent Verdict

### STRONG PARENT

Target M/T/G：

\[
mean_{M/T/G}(B0-A0)\ge +0.30pp
\]

且：

```text
>=2/3 target datasets mean positive
对应 positive dataset >=2/3 seeds positive
```

guards：

```text
ele-fashion mean delta >= -0.20pp
Reddit-S mean delta >= -0.20pp
```

且任何 guard 单 seed：

```text
不得 < -0.50pp
```

### ACCEPTABLE PARENT

\[
mean_{M/T/G}\ge0
\]

且：

```text
无 target dataset mean < -0.30pp
guards mean >= -0.20pp
```

### UNSTABLE

```text
M/T/G mean in [-0.15, 0)
或 seed direction highly inconsistent
```

### REJECT

\[
mean_{M/T/G}<-0.15pp
\]

或：

```text
任一 target mean < -0.50pp
任一 guard mean < -0.30pp
```

---

## 5.4 Gate

若：

```text
B0 = STRONG / ACCEPTABLE
```

才允许后面的 frozen-B0 adapter training。

若：

```text
B0 = REJECT
```

仍可完成 D1.5-B counterfactual，但停止 C/D 新训练，等待人工复审。

---

# 6. D1.5-B — R2-F / R2-S Counterfactual Failure Decomposition

这一阶段 **不重新训练 F/S**。

只读取已有：

```text
Movies/Toys/Grocery seed42
B0 / F / S best checkpoints
```

---

# 7. R2-F Counterfactuals

对同一 trained F checkpoint：

## F0 — Full

正常 F。

## F1 — Functional OFF

\[
\rho_{func}=0
\]

其他 trained weights 全不变。

## F2 — Diagonal Functional Only

仅：

\[
C\to C,\quad Pt\to Pt,\quad Pv\to Pv.
\]

## F3 — Off-Diagonal Functional Only

仅：

\[
a\neq b.
\]

## F4/F5/F6 — Source-row only

分别只保留：

```text
C-source row
Pt-source row
Pv-source row
```

base B0 diagonal path 始终保留。

---

## 7.1 定义三个关键量

### Forward Functional Effect

\[
E_{forward}^{F}
=
Acc(F_{full})
-
Acc(F_{func-off}).
\]

如果显著负：

```text
functional forward branch itself harmful
```

### Co-adaptation Gap

\[
G_{coadapt}^{F}
=
Acc(F_{func-off})
-
Acc(B0).
\]

如果显著负：

```text
joint training changed the shared backbone unfavorably
```

### Off-Diagonal Value

\[
E_{offdiag}^{F}
=
Acc(F_{offdiag})
-
Acc(F_{func-off}).
\]

注意：

`diag-only/offdiag-only` 是 post-hoc counterfactual，不等价于重新训练该结构。

---

# 8. R2-S Counterfactuals

同一个 trained S checkpoint：

## S0 — Full

Adaptive Common + Semantic Residual。

## S1 — Common Only

```text
learned adaptive common ON
semantic residual OFF
```

## S2 — Residual Only / Fixed Common

强制：

\[
w_t=w_v=0.5
\]

但保留 trained semantic residual。

## S3 — Both OFF

```text
fixed common 0.5/0.5
semantic residual OFF
```

然后比较独立训练的 B0。

---

## 8.1 关键量

### Semantic total forward effect

\[
E_{forward}^{S}
=
Acc(S_{full})
-
Acc(S_{both-off}).
\]

### Co-adaptation gap

\[
G_{coadapt}^{S}
=
Acc(S_{both-off})
-
Acc(B0).
\]

### Common effect

近似：

\[
E_{common}
=
Acc(S_{common-only})
-
Acc(S_{both-off}).
\]

### Residual effect

近似：

\[
E_{residual}
=
Acc(S_{full})
-
Acc(S_{common-only}).
\]

同时报告：

\[
Acc(S_{residual-only})
\]

作为交叉参考。

必须注明：

```text
post-hoc module masking causes distribution shift;
these are diagnostic counterfactuals, not retrained causal effects.
```

---

# 9. D1.5-B Optimization Audit

对：

```text
F full vs func-off
S full vs both-off
```

在 best checkpoint 计算：

```text
CE-only train gradient
shared group gradient cosine
branch-induced gradient norm ratio
parameter drift vs B0
representation drift vs B0
```

同时输出：

```text
Val Accuracy
Macro-F1
per-class F1
confusion shift
```

---

# 10. Failure Classification

对每 dataset/variant分类：

### TYPE-A：Forward Harm

\[
E_{forward}<-0.20pp
\]

且：

\[
G_{coadapt}\approx0.
\]

### TYPE-B：Co-adaptation Harm

\[
G_{coadapt}<-0.20pp
\]

即使 branch OFF 也救不回来。

### TYPE-C：Both

两者都显著负。

### TYPE-D：Optimization masking

branch forward 本身正：

\[
E_{forward}>0
\]

但：

\[
G_{coadapt}<0
\]

导致整体失败。

---

# 11. D1.5-C — Frozen-B0 Propagation Basis Audit

目的：

重新检验：

```text
1-hop
2-hop
high-pass/diversification
```

而不是重新造 latent relation prototypes。

本阶段 **先 frozen probe，不训练新 GNN**。

---

# 12. Propagation Signals

从 frozen B0 factor states：

\[
H_0^a=F^a
\]

\[
H_1^a=P H_0^a
\]

\[
H_2^a=P H_1^a
\]

定义一阶 high-pass/diversification：

\[
H_{HP}^a
=
H_0^a-H_1^a.
\]

暂不增加：

```text
P3/P4
PPR
learned attention
edge roles
```

先判断最基本的多尺度证据。

---

# 13. Per-Factor Matched Probes

对每：

\[
a\in\{C,Pt,Pv\}
\]

统一 2d：

\[
X_{1}^{a}
=
[F^a|H_1^a]
\]

\[
X_{2}^{a}
=
[F^a|H_2^a]
\]

\[
X_{HP}^{a}
=
[F^a|H_{HP}^a].
\]

定义：

\[
\Delta_{2-1}^{a}
=
Probe(X_2^a)-Probe(X_1^a)
\]

\[
\Delta_{HP-1}^{a}
=
Probe(X_{HP}^a)-Probe(X_1^a).
\]

---

# 14. Joint Matched Probes

\[
L=[C|Pt|Pv].
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

全部 6d。

---

# 15. Final-Residual Matched Probes

以 B0：

\[
Z=z_{B0}.
\]

构造：

\[
[Z|H_1^C|H_1^{Pt}|H_1^{Pv}]
\]

\[
[Z|H_2^C|H_2^{Pt}|H_2^{Pv}]
\]

\[
[Z|H_{HP}^C|H_{HP}^{Pt}|H_{HP}^{Pv}].
\]

定义：

\[
\Delta_{2-1}^{final}
\]

和：

\[
\Delta_{HP-1}^{final}.
\]

---

# 16. Multi-Scale Upper Bound

仅作 upper bound：

\[
X_{MS}
=
[L|
H_1^C,H_1^{Pt},H_1^{Pv}|
H_2^C,H_2^{Pt},H_2^{Pv}|
H_{HP}^C,H_{HP}^{Pt},H_{HP}^{Pv}
].
\]

同时做固定 permutation control：

```text
seed=20260904
```

shuffle H2/HP node rows。

不要用 high-dimensional upper bound 单独做 GO。

---

# 17. Propagation Basis Verdict

Primary target datasets：

```text
Movies/Toys/Grocery
```

所有 42/43/44。

## 2-hop FACTOR GO

至少一个 factor \(a\)：

\[
mean_{>=2 datasets}\Delta_{2-1}^a\ge+0.30pp
\]

且：

```text
这些 dataset >=2/3 seeds positive
```

## 2-hop FINAL GO

\[
mean_{M/T/G}\Delta_{2-1}^{final}\ge+0.20pp
\]

且至少 2/3 dataset positive。

## High-pass GO

同样规则：

\[
\Delta_{HP-1}
\]

达到上述 factor/final 门槛。

如果：

```text
factor evidence strong
final residual weak
```

结论写：

```text
factor-specific inductive-bias evidence
not final missing-feature evidence
```

不要过度解释。

---

# 18. D1.5-C2 — Conditional Minimal Propagation Training

只有 C 出现 GO 才运行。

不要同时组合多个新 channel。

---

## 18.1 若只有 2-hop GO

Variant：

\[
\boxed{\text{B0 + H2 correction}}
\]

对每 factor：

\[
\Delta_{2}^a
=
\alpha_2^a
LN(W_2^a H_2^a)
\]

其中：

```text
W2_a = Linear(d,d,bias=False)
alpha2_a = 0 init
```

最终：

\[
F_{out}^{a}
=
F_{B0}^{a}
+
\Delta_2^a.
\]

不是替换 B0 1-hop，而是 zero-init correction。

---

## 18.2 若只有 high-pass GO

Variant：

\[
\boxed{\text{B0 + HP correction}}
\]

\[
\Delta_{HP}^{a}
=
\alpha_{HP}^a
LN(W_{HP}^a(H_0^a-H_1^a)).
\]

同样：

```text
alpha=0 init
```

---

## 18.3 如果二者都 GO

分别训练：

```text
B0+H2
B0+HP
```

本阶段不组合。

---

## 18.4 seed42 screen

Movies/Toys/Grocery seed42。

GO：

\[
mean(candidate-B0)\ge+0.20pp
\]

且 2/3 positive。

Safety：

```text
Macro-F1 不得任一 dataset < -0.50pp
```

只有通过后，才补：

```text
ele-fashion / Reddit-S seed42
```

以及：

```text
42/43/44 M/T/G
```

---

# 19. D1.5-D — Frozen-B0 Interaction Realization Audit

这是本阶段最关键实验。

目的：

把：

\[
\boxed{
\text{information existence}
}
\]

与：

\[
\boxed{
\text{joint-training co-adaptation}
}
\]

彻底拆开。

---

# 20. Frozen-B0 Training Protocol

对：

```text
Movies/Toys/Grocery seed42
```

加载 B0 best checkpoint。

冻结：

```text
factorizer
source transforms
B0 graph path
fusion
all B0 parameters
```

训练：

```text
adapter
+
fresh linear classifier
```

B0 model 全程 eval 模式：

```text
dropout off
```

---

# 21. Head-Only Control

必须先有：

\[
\boxed{\text{HEAD}}
\]

使用 frozen B0：

\[
z_{B0}
\]

只训练一个新 linear classifier。

Adapter variants 必须使用：

```text
同一 classifier initialization
同一 optimizer
同一 train/val
同一 early stop
```

建议每个 dataset 保存一次 head initial state，并在各 variant 复用。

这样比较的是：

\[
\boxed{
\text{adapter 带来的 representational value}
}
\]

而不是不同 classifier init。

---

# 22. Adapter 插入位置

从 frozen B0 提取：

### Pre-graph ownership factors

\[
F_i^b
\]

### 1-hop contexts

\[
N_i^a=P F^a.
\]

### B0 graph-updated factors

\[
F_{B0,out}^b.
\]

Adapter 学习：

\[
\Delta_i^b.
\]

最终：

\[
\hat F_i^b
=
F_{B0,out}^b
+
\Delta_i^b.
\]

然后使用 **frozen B0 fusion**：

\[
\hat z_i
=
Fusion_{B0}(
\hat F_i^C,\hat F_i^{Pt},\hat F_i^{Pv}
).
\]

再进入 fresh classifier。

---

# 23. 四个 Realization Candidates

---

## D0 — HEAD

无 adapter。

---

## D1 — SCALAR（decoupled scalar control）

用于回答：

> scalar gate 失败主要来自 joint training，还是表达能力本身不足？

\[
u_{ab}
=
[
F_b,N_a,F_b\odot N_a,|F_b-N_a|,e_a,e_b
].
\]

\[
g_{ab}=\sigma(MLP(u_{ab})).
\]

独立 adapter source transform：

\[
U_aN_a.
\]

\[
\Delta_{ab}
=
\alpha_b
g_{ab}U_aN_a.
\]

注意：

```text
U_a 不与 B0 共用
alpha_b = 0 init
```

所以严格 zero correction 起步。

---

## D2 — CONCAT-VECTOR（capacity control）

不使用显式 product/difference：

\[
u_{ab}^{concat}
=
[
F_b,N_a,e_a,e_b
].
\]

共享：

```text
Linear(2d+2type, 128)
GELU
Linear(128,d)
```

最后层：

```text
zero-init
```

得到：

\[
\Delta_{ab}^{concat}\in\mathbb R^d.
\]

---

## D3 — PRODDIFF-VECTOR（核心候选）

只使用 R2-0C 直接支持的 interaction：

\[
u_{ab}^{int}
=
[
F_b\odot N_a,
|F_b-N_a|,
e_a,e_b
].
\]

使用与 CONCAT **完全同维度、同 hidden、同 output** 的 MLP：

```text
Linear(2d+2type,128)
GELU
Linear(128,d)
```

最后层：

```text
zero-init
```

因此：

\[
\boxed{
D2 vs D3 = parameter-matched comparison
}
\]

这是本轮非常重要的控制。

---

## D4 — FiLM-VECTOR

更完整借鉴 target-conditioned feature-wise modulation。

输入：

\[
u_{ab}
=
[
F_b,N_a,F_b\odot N_a,|F_b-N_a|,e_a,e_b
].
\]

网络：

```text
Linear(4d+2type,128)
GELU
Linear(128,2d)
```

输出：

\[
\Delta\gamma_{ab},\beta_{ab}\in\mathbb R^d.
\]

最后层 zero-init：

\[
\Delta\gamma=0,\quad\beta=0.
\]

独立 source projection：

\[
v_a=U_aN_a.
\]

定义 **correction**：

\[
\Delta_{ab}^{film}
=
\Delta\gamma_{ab}\odot v_a+\beta_{ab}.
\]

注意：

这不是：

\[
\gamma\odot v+\beta
\]

直接替代 B0 message，

而是对 B0 的 zero-init correction。

---

# 24. Cell Aggregation

D2/D3/D4：

\[
\Delta_i^b
=
\frac13\sum_a\Delta_i^{a\to b}.
\]

暂不使用：

```text
softmax
MoE top-k
competitive router
```

本轮先验证 expert output 是否真的有价值。

---

# 25. Adapter Training

只训练：

```text
adapter
fresh classifier
```

默认：

```text
AdamW
lr=1e-3
wd=1e-4
300 epochs
patience=30
best by Val Accuracy
```

不扫参。

因为 backbone frozen，训练成本很低。

---

# 26. Interaction Negative Control

所有 D2/D3/D4 best checkpoint：

固定：

```text
perm seed = 20260904
```

沿 node 维度 permutation source contexts：

\[
N^a\rightarrow N_{\pi(i)}^a.
\]

重新 forward，不训练。

输出：

\[
Real-Mismatch.
\]

对于 D3/FILM：

如果真实 interaction 有意义，应：

\[
Real > Mismatch.
\]

---

# 27. Interaction Diagnostics

必须输出：

## 27.1 Performance

```text
Acc
Macro-F1
per-class F1
```

## 27.2 Adapter residual ratio

\[
\|\Delta^b\|/\|F_{B0,out}^b\|.
\]

## 27.3 Message novelty

每 9 cells：

```text
cos(delta_ab, B0 base message_b)
orthogonal novelty ratio
```

## 27.4 Expert specialization

9 cell outputs：

```text
pairwise cosine 9x9
mean off-diagonal cosine
effective rank
per-cell norm
```

## 27.5 Source/target contribution

不是 gate contribution，而是：

\[
\mathbb E\|\Delta_{ab}\|.
\]

输出 3×3 matrix。

---

# 28. Interaction Realization Verdict

Primary：

\[
Gain_D
=
mean_{M/T/G}
[
Acc(D)-Acc(HEAD)
].
\]

## STRONG

\[
Gain_D\ge+0.50pp
\]

且：

```text
>=2/3 datasets positive
real > mismatch
无 Macro-F1 safety warning
```

## GO

\[
Gain_D\ge+0.30pp
\]

且：

```text
>=2/3 datasets positive
```

并且至少满足一个：

### Interaction specificity

\[
D3-D2\ge+0.15pp
\]

macro；

或：

### Correspondence evidence

\[
Real-Mismatch\ge+0.20pp
\]

macro。

## WEAK

\[
+0.15\sim+0.30pp.
\]

## NO-GO

\[
<+0.15pp
\]

或高度不稳定。

---

# 29. 对四个 realization 的解释矩阵

## SCALAR GO，D3/D4 无额外价值

说明：

```text
主要问题是 joint-training coupling，
不是 scalar expression。
```

## SCALAR NO，D3 GO

说明：

\[
\boxed{
\text{scalar bottleneck}
}
\]

是 D1 失败核心之一。

## D3 > D2

说明：

```text
explicit product/difference interaction
比同参数量 concat transformation 更有价值
```

这是最强的 R2-0C→trainable-mechanism bridge。

## D4 > D3

说明：

```text
feature-wise target-conditioned modulation
比 direct interaction residual 更适合
```

## D2≈D3≈D4 全正

说明：

```text
capacity / vector correction 更重要，
interaction form 本身未被识别
```

## 全部 NO-GO

说明：

```text
R2-0C frozen interaction headroom
可能主要是 probe feature exposure，
并不容易变成 message correction。
```

---

# 30. D1.5-D2 — Interaction 3-Seed Confirmation

只有：

```text
D3 or D4
```

达到 GO 才执行。

不确认 scalar-only candidate 作为最终方法。

---

## 30.1 Guards

先：

```text
ele-fashion
Reddit-S
seed42
```

要求：

```text
Acc delta vs HEAD >= -0.20pp
Macro-F1 delta >= -0.50pp
```

---

## 30.2 Formal

再：

```text
Movies/Toys/Grocery
seeds42/43/44
```

每 seed 对应加载同 seed 的 frozen B0。

比较：

```text
candidate vs HEAD
candidate vs D2 CONCAT control
real vs mismatch
```

Formal GO：

\[
mean_{M/T/G}(candidate-HEAD)\ge+0.30pp
\]

且：

```text
>=2/3 datasets positive
对应 dataset >=2/3 seeds positive
guards safe
```

---

# 31. D1.5-E — Final Route Decision

最后不再自动实现新 architecture。

只做 synthesis。

必须回答：

1. B0 是否 formal stable？
2. R2-F 失败主要是：
   - forward harm？
   - co-adaptation？
   - both？
3. R2-S 失败中：
   - adaptive common 是不是主要问题？
   - factor interaction residual 是否仍可能有价值？
4. 2-hop 是否有 factor-specific stable evidence？
5. high-pass 是否有 stable evidence？
6. propagation candidate 是否 end-to-end GO？
7. scalar adapter 在 frozen B0 上是否有效？
8. vector residual 是否优于 scalar？
9. PRODDIFF 是否优于 parameter-matched CONCAT？
10. FiLM 是否优于 direct vector residual？
11. message novelty 是否真实存在？
12. expert outputs 是否真的 specialization？
13. Macro-F1 是否安全？
14. 下一步应该进入哪条路线？

---

# 32. Final Route Matrix

## Route A — Vector Functional Transfer

条件：

```text
B0 stable
D3/D4 GO
propagation basis weak/optional
```

进入：

\[
\boxed{
R2\text{-Design-2:
Vector-valued Functional Semantic Transfer
}
\]

并使用 staged training：

```text
B0 warm start
adapter zero-init
optional gradual unfreeze
```

---

## Route B — Factor-Specific Multi-Scale

条件：

```text
B0 stable
propagation basis GO
interaction realization weak
```

进入：

\[
\boxed{
R2\text{-Design-2:
Factor-Specific Multi-Scale Propagation
}
\]

重点借鉴：

```text
MixHop / GPR-GNN / ACM style
```

但保持 semantic-factor specific。

---

## Route C — Both

条件：

```text
Propagation GO
Interaction GO
```

不要立刻 MoE。

先分别 formal-confirm。

只有两类 expert 都独立有价值后，
Design-2 才考虑：

```text
multi-expert composition / MoE
```

---

## Route D — Both Weak

若：

```text
B0 stable
propagation weak
interaction weak
```

下一阶段才正式进入：

\[
\boxed{
\textbf{Task-aware Relation Learning Audit}
}
\]

重点：

```text
RoleMAG
NRI
IDGL
heterophily role learning
semantic-aware edge relation
```

此时才考虑 edge-level relation learning，
而不是现在提前加入。

---

## Route E — B0 Unstable

先停止所有 R2 architecture expansion。

重新确定：

```text
A0 vs B0
```

哪个才是可靠 parent。

---

# 33. Prompt 1 — D1.5-0 Audit + Infrastructure

```text
我们进入 Bi-Axis R2-Design-1.5。

阶段目标不是继续堆模型，而是做 Hypothesis Recalibration：

A. B0 formal stability
B. R2-F/R2-S failure decomposition
C. factor-specific propagation basis
D. interaction realization

本 Prompt 只做 repository audit + analysis infrastructure。
不要运行正式训练。

请审查：

src/models/biaxis_r2.py
src/models/biaxis_r2_components.py
src/models/biaxis_p0.py
src/models/biaxis_p1_components.py
src/tasks/nc.py
R2D1_FINAL_DIAGNOSIS.md
existing B0/F/S checkpoints and outputs

明确确认：

1. current commit 与 R2D1 报告一致；
2. B0/F/S checkpoint 路径；
3. 如何在不修改 trained weights 的前提下：
   - F functional OFF
   - diagonal-only
   - off-diagonal-only
   - source-row only
   - S common-only
   - fixed-common + residual
   - both-off
4. 如何提取：
   F pre-graph
   N=PF
   B0 graph-updated factors
   z_final
5. 如何计算 Val per-class F1 / confusion matrix；
6. 如何计算 CE-only gradient diagnostics；
7. 如何计算 linear CKA / parameter drift；
8. 全程禁止 Test。

新增：

src/analysis/perf_r2d15_utils.py
src/models/biaxis_r2d15_adapters.py
tests/test_perf_r2d15_utils.py
tests/test_biaxis_r2d15_adapters.py

先实现 helper / adapter modules / unit tests，
但不训练 adapter。

Unit tests 至少覆盖：

- counterfactual masks 数学正确；
- F func-off 等价 rho_func=0；
- diagonal/offdiag cell mask 不重叠且 union=all；
- source-row masks 正确；
- S fixed common exactly .5/.5；
- B0 factor/context extraction 与 forward 对齐；
- H1/H2/HP shape finite；
- HEAD/D2/D3/D4 adapter zero-init -> exact B0 factor output；
- D2/D3 parameter count matched；
- FiLM delta gamma/beta zero-init；
- permutation deterministic；
- no Test access。

输出：

outputs/perf_r2d15/audit/R2D15_AUDIT.md

然后停止。
```

---

# 34. Prompt 2 — D1.5-A B0 Formal Confirmation

```text
D1.5-0 Audit PASS。

现在只执行 B0 Formal Confirmation。

Datasets：
Movies
Toys
Grocery
ele-fashion
Reddit-S

Seeds：
42
43
44

Val only。
禁止 Test。

已存在且与当前 commit/config 完全一致的 seed42 B0 run 可复用，
不要重复。

A0 使用已有 formal 42/43/44 Val reference，
不要重新训练。

Protocol 保持：
300 epochs
patience30
AdamW lr=1e-3 wd=1e-4
best by Val Accuracy

输出：

outputs/perf_r2d15/b0_confirm/
  b0_confirm_results.csv
  b0_confirm_resource.csv
  R2D15_B0_CONFIRM_REPORT.md

必须包含：

per seed
3-seed mean
population std ddof=0
paired seed delta vs A0
positive seed count
Val Macro-F1 delta
best epoch
train-val gap

Verdict：

STRONG PARENT：
M/T/G macro >= +0.30pp
>=2/3 target means positive
对应 positive dataset >=2/3 seeds positive
guards mean >= -0.20pp
guard single seed >= -0.50pp

ACCEPTABLE：
M/T/G macro >=0
无 target mean < -0.30
guards mean >= -0.20

UNSTABLE：
macro [-0.15,0) 或 seed direction inconsistent

REJECT：
macro < -0.15
或 target mean < -0.50
或 guard mean < -0.30

完成后停止。
```

---

# 35. Prompt 3 — D1.5-B F/S Counterfactual + Optimization Diagnosis

```text
现在只执行 trained-checkpoint counterfactual。

不要重新训练 F/S。
不要 Test。

Datasets：
Movies/Toys/Grocery
seed42

F：

F_full
F_func_off
F_diag_only
F_offdiag_only
F_src_C
F_src_Pt
F_src_Pv

base diagonal path 始终保留。

S：

S_full
S_common_only
S_fixed_common_plus_residual
S_both_off

比较独立训练 B0 seed42。

输出：

ForwardEffect_F = full - func_off
CoAdaptGap_F = func_off - B0
OffdiagEffect_F = offdiag - func_off

SemanticForwardEffect =
S_full - S_both_off

CoAdaptGap_S =
S_both_off - B0

CommonEffect approx =
S_common_only - S_both_off

ResidualEffect approx =
S_full - S_common_only

必须注明 post-hoc masking distribution shift。

此外计算：

1. CE-only gradient diagnostics：
   full vs branch-off
   groups:
   factorizer/source_transforms/fusion/classifier

输出：
norm_off
norm_delta
delta/off ratio
cos(off,delta)

2. parameter drift vs B0：
factorizer
source transforms
fusion

3. representation drift：
mean cosine
linear CKA
relative L2

4. Val:
Acc
Macro-F1
per-class F1
confusion matrix

输出：

outputs/perf_r2d15/counterfactual/
  f_counterfactual.csv
  s_counterfactual.csv
  gradient_diagnostics.csv
  parameter_drift.csv
  representation_drift.csv
  per_class_metrics.csv
  R2D15_COUNTERFACTUAL_REPORT.md

报告必须按：

TYPE-A Forward Harm
TYPE-B Co-adaptation Harm
TYPE-C Both
TYPE-D Optimization masking

分类。

完成后停止。
```

---

# 36. Prompt 4 — D1.5-C Frozen-B0 Propagation Basis Audit

```text
如果 B0 = STRONG/ACCEPTABLE，执行本 Prompt。
若 B0=REJECT，不执行。

使用 B0 formal checkpoints。

Datasets：
Movies/Toys/Grocery
seeds42/43/44

guards ele-fashion/Reddit-S 可同时做 frozen probe，
但 primary verdict 只看 M/T/G。

Val only。
禁止 Test。
不训练新 GNN。

提取 pre-graph factors：

H0^a = F^a
H1^a = P H0^a
H2^a = P H1^a
HP^a = H0^a - H1^a

Fixed Ridge：
StandardScaler
RidgeClassifier(alpha=1.0)
TRAIN fit
VAL eval

Per-factor matched 2d：

[F|H1]
[F|H2]
[F|HP]

输出：

Delta_2_1_factor
Delta_HP_1_factor

Joint matched 6d：

[L|H1_C|H1_Pt|H1_Pv]
[L|H2_C|H2_Pt|H2_Pv]
[L|HP_C|HP_Pt|HP_Pv]

Final-residual matched：

[z_B0|H1_all]
[z_B0|H2_all]
[z_B0|HP_all]

Upper bound：

[L|H1_all|H2_all|HP_all]

并做 fixed permutation seed=20260904
shuffle H2/HP rows。

输出：

outputs/perf_r2d15/propagation/
  propagation_factor_probe.csv
  propagation_joint_probe.csv
  propagation_final_probe.csv
  propagation_upper_bound.csv
  propagation_shuffle.csv
  R2D15_PROPAGATION_REPORT.md

GO rule：

2-hop factor GO：
至少一个 factor
在 >=2 datasets mean(H2-H1)>=+0.30pp
且对应 dataset >=2/3 seeds positive。

2-hop final GO：
M/T/G macro final(H2-H1)>=+0.20
且 2/3 datasets positive。

High-pass 使用同规则。

不要因为 upper-bound 高就 GO。

完成后停止。
```

---

# 37. Prompt 5 — D1.5-C2 Conditional Propagation Training

```text
读取 D1.5-C verdict。

如果 2-hop / HP 都未 GO：
不执行任何 propagation training，
只写 NOT ENTERED 并停止。

如果只有 2-hop GO：

实现：
B0 + H2 zero-init correction

Delta2_a =
alpha2_a * LN(W2_a(H2_a))

W2 = Linear(d,d,bias=False)
alpha2 init=0

如果只有 HP GO：

DeltaHP_a =
alphaHP_a * LN(WHP_a(H0_a-H1_a))

alpha init=0。

如果二者都 GO：
分别作为两个独立 variants；
不要组合。

第一轮：

Movies/Toys/Grocery
seed42
Val only

GO：
mean(candidate-B0)>=+0.20pp
>=2/3 datasets positive
且任一 dataset Macro-F1 delta 不得 < -0.50pp。

若 seed42 GO：
补 ele-fashion/Reddit-S seed42；
若 guards safe，再补 M/T/G seeds43/44。

输出：

outputs/perf_r2d15/propagation_train/
  propagation_train_results.csv
  propagation_train_mechanism.csv
  R2D15_PROPAGATION_TRAIN_REPORT.md

diagnostics：
alpha per factor
new message norm
message vs B0 base cosine
novelty ratio
Macro-F1 safety

完成后停止。
```

---

# 38. Prompt 6 — D1.5-D Frozen-B0 Interaction Realization Screen

```text
只有 B0 STRONG/ACCEPTABLE 才执行。

目标：
在冻结 B0 的条件下，
区分：
joint-training coupling
vs
scalar expression bottleneck
vs
真正 interaction representation value。

Datasets：
Movies/Toys/Grocery
seed42
Val only
No Test。

加载每个 dataset 的 B0 best checkpoint。

冻结：
整个 B0 model including fusion。

训练：
adapter + fresh classifier only。

必须先保存一个 classifier initial state，
HEAD/D1/D2/D3/D4 完全复用。

Variants：

D0 HEAD：
frozen z_B0 + fresh classifier。

D1 SCALAR：
独立 scalar functional adapter，
不能共用 B0 source transforms；
adapter final residual scale alpha=0 init。

D2 CONCAT-VECTOR：
input=[F_b,N_a,e_src,e_tgt]
shared MLP:
Linear(2d+2type,128)
GELU
Linear(128,d)
last layer zero-init。

D3 PRODDIFF-VECTOR：
input=[F_b*N_a,abs(F_b-N_a),e_src,e_tgt]
与 D2 完全相同 hidden/output/parameter count。

D4 FiLM-VECTOR：
input=[F_b,N_a,F_b*N_a,abs(F_b-N_a),types]
Linear(...,128)
GELU
Linear(128,2d)
输出 delta_gamma,beta
last layer zero-init
delta_ab =
delta_gamma * U_a(N_a) + beta

所有 vector adapter：

Delta_b = mean_a Delta_ab

最终：

Fhat_b =
F_B0_out_b + Delta_b

然后：
frozen B0 fusion
fresh classifier。

训练：
AdamW lr=1e-3 wd=1e-4
300 epochs
patience30
best Val Accuracy

不扫参。

Best checkpoint 做 mismatch：
perm seed=20260904
N_a -> N_perm_a
不训练。

输出：

outputs/perf_r2d15/interaction/
  interaction_results.csv
  interaction_resource.csv
  interaction_perclass.csv
  interaction_mismatch.csv
  interaction_message_novelty.csv
  interaction_expert_similarity.csv
  interaction_effective_rank.csv
  R2D15_INTERACTION_REPORT.md

必须计算：

Gain_D = D - HEAD

D3-D2
D4-D2
Real-Mismatch

residual ratios

9-cell:
norm matrix
cosine to B0 base message
orthogonal novelty
pairwise cosine 9x9
effective rank

Verdict：

STRONG：
M/T/G Gain>=+0.50pp
2/3 positive
real>mismatch
no F1 safety warning

GO：
Gain>=+0.30pp
2/3 positive
并且：
D3-D2>=+0.15pp
或 Real-Mismatch>=+0.20pp

WEAK：
+0.15~+0.30

NO-GO：
<+0.15

解释矩阵必须严格按计划执行。

完成后停止。
```

---

# 39. Prompt 7 — D1.5-D2 Interaction Confirmation

```text
读取 D1.5-D。

只有 D3 或 D4 达到 GO 才执行。

若两者都 GO：
两者都确认，不按 seed42 最佳值临时只挑一个。

Step A guards：

ele-fashion
Reddit-S
seed42

比较：
candidate vs HEAD

要求：
Acc >= -0.20pp
Macro-F1 >= -0.50pp

Step B formal：

Movies/Toys/Grocery
seeds42/43/44

每 seed 使用对应 frozen B0。

输出：

candidate vs HEAD
candidate vs D2 CONCAT control
real vs mismatch

3-seed mean/std(ddof=0)
positive seed count

Formal GO：

M/T/G candidate-HEAD macro >= +0.30pp
>=2/3 datasets positive
对应 dataset >=2/3 seeds positive
guards safe

输出：

outputs/perf_r2d15/interaction_confirm/
  interaction_confirm_results.csv
  interaction_confirm_mechanism.csv
  R2D15_INTERACTION_CONFIRM_REPORT.md

禁止 Test。

完成后停止。
```

---

# 40. Prompt 8 — Final Synthesis

```text
R2-Design-1.5 所有允许执行的阶段已完成。

不要跑新实验。
不要 Test。
不要调参。
不要实现 R2-Design-2。

读取：
B0 confirm
counterfactual
propagation
conditional propagation train
interaction
interaction confirm

输出：

outputs/perf_r2d15/summary/
  R2D15_MASTER_TABLE.csv
  R2D15_HYPOTHESIS_LEDGER.csv
  R2D15_FINAL_DIAGNOSIS.md

Hypothesis ledger 至少包含：

Current topology-prototype relation
Task-aware relation learning
Scalar functional routing
Vector functional interaction
FiLM-style modulation
Adaptive scalar common
Factor interaction residual
1-hop propagation
factor-specific 2-hop
high-pass/diversification
MoE
edge-level relation learning

status 只能使用：

SUPPORTED
OPEN
CONDITIONAL
WEAK
CLOSED

必须回答：

1. B0 是否 formal stable？
2. F failure type？
3. S failure type？
4. adaptive common 是否主要伤害源？
5. factor residual 是否仍 OPEN？
6. 2-hop evidence？
7. high-pass evidence？
8. propagation end-to-end 是否兑现？
9. frozen scalar adapter 是否有效？
10. D3 是否超过 D2 parameter-matched control？
11. FiLM 是否超过 D3？
12. mismatch 是否证明 correspondence？
13. message novelty / effective rank 是否证明 experts 真正不同？
14. 是否存在 Macro-F1 safety issue？
15. 下一阶段选择 Route A/B/C/D/E 中哪一个？

最后给：

R2-Design-1.5:
PASS / PARTIAL / NO-GO

Recommended next route:
A / B / C / D / E

不要写正式 R2 方法。
等待 ChatGPT / 人工审查。
```

---

# 41. 最终需要返给我的材料

完整执行后请返回：

```text
outputs/perf_r2d15/audit/R2D15_AUDIT.md

outputs/perf_r2d15/b0_confirm/
outputs/perf_r2d15/counterfactual/
outputs/perf_r2d15/propagation/
outputs/perf_r2d15/propagation_train/    # 若进入
outputs/perf_r2d15/interaction/
outputs/perf_r2d15/interaction_confirm/ # 若进入
outputs/perf_r2d15/summary/

R2D15_FINAL_DIAGNOSIS.md
R2D15_MASTER_TABLE.csv
R2D15_HYPOTHESIS_LEDGER.csv

gradient_diagnostics.csv
parameter_drift.csv
representation_drift.csv
per_class_metrics.csv

propagation_factor_probe.csv
propagation_final_probe.csv

interaction_results.csv
interaction_mismatch.csv
interaction_message_novelty.csv
interaction_expert_similarity.csv
interaction_effective_rank.csv

最新 GitHub commit
```

---

# 42. 本阶段最重要的纪律

不要因为某一个 variant seed42 高，就立刻扩展模型。

不要因为：

```text
gate non-uniform
alpha non-zero
expert utilization diverse
```

就说机制有效。

必须同时有：

\[
\boxed{
\text{performance evidence}
+
\text{functional novelty evidence}
+
\text{stability evidence}
}
\]

才允许进入 R2-Design-2。

---

# 43. R2-Design-1.5 最终科学问题

本阶段最终不是问：

> 哪个模块最好？

而是问：

\[
\boxed{
\textbf{
What is the true missing computation beyond a strong
factor-preserving 1-hop backbone:
propagation scale,
interaction representation,
or task-aware relation structure?
}
}
\]

只有回答这个问题以后，才进入正式 R2-Design-2。
