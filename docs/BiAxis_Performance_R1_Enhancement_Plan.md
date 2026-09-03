# Bi-Axis Performance-R1 — 性能补强实验推进计划
## 主题：先修 Relation/Context，再修 Γ，最后补 Movies/Toys 的 2-hop 能力

> Repository: `CrisRipper777/0901`  
> 基准模型：当前 frozen `biaxis_final` / P3 OFR  
> 当前任务：仅 Node Classification  
> 数据集：Movies / Toys / Grocery / ele-fashion / Reddit-S  
> 正式 seeds：42 / 43 / 44  
> **模型开发阶段只依据 Validation 选择方案，不依据 Test 继续改结构。**

---

# 0. R1 为什么这样推进？

Performance-R0 已经把当前瓶颈定位得比较清楚：

## Healthy / 暂时不动

- Semantic Factorizer 没有明显共同信息损失；
- Graph propagation 本身非常有效；
- Local / No-Transport state 有真实功能；
- node-specific graph demand 有真实功能；
- Hierarchical Operator 已经过 P3 验证，当前不是首要瓶颈。

## 当前主要问题

### 1. Relation 不塌缩，但缺少 task-semantic relevance

当前 `K=4` relation prototypes 能形成结构划分，但不同 relation 的：

```text
semantic edge similarity
train-only homophily
relation-conditioned context utility
```

差异普遍偏弱。

因此问题首先不是：

```text
K 太少
```

而是：

```text
Relation / Context 不够“值得选”
```

---

### 2. Γ 的 Local / graph-demand 分支有效，但 relation-selection 分支基本惰性

R0 frozen counterfactual：

```text
Current Γ ≈ Uniform relation routing
```

几乎 5/5 数据集成立。

因此：

\[
\boxed{
\Gamma\text{ 本身不需要推翻，重点是 relation-side evidence 与 score 输入。}
}
\]

---

### 3. Movies / Toys 存在额外 2-hop potential

R0 中相对 1-hop probe：

```text
Movies second-hop extra ≈ +0.60 pp
Toys   second-hop extra ≈ +0.55 pp
```

而 Grocery / ele-fashion / Reddit-S 的额外 2-hop 潜力很弱。

因此 multi-hop 应做成：

```text
adaptive / residual / optional
```

而不是所有节点强制加深。

---

# 1. R1 总体路线

R1 不做“大 V2 一次性堆模块”，而是按因果链逐层修：

```text
R1-0  Repository / Same-code-path Baseline Audit
  ↓
R1-A  Factor-conditioned Reliable Context
  ↓
  若 relation/context 仍弱
  └── R1-A2 Semantic-calibrated Relation Posterior（条件执行）
  ↓
R1-B  Evidence-aware Γ
  ↓
R1-C  Adaptive 2-hop Trajectory
  ↓
R1-F  Final combination / freeze
```

其中：

\[
\boxed{
\textbf{R1-A 是第一优先级，必须先单独验证。}
}
\]

---

# 2. 研究纪律

## 2.1 保留当前 Final 作为 V1 Frozen Reference

禁止修改：

```text
biaxis_p0
biaxis_p1
biaxis_p2
biaxis_p3
biaxis_final
及其 frozen configs
```

新增独立性能分支：

```text
src/models/biaxis_perf_r1.py
src/models/biaxis_perf_r1_components.py
configs/model/biaxis_perf_r1.yaml
tests/test_biaxis_perf_r1.py
```

---

## 2.2 Same-code-path baseline 必须存在

新模型支持：

```text
mode = baseline
```

其中：

```text
reliability OFF
evidence-aware Γ OFF
multi-hop OFF
```

必须与 `biaxis_final` 在**相同权重**下数学等价。

这个 `R1-B0` 是后续所有 paired comparison 的基线。

---

## 2.3 不重新开启这些方向

R1 禁止：

```text
重新调 UOT
改 Operator family
直接 K=8/16 sweep
改 Semantic Factorizer loss
加入完整 MoE
加入 topology rewiring
加入 pseudo nodes
```

除非 R1-A/B/C 全部失败，再重新规划 R2。

---

# 3. R1 的统一实验门槛

弱项数据集：

```text
Movies
Toys
Grocery
```

guard 数据集：

```text
ele-fashion
Reddit-S
```

---

## 3.1 Seed42 Screening GO

一个候选进入 multi-seed confirm，至少满足：

\[
\boxed{
\text{mean }\Delta ValAcc_{\{M,T,G\}}\ge+0.30\text{ pp}
}
\]

并且：

```text
至少 2/3 个弱项数据集 > +0.20 pp
```

同时：

```text
ele-fashion / Reddit-S 均不得 < -0.20 pp
```

Secondary：

```text
Val Macro-F1 不得出现明显系统性下降
```

---

## 3.2 3-seed Formal GO

paired seeds 42/43/44：

### Strong GO

满足任一：

```text
至少 2/3 弱项 dataset mean gain ≥ +0.50 pp
```

或：

```text
M/T/G 平均 gain ≥ +0.50 pp
```

并且 guard 不系统退化。

### GO

```text
M/T/G 平均 gain ≥ +0.30 pp
至少 2/3 为正
```

### Borderline

```text
+0.15 ~ +0.30 pp
```

且 2/3 seeds positive：

可以加 seeds45/46；
不要开启 deterministic mode。

### NO-GO

```text
平均 < +0.15 pp
或仅单 dataset 有效且 guards 明显退化
```

不继续“救”该模块。

---

# 4. R1-A — Factor-conditioned Semantic Edge Reliability
## 第一刀：提升 relation-conditioned neighborhood context 的质量

---

# 5. R1-A 的核心问题

当前：

\[
g_{ik}^{f}
=
\frac{
\sum_j r_{ji,k}^{str} f_j
}{
\sum_j r_{ji,k}^{str}+\epsilon
}.
\]

问题：

> 一旦一个邻居被 structural relation \(R_k\) 接收，
> relation 内部所有邻居仍然只按 \(r^{str}\) 加权；
> 没有判断该邻居对当前 semantic factor \(f\) 是否可信。

R0 表明：

```text
Graph propagation 很强，
但 Relation Context 比 plain neighbor mean 的额外价值很有限。
```

所以 R1-A 不改变 relation prototypes，
只修 Context。

---

# 6. R1-A1：Factor-conditioned Edge Reliability

定义：

\[
\eta_{ji}^{f}\in(0,2)
\]

表示：

> 边 \(j\to i\) 对 factor \(f\) 的 semantic reliability。

新的 context：

\[
\boxed{
g_{ik}^{f}
=
\frac{
\sum_j
r_{ji,k}^{str}
\eta_{ji}^{f}
f_j
}{
\sum_j
r_{ji,k}^{str}
\eta_{ji}^{f}
+\epsilon
}
}
\]

注意：

```text
r_str 决定 structural relation prior
η_f 决定 neighbor reliability
```

不改变 Relation prototype identity。

---

# 7. Reliability Scorer 推荐实现

为了控制显存和参数，不直接在 E×F 上拼 4×128 高维特征。

先每 factor 投影：

\[
u_i^f=P_f f_i,\qquad
P_f:\mathbb R^{128}\to\mathbb R^{32}.
\]

推荐三个 factor-specific projection：

```text
P_C
P_Pt
P_Pv
```

然后构造 symmetric pair token：

\[
v_{ji}^{f}
=
[
u_i^f+u_j^f
\Vert
|u_i^f-u_j^f|
\Vert
u_i^f\odot u_j^f
\Vert
\cos(u_i^f,u_j^f)
].
\]

共享小 MLP：

\[
\delta_{ji}^{f}=MLP_\eta(v_{ji}^{f})
\]

并定义：

\[
\boxed{
\eta_{ji}^{f}
=
2\sigma(\delta_{ji}^{f})
}
\]

### 初始化

最后一层：

```text
weight = 0
bias = 0
```

所以：

\[
\eta_{ji}^{f}=1
\]

严格从当前模型开始。

---

# 8. 为什么先用 symmetric reliability？

第一阶段只回答：

> 某条 observed edge 对 factor \(f\) 是否可靠？

不同时加入：

```text
directional attention
relation-specific semantic logits
topology rewiring
```

减少变量。

对于无向图：

\[
\eta_{ij}^{f}=\eta_{ji}^{f}
\]

更容易解释和测试。

---

# 9. R1-A Memory Discipline

禁止保存：

```text
[E,F,d]
[E,F,K,d]
```

Reliability 必须：

```text
edge chunk
× factor loop
```

即时计算。

允许持久：

```text
r            [E,K]
g_perm       [N,F,K,d]
effective_mass [N,F,K]
```

不要比当前 P3 增加一个新的超大 edge-level tensor。

---

# 10. R1-A 初始版本不要加任何新正则

不要第一轮加入：

```text
reliability entropy loss
sparsity loss
mean-to-one penalty
contrastive loss
```

先看 supervised NC 是否自然学出：

\[
\eta\neq1.
\]

诊断记录：

```text
mean / std / p10 / p50 / p90
frac η<0.5
frac η>1.5
per factor
corr(η, semantic cosine)
effective relation mass
```

---

# 11. R1-A 必须重新计算的机制诊断

和 R0 做可比：

### Context Diversity

\[
D_{ctx}^{f}
\]

### Relation Context Potential

\[
\Delta_{relctx}^{f}
\]

### Weighted semantic coherence

用：

\[
r_{ji,k}\eta_{ji}^{f}
\]

重新计算每 relation 的 semantic edge coherence。

### Routing counterfactual

至少重新计算：

```text
Current
Uniform
Availability
```

看 A1 是否让：

\[
Current-Uniform
\]

开始变得可利用。

---

# 12. R1-A 期待的机制修复信号

Grocery 是首要观察对象。

理想情况：

```text
Δ_relctx^C 从约 +0.40 进一步增加
D_ctx 不 collapse
Current-Uniform 从约 0 变成明显正值
```

Movies/Toys：

即使 relation selection 仍弱，只要可靠 context 直接提升 Val，也属于有效。

---

# 13. R1-A0 / A1 实验

Variants：

```text
A0 = R1 same-code-path baseline
A1 = Factor-conditioned Edge Reliability
```

### Smoke

```text
Movies seed42
Grocery seed42
5 epochs
```

只验证代码，不判断性能。

### 正式 seed42 screen

```text
5 datasets × A0/A1 × seed42 = 10 runs
```

如果 A1 GO：

追加 seeds43/44：

```text
5 datasets × A0/A1 × 2 seeds = 20 runs
```

最终 A0/A1 都有 3 seeds。

---

# 14. R1-A2 — Semantic-calibrated Relation Posterior
## 只有 A1 无法充分修复 relation semantics 时才执行

不要一开始实现。

触发条件：

```text
A1 Val 有正收益，
但 relation semantic range / Δrelctx 仍然很弱，
或 Grocery Current≈Uniform 仍然成立。
```

---

# 15. A2 的思想

当前：

\[
r_{ij,k}^{str}
=
Softmax_k(\ell_{ij,k}^{str}).
\]

加入 factor-conditioned semantic residual：

\[
\delta_{ij,k}^{f}.
\]

然后：

\[
\boxed{
r_{ij,k}^{f}
=
Softmax_k(
\log(r_{ij,k}^{str}+\epsilon)
+
\lambda\delta_{ij,k}^{f}
)
}
\]

### 初始化

semantic residual output zero-init：

\[
\delta=0
\Rightarrow
r^f=r^{str}.
\]

---

# 16. A2 与 A1 的区别

A1：

```text
relation membership 不变
只判断边是否可靠
```

A2：

```text
允许同一条边对 C/Pt/Pv 有不同 relation posterior
```

A2 更强但也更复杂，所以只在 A1 后执行。

---

# 17. R1-B — Evidence-aware Unified Γ
## 第二刀：保留统一 Γ，但给它更强的 evidence

R0 已经证明：

```text
Local state 有价值
node-specific graph demand 有价值
relation selection 几乎没价值
```

所以不要推翻：

\[
\Gamma.
\]

应该改善 score parameterization。

---

# 18. R1-B0

Parent：

```text
R1-A 最佳 variant
```

如果 A1 NO-GO：

```text
退回 A0 baseline
```

---

# 19. R1-B1：Dynamic Local + Support-aware Relation Scores

保留当前 relation compatibility scorer：

\[
s_{ifk}^{base}
=
MLP([f_i,g_{ik}^{f},f_i\odot g_{ik}^{f}]).
\]

增加 zero-init residual score。

---

## Dynamic Local Score

当前只有 global scalar：

\[
z_f.
\]

改成：

\[
\boxed{
s_{if0}
=
z_f
+
\delta_{if0}
}
\]

其中：

\[
\delta_{if0}
=
MLP_0(
[
f_i
\Vert
\bar g_i^f
\Vert
|f_i-\bar g_i^f|
\Vert
f_i\odot\bar g_i^f
]
).
\]

最后一层 zero-init：

\[
\delta_{if0}=0
\]

开始时严格等价当前模型。

---

## Support-aware Relation Residual

\[
\boxed{
s_{ifk}
=
s_{ifk}^{base}
+
\delta_{ifk}^{evidence}
}
\]

推荐输入：

\[
[
\log(1+\tilde m_{ifk})
\Vert
a_{ik}^{str}
]
\]

如果 A1 开启：

\[
\tilde m_{ifk}
=
\sum_jr_{ji,k}\eta_{ji}^{f}.
\]

否则使用原：

\[
m_{ik}.
\]

---

# 20. 非常重要：availability 只作为 feature

R0 已经表明：

\[
\Gamma\propto a
\]

并不可靠。

所以：

```text
a_ik / mass
```

只作为 scorer 的 evidence / confidence feature，

禁止重新变成：

```text
hard capacity prior
强制 routing proportional to availability
```

---

# 21. R1-B2：DMCAR-style Global Factor Context
## 只有 B1 有正信号再测

构造节点级 global factor context：

\[
c_i^{F}
=
\frac{1}{3}
\sum_{f}
\phi(f_i).
\]

投影到小维度：

\[
\phi:\mathbb R^{128}\to\mathbb R^{32}.
\]

然后加入 Local 与 Relation residual score：

\[
\delta_{if0}
=
MLP_0(...,c_i^F)
\]

\[
\delta_{ifk}
=
MLP_R(...,c_i^F).
\]

它回答：

> 当前 factor 的 routing 是否需要知道其他 semantic factors 的整体状态？

不要直接上完整 MoE。

---

# 22. R1-B 实验顺序

Seed42：

```text
Parent
B1 Evidence-aware
```

先跑 5 datasets。

如果 B1 达 GO：

追加 seeds43/44。

如果 B1 有明显正信号但仍 < Strong GO：

再 seed42 测 B2 Global Context。

不要：

```text
B1/B2/B3 同时大 sweep
```

---

# 23. R1-B 的关键机制判据

最看 Grocery：

当前大约：

```text
Current - Uniform ≈ +0.03 pp
```

希望新 router 让：

\[
\boxed{
\Delta_{select}
=
Current-Uniform
}
\]

稳定进入：

```text
+0.2 ~ +0.5 pp
```

同时：

```text
Δ_demand
Δ_local
```

不能被破坏。

如果 B1/B2 只改变 score entropy，却：

```text
Current ≈ Uniform
```

仍成立，则 Router enhancement NO-GO。

---

# 24. R1-C — Adaptive 2-hop Trajectory
## 第三刀：只解决 Movies/Toys 的 receptive-field 缺口

R0 证据不是 universal multi-hop。

所以：

\[
\boxed{
\textbf{不要简单堆两层后只取最后一层。}
}
\]

必须让模型能够退回 1-hop。

---

# 25. R1-C1 推荐设计

Parent：

```text
R1-A/B 最佳版本
```

第一跳：

\[
F^{(1)}
=
GraphBlock(F^{(0)}).
\]

第二跳：

\[
F^{(2)}
=
GraphBlock(F^{(1)}).
\]

### 第一版建议共享参数

```text
same relation module
same reliability module
same Γ scorer
same operator
```

目的：

> 测 receptive field，不把参数量增长混进来。

---

# 26. 2-hop trajectory readout

不要直接：

\[
F^{out}=F^{(2)}.
\]

推荐 zero-init residual trajectory：

\[
\boxed{
F_i^{out,f}
=
LN(
F_i^{(1),f}
+
\lambda_i^f
W_{traj}
(
F_i^{(2),f}-F_i^{(1),f}
)
)
}
\]

其中：

\[
\lambda_i^f
=
\tanh(
MLP_d(
[
F_i^{(0),f}
\Vert
F_i^{(1),f}
\Vert
F_i^{(2),f}
]
)
).
\]

### 初始化

`MLP_d` 最后一层 zero-init：

\[
\lambda=0
\]

因此开始时：

\[
F^{out}=F^{(1)}
\]

即 current 1-hop parent。

`W_traj` 可以 Xavier 初始化。

---

# 27. 为什么允许 λ 为负？

第一阶段目的是：

```text
adaptive residual correction
```

而不是强制 convex depth mixture。

如果后续观察：

```text
λ 大量负值且性能有益
```

说明 second-hop difference 本身起 correction 作用。

如果想做严格 convex depth mixture，可在 C1 GO 后再设计，不是第一版。

---

# 28. R1-C 重点诊断

per dataset / factor：

```text
mean λ
std λ
p10/p50/p90
frac |λ|<0.1
```

期望：

```text
Movies/Toys |λ| 更大
Grocery/ele/Reddit 自动接近 0
```

如果所有 dataset 都大 gate：

> 可能只是增加了 capacity，而非 adaptive depth。

---

# 29. R1-C 实验

Parent 已有 3-seed结果。

先：

```text
C1 × 5 datasets × seed42 = 5 runs
```

如果 GO：

追加：

```text
C1 × 5 datasets × seeds43/44 = 10 runs
```

重点：

```text
Movies
Toys
```

但 guard 仍必须检查全部 5 datasets。

---

# 30. R1-F — 最终组合原则

R1 不是：

```text
A+B+C 全部无脑打开。
```

只组合已经单独 GO 的机制。

示例：

### 如果 A GO，B GO，C GO

Parent chain 已经是：

```text
A
A+B
A+B+C
```

不需要再做 2^3 factorial。

### 如果 A GO、B NO-GO、C GO

最终：

```text
A+C
```

### 如果 A NO-GO、B weak、C GO

最终：

```text
C
```

不要为了故事完整保留无性能作用的机制。

---

# 31. R1 最终成功标准

R1-Frozen candidate 相对 R1-A0：

## 最低目标

在 Movies/Toys/Grocery：

```text
至少 2/3 dataset Val Acc +0.50 pp
第三个不系统下降
```

同时：

```text
ele-fashion / Reddit-S 不明显退化
```

---

## 理想目标

使：

```text
Movies / Toys / Grocery
```

在 Validation 上至少接近当前 DiP：

```text
|gap| <= 0.30 pp
```

并保留 ele-fashion / Reddit-S 的优势。

---

# 32. R1 中 Test 的使用规则

R1-A / B / C screening 和 confirm：

```text
只汇总 Val Acc / Val Macro-F1
```

即使标准 trainer 生成 Test：

```text
runner / summarizer 不读取 Test 进行任何选择。
```

只有：

\[
\boxed{
\text{R1 final architecture 完全 freeze}
}
\]

后，再进行一次正式 Test evaluation。

如果最终 Test 不理想：

```text
不能用该 Test 再反推 R1 结构
```

后续 R2 仍基于 Val bottleneck 设计。

---

# 33. 推荐代码结构

```text
src/models/
  biaxis_perf_r1.py
  biaxis_perf_r1_components.py

configs/model/
  biaxis_perf_r1.yaml

tests/
  test_biaxis_perf_r1.py

scripts/
  run_perf_r1_screen.py
  analyze_perf_r1_checkpoint.py
  summarize_perf_r1.py
```

输出：

```text
outputs/perf_r1/
  baseline/
  reliability/
  relation_calibration/
  routing/
  multihop/
  final/
  summary/
```

---

# 34. R1 必做测试

## Baseline equivalence

同一 weights：

```text
biaxis_perf_r1 mode=baseline
==
biaxis_final
```

forward embedding：

```text
torch.allclose / ideally exact where possible
```

---

## Reliability

1. zero-init η == 1
2. η in (0,2)
3. undirected reverse edges η symmetric
4. η=1 时 g 与当前 relation_weighted_mean 等价
5. gradients finite
6. no E×F×d persistent tensor
7. isolated nodes
8. inference

---

## Evidence-aware Γ

1. residual score zero-init == parent Γ
2. gamma row sum == 1
3. Local dynamic score shape
4. support feature finite
5. isolated nodes local-only
6. no availability hard constraint

---

## Multi-hop

1. depth gate zero-init => parent 1-hop output
2. second hop shares parameters
3. gradients reach second-hop path
4. no recursive state cache bug
5. inference equivalence
6. memory peak controllable

---

# 35. AI / Codex Prompt 1 — R1 Repository Audit
## 现在先只执行这个 Prompt

```text
我们进入 Bi-Axis Performance-R1。

背景：
Performance-R0 已完成，结论是：

1. Factorizer healthy。
2. Graph propagation strongly useful。
3. Structural relations non-collapsed but weakly task-semantic。
4. Relation-conditioned contexts only have limited extra utility over plain neighbor mean，
   Grocery 最明显。
5. Gamma 的 Local / node-specific graph-demand 分支有效，
   但 relation-selection branch 基本 inactive：
   Current ≈ Uniform。
6. Movies/Toys 存在约 +0.5pp 级额外 second-hop probe signal。
7. 不再调 UOT/K/operator/factorizer。

R1 顺序：
A. Factor-conditioned semantic edge reliability
B. Evidence-aware Gamma
C. Adaptive 2-hop trajectory

先不要写代码。

请审计当前：
- biaxis_p1/p2/p3/final
- relation_weighted_mean
- _decompose_relations
- FactorRelationScore
- P3 _graph_update
- current edge_chunk_size / ele-fashion memory path
- model registry / config system
- existing tests
- P3 diagnostics / R0 analysis scripts

回答：

1. 新 biaxis_perf_r1 最安全的继承层次是什么？
2. 如何实现 same-code-path baseline 且保证 same-weights == biaxis_final？
3. R1-A reliability 应插入在 relation_weighted_mean 的哪个位置？
4. 如何避免存 E×F reliability tensor？
5. g_perm / effective_mass 应怎样计算？
6. structural availability 是否继续保持原 a_ik 不变？
7. Reliability gate zero-init η=1 如何严格实现？
8. factor-specific 128->32 projection + shared MLP 是否有实现/显存问题？
9. R1-B future dynamic Local score 最安全插在哪里？
10. 共享第二 hop 时，当前 raw topology cache 是否可安全复用？
11. 给代码文件计划、测试计划、显存风险。
12. 不修改任何 frozen model file。

输出：
docs/Performance_R1_Audit.md

不要训练。
不要实现模块。
```

---

# 36. Prompt 2 — 实现 R1-A Components

```text
R1 audit 已通过。

只实现 R1-A reliability components：

新增：
src/models/biaxis_perf_r1_components.py

实现：
FactorConditionedEdgeReliability

设计：
- input factors [N,F,d], F=3,d=128
- per-factor projection P_f: d->32
- edge symmetric token:
  [u_i+u_j | abs(u_i-u_j) | u_i*u_j | cosine(u_i,u_j)]
- shared MLP -> scalar delta
- eta = 2*sigmoid(delta)
- final scalar layer zero-init => eta exactly 1 at initialization
- edge chunk compatible
- 不永久 materialize E×F×d
- 不读 labels
- 不读 raw modalities
- factor semantics only

同时实现一个 reliable relation aggregation helper：

输入：
edge_index
r [E,K]
f_block [N,F,d]
reliability module

输出：
g_perm [N,F,K,d]
effective_mass [N,F,K]

数学：
g_ifk =
sum_j r_jik * eta_jif * f_j /
(sum_j r_jik * eta_jif + eps)

结构 availability 仍然使用原始 r 的：
a_ik = mass_ik / degree_i
不要改。

写 unit tests：
eta neutral
symmetry
range
aggregation eta=1 equivalence
chunk/full equivalence
gradients
isolated
memory discipline
```

---

# 37. Prompt 3 — Integrate R1-A Model

```text
现在新增：

src/models/biaxis_perf_r1.py
configs/model/biaxis_perf_r1.yaml
tests/test_biaxis_perf_r1.py

继承 audit 推荐的 frozen parent。

支持 mode：

baseline
semantic_reliability

baseline：
必须 same weights 与 biaxis_final 等价。

semantic_reliability：
只替换 relation-conditioned context aggregation：
r_str 不变
Gamma scorer 不变
operator 不变
Local 不变
P0 aux losses 不变
K=4
NullSoftmax
deterministic=false

返回额外 diagnostics：
eta mean/std/quantiles
per-factor eta
effective mass
context D_ctx
weighted semantic coherence

不要实现 evidence-aware Gamma。
不要实现 multi-hop。
不要实现 semantic relation calibration。
```

---

# 38. Prompt 4 — R1-A Smoke

```text
只做 smoke：

Movies seed42
Grocery seed42

modes:
baseline
semantic_reliability

epochs=5

验证：
- no NaN
- no OOM
- baseline config correctness
- eta starts near/exact 1
- eta receives gradient
- r_str unchanged
- gamma valid
- full inference valid
- memory

不要用 smoke metric 做选择。
```

---

# 39. Prompt 5 — R1-A Seed42 Screen

```text
Smoke PASS。

正式 seed42 screen：

datasets:
Movies,Toys,Grocery,ele-fashion,Reddit-S

modes:
A0 baseline
A1 semantic_reliability

seed:
42

完整 300ep / patience30 / ValAcc checkpoint protocol。

输出：
Val Acc
Val Macro-F1
params
runtime
peak memory

paired:
A1-A0

并在 best checkpoint 做：
eta statistics
D_ctx
Δ_relctx fixed Ridge probe
Current-vs-Uniform Gamma counterfactual
weighted semantic coherence

生成：
outputs/perf_r1/reliability/R1_A_SEED42_REPORT.md

根据预注册 GO：
M/T/G mean >= +0.30pp
至少2/3 > +0.20
guards >= -0.20

不要读取 Test 进行判断。
```

---

# 40. Prompt 6 — R1-A 3-seed Confirm

```text
A1 seed42 达 GO。

追加 seeds43/44：

A0
A1

5 datasets。

与 seed42 合并成 3-seed paired report：

mean±population std
paired delta mean/std
positive_seed_count

Val Acc primary
Val Macro-F1 secondary

机制诊断只需：
seed42 全量
seeds43/44 聚合关键 eta / D_ctx 即可。

最终判定：
Strong GO / GO / Borderline / NO-GO。

不要进入 R1-B，先把完整报告给我审查。
```

---

# 41. Prompt 7 — 条件实现 R1-A2 Semantic Relation Calibration

只有我/ChatGPT 审查 A1 后明确要求再执行。

```text
实现 factor-conditioned semantic residual relation posterior。

保持 r_str 为 prior：

r_f ∝ exp(log(r_str+eps) + lambda*delta_f)

要求：
- delta zero-init
- initial r_f == r_str
- K=4
- no topology rewiring
- no labels
- chunked edge processing
- 与 A1 reliability 可单独开关
- 第一轮不要二者同时训练，先测试 A2-only
```

---

# 42. Prompt 8 — R1-B Evidence-aware Γ

只有 R1-A 结论确认后执行。

```text
Parent = R1-A 最佳 frozen-in-stage variant。

新增 router mode：

base
evidence

Evidence mode：

s_local =
z_f + delta_local(
[f | g_bar | abs(f-g_bar) | f*g_bar]
)

s_rel =
s_rel_base +
delta_evidence(
[log1p(effective_mass_ifk) | structural_availability_ik]
)

两个 residual scorer 最后一层 zero-init。

保持：
Gamma = row-wise NullSoftmax
epsilon=.2
Local state
operator
relation contexts
全部不变。

禁止：
UOT
hard availability prior
top1 routing

先做 same-weight zero-init equivalence tests。
然后 Movies/Grocery smoke。
再 seed42 5 datasets screen。
达 GO 才追加 43/44。
```

---

# 43. Prompt 9 — R1-B2 Global Factor Context

仅 B1 有正信号但不够强时执行。

```text
在 B1 上加入 DMCAR-style global factor context：

cF_i = mean_f phi(f_i), phi:128->32

将 cF_i 输入：
dynamic Local residual scorer
relation evidence residual scorer

不要引入 experts/MoE。

先 seed42 5 datasets。
只有明显超过 B1 才做 multi-seed。
```

---

# 44. Prompt 10 — R1-C Adaptive 2-hop

只有 A/B 阶段审查完成后执行。

```text
Parent = 当前 R1 最佳 variant。

实现 shared-parameter second hop：

F1 = GraphBlock(F0)
F2 = same GraphBlock(F1)

relation topology cache复用。
权重共享。
不要增加第二套 operator/router。

trajectory readout：

lambda_if =
tanh(MLP_depth([F0_if | F1_if | F2_if]))

MLP final layer zero-init => lambda=0

Fout =
LN(F1 + lambda_if * Wtraj(F2-F1))

第一版：
Wtraj shared across factors
Xavier init

diagnostics：
lambda per factor/dataset
|lambda| distribution
second-hop message norm
F1/F2 cosine

先 seed42 5 datasets。
重点 Movies/Toys，但必须 guard all 5。
达 GO 才追加43/44。
```

---

# 45. Prompt 11 — R1 Final Synthesis

```text
读取：
R1-A
R1-A2（若执行）
R1-B
R1-B2（若执行）
R1-C

以 parent-child paired Val delta 形成 stage tree。

输出：
R1_MASTER_TABLE.csv
R1_MECHANISM_TABLE.csv
R1_FINAL_REPORT.md

要求：

1. 不用 Test 排序。
2. 每个模块标：
Strong GO / GO / Borderline / NO-GO。
3. 只保留真正 GO 的模块形成 R1 final candidate。
4. 比较 final candidate vs R1-A0：
   Movies/Toys/Grocery
   ele-fashion/Reddit guard
5. 检查参数量/内存/训练时间。
6. 如果达到 R1 final success criteria：
   标记 PERFORMANCE-R1 FROZEN。
7. 如果没有：
   不自动堆更多模块，
   给出最主要 remaining bottleneck，
   等待下一阶段人工审查。
```

---

# 46. 什么时候把结果返给 ChatGPT？

不要一次把整个 R1 全跑完。

按以下 checkpoint 推进：

```text
Checkpoint 1
R1 Audit
→ 发回审查

Checkpoint 2
R1-A seed42 screen
→ 发回审查

Checkpoint 3
R1-A 3-seed confirm（如果进入）
→ 发回审查

Checkpoint 4
R1-B screen/confirm
→ 发回审查

Checkpoint 5
R1-C screen/confirm
→ 发回审查

Checkpoint 6
R1 final synthesis
```

每一步都可能改变下一步是否值得做。

---

# 47. 当前立即执行

现在只执行：

\[
\boxed{
\textbf{Prompt 1 — R1 Repository Audit}
}
\]

不要直接让 AI 写全部 R1-A/B/C。

R1 的核心原则是：

\[
\boxed{
\textbf{
先让 relation/context 变得值得选择，
再强化 Γ 的选择能力，
最后只对确有证据的数据集补充 second-hop。
}
}
\]
