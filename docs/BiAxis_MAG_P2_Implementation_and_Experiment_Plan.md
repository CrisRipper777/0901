# Bi-Axis MAG — P2 阶段实现与实验推进文档
## 阶段目标：从独立 Gate 转向可退化的 Adaptive Factor–Relation Allocation

> 代码仓库：`CrisRipper777/0901`  
> 前置阶段：P0 已冻结；P1 已冻结并判定 **Conditional GO**  
> P2 任务范围：**仅 Node Classification**  
> 数据集：Movies / Toys / Grocery / ele-fashion / Reddit-S  
> 主协议：继续沿用已冻结的 NC full-graph protocol  
> P2 暂不实现：Low-rank Factor–Relation Operator / Relation-specific `W_{f,k}` / Pseudo Nodes / Diffusion / MoE / Rewiring

---

# 0. P2 的研究起点：P1 到底暴露了什么问题？

P1 最稳定的结果不是 “selector 一定有效”，而是三层事实：

1. **Factor-dependent Graph Demand 稳定存在**
   `beta_i^C, beta_i^Pt, beta_i^Pv`
   在 15/15 F1R1 runs 中形成稳定分化。

2. **Structural Relation Specialization 是 graph-dependent 的**
   `S_R = 1 - H_R / log(K)`
   在 Grocery / ele-fashion 明显，在 Movies / Toys 中等，而 Reddit-S 接近 0。

3. **Factor-specific Relation Selectivity 是 conditional 的**
   - Grocery：factor-specific selector 明显有效；
   - Movies：`Delta FR > 0`，但 factor-specific JS 很弱；
   - Toys：二维 interaction 稳定为负；
   - Reddit-S：relation decomposition 本身接近 per-edge uniform。

因此，P2 不能简单理解为：

`beta_if * alpha_ifk -> UOT`

或者“普通 gate 不够高级，所以换 OT”。

P2 真正要回答的是：

**Can a single coupler adaptively decide**
1. whether graph evidence is needed,
2. whether relation differentiation is needed,
3. how relation capacity should be shared across semantic factors?

---

# 1. P2 的核心设计原则

P2 需要一个 coupling mechanism，能够自动退化到不同工作状态。

## 状态 A：Local Preservation

当某个 semantic factor 不需要 graph evidence 时：

`Gamma_if,null -> 1`

## 状态 B：Relation-agnostic Graph Use

当图有用，但 latent relation differentiation 本身不可靠时：

`Gamma_if,1:K`

可以接近普通 relation-agnostic / factor-similar allocation。

Reddit-S 应该允许出现这种退化。

## 状态 C：Factor-specific Relation Allocation

当 structural relation 有意义，且不同 factors 对 relation 的作用确实不同，例如 Grocery：

`Gamma_i^C != Gamma_i^Pt != Gamma_i^Pv`

因此 P2 的目标不是“让 transport 更尖锐”，而是：

**让 coupling 在不需要复杂关系时自动变简单，在需要时才产生 factor-specific relation allocation。**

---

# 2. P2 的主计算对象：Null-Augmented Factor–Relation Plan

P1 使用两个独立量：

`beta_i^f in [0,1]`

回答 how much graph evidence，

以及：

`alpha_i,f,k`, `sum_k alpha_i,f,k = 1`

回答 which relation。

P2 将它们统一为：

`Gamma_i in R_+^{F x (K+1)}`

其中：
- `F = 3`
- `f in {C, Pt, Pv}`
- 第 0 列是 `Local / No-Transport State`
- 第 1..K 列是 latent structural relations

强制每个 factor row：

`sum_{k=0}^K Gamma_i,f,k = 1`

于是：

`beta_i^f = 1 - Gamma_i,f,0`

以及：

`alpha_i,f,k = Gamma_i,f,k / (sum_{l=1}^K Gamma_i,f,l + eps)`

所以：

`Gamma_i,f,k = beta_i^f * alpha_i,f,k`, `k >= 1`

这个等价关系是 P2 最重要的理论连接之一。

---

# 3. 为什么需要 Null / Local State？

P1 budget ablation 已经表明 `B0 ~= B2`，显式 beta gating 并不是普遍的精度来源。

但 factor-dependent graph demand 的机制差异又稳定存在。

因此 P2 不应该继续单独保留一个大型 Budget MLP，而是把：

`use graph / do not use graph`

变成 coupling 本身的一个 assignment option。

这类 unmatched / discard / dustbin 状态在 CV matching / feature aggregation 中是成熟 primitive；P2 只借用这个思想，不把它当作创新本身。

---

# 4. P2 冻结边界

P2 只研究 **Coupling**。

必须冻结：

## M1 Semantic Factorization

继续使用 P0/P1 的 `C, Pt, Pv`。

保持：

```text
hidden_dim = 256
factor_dim = 128
lambda_common = 0.02
lambda_orth = 0.01
lambda_recon = 0.3
```

architecture/objective unchanged，继续 joint optimization。

## M2 Structural Relation Decomposition

继续使用 P1 的 topology-only：

```text
TopologyDiffusionSignature
EdgeStructuralToken
RelationPrototypes
K = 4
relation_dim = 32
relation_temperature = 0.5
```

禁止在 P2 修改 relation encoder、relation prototype 数量或 topology signature。

## Graph Transformation

继续只用 P1 的共享 `W0`。

P2 不允许实现：

```text
W_f
W_k
W_fk
Low-rank operator
MoE operator
```

这些全部属于 P3。

---

# 5. P2 的 relation-specific factor contexts 直接复用 P1

P1 已经得到 relation assignment `r_ij,k` 以及 relation weighted mean：

`g_i,k^f = sum_j r_ji,k f_j / (sum_j r_ji,k + eps)`

同时：

`a_i,k = m_i,k / (d_i + eps)`

作为 relation availability。

P2 不重新定义这两个量。

P2 的输入固定为：

`{f_i, g_i,1:K^f, a_i,1:K, r_ij,1:K}`

---

# 6. P2 Relation Compatibility Scorer

P2 不应该因为换 transport 就同时换一个很复杂的 scorer。

第一版使用共享 scorer：

`s_i,f,k = g_C([f_i || g_i,k^f || f_i * g_i,k^f])`

第一版**不把 availability 放入 score**。

原因：

> availability 在 P2 中承担 relation-side supply prior；若同时塞入 score，会把 compatibility 和 capacity 混为一谈。

推荐：

```python
class FactorRelationScore(nn.Module):
    Linear(3*factor_dim, 64)
    GELU
    Linear(64, 1)
```

同一 scorer 对 C / Pt / Pv 和 R1..RK 全部共享。

---

# 7. Null / Local Score

Local state 使用：

`s_i,f,0 = z_f`

其中 `z_C, z_Pt, z_Pv` 是 3 个 learnable scalars。

推荐初始化：

`z_f = 0`

设计动机：

- 不重新引入 node-wise Budget MLP；
- factor-specific global threshold 允许 Common / Private 有不同 graph propensity；
- node-level graph demand 仍会因为 relation compatibility scores 不同而变化。

后续若 P2 失败，再考虑 node-adaptive null scorer；第一版禁止加入。

---

# 8. Augmented Score Matrix

形成：

`S_i = [s_i,f,0, s_i,f,1, ..., s_i,f,K]`

shape：

`[N, F, K+1]`

所有 P2 coupler 使用**完全相同的 score matrix**。

因此 NullSoftmax vs Transport 的差异只来自 coupling solver，而不是 scorer。

---

# 9. Relation-side Reference Capacity

P1 已有 availability：

`a_i = [a_i1, ..., a_iK]`, `sum_k a_i,k = 1`

P2 把它解释成 topology-side relation availability prior。

加入 Local 状态后，定义：

`nu_tilde_i = [pi0, (1-pi0)*a_i1, ..., (1-pi0)*a_iK]`

由于每个节点有 F=3 个 factor rows，每行质量为 1，总质量为 F。

所以 target reference：

`nu_i = F * nu_tilde_i`

第一版固定：

`pi0 = 0.5`

解释：

> 这是 Local-vs-Graph 的中性宏观 prior：一半 capacity 给 Local，一半 graph capacity 再按 topology availability 分到 K 个 relations。

它只是 soft prior，不是 hard budget。

P2-screen 阶段禁止 dataset-specific 调 `pi0`。

---

# 10. 一个重要实现约束：capacity prior stop-gradient

推荐：

```python
nu = build_reference(availability.detach())
```

原因：

如果 `nu` 对 relation availability 直接反向传播，relation module 可以通过修改自己的 occupancy 来“迁就” transport capacity penalty。

这会使 relation discovery 与 coupling constraint 发生不必要的 shortcut。

因此：
- relation contexts `g_i,k^f` 仍然对 `r` 可导；
- task gradient 仍可通过 message path 更新 relation module；
- 但 capacity reference `nu` 本身 stop-gradient。

---

# 11. P2-Variant 0：P1 Gate Baseline

不重新跑。

直接复用已冻结 P1 F1R1 confirm。

概念上：

`Gamma_if,0^P1 = 1 - beta_i^f`

`Gamma_if,k^P1 = beta_i^f * alpha_i,f,k`

这只是用于 conceptual comparison 和 performance reference。

---

# 12. P2-Variant 1：Null-Augmented Independent Softmax

P2 必须实现的最关键 baseline：

`Gamma_i,f,:^NS = Softmax(S_i,f,: / epsilon)`

每个 factor 独立选择：

```text
Local
R1
...
RK
```

特点：
- 一个 plan 同时表示 how much + which relation；
- 不同 factors 之间没有 relation-side coupling；
- relation capacity 没有约束。

它回答：

> 仅仅把 Budget + Selector 统一成一个 null-augmented row-softmax 是否已经足够？

如果 NullSoftmax 已经最好，则 P2 不需要 OT。

---

# 13. P2-Variant 2：Fixed Semi-Relaxed UOT

P2 主候选之一。

对每个节点：

`Gamma_i* = argmin_Gamma <C_i,Gamma> - epsilon H(Gamma) + tau_R KL(Gamma^T 1 || nu_i)`

subject to：

`Gamma_i 1 = 1_F`

其中：

`C_i = -S_i`

解释：

## Factor-side row marginal：hard

每个 factor 必须把 unit decision mass 分给：

```text
Local or graph relations
```

## Relation-side column marginal：soft

`Gamma_i^T 1` 只被 KL penalty 软约束到 topology reference `nu_i`。

因此同一个 relation 可以被多个 factors 同时使用，不会像 hard balanced matching 那样强制“一人一个坑”。

---

# 14. Semi-relaxed Sinkhorn 的连续关系

令：

`K_i = exp(S_i / epsilon)`

generalized Sinkhorn：

`u <- mu / (K v)`

其中：

`mu = 1_F`

relation-side：

`v <- (nu / (K^T u))^theta`

其中：

`theta = tau_R / (tau_R + epsilon)`

于是：

## tau_R = 0

`theta = 0`，退化为 NullSoftmax。

## 0 < tau_R < inf

得到 Semi-relaxed UOT。

## tau_R -> inf

`theta -> 1`，趋向 hard relation-capacity Sinkhorn。

因此 P2 实际研究的是：

**relation-side constraint strength 的连续变化。**

---

# 15. P2-Variant 3：Confidence-Adaptive Semi-Relaxed UOT

这是当前 P2 最值得重点验证的候选，但必须先作为可替换积木，不预设最终一定保留。

P1 已证明 `S_R` 具有明显 graph dependence。

因此固定 `tau_R` 有潜在问题：

> Reddit-S 上 relation decomposition 几乎 uniform，却仍被同样强度的 relation-side constraint 约束。

---

# 16. Node-wise Relation Specialization Confidence

对于每条 incoming edge：

`h_ji = -sum_k r_ji,k log(r_ji,k + eps)`

节点平均 edge entropy：

`hbar_i = (1/d_i) sum_{j in N(i)} h_ji`

定义：

`q_i^R = 1 - hbar_i / log(K)`

并 clamp 到 `[0,1]`。

解释：

```text
q_i^R ~= 0:
incident edges 的 relation assignment 近 uniform；
当前节点附近 relation differentiation 不可信。

q_i^R 较高:
邻域 edges 有更明确的 latent relation specialization。
```

孤立节点：

`q_i^R = 0`

---

# 17. Relation confidence 必须 stop-gradient

使用：

```python
q_rel = q_rel.detach()
```

原因：

如果 gradient 可以通过 `q_i^R` 回到 relation assignments，模型可能通过人为增加 relation entropy：

`r -> Uniform`

来主动减弱 OT constraint。

因此 `q_i^R` 只作为 topology relation decomposition 的 confidence indicator。

---

# 18. Adaptive Constraint Strength

定义：

`tau_i = tau0 * q_i^R`

于是：

`theta_i = tau0*q_i^R / (tau0*q_i^R + epsilon)`

这产生预期退化：

## Reddit-S-like

`q_i^R ~= 0 => theta_i ~= 0`

Adaptive-UOT 自动趋向 NullSoftmax。

## Grocery-like

`q_i^R > 0`

relation-side capacity constraint 自动变强。

所以 Adaptive-UOT 的核心不是“更复杂”，而是：

**constraint strength follows structural relation confidence.**

---

# 19. 第一版超参数

建议：

```yaml
p2:
  mode: adaptive_uot

  score_hidden_dim: 64

  epsilon: 0.2
  tau_base: 1.0
  sinkhorn_iters: 10

  null_prior: 0.5
  null_score_init: 0.0

  detach_capacity_prior: true
  detach_relation_confidence: true

  eps: 1.0e-8
```

第一轮禁止 dataset-specific tuning。

若整体过于 diffuse，再统一测试：

```text
epsilon = 0.1 / 0.2 / 0.4
```

但必须先完成默认 screen。

---

# 20. Log-domain Solver

必须实现 log-domain。

核心形状：

```text
logK: [N,F,K+1]
log_u: [N,F]
log_v: [N,K+1]
```

伪代码：

```python
for _ in range(T):
    log_u = log_mu - logsumexp(logK + log_v[:, None, :], dim=-1)

    log_col = logsumexp(
        logK + log_u[:, :, None],
        dim=1
    )

    log_v = theta * (log_nu - log_col)

# 最后再更新一次 u，保证 row marginal
log_u = log_mu - logsumexp(
    logK + log_v[:, None, :],
    dim=-1
)

log_gamma = (
    log_u[:, :, None]
    + logK
    + log_v[:, None, :]
)

gamma = exp(log_gamma)
```

其中：
- fixed mode：`theta` 为 scalar
- adaptive mode：`theta` 为 `[N,1]`

最终必须检查：

`max_abs(Gamma 1 - 1_F) < 1e-5`

---

# 21. Isolated Node Fast Path

对 `degree_i = 0`：

```text
Gamma_if,0 = 1
Gamma_if,1:K = 0
f_tilde = f
```

不运行 transport。

禁止产生 `log(0)` / NaN。

---

# 22. P2 Message Passing

graph part：

`Gamma_graph = Gamma[...,1:]`

然后：

`g_i^f = sum_k Gamma_i,f,k * g_i,k^f`

再：

`m_i^f = W0 g_i^f`

更新：

`f_tilde_i = LayerNorm(f_i + m_i^f)`

不再单独乘 beta。

因为：

`sum_{k=1}^K Gamma_i,f,k = 1 - Gamma_i,f,0`

已经是 graph mass。

---

# 23. P2 推荐代码组织

新增：

```text
src/models/biaxis_p2_components.py
src/models/biaxis_p2.py
configs/model/biaxis_p2.yaml

tests/test_biaxis_p2.py

scripts/analyze_p2_checkpoint.py
scripts/run_p2_screen.py
scripts/run_p2_confirm.py
scripts/summarize_p2.py
```

原则：

- 不修改 `biaxis_p1.py`
- 不修改 `biaxis_p1_components.py`
- P2 直接复用 P1 relation decomposition / aggregation
- P1 results 作为 frozen baseline

---

# 24. P2 Model 推荐继承 P1，但删除不用的 Gate 模块

推荐：

```python
from .biaxis_p1 import Model as P1Model

class Model(P1Model):
    def __init__(self, cfg, data_info):
        super().__init__(cfg, data_info)

        assert self.factor_aware
        assert self.num_relations == 4

        del self.graph_budget
        del self.factor_selector

        self.transport_scorer = ...
        self.null_score = ...
        self.transport_coupler = ...
```

原因：
- 直接复用 P1 M1/M2/W0；
- 不修改冻结的 P1 文件；
- 删除不用的 P1 gate params，避免参数量虚增；
- P2 override `_graph_update()` 即可被继承的 `forward()` 动态调用。

---

# 25. P2 `_graph_update()` 返回兼容字段

建议：

```python
{
    "f_tilde": ...,
    "beta": graph_mass,
    "alpha": conditional_relation_plan,
    "r": ...,
    "availability": ...,

    "gamma": ...,
    "null_mass": ...,
    "relation_confidence": ...,
    "theta": ...,
}
```

注意：

> P2 的 beta/alpha 是 unified transport plan 的派生 diagnostics，不再是两个独立 predictor。

---

# 26. P2 必须实现的三个主要 mode

统一 config：

```text
model.p2.mode=null_softmax
model.p2.mode=fixed_uot
model.p2.mode=adaptive_uot
```

所有 mode：
- scorer 相同；
- null score 相同；
- relation decomposition 相同；
- W0 相同。

只改变 plan solver。

---

# 27. Optional diagnostic：Balanced OT

只有 screen 后有必要再跑。

mode：

`balanced_ot`

使用 hard column marginal：

`Gamma^T 1 = nu`

优先只跑：

```text
Grocery
Toys
Reddit-S
seed42
```

目的：

> 验证 hard capacity 是否比 soft capacity 更容易 over-constrain。

不把 Balanced-OT 当主模型。

---

# 28. P2 的核心理论关系

## 28.1 P1 Gate Decomposition

任意 P1 beta/alpha 都可写成：

`Gamma_f0 = 1 - beta_f`

`Gamma_fk = beta_f * alpha_fk`

所以 P2 unified plan 覆盖 P1 的表示形式。

## 28.2 NullSoftmax Special Case

`tau_R = 0` 时 semi-relaxed OT 退化为 independent row-softmax。

## 28.3 Balanced Limit

`tau_R -> inf` 时 relation-side marginal 趋近 hard capacity。

## 28.4 Local-only Special Case

`Gamma_f0 = 1` 时：

`f_tilde = f`

P2 显式退化为 topology-free semantic factor。

---

# 29. P2 Complexity

每个节点只求：

`3 x 5`

transport plan。

Sinkhorn：

`T = 10`

复杂度：

`O(N * F * (K+1) * T)`

其中 `F=3, K=4`。

相比 relation-weighted edge aggregation：

`O(|E| K d)`

transport solver 成本很小。

禁止把 OT 放在：
- node × node
- edge × edge

空间中。

---

# 30. P2 Diagnostics

## 30.1 Local / Graph Mass

每 factor：

```text
null_mean_C
null_mean_Pt
null_mean_Pv

graph_mass_C
graph_mass_Pt
graph_mass_Pv
```

以及 p10/p50/p90/high_frac。

## 30.2 Plan Entropy

每 factor 记录：

`H_Gamma^f = -sum_{k=0}^K Gamma_ifk log Gamma_ifk`

不要把低 entropy 自动解释成好。

## 30.3 Conditional Relation Selectivity

对 graph part normalize 后计算：

```text
JS(C,Pt)
JS(C,Pv)
JS(Pt,Pv)
```

这样才与 P1 alpha JS 可比。

## 30.4 Column Capacity Deviation

记录：

```text
capacity_kl
capacity_l1
```

检查 UOT 是否真的利用了 relation-side constraint。

## 30.5 Relation Confidence

adaptive mode：

```text
rel_conf_mean
rel_conf_p10/p50/p90
theta_mean
theta_p10/p50/p90
```

---

# 31. P2 必须加入的 unit tests

新增：

`tests/test_biaxis_p2.py`

至少覆盖：

1. Gamma shape `[N,3,K+1]`
2. row marginal sum = 1
3. NullSoftmax 与直接 softmax 一致
4. `tau=0` 的 semi-UOT 数值等价 NullSoftmax
5. large tau 时 columns 更接近 reference `nu`
6. 增加 null score 会单调增加 null mass
7. isolated node all-null
8. uniform `r` -> relation confidence ≈ 0
9. one-hot `r` -> relation confidence ≈ 1
10. `nu` / confidence stop-gradient
11. Sinkhorn scorer/null score finite nonzero gradient
12. large positive/negative score 数值稳定
13. no NaN
14. no dense adjacency
15. forward/inference equivalence
16. shared W0 仍是唯一 graph operator

---

# 32. P2 阶段推进漏斗

## P2-D0：Repository Audit

不改代码。

目标：
- 确认 P1 当前冻结状态；
- 确认哪些函数可以直接复用；
- 确认 subclass + delete gate params 是否安全；
- 确认 P1 forward 对 `_graph_update()` 的动态 dispatch；
- 确认 diagnostics / checkpoint 如何扩展。

## P2-D1：纯数学 Transport Layer

只实现：
- FactorRelationScore
- NullAugmentedSoftmax
- SemiRelaxedTransport
- relation_confidence()

先 synthetic test，不接模型。

## P2-D2：集成 biaxis_p2

接入：
- P1 M1
- P1 M2
- P1 W0

实现：
- null_softmax
- fixed_uot
- adaptive_uot

## P2-D3：Mechanism Diagnostics

实现：
- gamma
- null mass
- graph mass
- plan entropy
- conditional JS
- capacity deviation
- relation confidence
- theta

---

# 33. P2-Screen：先 15 runs，不重新跑 P1

P1 F1R1 seed42 已有。

新跑：

```text
5 datasets
x 3 P2 modes
x seed42
= 15 runs
```

modes：

```text
NS   = null_softmax
FUOT = fixed_uot
AUOT = adaptive_uot
```

---

# 34. P2-Screen 主表

| Dataset | P1 F1R1 | NullSoftmax | Fixed-UOT | Adaptive-UOT |
|---|---:|---:|---:|---:|

同时记录：
- Val Acc
- Test Acc
- Test Macro-F1
- params
- epoch time
- peak memory

模型选择仍以 validation Accuracy 为主。

---

# 35. P2-Screen Hypotheses

## H1：Unified Plan 是否有价值？

比较：

`NullSoftmax vs P1 beta*alpha`

如果 NullSoftmax 与 P1 相当或更好：

> Budget + Selector 分开预测并非必要；统一 Local+Relation plan 是可行方向。

## H2：Relation-side Constraint 是否有价值？

比较：

`Fixed/Adaptive-UOT vs NullSoftmax`

重点看 Grocery / ele-fashion / Movies。

如果 transport 不能超过 NullSoftmax：

> OT constraint 没有实证必要性。

## H3：Adaptive Constraint 是否真的必要？

比较：

`Adaptive-UOT vs Fixed-UOT`

重点看 Reddit-S / Toys。

如果 adaptive 能减少 low-specialization graph 的负迁移，同时保留 Grocery 收益：

> P1 暴露的 conditional relation specialization 得到了真正修复。

如果 fixed≈adaptive，则删除 adaptive complexity。

---

# 36. P2-Screen Decision

## Strong GO

一个 transport variant 满足大部分：

1. 相比 P1 F1R1，Val Acc 在 ≥3/5 datasets 改善；
2. 相比 NullSoftmax，在 ≥2 个 `S_R` 较高数据集有正收益；
3. Toys/Reddit-S 不产生明显大幅退化；
4. plan 不 collapse 到全 Null；
5. capacity deviation 相比 NullSoftmax 下降；
6. Grocery 等有效图中 conditional JS 保持非零；
7. Reddit-S 不被强制产生虚假 factor-relation differentiation。

## Soft GO

transport 与 P1 性能相近，但：
- NullSoftmax 明显优于 P1 gate；
- unified plan 更简洁；
- 有清晰 Local mass；
- 不出现负迁移。

此时先确认 NullSoftmax。

## NO-GO for OT

如果：

`NullSoftmax >= Fixed/Adaptive-UOT`

几乎所有数据集成立，则不保留 OT。

最终 coupler 可以收敛为：

`Null-Augmented Factor–Relation Softmax`

然后进入 P3。

## NO-GO for P2 Revision

如果 P1 Gate 明显优于所有 P2 方案：

冻结 P1 coupler，P2 结束，直接进入 P3。

---

# 37. Optional Balanced-OT Diagnostic

只有 transport screen 有信号才跑：

```text
Grocery
Toys
Reddit-S
seed42
```

3 runs。

目的：
- Grocery：高 specialization / positive interaction
- Toys：negative interaction
- Reddit-S：relation-uniform

验证 hard capacity 是否 over-constrain。

---

# 38. P2-Confirm

不重新跑 P1 F1R1，复用 P1 confirm 15 runs。

## 情况 A：Adaptive-UOT 明显最好

确认：

```text
NullSoftmax
Adaptive-UOT
```

补跑 seeds 43/44：

`5 datasets x 2 modes x 2 seeds = 20 new runs`

## 情况 B：Fixed ~= Adaptive

删除 adaptive complexity。

确认：

```text
NullSoftmax
Fixed-UOT
```

同样 20 new runs。

## 情况 C：NullSoftmax 最好

只补 NullSoftmax seeds43/44：

`5 x 2 = 10 new runs`

---

# 39. P2 Confirm 最终 GO 条件

最终 P2 主候选至少应满足：

## Performance
- 相对 P1 F1R1，多数 dataset 不退化；
- 至少部分 dataset 稳定提升；
- Macro-F1 不系统性下降。

## Mechanism
能清楚表现：
- Local mass
- Graph mass
- Conditional relation use

且不同 dataset 自动进入不同 regime。

## Simplicity

如果两个机制性能相同：

选择更简单的那个。

复杂度优先顺序：

`NullSoftmax < Fixed Semi-UOT < Adaptive Semi-UOT`

---

# 40. P2 最终可能出现的四种结果

## Outcome A：Adaptive-UOT 赢
最终核心：Confidence-Adaptive Factor–Relation Transport。

## Outcome B：Fixed-UOT 赢
relation-side soft capacity 有价值，但不需要 confidence modulation。

## Outcome C：NullSoftmax 赢
P1 的主要问题是 Budget/Selector 分裂，而不是缺少 OT constraint。

## Outcome D：P1 Gate 赢
P2 transport idea NO-GO，直接进入 P3 operator study。

---

# 41. AI/Codex 推进方式

推荐拆成 7 个 Prompt：

```text
Prompt 1  P2 audit，不改代码
Prompt 2  transport 数学层 + synthetic tests
Prompt 3  biaxis_p2 集成
Prompt 4  P2 diagnostics
Prompt 5  Movies smoke
Prompt 6  P2 screen
Prompt 7  confirm（根据 screen 决定）
```

---

# 42. Prompt 1：P2 Repository Audit

```text
你现在协助我在 CrisRipper777/0901 中正式进入 Bi-Axis P2。

先不要修改任何代码。

P0/P1 已冻结。P2 只做 NC，冻结实验协议和数据集：
Movies, Toys, Grocery, ele-fashion, Reddit-S。

P1 最终结论：
1. Factor-dependent Graph Demand 稳定存在；
2. Structural Relation Specialization graph-dependent；
3. Factor-specific Relation Selectivity conditional；
4. Reddit-S relation specialization 接近 uniform；
5. Budget B0≈B2，budget 的价值主要是机制，不是稳定精度增益。

P2 科学目标：
用一个统一 plan Gamma_i ∈ R^{3×(K+1)} 替代独立 beta 和 alpha。
column 0 = Local/No-Transport；
columns 1..K = latent structural relations。

P2 不允许修改：
- src/tasks/nc.py
- configs/task/nc.yaml
- src/data/loaders.py / splits
- dataset configs
- biaxis_p0.py / biaxis_components.py
- biaxis_p1.py / biaxis_p1_components.py
- P1 relation definition
- K=4
- W0 shared operator

请仔细审查当前：
- src/models/biaxis_p1.py
- src/models/biaxis_p1_components.py
- configs/model/biaxis_p1.yaml
- tests/test_biaxis_p1.py
- scripts/analyze_p1_checkpoint.py
- scripts/run_p1_screen.py
- scripts/summarize_p1.py

重点回答：
1. biaxis_p2 是否可以安全 subclass P1Model 并 override _graph_update？
2. 能否在 P2 __init__ 中删除 graph_budget / factor_selector，避免 unused trainable params？
3. 哪些 P1 functions 可以原样复用？
4. P1 forward 对 graph_out 的字段有哪些硬依赖？
5. P2 diagnostics 应如何扩展而不破坏 P1？
6. full-graph 下 Gamma[N,3,5] 的显存是否可接受？
7. 哪些路径可能造成 semantic leakage into relation decomposition？
8. 哪些路径可能造成 capacity prior 被模型 gaming？
9. stop-gradient 应放在哪里？
10. 给出最小侵入 implementation plan。

不要写代码。
```

---

# 43. Prompt 2：只实现 Transport 数学层

```text
基于 P2 audit，现在只实现数学组件，不接 biaxis_p2 Model。

新增：
src/models/biaxis_p2_components.py
tests/test_biaxis_p2.py

实现：

1. FactorRelationScore
   输入 f:[N,F,d], g:[N,F,K,d]
   score_ifk = shared MLP([f, g_k, f*g_k])
   输出 [N,F,K]
   不使用 availability。

2. Null score
   3 个 learnable scalar z_f，初始化 0。

3. build_augmented_scores:
   [null_score, relation_scores] -> [N,F,K+1]

4. build_reference_capacity:
   nu_i = F * [pi0, (1-pi0)*availability_i]
   pi0=0.5
   availability 必须 detach。
   isolated nodes 单独处理。

5. compute_node_relation_confidence:
   q_i = 1 - mean_incoming_edge_entropy/log(K)
   基于 r:[E,K]
   isolated q=0
   clamp[0,1]
   q 必须 detach。

6. NullSoftmax:
   gamma = softmax(scores_aug/epsilon)

7. SemiRelaxedTransport:
   hard row marginal mu=ones(F)
   soft column KL
   theta=tau/(tau+epsilon)
   log-domain generalized Sinkhorn
   fixed iteration count=10
   final recompute log_u。

8. Adaptive SemiRelaxedTransport:
   tau_i=tau_base*q_i
   theta_i=tau_i/(tau_i+epsilon)

不要接 graph message passing。

单元测试必须覆盖：
- shape
- row sum
- null-softmax exactness
- tau=0 == null-softmax
- large tau 更接近 target columns
- null score monotonicity
- isolated node all-null
- uniform r -> q≈0
- one-hot r -> q≈1
- q/nu stop-gradient
- score/null gradient finite and nonzero
- numerical stability for large positive/negative scores
- no NaN

实现后运行全部 P0/P1/P2 tests。
不要进入模型集成。
```

---

# 44. Prompt 3：集成 biaxis_p2.py

```text
P2 transport components 已通过测试。现在实现 biaxis_p2 Model。

新增：
src/models/biaxis_p2.py
configs/model/biaxis_p2.yaml

要求：

1. class Model 继承 biaxis_p1.Model。
2. P1 M1/M2/W0 原样复用。
3. assert factor_aware=true, num_relations=4。
4. 删除 inherited graph_budget / factor_selector。
5. override _graph_update。

_graph_update 流程：
a. P1 _decompose_relations -> r, availability
b. P1 relation_weighted_mean -> g:[N,3,K,d]
c. relation scores
d. augment null scores
e. 根据 mode：
   null_softmax
   fixed_uot
   adaptive_uot
   得 gamma:[N,3,K+1]
f. graph_mass = 1-gamma[...,0]
g. alpha_diag = gamma[...,1:] / (graph_mass+eps)
h. g_mix = sum_k gamma[...,k+1] * g_k
i. m = W0(g_mix)
j. f_tilde = LayerNorm(f + m)
k. 返回兼容 + P2-specific fields

禁止：
- 新 W_f/W_k/W_fk
- 新 budget MLP
- 新 selector MLP
- pseudo node
- diffusion
- 新 loss

保持 P0 aux loss 不变。
P2 默认不增加 transport auxiliary loss。

inference 继续 full-graph exact。

测试：
- 三 mode forward
- shapes
- isolated node
- gamma row sum
- graph_mass/null relation
- derived alpha finite
- shared W0 唯一
- P1 relation decomposition 不变
- forward/inference equivalence
- gradient flow through scorer/transport/W0/relation context
- inherited budget/selector 不再存在/不计参数
```

---

# 45. Prompt 4：P2 Diagnostics

```text
P2 Model 已通过测试。现在实现 P2 best-checkpoint mechanism diagnostics。

新增：
scripts/analyze_p2_checkpoint.py
并在 Model 中实现 compute_p2_diagnostics。

输出：

Relation side（继续保留 P1）：
- K_eff
- H_R
- S_R

Plan：
- null mean/std/p10/p50/p90/high_frac for C/Pt/Pv
- graph mass mean for C/Pt/Pv
- plan entropy C/Pt/Pv
- conditional relation alpha entropy
- conditional relation JS C/Pt / C/Pv / Pt/Pv
- capacity KL
- capacity L1

Adaptive only：
- relation confidence mean/p10/p50/p90
- theta mean/p10/p50/p90

保存：
diagnostics.json
transport_plan_summary.csv
conditional_usage_matrix.csv

不要保存完整 gamma，除非 debug flag 显式开启。
diagnostics 不使用 labels。
```

---

# 46. Prompt 5：Movies Smoke

```text
现在只做 P2 smoke。

先运行全部 tests。

Movies seed=42：

1. null_softmax 5 epochs
2. fixed_uot 5 epochs
3. adaptive_uot 5 epochs

检查：
- 无 NaN
- row residual
- null mass 不全 0/1
- graph mass 合理
- plan entropy
- capacity KL
- relation confidence / theta
- epoch time
- peak memory
- gradient norm
- checkpoint diagnostics 可读取

然后各跑 10 epochs 做第二轮 smoke。

如果出现问题：
只定位实现/数值原因；
不加新的 loss；
不改 relation decomposition；
不调 dataset-specific 参数；
不进入 screen。
```

---

# 47. Prompt 6：P2 Screen 15 Runs

```text
P2 smoke 已通过。现在执行 P2-screen。

不要重新跑 P1。
P1 F1R1 seed42 直接读取已有结果作为 reference。

新跑：

datasets:
Movies
Toys
Grocery
ele-fashion
Reddit-S

seed:
42

modes:
NS   = null_softmax
FUOT = fixed_uot
AUOT = adaptive_uot

共 15 runs。

统一：
epsilon=0.2
tau_base=1.0
null_prior=0.5
sinkhorn_iters=10
其它 P0/P1 config 全部冻结。

生成：
outputs/p2/screen/p2_screen_results.csv
outputs/p2/screen/p2_screen_mechanism.csv
outputs/p2/screen/P2_SCREEN_REPORT.md

报告必须比较：
P1 F1R1 vs NS vs FUOT vs AUOT

重点回答：
1. unified plan 是否优于/不弱于 P1 beta×alpha？
2. UOT 是否优于 NullSoftmax？
3. AUOT 是否比 FUOT 更适合 low-S_R 图？
4. Grocery/ele 是否从 relation-side capacity 得益？
5. Reddit-S 是否自动接近 weak-constraint regime？
6. Toys 的 P1 negative interaction 是否得到缓解？
7. 是否存在全-null / uniform-plan collapse？

只给 GO / REVISE / NO-GO。
不要实现 P3。
```

---

# 48. Prompt 7：Confirm 模板

根据 screen 再决定。

## 如果 AUOT 明显最好

```text
确认 NullSoftmax + Adaptive-UOT。
复用 seed42。
补跑 seeds43/44：
5 datasets × 2 modes × 2 seeds = 20 new runs。
P1 使用已有 confirm 3-seed 结果。
```

## 如果 Fixed≈Adaptive

```text
删除 adaptive complexity。
确认 NullSoftmax + Fixed-UOT。
```

## 如果 NullSoftmax 最好

```text
只确认 NullSoftmax seeds43/44。
不要继续救 OT。
```

---

# 49. P2 完成 Definition of Done

- [ ] P1 frozen files 未被修改
- [ ] NC protocol 未被修改
- [ ] score / null / solver 单元测试通过
- [ ] tau=0 数值退化到 NullSoftmax
- [ ] adaptive confidence 在 uniform r 上退化
- [ ] isolated node all-local
- [ ] capacity prior / confidence stop-gradient
- [ ] P2 Model 只保留 shared W0
- [ ] 15-run screen 完成
- [ ] screen GO/REVISE/NO-GO 完成
- [ ] 必要时 balanced diagnostic 完成
- [ ] 必要时 multi-seed confirm 完成
- [ ] P2_REPORT.md 完成
- [ ] 最终明确选择：P1 Gate / NullSoftmax / Fixed-UOT / Adaptive-UOT
- [ ] 未提前实现 P3 operator

---

# 50. P2 最终要回答的五句话

完成 P2 后，必须能够用数据回答：

1. **P1 的 how-much 与 which-relation 是否需要两个独立 predictor，还是一个 unified plan 更好？**
2. **relation-side capacity constraint 是否真的提供额外收益？**
3. **在 relation specialization 较弱的图中，coupler 能否自动退化而不制造虚假 relation selectivity？**
4. **Local / No-Transport state 是否能用一个统一 plan 表达 Factor-dependent Graph Demand？**
5. **最终最简单且稳定的 coupler 是 P1 Gate、NullSoftmax、Fixed-UOT 还是 Adaptive-UOT？**

只有 P2 明确回答完这五个问题，才进入 P3：

`How should a structural relation transform each semantic factor?`

P3 才研究 `W_f,k` 与 Low-rank Factor–Relation Operator。
