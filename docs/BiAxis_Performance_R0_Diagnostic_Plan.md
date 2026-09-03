# Bi-Axis Performance-R0 — 性能瓶颈诊断实验推进计划
## 目标：不修改模型、不重新设计模块，先定位 Factor → Relation → Context → Γ → Hop 的真实性能瓶颈

> Repository: `CrisRipper777/0901`  
> 当前最终模型：`biaxis_final` / P3 OFR  
> 当前任务范围：**仅 Node Classification**  
> 数据集：Movies / Toys / Grocery / ele-fashion / Reddit-S  
> 主诊断 seeds：42 / 43 / 44  
> **本阶段禁止修改最终模型结构，禁止基于 Test 指标做任何结构选择。**

---

# 0. 为什么先做 R0 Diagnostic，而不是直接继续加模块？

当前模型已经形成完整的：

\[
\text{Semantic Factor}
\rightarrow
\text{Structural Relation}
\rightarrow
\text{Relation-conditioned Context}
\rightarrow
\Gamma
\rightarrow
T_{fk}
\]

计算链，但正式 NC Benchmark 显示：

- Movies / Toys / Grocery 在 Val 上仍落后 DiP；
- ele-fashion / Reddit-S 已经接近或达到强基线水平；
- Semantic-factor-aware graph modeling 和 Hierarchical Operator 有较明确贡献；
- Structural Relation Axis 与 learned allocation 的消融贡献偏弱。

因此当前最关键的问题不是：

> “再加什么新模块？”

而是先回答：

\[
\boxed{
\textbf{
性能信息到底在哪一层被损失、弱化或没有被利用？
}
}
\]

R0 的唯一任务是把这条链拆开诊断：

```text
Factor 质量够不够？
        ↓
Relation 真分出有意义的关系了吗？
        ↓
relation-conditioned contexts 真不同且有用吗？
        ↓
Γ 真会选择好 context 吗？
        ↓
当前 1-hop 是否缺失可利用的 2/3-hop 信息？
```

R0 完成后，再根据证据决定第一刀应该改：

```text
Relation
Context
Γ
Multi-hop
Factorizer
```

而不是并行堆模块。

---

# 1. 诊断纪律

## 1.1 不改 Frozen Model

本阶段不得修改：

```text
src/models/biaxis_final.py
src/models/biaxis_p0.py
src/models/biaxis_p1.py
src/models/biaxis_p2.py
src/models/biaxis_p3.py
其对应 frozen configs
```

所有诊断代码放在独立：

```text
scripts/perf_r0_*.py
```

必要时新增：

```text
src/analysis/perf_r0_utils.py
```

---

## 1.2 不重新训练主模型

优先直接复用 P3 OFR best checkpoints：

```text
outputs/p3/operator/<dataset>/OFR/seed_<seed>/model.pt
```

原因：

- OFR 与当前 `biaxis_final` 数学结构相同；
- checkpoint 中同时保存 `model_state` 与 `head_state`；
- seeds 42/43/44 已存在；
- P3 runner 已按 frozen NC protocol 训练并保存 best checkpoint。

R0 的 counterfactual / mechanism 结果全部**相对于其自身 OFR checkpoint baseline**比较，
不要与 fresh `biaxis_final` benchmark run 做 0.1pp 级直接数值比较。

---

## 1.3 Test labels 在 R0 中封存

### 可以使用

- 所有节点 features；
- 完整 transductive topology；
- train labels；
- val labels；
- train/val split。

### 禁止使用

```text
test labels
test accuracy
test Macro-F1
```

做：

- probe 选择；
- bottleneck 判断；
- candidate 排序；
- 模块设计。

所有 supervised diagnostics：

```text
train labels -> fit
val labels   -> diagnostic evaluation
```

---

## 1.4 3 seeds 统一汇总

主诊断：

```text
42 / 43 / 44
```

报告：

\[
mean \pm population\ std.
\]

任何单 seed 异常都不直接做模型决策。

---

# 2. R0 总流程

```text
R0-D0  Repository / Checkpoint Audit
        ↓
R0-D1  Frozen Baseline Snapshot
        ↓
R0-D2  Semantic Factor Quality
        ↓
R0-D3  Structural Relation Quality
        ↓
R0-D4  Relation-conditioned Context Quality
        ↓
R0-D5  Γ Routing / Local-Graph Utility
        ↓
R0-D6  Multi-hop Potential
        ↓
R0-D7  Bottleneck Synthesis
        ↓
决定 Performance-R1 第一刀
```

建议严格按顺序推进。

---

# 3. R0-D0 — Repository / Checkpoint Audit

## 目标

先确认诊断基础完全可靠，不写诊断数学、不跑大分析。

需要确认 15 个 checkpoint：

\[
5\ datasets \times 3\ seeds
\]

全部存在：

```text
outputs/p3/operator/<dataset>/OFR/seed_<seed>/model.pt
```

并且：

```text
task = nc
p2.mode = null_softmax
p2.deterministic = false
p3.operator_mode = full_interaction
K = 4
factor_dim = 128
```

---

## 必须验证 checkpoint 内容

```python
ckpt["model_state"]
ckpt["head_state"]
ckpt["data_info"]
ckpt["seed"]
ckpt["task"]
```

并检查：

- head_state 可以恢复；
- model forward 可运行；
- 原始 current-\(\Gamma\) forward 的 Val Acc 能复现该 checkpoint 对应训练结果；
- 不使用当前 `biaxis_final` fresh benchmark checkpoint，因为默认 benchmark 并未保存 checkpoint。

---

## 输出

```text
outputs/perf_r0/audit/
  checkpoint_audit.csv
  R0_AUDIT.md
```

字段：

```text
dataset
seed
checkpoint_exists
config_match
model_load_ok
head_load_ok
val_reproduce_ok
checkpoint_val_acc
recomputed_val_acc
absolute_diff
```

---

# 4. R0-D1 — Frozen Baseline Snapshot

这一步不增加新指标逻辑，只把已有 P2/P3 diagnostics 在最终 OFR checkpoints 上统一重新汇总。

当前已有：

```text
K_eff
S_R
relation occupancy
null/graph mass
plan entropy
conditional relation usage
operator residual/message deviation
```

直接复用已有 `compute_p3_diagnostics()` 数学。

---

## 新增两个简单但重要的 relation assignment 指标

对：

\[
r_{e,k}
\]

记录：

### Edge assignment confidence

\[
Conf_R
=
\mathbb E_e\max_k r_{e,k}
\]

### Normalized relation entropy

\[
H_R^{norm}
=
\frac{
\mathbb E_e[-\sum_k r_{e,k}\log r_{e,k}]
}{
\log K
}.
\]

---

## Prototype separation 只允许 within-seed

对当前 seed 内：

\[
\rho_k
\]

计算 pairwise cosine。

记录：

```text
mean off-diagonal cosine
max off-diagonal cosine
min pairwise cosine distance
```

### 禁止

直接比较不同 seed 的 prototype vector：

```text
cos(rho_seed42, rho_seed43)
```

因为 relation embedding basis 本身可旋转，跨 seed 原始 prototype vector 不具备直接可比性。

---

## 输出

```text
outputs/perf_r0/baseline_snapshot/
  per_seed_snapshot.csv
  dataset_summary.csv
  R0_BASELINE_SNAPSHOT.md
```

---

# 5. R0-D2 — Semantic Factor Quality Diagnostic

## 核心问题

\[
\boxed{
C,\ P_t,\ P_v
\text{ 本身是否已经拥有足够的任务判别信息？}
}
\]

如果 factor 本身弱，后面 Relation / Γ 再复杂也救不了。

---

## 5.1 提取 frozen representations

从 checkpoint 提取：

```text
h_t
h_v
c_t
c_v
C
Pt
Pv
z_local
```

同时提取最终 graph updated：

```text
C'
Pt'
Pv'
z_final
```

---

## 5.2 Label-free factor statistics

记录：

### Common alignment

\[
\mathbb E_i\cos(c_i^t,c_i^v)
\]

### Private cross-modal similarity

\[
\mathbb E_i\cos(p_i^t,p_i^v)
\]

### Common-private overlap

```text
cos / cross-cov overlap:
C vs Pt
C vs Pv
Pt vs Pv
```

### Norm / variance

对 C/Pt/Pv：

```text
mean norm
feature variance
effective rank（可选）
```

---

## 5.3 Fixed Linear Probe

不要用复杂 probe。

推荐固定：

```python
StandardScaler()
RidgeClassifier(alpha=1.0)
```

只：

```text
fit(train)
evaluate(val)
```

不调 alpha。

Probe：

```text
h_t
h_v
[C_t|C_v]   # 可选
C
Pt
Pv
[C|Pt|Pv]
z_local
C'
Pt'
Pv'
[C'|Pt'|Pv']
z_final
```

记录：

```text
Val Accuracy
Val Macro-F1
```

---

## 5.4 最关键的差值

### Factorization compression gap

\[
\Delta_{fact}
=
Probe([C|P_t|P_v])
-
Probe([h_t|h_v]).
\]

如果明显为负：

> factorization 可能丢失 discriminative information。

---

### Graph gain

\[
\Delta_{graph}
=
Probe(z_{final})
-
Probe(z_{local}).
\]

以及每 factor：

\[
Probe(f')-Probe(f).
\]

如果某个 factor graph 后反而显著下降：

> downstream propagation 对该 factor 可能存在污染。

---

## 5.5 Factor bottleneck 判据

### Strong Factorizer concern

若在 Movies/Toys/Grocery 中至少 2 个：

\[
\Delta_{fact}<-0.3\text{ pp}
\]

或某个原始 modality representation 明显强于 factorized concat：

\[
>0.5\text{ pp}
\]

则 Factorizer 升为 Performance-R1 候选。

### Otherwise

如果 factorized representation 不弱：

> 暂不改 P0，优先查 Relation/Context/Γ。

---

## 输出

```text
outputs/perf_r0/factor/
  factor_stats_per_seed.csv
  factor_probe_per_seed.csv
  factor_probe_summary.csv
  R0_FACTOR_REPORT.md
```

---

# 6. R0-D3 — Structural Relation Quality Diagnostic

## 核心问题

当前：

\[
R=f(A)
\]

只基于结构 signature。

需要判断：

\[
\boxed{
R_1,\dots,R_4
\text{ 是否真的形成了互相不同、稳定、有任务潜力的 relation basis？}
}
\]

---

# 7. R0-D3.1 Relation occupancy / confidence / support

每 seed 计算：

### Global occupancy

\[
p_k
=
\frac1E\sum_e r_{e,k}.
\]

### Effective relation number

\[
K_{eff}
=
\exp(H(p)).
\]

### Edge entropy

\[
H(r_e)
\]

### Max assignment confidence

\[
\max_k r_{e,k}.
\]

### Node relation mass

\[
m_{ik}
=
\sum_{j\in N(i)}
r_{ji,k}.
\]

### Availability

\[
a_{ik}
=
\frac{m_{ik}}{d_i+\epsilon}.
\]

对 \(m,a\) 记录：

```text
mean
p10 / p50 / p90
fraction mass < 0.5
fraction availability < 0.05
```

目的：

> 检查是否存在大量极低 support relation contexts。

---

# 8. R0-D3.2 Relation 是否只是在重复 degree pattern？

对每个 relation k，用 \(r_{e,k}\) 作为 soft weight，统计：

```text
src log-degree
dst log-degree
|log d_src - log d_dst|
u1/u2 structural signature statistics
```

然后计算 relation 之间这些 behavioral signatures 的 pairwise distance。

如果四个 relation 的 structural profile 几乎相同：

> prototype decomposition 没有真正分出结构角色。

---

# 9. R0-D3.3 Relation semantic coherence potential

虽然 Relation 学习本身不读 semantics，但诊断时允许检查：

> 学出的 relation 是否偶然对应不同 semantic edge patterns？

对：

\[
f\in\{C,P_t,P_v\}
\]

定义：

\[
Sim_{f,k}
=
\frac{
\sum_{(j,i)}
r_{ji,k}
\cos(f_j,f_i)
}{
\sum_{(j,i)}
r_{ji,k}+\epsilon
}.
\]

再记录：

```text
std across k
max-min across k
```

如果：

\[
Sim_{f,1}\approx\cdots\approx Sim_{f,K}
\]

说明 structural relations 对 semantic usefulness 没有分层。

---

# 10. R0-D3.4 Train-only relation homophily

只使用：

```text
两端都在 train split 的 edges
```

定义：

\[
Hom_k
=
\frac{
\sum_{(j,i)\in E_{train}}
r_{ji,k}\mathbf 1[y_j=y_i]
}{
\sum_{(j,i)\in E_{train}}
r_{ji,k}+\epsilon
}.
\]

记录：

```text
Hom_k
range(Hom_k)
std(Hom_k)
```

### 解释

- 不是要求 relation 必须 homophilous；
- 只检查不同 relation 是否对应不同 label relation regimes；
- 不允许使用 val/test labels。

---

# 11. R0-D3.5 Cross-seed relation stability

### 禁止

prototype vector 直接跨 seed cosine。

### 使用 relation behavioral signature

每个 relation 建立一个 label-free signature：

```text
occupancy
src/dst degree
degree gap
C/Pt/Pv edge similarity
availability
```

对 seed pair：

```text
42↔43
42↔44
43↔44
```

使用 Hungarian matching 匹配 relation signatures。

报告：

```text
matched cosine similarity
matched L2 distance
mean / worst match
```

它回答：

> relation 的“行为”是否跨 seed 稳定？

---

## Relation bottleneck 主要判据

Relation Axis 可疑，如果出现以下组合：

```text
1. K_eff 看似正常
BUT
2. semantic coherence across k 差异很小
AND/OR
3. train homophily across k 差异很小
AND/OR
4. cross-seed behavioral stability 很低
```

这意味着：

> prototype 可能只是在结构空间中形成 soft partition，
> 但没有形成 downstream-useful relation basis。

---

## 输出

```text
outputs/perf_r0/relation/
  relation_assignment_per_seed.csv
  relation_support_per_seed.csv
  relation_structural_profiles.csv
  relation_semantic_profiles.csv
  relation_train_homophily.csv
  relation_seed_stability.csv
  R0_RELATION_REPORT.md
```

---

# 12. R0-D4 — Relation-conditioned Context Quality

## 核心问题

即使：

\[
r_{ij,k}
\]

看起来分开了，真正送给 GNN 的：

\[
g_{ik}^{f}
=
\frac{\sum_jr_{ji,k}f_j}{m_{ik}+\epsilon}
\]

是否仍然高度相似？

如果是：

\[
\boxed{
\text{Relation Assignment 有差异，但 Relation Context 已经 collapse。}
}
\]

---

# 13. R0-D4.1 Context Diversity

对 factor \(f\)：

\[
D_{ctx}^{f}
=
\frac{2}{K(K-1)}
\sum_{k<l}
\mathbb E_i
[
1-\cos(g_{ik}^{f},g_{il}^{f})
].
\]

计算时：

- 排除 isolated nodes；
- 对某 relation support 太低的 node-cell 做 mask；
- 默认有效条件建议：

\[
m_{ik}\ge0.5.
\]

同时输出 K×K cosine redundancy matrix。

---

# 14. R0-D4.2 Context 与 plain neighbor mean 的差异

plain：

\[
\bar g_i^f
=
\frac1{d_i}
\sum_j f_j.
\]

计算：

\[
D(g_{ik}^{f},\bar g_i^f)
=
1-\cos(g_{ik}^{f},\bar g_i^f).
\]

如果所有：

\[
g_{ik}^{f}\approx \bar g_i^f,
\]

则 relation decomposition 在真正 message 层基本没有产生新信息。

---

# 15. R0-D4.3 Context-local agreement

定义：

\[
Q_{ifk}
=
\cos(f_i,g_{ik}^{f}).
\]

记录：

```text
mean / std per factor-relation
range across k
```

并计算：

\[
Corr(m_{ik},Q_{ifk})
\]

\[
Corr(a_{ik},Q_{ifk})
\]

检查低 support context 是否更 noisy。

---

# 16. R0-D4.4 Context Probe — 这是 R0 最重要的诊断之一

使用和 Factor Probe 相同的固定 Ridge protocol。

对每个 factor f：

### Local

\[
Probe(f_i)
\]

### Plain neighbor

\[
Probe(\bar g_i^f)
\]

### Relation contexts concatenated

\[
Probe([g_{i1}^f|\cdots|g_{iK}^f])
\]

### Local + plain neighbor

\[
Probe([f_i|\bar g_i^f])
\]

### Local + relation contexts

\[
Probe([f_i|g_{i1}^f|\cdots|g_{iK}^f]).
\]

---

## 核心量：Relation Context Potential Gain

\[
\boxed{
\Delta_{relctx}^{f}
=
Probe([f|g_1|\cdots|g_K])
-
Probe([f|\bar g]).
}
\]

### 解释

如果：

\[
\Delta_{relctx}^{f}\approx0
\]

那么即使后面的 \(\Gamma\) 再聪明，也几乎没有“更好的 relation context”可以选择。

如果：

\[
\Delta_{relctx}^{f}>0.3\text{ pp}
\]

甚至：

\[
>0.5\text{ pp},
\]

但最终 Relation Axis 消融仍接近 0：

> Context 有潜力，但 Router/Operator 没把潜力利用出来。

这是非常关键的分流证据。

---

## 输出

```text
outputs/perf_r0/context/
  context_diversity.csv
  context_redundancy_matrices/
  context_agreement.csv
  context_probe_per_seed.csv
  context_probe_summary.csv
  R0_CONTEXT_REPORT.md
```

---

# 17. R0-D5 — Γ Routing / Local-Graph Utility Diagnostic

## 核心问题

\[
\boxed{
\Gamma
\text{ 是否真的把更多质量分给了更有用的 context？}
}
\]

以及：

\[
\boxed{
当前 Local score / graph-demand parameterization 是否太弱？
}
\]

---

# 18. R0-D5.1 Score-level diagnostics

当前 relation score：

\[
s_{ifk}
=
MLP([f_i,g_{ik}^f,f_i\odot g_{ik}^f])
\]

Local score 是每个 factor 一个 global scalar。

诊断脚本中重新计算：

```text
s_rel
s_local
gamma
```

记录：

### Local scalar

```text
null_score_C
null_score_Pt
null_score_Pv
```

### Local-vs-best-relation margin

\[
M_{if}
=
\max_k s_{ifk}
-
s_{if0}.
\]

记录：

```text
mean/std/p10/p50/p90
```

### Gamma top margin

\[
\Gamma_{top1}-\Gamma_{top2}.
\]

### Local win fraction

\[
P[
\Gamma_{if0}
=
\max_c\Gamma_{ifc}
].
\]

---

# 19. R0-D5.2 Routing–quality alignment

使用前面：

\[
Q_{ifk}=\cos(f_i,g_{ik}^{f})
\]

计算：

\[
Corr(
\Gamma_{ifk},
Q_{ifk}
).
\]

同时：

\[
Corr(
\Gamma_{ifk},
a_{ik}
)
\]

以及 conditional：

\[
Corr(
\alpha_{ifk},
Q_{ifk}
).
\]

### 解释

如果 contexts 有明显质量差异，但：

\[
Corr(\alpha,Q)\approx0,
\]

说明 router 没有利用 context quality。

---

# 20. R0-D5.3 Frozen Counterfactual Routing

这是 R0 最关键的 routing functional test。

### 必须：

- 使用当前 checkpoint 的 model weights；
- 使用同一个 `head_state`；
- **不重新训练**；
- 只评价 Val；
- current-\(\Gamma\) sanity 必须复现原 checkpoint Val Acc。

---

## CF0 — Current

\[
\Gamma^{cur}
\]

sanity baseline。

---

## CF1 — Uniform Relation，保留当前 graph mass

保留：

\[
\beta_i^f=1-\Gamma_{if0}.
\]

但：

\[
\Gamma_{ifk}^{uni}
=
\frac{\beta_i^f}{K}.
\]

回答：

> learned relation selection 是否真的有用？

---

## CF2 — Availability Relation，保留当前 graph mass

\[
\Gamma_{ifk}^{avail}
=
\beta_i^f a_{ik}.
\]

回答：

> semantic learned relation selection 是否优于纯 topology availability？

---

## CF3 — Hard Top-1 Relation，保留当前 graph mass

\[
k^*=\arg\max_k\alpha_{ifk}.
\]

\[
\Gamma_{ifk^*}=\beta_i^f,
\quad
others=0.
\]

回答：

> 当前 soft mixture 是否必要？

---

## CF4 — No Local

保留 learned conditional alpha：

\[
\Gamma_{if0}=0,
\qquad
\Gamma_{ifk}=\alpha_{ifk}.
\]

回答：

> Local / No-Transport 状态是否真的承担功能？

---

## CF5 — Factor-mean Fixed Graph Mass

定义：

\[
\bar\beta^f
=
\mathbb E_i\beta_i^f.
\]

然后：

\[
\Gamma_{if0}=1-\bar\beta^f,
\]

\[
\Gamma_{ifk}
=
\bar\beta^f\alpha_{ifk}.
\]

回答：

> 当前 node-specific graph demand 是否真的有作用？

---

## CF6 — Local Only

\[
\Gamma_{if0}=1,
\qquad
\Gamma_{ifk}=0.
\]

回答：

> graph branch 对当前 trained representation 的整体依赖程度。

---

# 21. Counterfactual 的关键差值

### Learned relation selection utility

\[
\Delta_{select}
=
Val(Current)
-
Val(UniformRel).
\]

---

### Semantic selection beyond topology

\[
\Delta_{sem-select}
=
Val(Current)
-
Val(AvailabilityRel).
\]

---

### Dynamic graph-demand utility

\[
\Delta_{demand}
=
Val(Current)
-
Val(FixedMeanBeta).
\]

---

### Local-state utility

\[
\Delta_{local}
=
Val(Current)
-
Val(NoLocal).
\]

---

## Routing bottleneck 判据

### Router under-utilization

如果：

\[
\Delta_{relctx}>0.3\text{ pp}
\]

但是：

\[
|\Delta_{select}|<0.1\text{ pp}
\]

且：

\[
Corr(\alpha,Q)\approx0,
\]

则：

\[
\boxed{
\textbf{Context 有潜力，但 \(\Gamma\) 没有利用。}
}
\]

Performance-R1 应优先做：

```text
Evidence-aware Γ
Dynamic Local score
Context-aware routing
```

---

### Relation/context bottleneck

如果：

\[
\Delta_{relctx}\approx0
\]

且：

\[
Current\approx Uniform\approx Availability,
\]

则：

\[
\boxed{
\textbf{Router 不是根因，Relation/Context 才是。}
}
\]

---

## 输出

```text
outputs/perf_r0/routing/
  routing_scores.csv
  routing_alignment.csv
  routing_counterfactual_per_seed.csv
  routing_counterfactual_summary.csv
  R0_ROUTING_REPORT.md
```

---

# 22. R0-D6 — Multi-hop Potential Diagnostic

## 核心问题

当前模型真正 semantic message propagation 主要是一轮 1-hop。

先不写 multi-hop model，先问：

\[
\boxed{
2/3-hop 信息是否真的能提升 frozen representation 的 Val separability？
}
\]

---

# 23. R0-D6.1 Plain diffusion trajectory

使用与当前 message direction 一致的 row-normalized：

\[
P=D^{-1}A.
\]

对 frozen representation：

\[
Z^{(0)}=Z.
\]

计算：

\[
Z^{(1)}=PZ^{(0)},
\]

\[
Z^{(2)}=PZ^{(1)},
\]

\[
Z^{(3)}=PZ^{(2)}.
\]

---

# 24. 两套 trajectory 都测试

## Local trajectory

\[
Z=z_{local}.
\]

测试：

```text
Z0
[Z0|Z1]
[Z0|Z1|Z2]
[Z0|Z1|Z2|Z3]
```

---

## Final embedding trajectory

\[
Z=z_{final}.
\]

同样测试：

```text
Z0
[Z0|Z1]
[Z0|Z1|Z2]
[Z0|Z1|Z2|Z3]
```

全部用固定 Ridge probe：

```text
fit train
eval val
```

---

# 25. Multi-hop gain

定义：

\[
\Delta_{hop2}
=
Probe([Z^0|Z^1|Z^2])
-
Probe(Z^0).
\]

\[
\Delta_{hop3}
=
Probe([Z^0|Z^1|Z^2|Z^3])
-
Probe(Z^0).
\]

同时记录：

```text
1-hop / 2-hop / 3-hop feature cosine convergence
feature variance
```

用于判断 oversmoothing。

---

## Multi-hop bottleneck 判据

### Strong

如果 Movies/Toys/Grocery 至少 2/3：

\[
\Delta_{hop2/3}
\ge+0.5\text{ pp},
\]

则：

\[
\boxed{
\textbf{Multi-hop 升为 Performance-R1 S 级候选。}
}
\]

### Moderate

平均：

\[
+0.3\text{ pp}
\]

以上：

> 值得测试 adaptive trajectory。

### Weak

全部：

\[
<0.2\text{ pp}
\]

则暂时不做复杂 multi-hop model。

---

## 输出

```text
outputs/perf_r0/hop/
  hop_probe_per_seed.csv
  hop_probe_summary.csv
  hop_smoothing_stats.csv
  R0_HOP_REPORT.md
```

---

# 26. R0-D7 — Bottleneck Synthesis

最终生成一张 dataset-level matrix：

| Dataset | Factor | Relation | Context | Γ | Multi-hop | Main Bottleneck |
|---|---|---|---|---|---|---|
| Movies | ... | ... | ... | ... | ... | ... |
| Toys | ... | ... | ... | ... | ... | ... |
| Grocery | ... | ... | ... | ... | ... | ... |
| ele-fashion | ... | ... | ... | ... | ... | ... |
| Reddit-S | ... | ... | ... | ... | ... | ... |

---

# 27. 建议的 summary quantities

每 dataset 至少汇总：

```text
Factor:
  Δ_fact
  Δ_graph

Relation:
  K_eff
  relation confidence
  semantic coherence range
  train homophily range
  seed behavioral stability

Context:
  D_ctx
  Δ_relctx

Γ:
  Δ_select
  Δ_sem-select
  Δ_demand
  Δ_local
  Corr(alpha,Q)

Hop:
  Δ_hop2
  Δ_hop3
```

---

# 28. Bottleneck 分类规则

## A. Factor-Limited

满足明显：

```text
factorized concat < projected raw modalities
或 graph前 factor 本身判别性弱
```

→ 下一阶段优先：

```text
adaptive common fusion
Common InfoNCE/VICReg
Private denoising
```

---

## B. Relation-Limited

表现：

```text
relation assignment confidence低
relation behavioral profile差异小
relation semantic coherence差异小
relation seed stability差
```

→ 优先：

```text
rich structural encoding
semantic-calibrated structural relation
```

---

## C. Context-Limited

表现：

```text
relation assignment有差异
BUT
D_ctx低
Δ_relctx≈0
```

→ 优先：

```text
factor-conditioned edge reliability
neighbor attention
support-aware context
```

---

## D. Router-Limited

表现：

```text
Δ_relctx明显正
BUT
Current≈Uniform/Availability
Corr(alpha,Q)低
```

→ 优先：

```text
dynamic Local score
support/evidence-aware relation score
DMCAR-style global factor context
```

---

## E. Receptive-field-Limited

表现：

```text
2/3-hop frozen probe提升明显
```

→ 优先：

```text
adaptive multi-hop trajectory
```

---

# 29. R0 最终不应该做什么

禁止在 R0：

```text
增加新 relation MLP
增加 semantic reliability gate
增加 multi-hop layer
修改 Γ scorer
改 Common loss
改 operator
重新调 K
重新调 UOT
```

R0 只诊断。

否则无法知道原模型真正的问题在哪。

---

# 30. 推荐代码组织

```text
src/analysis/
  perf_r0_utils.py

scripts/
  perf_r0_audit.py
  perf_r0_snapshot.py
  perf_r0_factor.py
  perf_r0_relation_context.py
  perf_r0_routing_counterfactual.py
  perf_r0_hop_probe.py
  summarize_perf_r0.py
```

输出：

```text
outputs/perf_r0/
  audit/
  baseline_snapshot/
  factor/
  relation/
  context/
  routing/
  hop/
  summary/
```

---

# 31. AI / Codex Prompt 1 — Repository & Checkpoint Audit

> **现在只执行这一 Prompt。完成后先把 audit 结果给我/ChatGPT 审查，不要继续实现 D1-D6。**

```text
我们现在进入 Bi-Axis Performance-R0 性能瓶颈诊断阶段。

Repository:
CrisRipper777/0901

目标：
不修改模型、不训练新模型，先诊断当前 frozen OFR 的
Factor -> Relation -> Context -> Gamma -> Hop 链条。

重要纪律：
1. 不修改 biaxis_p0/p1/p2/p3/final。
2. 不使用 test labels / test metrics 做任何诊断决策。
3. 主诊断 seeds = 42,43,44。
4. 优先复用 P3 OFR checkpoints：
   outputs/p3/operator/<dataset>/OFR/seed_<seed>/model.pt
5. checkpoint 内应有 model_state + head_state。
6. 所有 counterfactual 未来必须相对同一个 checkpoint 自身 current-Gamma baseline 比较，
   不与 fresh final benchmark 的 0.1pp 差异直接比较。

先不要写诊断模块。

请审查：
- outputs/p3/operator 目录结构
- 5 datasets × seeds42/43/44 的 OFR model.pt 是否 15/15 存在
- 每个 checkpoint 的字段
- 对应 .hydra/config.yaml
- src/tasks/nc.py checkpoint 保存/加载协议
- src/models/biaxis_p0.py
- biaxis_p1.py / biaxis_p1_components.py
- biaxis_p2.py / biaxis_p2_components.py
- biaxis_p3.py
- scripts/analyze_p3_checkpoint.py

回答：
1. 15 个 OFR checkpoint 是否都可直接作为 R0 基础？
2. OFR 与当前 biaxis_final 的数学结构是否完全一致？
3. checkpoint 是否包含 classifier head？
4. 如何加载 saved Hydra config，避免当前 config 漂移？
5. 如何重新计算 current forward 并用 head_state 复现 checkpoint Val Acc？
6. 现有 compute_p3_diagnostics 已经覆盖哪些指标？
7. 为 R0 新增哪些值必须手动重算：
   factors / r / mass / g_perm / scores / gamma / z_local / z_final？
8. 哪些 tensor 对 ele-fashion 可能导致显存问题？
9. 推荐怎样 chunk，不改变数学？
10. 给 PASS/FAIL audit 和最小实现结构。

输出：
outputs/perf_r0/audit/R0_AUDIT.md

不要修改 frozen model。
不要跑新训练。
```

---

# 32. Prompt 2 — 实现共享诊断基础层

> Audit PASS 后执行。

```text
R0 audit 已通过。

现在只实现共享 diagnostic extraction，不做 probe/counterfactual。

新增：
src/analysis/perf_r0_utils.py
scripts/perf_r0_snapshot.py

要求：
1. 从 P3 OFR checkpoint + saved Hydra config 加载 model/head/data。
2. 只使用 seeds42/43/44。
3. test labels 完全不访问。
4. current forward + head 在 val split 上必须复现 checkpoint Val Acc。
5. 提取但不要永久保存超大 node/edge tensor：
   factors h_t,h_v,c_t,c_v,C,Pt,Pv
   z_local
   r
   mass / availability
   g_perm
   s_rel / s_aug
   gamma
   f_tilde
   z_final
6. 复用 compute_p3_diagnostics 的已有统计。
7. 新增：
   edge max relation confidence
   normalized relation entropy
   within-seed prototype pairwise cosine
8. 大 tensor 只在单 dataset/seed 生命周期内存在。
9. ele-fashion 使用 chunk / streaming statistics，禁止保存完整 E×K×d。
10. 输出 per-seed JSON/CSV 和 dataset summary。

tests/sanity：
- current val reproduction
- gamma row sum == 1
- r row sum == 1
- availability nonisolated row sum ≈1
- current z_final 与 model.forward output一致
- no test labels read

不要实现 Factor probe。
不要实现 Counterfactual。
```

---

# 33. Prompt 3 — Factor Quality Diagnostic

```text
共享 extraction 已通过。

实现：
scripts/perf_r0_factor.py

对每个 OFR checkpoint 提取：
h_t,h_v,C,Pt,Pv,z_local,C',Pt',Pv',z_final。

Label-free：
- common similarity c_t/c_v
- private similarity Pt/Pv
- pairwise factor overlap
- norm / variance

固定 supervised probe：
StandardScaler + RidgeClassifier(alpha=1.0)
fit TRAIN only
eval VAL only

Probe：
h_t
h_v
[h_t|h_v]
C
Pt
Pv
[C|Pt|Pv]
z_local
C'
Pt'
Pv'
[C'|Pt'|Pv']
z_final

输出：
factor_stats_per_seed.csv
factor_probe_per_seed.csv
factor_probe_summary.csv
R0_FACTOR_REPORT.md

报告核心：
Δ_fact = Probe([C|Pt|Pv])-Probe([h_t|h_v])
Δ_graph = Probe(z_final)-Probe(z_local)
每 factor graph前后 probe差异

不要使用 test。
不要调 probe hyperparameter。
```

---

# 34. Prompt 4 — Relation + Context Diagnostic

```text
实现：
scripts/perf_r0_relation_context.py

Relation：
- occupancy p_k
- K_eff
- edge entropy / normalized entropy
- max assignment confidence
- relation mass m_ik statistics
- availability statistics
- within-seed prototype separation
- r-weighted structural profile
- r-weighted semantic edge cosine for C/Pt/Pv
- train-train edge weighted homophily（TRAIN LABELS ONLY）

Cross-seed：
不要直接 cosine prototype vectors。
构造 label-free behavioral signature：
occupancy
src/dst degree
degree gap
C/Pt/Pv edge similarity
availability
用 Hungarian matching 比较 42/43/44 的 relation behavior stability。

Context：
- D_ctx^C/Pt/Pv
- K×K context cosine redundancy
- g_fk vs plain neighbor mean
- Q_ifk=cos(f_i,g_ifk)
- corr(m,Q), corr(a,Q)
- support mask default m>=0.5

Context fixed Ridge probe：
对 C/Pt/Pv 分别：
f
g_bar
concat(g1..gK)
[f|g_bar]
[f|g1..gK]

核心：
Δ_relctx^f =
Probe([f|g1..gK]) - Probe([f|g_bar])

train only fit / val only eval。
不用 test。

输出 relation/ 和 context/ 全部 csv + 两份报告。
```

---

# 35. Prompt 5 — Γ Counterfactual Diagnostic

```text
实现：
scripts/perf_r0_routing_counterfactual.py

必须加载：
OFR checkpoint model_state + head_state。

先 current-Gamma sanity：
重新算 embedding + classifier，
Val Acc 必须复现原 checkpoint。

只改变 Gamma，不训练任何参数。

CF0 current

CF1 uniform relations, preserve current beta:
gamma0=current gamma0
gamma_k=beta/K

CF2 topology availability, preserve current beta:
gamma0=current gamma0
gamma_k=beta*a_ik

CF3 hard top1 relation, preserve current beta

CF4 no-local:
gamma0=0
gamma_graph=alpha

CF5 fixed factor-mean beta:
beta_bar_f=mean_i beta_if
gamma0=1-beta_bar_f
gamma_graph=beta_bar_f*alpha

CF6 local-only:
gamma0=1
gamma_graph=0

对于每个 counterfactual：
- 使用同一个 frozen operator T_fk
- 同一个 graph_norm / fusion
- 同一个 classifier head
- eval VAL only

同时计算：
- null scores
- max relation score - local score margin
- gamma top1-top2 margin
- local win fraction
- corr(gamma_graph,Q)
- corr(alpha,Q)
- corr(gamma_graph,availability)

输出：
routing_scores.csv
routing_alignment.csv
routing_counterfactual_per_seed.csv
routing_counterfactual_summary.csv
R0_ROUTING_REPORT.md

严禁：
重新训练
oracle routing
使用 val labels 为每个节点挑 relation
读取 test metric
```

---

# 36. Prompt 6 — Multi-hop Potential Probe

```text
实现：
scripts/perf_r0_hop_probe.py

目的：
不修改模型，检查 frozen representation 中是否还存在可利用的 2/3-hop 信息。

使用 row-normalized message-direction P=D^{-1}A。

对两套 embedding：

A. z_local
B. z_final

构造：
Z0
Z1=PZ0
Z2=PZ1
Z3=PZ2

固定 Ridge probe：
Z0
[Z0|Z1]
[Z0|Z1|Z2]
[Z0|Z1|Z2|Z3]

fit train only
eval val only

记录：
Val Acc / Macro-F1
Δ_hop1/2/3
hop-wise mean pair cosine / variance（oversmoothing diagnostic）

注意：
这只是“multi-hop information potential”诊断，
不是 multi-hop 模型性能，
不要把 probe gain 当成最终 architecture gain。

输出：
hop_probe_per_seed.csv
hop_probe_summary.csv
hop_smoothing_stats.csv
R0_HOP_REPORT.md
```

---

# 37. Prompt 7 — Final R0 Synthesis

```text
现在 R0 各阶段已全部完成。

实现：
scripts/summarize_perf_r0.py

读取：
audit
baseline_snapshot
factor
relation
context
routing
hop

输出：
outputs/perf_r0/summary/
  R0_MASTER_TABLE.csv
  R0_BOTTLENECK_MATRIX.csv
  R0_FINAL_DIAGNOSIS.md

每 dataset 汇总：

Factor:
Δ_fact
Δ_graph

Relation:
K_eff
assignment confidence
semantic coherence range
train-homophily range
seed behavioral stability

Context:
D_ctx
Δ_relctx

Gamma:
Δ_select = Current-Uniform
Δ_sem_select = Current-Availability
Δ_demand = Current-FixedBeta
Δ_local = Current-NoLocal
Corr(alpha,Q)

Hop:
Δ_hop2
Δ_hop3

按证据分类：
Factor-Limited
Relation-Limited
Context-Limited
Router-Limited
Receptive-field-Limited

不要自动建议新模块。
只做 bottleneck ranking 和证据强度：
STRONG / MODERATE / WEAK / NO EVIDENCE。

最后给出：
1. Movies 主瓶颈
2. Toys 主瓶颈
3. Grocery 主瓶颈
4. ele-fashion 为什么当前已强
5. Reddit-S 为什么当前已强
6. 跨数据集最值得优先修的 1~2 个共同瓶颈
```

---

# 38. R0 完成后你需要返给我什么？

不用把所有原始 tensor 给我。

请把以下文件发回来：

```text
R0_AUDIT.md
R0_FACTOR_REPORT.md
R0_RELATION_REPORT.md
R0_CONTEXT_REPORT.md
R0_ROUTING_REPORT.md
R0_HOP_REPORT.md
R0_FINAL_DIAGNOSIS.md
R0_MASTER_TABLE.csv（如果方便）
```

最重要的是：

```text
factor_probe_summary.csv
relation_semantic_profiles.csv
context_probe_summary.csv
routing_counterfactual_summary.csv
hop_probe_summary.csv
```

我会基于这些结果决定：

\[
\boxed{
Performance-R1
\text{ 第一刀到底改 Relation、Context、Γ、Multi-hop 还是 Factorizer。}
}
\]

---

# 39. 最终执行顺序

现在严格按下面做：

```text
Step 1:
Prompt 1 — Audit
→ 把 R0_AUDIT.md 发回来先审查

Step 2:
Prompt 2 — Shared extraction
→ sanity 全 PASS

Step 3:
Prompt 3 + Prompt 4
→ Factor / Relation / Context

Step 4:
Prompt 5
→ Gamma functional diagnosis

Step 5:
Prompt 6
→ Multi-hop potential

Step 6:
Prompt 7
→ Final synthesis

Step 7:
把最终结果返给 ChatGPT
→ 决定 Performance-R1
```

**不要跳过 R0 直接开始 V2 模型。**
