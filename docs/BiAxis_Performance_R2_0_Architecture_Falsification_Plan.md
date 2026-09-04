# Bi-Axis Performance-R2-0
## Architecture Falsification & Headroom Probe Experiment Plan

**Repository:** `CrisRipper777/0901`  
**Current retained parent:** `biaxis_final / A0`  
**Stage:** `Performance-R2-0`  
**Purpose:** 在正式进入 R2 架构重构前，用 frozen A0 checkpoints 对三个核心架构假设做定向证伪与 headroom 探测，而不是继续堆叠新模块。

---

# 0. 阶段背景与当前结论

截至目前：

```text
P0/P1/P2/P3   已完成
Final A0      已完成正式 NC Benchmark + Ablation
R0            已完成系统瓶颈诊断
R1-A          NO-GO
R1-B          NO-GO
R1-C          NO-GO
R1.5          未发现足以解释主要性能差距的常规训练/优化异常
```

当前保留模型：

\[
\boxed{\textbf{A0 = biaxis\_final}}
\]

R0→R1→R1.5 的稳定结论：

1. Semantic factorization 基本没有系统性丢失原始 multimodal semantic information；
2. Graph information 对 Movies / Toys / Grocery / Reddit-S 明显有价值；
3. 当前 latent relations 在 prototype 数量上没有 collapse，但 semantic differentiation、homophily differentiation、relation-conditioned context utility 都偏弱；
4. Router 本身不是主要问题，因为当前没有足够“值得选择”的 relation evidence；
5. 当前 graph propagation 只允许：
   \[
   C\rightarrow C,\quad P_t\rightarrow P_t,\quad P_v\rightarrow P_v
   \]
   不允许 graph-level cross-factor semantic transfer；
6. 当前 M2 structural relation construction 原始 topology observation 只有：
   \[
   [\log d,\;P\log d,\;P^2\log d]
   \]
   信息带宽有限；
7. 当前 P0 完成了 semantic ownership decomposition，但没有显式 node-local cross-modal / cross-factor semantic interaction enhancement。

因此 R2-0 只回答三个问题：

\[
\boxed{\textbf{R2-0A：Cross-Factor Transfer Headroom}}
\]

\[
\boxed{\textbf{R2-0B：Structural Function Basis Headroom}}
\]

\[
\boxed{\textbf{R2-0C：Semantic Interaction Headroom}}
\]

---

# 1. R2-0 的基本原则

## 1.1 R2-0 不是训练新模型

R2-0 禁止：

```text
finetune A0
训练新的 GNN
训练新的 relation encoder
训练新的 router
训练新的 semantic refiner
训练 cross-attention / Transformer
训练新的 graph backbone
组合 R1 的 A/B/C 模块
```

允许的唯一监督学习：

```text
StandardScaler
+
RidgeClassifier(alpha=1.0)
```

严格：

```text
fit TRAIN
eval VAL
```

完全复用 R0 fixed probe protocol。

## 1.2 全阶段禁止 Test

整个 R2-0：

```text
NO test_idx
NO test labels
NO test metrics
```

所有方向裁决只依据 Validation。

## 1.3 Frozen checkpoint

优先直接复用 R0 已审计的：

```text
outputs/p3/operator/<dataset>/OFR/seed_<42|43|44>/model.pt
```

原因：

- 与 R0 诊断连续；
- 不需要重新训练；
- 当前 memory optimization 只影响训练显存路径，不应影响 eval 函数；
- R2-0 的目标是 probe，不是刷新 A0 benchmark。

## 1.4 第一轮数据集

先只做：

```text
Movies
Toys
Grocery
```

seeds：

```text
42
43
44
```

总共：

\[
3\ datasets\times3\ seeds=9
\]

个 frozen lifecycle。

`ele-fashion / Reddit-S` 暂时不进入第一轮。只有 A/B/C 中某个方向出现明确 headroom 后，再补它们作为 preservation/control datasets。

---

# 2. 代码组织

优先复用：

```text
src/analysis/perf_r0_utils.py
scripts/perf_r0_factor.py
scripts/perf_r0_relation_context.py
scripts/perf_r0_hop_probe.py
src/models/biaxis_p1_components.py
```

建议新增：

```text
src/analysis/
  perf_r20_utils.py

scripts/
  perf_r20_a_cross_factor.py
  perf_r20_b_structural_basis.py
  perf_r20_c_semantic_interaction.py
  summarize_perf_r20.py

tests/
  test_perf_r20_utils.py

docs/
  Performance_R2_0_Audit.md

outputs/perf_r20/
  audit/
  cross_factor/
  structural_basis/
  semantic_interaction/
  summary/
```

禁止修改 frozen model files：

```text
src/models/biaxis_p0.py
src/models/biaxis_p1*.py
src/models/biaxis_p2*.py
src/models/biaxis_p3*.py
```

---

# 3. R2-0 Audit — 先验证分析链条

正式进入 A/B/C 之前，必须先确认：

1. old OFR checkpoints 可正常加载；
2. current main 下 eval 与 R0 一致；
3. `extract_forward()` 提供的中间量符合预期；
4. `neighbor_mean()` message direction 和 graph aggregation 一致；
5. no-Test discipline 可持续复用。

## 3.1 Reproduction targets

对 Movies seed42 先 smoke：

\[
Probe([C|P_t|P_v])
\]

\[
Probe(z_{local})
\]

\[
Probe(z_{final})
\]

与 R0 已保存结果比较。

要求：

```text
abs diff <= 1e-5
```

若 sklearn / numerical environment 有轻微差异：

```text
<= 1e-4
```

可接受。

之后建议把 M/T/G × 3 seeds 全部复现一次。

## 3.2 Audit 输出

```text
outputs/perf_r20/audit/reproduction.csv
docs/Performance_R2_0_Audit.md
```

Audit FAIL：

```text
停止，不执行 R2-0A/B/C。
```

---

# 4. R2-0A — Cross-Factor Transfer Probe

## 4.1 科学问题

当前 graph path 只允许：

\[
C_j\to C_i
\]

\[
P_j^t\to P_i^t
\]

\[
P_j^v\to P_i^v
\]

也就是说 graph propagation 是 **factor-diagonal** 的。

R2-0A 要回答：

> 对 target factor \(b\)，邻居 source factor \(a\neq b\) 是否具有 current same-factor propagation 没有利用的下游任务信息？

例如：

\[
P_j^v\rightarrow C_i
\]

\[
C_j\rightarrow P_i^t
\]

\[
P_j^t\rightarrow P_i^v
\]

是否有稳定增益。

## 4.2 A1 — Plain 1-hop source→target matrix

Frozen factors：

\[
F_i\in\{C_i,P_i^t,P_i^v\}
\]

对 source factor \(a\)：

\[
N_i^a=
\frac{1}{d_i}
\sum_{j\in\mathcal N(i)}F_j^a
\]

直接复用：

```python
neighbor_mean(edge_index, F_a, N, edge_chunk_size=...)
```

对每个 target \(b\) 和 source \(a\)：

\[
\boxed{
X_{a\to b}^{plain}
=
[F_i^b\Vert N_i^a]
}
\]

所有 9 cells 都是：

\[
2d_f
\]

严格 dimension-matched。

local baseline：

\[
X_b^{local}=F_i^b
\]

定义：

### Graph utility

\[
U_{a\to b}^{plain}
=
Probe(X_{a\to b}^{plain})
-
Probe(F^b)
\]

### Cross-factor advantage

\[
Adv_{a\to b}^{plain}
=
Probe(X_{a\to b}^{plain})
-
Probe(X_{b\to b}^{plain})
\]

其中对角线：

```text
C→C
Pt→Pt
Pv→Pv
```

是 current same-factor inductive bias 的最简对照。

## 4.3 A2 — Current relation-context transfer matrix

从 frozen A0 读取：

```text
g_perm [N,F,K,d]
```

对于 source factor \(a\)：

\[
G_i^a=
[g_{i1}^a\Vert\cdots\Vert g_{iK}^a]
\]

构造：

\[
\boxed{
X_{a\to b}^{rel}
=
[F_i^b\Vert G_i^a]
}
\]

所有 cells 都是：

\[
(1+K)d_f
\]

严格 dimension-matched。

定义：

\[
U_{a\to b}^{rel}
\]

和：

\[
Adv_{a\to b}^{rel}
\]

同 A1。

## 4.4 A3 — All-source upper bound

对每个 target \(b\)：

### Plain all-source

\[
X_{all\to b}^{plain}
=
[F_i^b
\Vert N_i^C
\Vert N_i^{Pt}
\Vert N_i^{Pv}]
\]

### Relation all-source

\[
X_{all\to b}^{rel}
=
[F_i^b
\Vert G_i^C
\Vert G_i^{Pt}
\Vert G_i^{Pv}]
\]

定义 relative to same-factor：

\[
\Delta_{all,b}^{plain}
\]

\[
\Delta_{all,b}^{rel}
\]

注意：

> ALL-source 维度更高，只用于 headroom upper bound，不能单独作为 GO 证据。

真正强证据必须来自 dimension-matched off-diagonal matrix。

## 4.5 R2-0A 判定

### STRONG

同时满足：

1. M/T/G 整体 all-source headroom：
   \[
   \ge+0.30pp
   \]
2. 至少两个稳定 off-diagonal transfer：
   \[
   Adv_{a\to b}\ge+0.20pp,\quad a\neq b
   \]
3. 对应 cell：
   ```text
   >=2/3 seeds positive
   ```
4. 不是由单一 dataset / 单一 cell 独占。

### MODERATE

整体：

```text
+0.15 ~ +0.30pp
```

且存在稳定 off-diagonal signal。

### NO EVIDENCE

```text
all-source < +0.15pp
且 off-diagonal 无稳定优势
```

## 4.6 A 的解释矩阵

### Plain positive + Relation positive

说明：

\[
\boxed{\text{factor-diagonal propagation 本身是瓶颈}}
\]

### Plain positive + Relation weak

说明：

\[
\boxed{
\text{cross-factor signal 存在，
但 current M2 relation pooling 损伤了它}
}
\]

### Plain weak + Relation positive

说明：

```text
current relation decomposition
对 source-factor-specific structure 有一定揭示能力，
但 P3 不允许 target cross-factor consumption。
```

### 两者都 weak

R2 不优先做 3×3 functional semantic transfer。

## 4.7 R2-0A 输出

```text
outputs/perf_r20/cross_factor/
  cross_factor_plain_cells.csv
  cross_factor_relation_cells.csv
  cross_factor_all_source.csv
  R20_A_CROSS_FACTOR_REPORT.md
```

报告必须包含：

```text
3×3 matrix
source rows × target columns
Acc
Macro-F1
per-seed
3-seed mean
population std ddof=0
positive seed count
```

---

# 5. R2-0B — Explicit Structural Function Basis Probe

## 5.1 科学问题

当前 M2：

\[
[\log d,P\log d,P^2\log d]
\rightarrow
K=4\text{ latent prototypes}
\rightarrow
g_{ik}^{f}
\]

R0 已显示：

```text
K_eff non-collapse
BUT semantic-range weak
BUT homophily-range weak
BUT D_ctx small
BUT Δ_relctx small
```

R2-0B 不训练新的 K-prototype。

它直接问：

> 如果构造 4 个真正具有不同 structural function 的 context channels，是否比 current 4 latent relation contexts 更有下游任务信息？

## 5.2 B1 — Rich topology-only signature

定义：

\[
u_0=\log(1+d)
\]

\[
u_1=Pu_0
\]

\[
u_2=Pu_1
\]

\[
u_3=Pu_2
\]

邻居 degree：

\[
\mu_d=P(d)
\]

\[
\sigma_d=
\sqrt{P(d^2)-\mu_d^2}
\]

edge degree-role gap：

\[
gap_{ji}=|u_{0j}-u_{0i}|
\]

聚合到 target：

\[
\mu_{gap}
\]

\[
\sigma_{gap}
\]

最终：

\[
\boxed{
S_i^+
=
[u_0,u_1,u_2,u_3,
\mu_d,\sigma_d,
\mu_{gap},\sigma_{gap}]
}
\]

处理：

```text
whole-graph column z-score
then row L2 normalize
```

严格 topology-only：

```text
不能读 x
不能读 h_t/h_v
不能读 C/Pt/Pv
不能读 labels
不能读 logits
```

## 5.3 B2 — 四个 structural-function channels

对每个 factor \(f\)：

### Channel 1：1-hop ordinary

\[
G_1^f=PF^f
\]

### Channel 2：2-hop diffusion

\[
G_2^f=P(PF^f)
\]

不 materialize 2-hop edge list。

### Channel 3：structurally-similar neighbors

对 observed edge：

\[
c_{ji}=\cos(S_j^+,S_i^+)
\]

\[
w_{ji}^{sim}
=
\frac{1+c_{ji}}{2}+\epsilon
\]

\[
G_{sim,i}^f
=
\frac{
\sum_jw_{ji}^{sim}F_j^f
}{
\sum_jw_{ji}^{sim}+\epsilon
}
\]

### Channel 4：structurally-contrastive neighbors

\[
w_{ji}^{diff}
=
\frac{1-c_{ji}}{2}+\epsilon
\]

\[
G_{diff,i}^f
=
\frac{
\sum_jw_{ji}^{diff}F_j^f
}{
\sum_jw_{ji}^{diff}+\epsilon
}
\]

不使用：

```text
threshold
temperature
learnable parameter
label
```

## 5.4 B3 — Per-factor matched probe

Current：

\[
X_{current}^{f}
=
[F^f
\Vert g_{R1}^f
\Vert g_{R2}^f
\Vert g_{R3}^f
\Vert g_{R4}^f]
\]

Basis：

\[
X_{basis}^{f}
=
[F^f
\Vert G_1^f
\Vert G_2^f
\Vert G_{sim}^f
\Vert G_{diff}^f]
\]

两边：

\[
5d_f
\]

严格同维。

定义：

\[
\Delta_B^f
=
Probe(X_{basis}^{f})
-
Probe(X_{current}^{f})
\]

同时测试：

\[
Probe([F^f|G_1^f])
\]

判断 explicit basis 相对普通 1-hop 多出来的 utility。

## 5.5 B4 — Joint factor probe（Primary）

Current joint：

\[
X_{current}^{joint}
=
[
C,Pt,Pv,
G_{R1..R4}^C,
G_{R1..R4}^{Pt},
G_{R1..R4}^{Pv}
]
\]

Explicit basis joint：

\[
X_{basis}^{joint}
=
[
C,Pt,Pv,
B_{1..4}^C,
B_{1..4}^{Pt},
B_{1..4}^{Pv}
]
\]

两者都是：

\[
15d_f
\]

Primary：

\[
\boxed{
\Delta_B^{joint}
=
Probe(X_{basis}^{joint})
-
Probe(X_{current}^{joint})
}
\]

## 5.6 Context diversity 只作 secondary

计算：

\[
D_{ctx}^f
\]

以及：

```text
4×4 cosine redundancy matrix
```

但：

\[
\boxed{
D_{ctx}\uparrow
\neq
task\ utility\uparrow
}
\]

R2-0B 的 GO/NO-GO：只由 fixed probe utility 决定。

## 5.7 B 判定

### STRONG

\[
mean_{M/T/G}\Delta_B^{joint}
\ge+0.50pp
\]

且至少：

```text
2/3 datasets positive
```

### GO

\[
\ge+0.30pp
\]

且至少 2/3 positive。

### WEAK

```text
+0.15 ~ +0.30pp
```

### NO-GO

```text
< +0.15pp
```

或仅一个 dataset 单点提升。

## 5.8 Memory discipline

禁止：

```text
[N,N]
[K,N,N]
explicit 2-hop edge list
永久保存每个 factor/channel 的 [E,d]
```

允许：

```text
edge_index [2,E]
Splus [N,8]
chunk edge weights
context accumulator [N,d]
```

建议：

```text
edge_chunk_size <= 500000
```

## 5.9 R2-0B 输出

```text
outputs/perf_r20/structural_basis/
  structural_signature_stats.csv
  structural_context_probe_per_factor.csv
  structural_context_probe_joint.csv
  structural_context_diversity.csv
  structural_context_redundancy.csv
  R20_B_STRUCTURAL_BASIS_REPORT.md
```

---

# 6. R2-0C — Semantic Interaction Headroom Probe

## 6.1 科学问题

当前 P0：

\[
(h_t,h_v)\rightarrow(C,P_t,P_v)
\]

但没有显式 cross-modal / cross-factor semantic interaction refinement。

R2-0C 不训练 Transformer。

它只测试：

> frozen factors 中是否存在显式 interaction feature 能暴露、而 current local representation 没充分利用的 task information。

## 6.2 C1 — Factor interaction feature

Base：

\[
X_{factor}^{base}
=
[C|Pt|Pv]
\]

Interaction：

\[
\boxed{
X_{factor}^{inter}
=
[
C,Pt,Pv,
C\odot Pt,
C\odot Pv,
Pt\odot Pv,
|C-Pt|,
|C-Pv|,
|Pt-Pv|
]
}
\]

维度：

\[
9d_f
\]

## 6.3 C2 — Modal interaction

Base：

\[
X_{modal}^{base}
=
[h_t|h_v]
\]

Interaction：

\[
X_{modal}^{inter}
=
[
h_t,h_v,
h_t\odot h_v,
|h_t-h_v|
]
\]

## 6.4 C3 — 必须与 z_local 比较

不能只看：

\[
Probe(X_{factor}^{inter})
>
Probe([C|Pt|Pv])
\]

因为 current `z_local` 已经是 learned fusion。

Primary：

\[
\boxed{
\Delta_C^{local}
=
Probe(X_{factor}^{inter})
-
Probe(z_{local})
}
\]

如果：

```text
interaction > raw factor concat
但 interaction ≈ z_local
```

则判：

\[
\boxed{\text{PARTIAL}}
\]

说明 current local fusion 已经吸收大部分 node-local interaction，不支持优先重构 P0。

## 6.5 C4 — Shuffled interaction negative control

保持：

\[
[C|Pt|Pv]
\]

不变。

interaction-only block 沿 node dimension 使用固定 permutation：

```text
seed = 20260904
```

shuffle。

得到：

\[
X_{factor}^{shuffle}
\]

比较：

\[
Probe(real\ interaction)
-
Probe(shuffled\ interaction)
\]

防止高维 feature map 产生假提升。

不要尝试多个 permutation。

## 6.6 C 判定

### STRONG

\[
mean_{M/T/G}
[
Probe(X_{factor}^{inter})
-
Probe(z_{local})
]
\ge+0.50pp
\]

且：

```text
至少2/3 datasets positive
real interaction > shuffled interaction
```

### GO

\[
\ge+0.30pp
\]

且至少 2/3 positive，real > shuffled。

### PARTIAL

```text
inter > factor concat
但 inter ≈ z_local
```

### NO-GO

```text
< +0.15pp vs z_local
```

或者 real ≈ shuffled。

## 6.7 C 输出

```text
outputs/perf_r20/semantic_interaction/
  semantic_factor_interaction_probe.csv
  semantic_modal_interaction_probe.csv
  semantic_interaction_shuffle_control.csv
  R20_C_SEMANTIC_INTERACTION_REPORT.md
```

---

# 7. R2-0 Final Decision Matrix

## Case 1 — A strong + B strong

R2 优先：

\[
\boxed{
\textbf{
Rich Structural Prior
+
Factor-Conditioned Cross-Factor Functional Transfer
}
}
\]

核心候选：

\[
q_{j\to i}^{a\to b}
=
\psi(F_i^b,F_j^a,e_{ij}^{str})
\]

## Case 2 — A strong + B weak

R2 优先：

```text
Semantic Ownership
→ target-conditioned cross-factor transfer
→ simple topology
```

不要重新造复杂 M2。

## Case 3 — A weak + B strong

说明：

```text
same-factor propagation restriction 不是主问题，
current structural relation basis 才是主瓶颈。
```

R2：

```text
保留 factor-diagonal propagation
替换 M2 为 richer structural function basis
简化/删除 K-prototype router
```

## Case 4 — C strong，A/B weak

R2：

```text
Semantic Ownership
→ Semantic Refinement
→ simple factor-preserving graph propagation
```

更接近 DecAlign-inspired semantic refinement。

## Case 5 — B strong + C strong，A moderate

优先：

```text
Semantic Refinement
+
Rich Structural Context Basis
```

Cross-factor transfer 放到后续 conditional experiment。

## Case 6 — A/B/C 全 weak

不要继续局部修改 Bi-Axis。

进入更高层：

```text
更强 semantic backbone
或 topology-conditioned representation learning
或成熟 local-global graph encoder
```

---

# 8. 执行顺序

严格分阶段：

```text
Prompt 1
R2-0 Repository Audit + Common Utilities

↓ 人工审查

Prompt 2
R2-0A Cross-Factor Transfer

↓ 回传结果

Prompt 3
R2-0B Structural Function Basis

↓ 回传结果

Prompt 4
R2-0C Semantic Interaction

↓ 回传结果

Prompt 5
R2-0 Final Synthesis
```

不要一次性全部实现和运行。

---

# 9. Prompt 1 — Repository Audit + Common Utilities
## 当前立即执行

```text
我们进入 Bi-Axis Performance-R2-0。

R2-0 不是训练新模型，而是使用 frozen A0/OFR checkpoints 做 architecture falsification / headroom probes，为正式 R2 架构重构决定方向。

当前科学假设：

1. current graph path 只做 C→C / Pt→Pt / Pv→Pv，可能遗漏 cross-factor semantic transfer；

2. current M2 只使用 [logd, P logd, P2 logd] + K=4 latent prototypes，R0 已证明 relation non-collapse，但 relation-context task differentiation 很弱；

3. current P0 完成 semantic ownership decomposition，但没有显式 semantic interaction/refinement。

本 Prompt 只做：
Repository Audit + Common Utilities。

不要执行 R2-0A/B/C 正式实验。
不要训练任何新模型。
不要读取 Test。

请仔细审查：

- src/analysis/perf_r0_utils.py
- scripts/perf_r0_factor.py
- scripts/perf_r0_relation_context.py
- scripts/perf_r0_hop_probe.py
- src/models/biaxis_components.py
- src/models/biaxis_p0.py
- src/models/biaxis_p1_components.py
- src/models/biaxis_p1.py
- src/models/biaxis_p2.py
- src/models/biaxis_p3.py
- configs/model/biaxis_final.yaml
- outputs/p3/operator checkpoint convention

输出：

docs/Performance_R2_0_Audit.md

必须明确回答：

1. 是否可以直接复用 R0 的 15 个 OFR checkpoints？

2. 当前 main 下 old checkpoint + eval 是否与 R0 数学一致？

3. extract_forward 是否已经提供：
   h_t,h_v,C,Pt,Pv,z_local,z_final,
   f_block,g_perm,edge_index,deg？

4. g_perm 的 factor order 和 relation order 是什么？

5. neighbor_mean message direction 是否为 src→dst？
   是否与 model aggregation 一致？

6. isolated node 当前怎样处理？

7. Grocery / ele-fashion safe chunking 应怎样设置？

8. 新 R2-0 scripts 如何显式禁止 test_idx / Test？

9. Fixed Ridge 必须完全复用 R0：
   StandardScaler
   RidgeClassifier(alpha=1.0)
   TRAIN fit
   VAL eval

10. 禁止修改任何 frozen model files。

新建：

src/analysis/perf_r20_utils.py
tests/test_perf_r20_utils.py

perf_r20_utils 只实现：

1. frozen setup wrapper，尽量直接复用 perf_r0_utils；

2. factor aliases；

3. weighted_neighbor_mean(
      edge_index,
      weights[E],
      features[N,d]
   )

   数学：

   g_i =
   sum_j w_ji F_j /
   (sum_j w_ji + eps)

   要求：
   - src→dst
   - edge chunk safe
   - isolated -> zero
   - 禁止 materialize [N,N]

4. topology-only Splus helper：

u0 = log(1+d)
u1 = P u0
u2 = P u1
u3 = P u2

mu_d = P d
std_d = sqrt(P(d^2) - mu_d^2)

gap_ji = abs(u0_j-u0_i)

mu_gap = incoming mean gap
std_gap = incoming gap std

Splus =
[u0,u1,u2,u3,mu_d,std_d,mu_gap,std_gap]

然后：
whole-graph column zscore
row L2 normalize

严格 topology-only。

5. context concat helper；

6. CSV helper；

7. explicit no-test guard。

Unit tests：

- changing node features / labels 不改变 Splus；
- edge weights 全 1 时 weighted_neighbor_mean == neighbor_mean；
- chunk/full equivalence；
- isolated node finite zero；
- wrapper 不读取 / 不暴露 test split；
- Ridge protocol 不重写，直接复用 R0 ridge_probe。

最后只做 reproduction smoke：

Movies seed42：

Probe([C|Pt|Pv])
Probe(z_local)
Probe(z_final)

与现有 R0 结果对比。

不要执行正式 R2-0A/B/C。
不要训练模型。
不要读取 Test。

输出 Audit 后停止，等待人工审查。
```

---

# 10. Prompt 2 — R2-0A Cross-Factor Transfer
## Prompt 1 审查 PASS 后执行

```text
R2-0 Audit 已人工通过。

现在只执行：
R2-0A Cross-Factor Transfer Probe。

不要执行 R2-0B/C。
不要训练主模型。
不要读取 Test。

Datasets：
Movies
Toys
Grocery

Seeds：
42
43
44

使用 frozen A0/OFR checkpoints。

对每 checkpoint：
extract C,Pt,Pv,g_perm,edge_index。

A1 Plain 1-hop matrix：

N^a = neighbor_mean(F^a)

for target b in [C,Pt,Pv]:
    local = Probe(F^b)

    for source a in [C,Pt,Pv]:
        Probe([F^b | N^a])

输出：

U_plain(a→b)
=
Probe([F^b|N^a])-Probe(F^b)

Adv_plain(a→b)
=
Probe([F^b|N^a])
-
Probe([F^b|N^b])

所有 cells 必须严格 2*d_f。

A2 Current-relation matrix：

G^a =
flatten(g_perm[:,a,:,:])
shape [N,K*d]

for target b:
    for source a:
        Probe([F^b | G^a])

输出 U_rel / Adv_rel。

所有 cells 必须严格 (1+K)*d_f。

A3 All-source upper bound：

for target b:

Probe(
  [F^b | N^C | N^Pt | N^Pv]
)

Probe(
  [F^b | G^C | G^Pt | G^Pv]
)

计算相对 same-factor delta。

注意：
ALL-source 维度更高，只作 upper bound，
不能单独作为 GO 依据。

必须：

- Ridge alpha=1.0；
- TRAIN fit / VAL eval；
- Acc + Macro-F1；
- per seed；
- 3-seed mean；
- population std ddof=0；
- positive seed count；
- 报告两个 3×3 matrix：source rows × target columns；
- 不以单一最佳 cell 作总体结论。

输出：

outputs/perf_r20/cross_factor/
  cross_factor_plain_cells.csv
  cross_factor_relation_cells.csv
  cross_factor_all_source.csv
  R20_A_CROSS_FACTOR_REPORT.md

预注册判定：

STRONG 需要同时满足：

1. M/T/G overall all-source headroom >= +0.30pp；

2. 至少两个稳定 dimension-matched off-diagonal：
   Adv >= +0.20pp；

3. 对应 cell 至少 2/3 seeds positive；

4. 不是单一 dataset / cell 驱动。

MODERATE：
overall +0.15~+0.30pp 且存在稳定 off-diagonal。

NO EVIDENCE：
overall < +0.15pp 且 off-diagonal 不稳定。

报告完成后停止。
等待 ChatGPT 审查。
```

---

# 11. Prompt 3 — R2-0B Structural Function Basis
## R2-0A 审查完成后执行

```text
现在只执行：
R2-0B Explicit Structural Function Basis Probe。

不要训练新模型。
不要执行 R2-0C。
禁止 Test。

Datasets：Movies / Toys / Grocery
Seeds：42 / 43 / 44

使用 frozen A0/OFR checkpoints。

使用 perf_r20_utils.Splus。

Splus 固定：

u0=log(1+d)
u1=P u0
u2=P u1
u3=P u2

mu_d=P d
std_d=sqrt(P(d^2)-mu_d^2)

gap_ji=abs(u0_j-u0_i)

mu_gap=incoming mean(gap)
std_gap=incoming std(gap)

Splus=[u0,u1,u2,u3,mu_d,std_d,mu_gap,std_gap]

whole-graph column z-score
row L2 normalize。

对 observed edge j→i：

c_ji = cos(Splus_j,Splus_i)

w_sim = (1+c_ji)/2 + 1e-8
w_diff = (1-c_ji)/2 + 1e-8

对每 factor f：

G1 = P F
G2 = P G1
Gsim = weighted_neighbor_mean(w_sim,F)
Gdiff = weighted_neighbor_mean(w_diff,F)

禁止 materialize explicit 2-hop edges。

Per-factor current：

X_current^f =
[F | g_R1 | g_R2 | g_R3 | g_R4]

Per-factor basis：

X_basis^f =
[F | G1 | G2 | Gsim | Gdiff]

两者严格 5*d_f。

计算：

Delta_B^f =
Probe(X_basis^f)-Probe(X_current^f)

同时计算：

Probe([F|G1])

用于判断 explicit basis 相比普通 1-hop 的额外 utility。

再做 joint：

X_current_joint =
[C|Pt|Pv |
 current 4 contexts of C |
 current 4 contexts of Pt |
 current 4 contexts of Pv]

X_basis_joint =
[C|Pt|Pv |
 explicit 4 contexts of C |
 explicit 4 contexts of Pt |
 explicit 4 contexts of Pv]

两者严格 15*d_f。

Primary：

Delta_B_joint =
Probe(X_basis_joint)-Probe(X_current_joint)

Secondary diagnostics：

D_ctx
4×4 cosine redundancy matrix

重要：

D_ctx 变大不能作为 GO 依据。
GO/NO-GO 只看 fixed probe utility。

输出：

outputs/perf_r20/structural_basis/
  structural_signature_stats.csv
  structural_context_probe_per_factor.csv
  structural_context_probe_joint.csv
  structural_context_diversity.csv
  structural_context_redundancy.csv
  R20_B_STRUCTURAL_BASIS_REPORT.md

判定：

STRONG：
M/T/G mean Delta_B_joint >= +0.50pp
且至少 2/3 datasets positive。

GO：
>= +0.30pp
且至少 2/3 positive。

WEAK：
+0.15~+0.30pp。

NO-GO：
< +0.15pp
或只有单一 dataset 提升。

完成后停止。
等待 ChatGPT 审查。
```

---

# 12. Prompt 4 — R2-0C Semantic Interaction
## R2-0B 审查后执行

```text
现在只执行：
R2-0C Semantic Interaction Headroom Probe。

不要训练新 semantic module。
不要执行 R2 正式模型。
禁止 Test。

Datasets：Movies / Toys / Grocery
Seeds：42 / 43 / 44

Frozen checkpoints only。

C1 Factor interaction：

base =
[C|Pt|Pv]

inter =
[C|Pt|Pv|
 C*Pt|
 C*Pv|
 Pt*Pv|
 abs(C-Pt)|
 abs(C-Pv)|
 abs(Pt-Pv)]

计算：

Probe(base)
Probe(inter)
Probe(z_local)

Primary：

Delta_inter_vs_local =
Probe(inter)-Probe(z_local)

Secondary：

Probe(inter)-Probe(base)

C2 Modal interaction：

base_modal = [h_t|h_v]

inter_modal =
[h_t|h_v|h_t*h_v|abs(h_t-h_v)]

计算：

Probe(inter_modal)-Probe(base_modal)

C3 Shuffled negative control：

保持 [C|Pt|Pv] 不动。

interaction-only columns 沿 node dimension
使用固定 permutation：

seed = 20260904

只使用一个 permutation。

得到：

Probe(base | shuffled_interaction)

并输出：

real_interaction - shuffled_interaction

所有 features 继续使用 R0 ridge_probe 内部：
StandardScaler
RidgeClassifier(alpha=1.0)
TRAIN fit
VAL eval。

输出：

outputs/perf_r20/semantic_interaction/
  semantic_factor_interaction_probe.csv
  semantic_modal_interaction_probe.csv
  semantic_interaction_shuffle_control.csv
  R20_C_SEMANTIC_INTERACTION_REPORT.md

判定：

STRONG：
M/T/G mean(inter - z_local) >= +0.50pp
至少 2/3 datasets positive
且 real > shuffled。

GO：
>= +0.30pp
至少 2/3 positive
real > shuffled。

PARTIAL：
inter > factor concat
但 inter ≈ z_local。

这种情况不支持优先重构 P0。

NO-GO：
inter-z_local < +0.15pp
或 real≈shuffled。

完成后停止。
等待 ChatGPT 审查。
```

---

# 13. Prompt 5 — R2-0 Final Synthesis
## A/B/C 全部完成后执行

```text
读取 R2-0A/B/C 已生成 CSV。
不要运行新实验。
不要读取 Test。

输出：

outputs/perf_r20/summary/
  R20_MASTER_TABLE.csv
  R20_FINAL_DIAGNOSIS.md

Master table dataset-level至少包含：

A:
plain all-source headroom
relation all-source headroom
stable off-diagonal count
strongest stable cross-factor cells

B:
Delta_B_joint
Delta_B_C
Delta_B_Pt
Delta_B_Pv
current D_ctx
basis D_ctx

C:
factor_inter - factor_base
factor_inter - z_local
modal_inter - modal_base
real_inter - shuffled_inter

每项必须给：

3-seed mean
population std ddof=0
positive seed count

还要计算：
M/T/G macro mean。

最后严格按以下 Case 分类：

Case 1:
A strong + B strong

Case 2:
A strong + B weak

Case 3:
A weak + B strong

Case 4:
C strong + A/B weak

Case 5:
B strong + C strong，A moderate

Case 6:
A/B/C 全 weak

只给 architecture evidence verdict。
不要设计 R2 具体模型。
等待人工 / ChatGPT 裁决。
```

---

# 14. 当前立即执行

现在只执行：

\[
\boxed{\textbf{Prompt 1 — Repository Audit + Common Utilities}}
\]

不要一次性让 AI 完成 A/B/C。

Prompt 1 返回后重点审查：

1. frozen checkpoint protocol 是否正确；
2. reproduction 是否复现 R0；
3. `g_perm` factor indexing 是否正确；
4. message direction 是否为 `src→dst`；
5. `Splus` 是否严格 topology-only；
6. `weighted_neighbor_mean` full/chunk 是否数学等价；
7. isolated node 是否 finite zero；
8. 是否存在任何 Test leakage。

Audit PASS 后再进入 R2-0A。

---

# 15. R2-0 最终目标

R2-0 的目标不是找一个“Val 更高的 feature concat”，而是确定：

\[
\boxed{
\textbf{R2 应该改变 WHAT evidence，
而不是继续优化 HOW to route the same evidence.}
}
\]

最终要在以下方向中给出有证据的优先级：

\[
\boxed{\text{Cross-Factor Functional Transfer}}
\]

\[
\boxed{\text{Richer Structural Function Basis}}
\]

\[
\boxed{\text{Semantic Interaction / Refinement}}
\]

只有 frozen probes 证明存在稳定 headroom 的方向，才允许进入正式 R2 架构开发。
