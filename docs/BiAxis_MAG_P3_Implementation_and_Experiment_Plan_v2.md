# Bi-Axis MAG — P3 阶段实现与实验推进文档
## 阶段目标：验证 Factor–Relation-specific Transformation，并收敛为参数高效 Operator

> 代码仓库：`CrisRipper777/0901`  
> 前置阶段：P0 / P1 / P2 已正式冻结  
> P3 任务范围：**仅 Node Classification**  
> 数据集：Movies / Toys / Grocery / ele-fashion / Reddit-S  
> Seeds：**42 / 43 / 44**  
> 主协议：继续沿用已冻结 NC full-graph protocol  
> P2 默认 coupler：**Unified Null-Augmented Plan + NullSoftmax**  
> P2 可选 coupler：Composition-UOT，仅作为补充，不参与 P3 主结构搜索  
> **P3 默认使用快速非确定性训练路径；不要启用 `p2.deterministic=true`。**  
> P3 暂不加入：Pseudo Nodes / Diffusion / MoE / Rewiring / 新 Relation Encoder / 新 Semantic Loss / 新 Coupler

---

# 0. 本版相对上一版的关键调整

当前仓库已经包含一个 opt-in deterministic verification mode：

```text
p2.deterministic=true
```

它的目的只是做严格复现验证，代价很高，训练速度可慢约 5–10×。

**P3 不使用它。**

P3 的统计策略改为：

\[
\boxed{
\textbf{Fast training + 3 seeds + mean ± population std}
}
\]

并采用 paired-seed comparison：

\[
\Delta_s
=
Metric(\text{variant},s)
-
Metric(\text{reference},s),
\qquad
s\in\{42,43,44\}.
\]

因此：

- 单 seed 只用于 smoke/debug；
- 任何科学 GO/NO-GO 不依赖单 seed；
- 正式结构比较必须基于 3 seeds；
- 报告 mean ± population std；
- 同时报告 paired delta 的均值、方差与 3 个 seed 的符号一致性；
- 对 `<0.2 pp` 的单次差异不作强解释；
- 对关键但边界的效应，优先增加 seeds，而不是切换到慢速 deterministic mode。

---

# 1. P3 的唯一科学问题

P0–P2 已经回答：

```text
P0:
不同 semantic factors 对 graph neighborhood 的 utility 不同。

P1:
Semantic Ownership 与 Structural Relation 是两个不同轴；
不同 factor 的 graph demand 稳定不同，
relation selectivity 具有条件性。

P2:
how much graph + which relation 可以统一成
Null-Augmented Factor–Relation Plan Γ。
```

当前 P2 的消息为：

\[
g_i^f
=
\sum_{k=1}^{K}
\Gamma_{ifk} g_{ik}^{f},
\]

然后所有 factor / relation 统一经过：

\[
m_i^f=W_0g_i^f.
\]

由于 \(W_0\) 是线性的，因此等价于：

\[
m_i^f
=
\sum_k
\Gamma_{ifk}
W_0g_{ik}^{f}.
\]

这意味着当前模型虽然已经区分：

```text
semantic factor identity
structural relation identity
```

但在真正解释 relation message 时仍然假设：

\[
T_{C,R_k}
=
T_{P_t,R_k}
=
T_{P_v,R_k}
=
W_0,
\]

并且：

\[
T_{f,R_1}
=
T_{f,R_2}
=
\cdots
=
T_{f,R_K}.
\]

P3 因此只回答：

\[
\boxed{
\textbf{
Should a structural relation transform different semantic factors differently?
}
}
\]

最终主公式：

\[
\boxed{
m_i^f
=
\sum_{k=1}^{K}
\Gamma_{ifk}
T_{f,k}
\left(
g_{i,k}^{f}
\right)
}
\]

其中：

- \(\Gamma_{ifk}\)：**how much**
- \(T_{f,k}\)：**how**

---

# 2. P3 的科学边界：只研究 Transformation

P3 必须冻结以下部分。

## 2.1 M1 Semantic Factorization

继续使用：

```text
C / Pt / Pv
hidden_dim = 256
factor_dim = 128
lambda_common = 0.02
lambda_orth = 0.01
lambda_recon = 0.3
```

architecture/objective unchanged，继续 joint optimization。

---

## 2.2 M2 Structural Relation Decomposition

继续使用 P1：

```text
K = 4
relation_dim = 32
relation_temperature = 0.5
TopologyDiffusionSignature
EdgeStructuralToken
RelationPrototypes
relation_weighted_mean
```

Relation 继续严格 topology-only。

---

## 2.3 M3 Unified Coupler

P3 主实验固定：

\[
\boxed{
\text{NullSoftmax}
}
\]

显式设置：

```yaml
p2:
  mode: null_softmax
  epsilon: 0.2
```

不再搜索：

```text
epsilon
tau
null_prior
transport solver
```

Composition-UOT 只允许在 P3 最终 operator 冻结后做少量 compatibility check。

---

## 2.4 禁止打开 deterministic mode

P3 config 必须显式：

```yaml
p2:
  deterministic: false
```

AI/Codex 不得：

```text
设置 p2.deterministic=true
打开 torch.use_deterministic_algorithms 作为默认训练路径
设置 CUBLAS_WORKSPACE_CONFIG 进行日常 P3 训练
把 Hillis-Steele deterministic aggregation 用于正式 P3 batch runs
```

仓库中的 deterministic mode 保留，不删除，只作为未来严格复现/排查工具。

---

## 2.5 其它冻结

禁止修改：

```text
src/tasks/nc.py
configs/task/nc.yaml
train/val/test split
dataset configs
P0/P1/P2 frozen model files
fusion head
semantic auxiliary losses
relation K
factor_dim
checkpoint criterion
seed 列表
```

---

# 3. P3 的可复现性与统计判读规则

由于默认快速 GPU 聚合存在轻微 numerical nondeterminism，因此 P3 不追求 bitwise identical。

正式统计原则：

## 3.1 每个正式 variant 都跑 3 seeds

```text
42
43
44
```

报告：

\[
mean \pm std_{population}.
\]

---

## 3.2 必须做 paired-seed delta

例如比较 OFR 和 OADD：

\[
\Delta_s^{INT}
=
ValAcc_s(OFR)
-
ValAcc_s(OADD).
\]

汇总：

```text
mean Δ
population std of Δ
positive seed count: 0/3, 1/3, 2/3, 3/3
```

paired delta 比只比较两组 mean 更适合当前实验。

---

## 3.3 单 seed 小差异不作为证据

已有复现性分析显示：

```text
Val 单-run <0.2 pp
```

可能与数值扰动同量级。

所以：

\[
\boxed{
|\Delta|<0.2\text{ pp 的单 seed 差异不独立解释}
}
\]

正式科学判断看：

```text
3-seed mean
paired signs
std
Acc + Macro-F1
```

---

## 3.4 边界结果怎么办？

如果某个最终关键比较出现：

```text
mean gain 0.15~0.30 pp
2/3 seeds 同号
std 较大
```

不要启用 deterministic mode。

优先补：

```text
seed45
seed46
```

扩展到 5 seeds。

只有最终论文复现实验或 debugging 才考虑 deterministic verification mode。

---

# 4. 为什么不能直接一上来做 Low-rank Operator？

P3 首先要回答：

```text
Factor identity 是否需要不同 transformation？
Relation identity 是否需要不同 transformation？
两者的 main effects 是否已经足够？
pair-specific transformation 是否还有额外价值？
```

如果直接做 Low-rank：

\[
T_{f,k}
=
W_0+
U\operatorname{Diag}(\cdots)V^\top,
\]

即使性能变好，也无法判断来自：

```text
factor-specific transformation
relation-specific transformation
factor+relation additive effect
真正 factor×relation interaction
参数量变化
```

因此 P3 分为：

```text
P3-A  Full Operator Necessity / Axis Attribution
P3-B  Parameter-Matched Low-rank Interaction
P3-C  Optional FiLM / Rank Study
P3-D  Final Selection
```

---

# 5. P3-A：Full Residual Operator Decomposition

使用：

\[
\boxed{
T_{f,k}
=
W_0
+
A_f
+
B_k
+
C_{f,k}
}
\]

其中：

- \(W_0\)：P2 shared graph operator
- \(A_f\)：semantic-factor main effect
- \(B_k\)：structural-relation main effect
- \(C_{f,k}\)：pair-specific residual

第一轮使用 full \(d\times d\) matrices。

注意：

> P3-A 是 mechanism upper-bound probe，不是最终参数高效模型。

而且 \(C_{fk}\) 与 \(A_f/B_k\) 存在表达冗余，因此：

\[
OFR>OADD
\]

只能说明“增加 pair-specific operator capacity 有价值”，**不能单独作为严格 interaction 因果证明**。

真正参数匹配的 interaction 证据留到 P3-B：

\[
LR\text{-}INT
\quad vs\quad
LR\text{-}ADD.
\]

---

# 6. P3-A 五个 Operator Variants

所有 variants 使用完全相同：

```text
M1 semantic factorization
M2 relation decomposition
NullSoftmax Γ
fusion
training protocol
seeds
```

只切换 transformation。

---

## O0 — Shared

\[
\boxed{
T_{f,k}=W_0
}
\]

即 P2 shared operator control。

---

## OF — Factor-specific

\[
\boxed{
T_{f,k}=W_0+A_f
}
\]

回答：

> semantic ownership 是否需要不同 message transformation？

---

## OR — Relation-specific

\[
\boxed{
T_{f,k}=W_0+B_k
}
\]

回答：

> structural relation identity 是否需要不同 transformation？

---

## OADD — Factor + Relation Additive

\[
\boxed{
T_{f,k}=W_0+A_f+B_k
}
\]

允许两个 main effects，但没有额外 pair residual。

---

## OFR — Full Pair-specific Upper Bound

\[
\boxed{
T_{f,k}=W_0+A_f+B_k+C_{f,k}
}
\]

回答：

> 在两个 main effects 之外，额外 cell-specific transformation capacity 是否有价值？

P3-A 中最重要的比较：

\[
\boxed{
OFR\quad vs\quad OADD
}
\]

但由于参数量和可辨识性问题，这只是 upper-bound evidence。

---

# 7. 初始化纪律

所有 residual：

\[
A_f,\ B_k,\ C_{fk}
\]

必须 zero initialization。

step 0：

\[
T_{fk}=W_0.
\]

这样所有 variants 都从 P2 Shared Operator 相同函数开始。

禁止：

```text
随机大尺度 residual 初始化
variant-specific pretrained operator
不同 variant 使用不同 residual initialization scale
```

第一轮：

```text
operator_reg_weight = 0
interaction_reg_weight = 0
```

只保留全局 AdamW weight decay。

---

# 8. 参数量与 confounding

设：

\[
F=3,\quad K=4,\quad d=128.
\]

单矩阵：

\[
d^2=16,384.
\]

额外 residual 参数约：

```text
OF    : 3*d^2  ≈ 49K
OR    : 4*d^2  ≈ 66K
OADD  : 7*d^2  ≈ 115K
OFR   : 19*d^2 ≈ 311K
```

因此：

\[
\boxed{
OFR\text{ 的优势不能直接解释成 interaction}
}
\]

因为它同时拥有更多 parameters。

P3-B 必须用 parameter-matched LR-ADD vs LR-INT 排除这个 confound。

---

# 9. 高效计算：禁止 node-wise operator tensor

严禁：

```text
[N,F,K,d,d]
```

也不要长期 materialize额外：

```text
[N,F,K,d]
```

transformed copy。

当前 P2 已有：

```text
g_perm      [N,F,K,d]
gamma_graph [N,F,K]
```

推荐分项累加。

---

## 9.1 Shared term

\[
m_i^{(0)}
=
W_0
\left(
\sum_k
\Gamma_{ifk}g_{ik}^{f}
\right).
\]

---

## 9.2 Factor term

因为 \(A_f\) 与 \(k\) 无关：

\[
m_i^F
=
A_f
\left(
\sum_k
\Gamma_{ifk}g_{ik}^{f}
\right).
\]

---

## 9.3 Relation term

\[
m_i^R
=
\sum_k
\Gamma_{ifk}
B_k g_{ik}^{f}.
\]

循环 4 个 relations。

---

## 9.4 Pair term

\[
m_i^{FR}
=
\sum_k
\Gamma_{ifk}
C_{fk}g_{ik}^{f}.
\]

最多 12 个 cell matmul。

每次只产生：

```text
[N,d]
```

临时 tensor 并立即累加。

---

# 10. P3-A 正式实验：75 runs

P3-A 是 P3 的 life-or-death scientific attribution，因此不再用单 seed Screen 做结论。

数据：

```text
Movies
Toys
Grocery
ele-fashion
Reddit-S
```

Variants：

```text
O0
OF
OR
OADD
OFR
```

Seeds：

```text
42
43
44
```

总计：

\[
\boxed{
5\times5\times3=75\text{ runs}
}
\]

其中 O0 必须在 P3 实现中重新跑，不直接复用历史 P2 数值。

原因：

- P3 operator 代码路径可能改变 floating-point execution order；
- 当前又不使用 deterministic mode；
- 0.1 pp 级比较必须使用 P3 内部同一实现的 control。

---

# 11. P3-A 主结果表

必须输出：

| Dataset | O0 | OF | OR | OADD | OFR |
|---|---:|---:|---:|---:|---:|
| Val Acc | mean±std | | | | |
| Val F1 | mean±std | | | | |
| Test Acc | mean±std | | | | |
| Test F1 | mean±std | | | | |

再输出 paired deltas：

```text
OF-O0
OR-O0
OADD-O0
OFR-OADD
OFR-O0
```

每个都记录：

```text
mean delta
std delta
positive seeds / 3
```

---

# 12. P3-A GO / REVISE / NO-GO

不使用单 seed 阈值。

## Strong Pair-specific Signal

若：

\[
\Delta_{INT}
=
OFR-OADD
\]

满足大部分：

```text
至少 2/5 datasets mean Val gain >= +0.30 pp
或多个 datasets mean gain >= +0.20 pp 且 3/3 seeds 同号
没有系统性 Macro-F1 下降
```

则进入 P3-B interaction branch。

---

## Main-effect GO

若：

```text
OADD > O0
```

在多数数据集/seed 稳定，

但：

```text
OFR ≈ OADD
```

则说明 factor/relation transformation main effects 有用，但 pair-specific capacity 不必要。

P3-B 重点做 low-rank additive。

---

## Single-axis GO

若：

```text
OF > O0
```

明显而 OR/OADD/OFR 没有进一步价值：

→ Factor-specific Operator。

若只有：

```text
OR > O0
```

明显：

→ Relation-specific Operator。

---

## Borderline

若关键比较：

```text
mean +0.15~0.30 pp
2/3 seeds 同号
```

且可能决定论文主结构：

优先：

```text
补 seeds45/46
```

而不是打开 deterministic mode。

---

## NO-GO

若：

```text
OF / OR / OADD / OFR
```

均无法在 multi-seed 上稳定超过 O0：

\[
\boxed{
P3 Operator Specialization NO-GO
}
\]

最终保留 shared \(W_0\)。

---

# 13. P3-A Operator Diagnostics

best checkpoint 后记录：

## 13.1 Residual Frobenius Norm

\[
r_F(f)
=
\frac{\|A_f\|_F}
{\|W_0\|_F+\epsilon}
\]

\[
r_R(k)
=
\frac{\|B_k\|_F}
{\|W_0\|_F+\epsilon}
\]

\[
r_{FR}(f,k)
=
\frac{\|C_{fk}\|_F}
{\|W_0\|_F+\epsilon}.
\]

---

## 13.2 Usage-weighted Pair Strength

P2 plan：

\[
u_{fk}
=
\frac1N\sum_i\Gamma_{ifk}.
\]

定义：

\[
S_{PAIR}
=
\sum_{f,k}
u_{fk}
\frac{\|C_{fk}\|_F}
{\|W_0\|_F+\epsilon}.
\]

避免过度解释几乎没被使用的 cell。

---

## 13.3 Operator Distance

记录：

```text
same relation / different factors
same factor / different relations
```

的 normalized Frobenius distance 与 flattened cosine。

---

## 13.4 Message-level Effect

\[
\delta_{ifk}
=
\frac{
\|T_{fk}(g_{ik}^f)-W_0g_{ik}^f\|_2
}{
\|W_0g_{ik}^f\|_2+\epsilon
}.
\]

报告 usage-weighted mean。

---

# 14. P3-B：Parameter-Matched Low-rank Operator

只有 P3-A 表明 transformation specialization 有价值后进入。

定义：

\[
T_{fk}(x)
=
W_0x
+
U
\left[
c_{fk}
\odot
(V^\top x)
\right],
\]

其中：

\[
U,V\in\mathbb R^{d\times r},
\quad
a_f,b_k\in\mathbb R^r.
\]

默认：

\[
r=16.
\]

---

# 15. Low-rank Additive

\[
\boxed{
c_{fk}^{ADD}=a_f+b_k
}
\]

因此：

\[
T_{fk}^{ADD}(x)
=
W_0x+
U[
(a_f+b_k)
\odot(V^\top x)
].
\]

---

# 16. Low-rank Interaction

\[
\boxed{
c_{fk}^{INT}
=
a_f+b_k+a_f\odot b_k
}
\]

因此：

\[
\boxed{
T_{fk}^{INT}(x)
=
W_0x+
U[
(a_f+b_k+a_f\odot b_k)
\odot(V^\top x)
]
}
\]

其中：

\[
a_f\odot b_k
\]

是显式 factor×relation interaction term。

---

# 17. 为什么 LR-ADD vs LR-INT 是 P3 最重要的 interaction test

LR-ADD 与 LR-INT 使用完全相同参数：

```text
W0
U
V
a_f
b_k
```

trainable parameter count 完全相同。

唯一区别：

```text
是否使用 a_f * b_k
```

因此：

\[
\boxed{
LR\text{-}INT
>
LR\text{-}ADD
}
\]

在 multi-seed 下稳定成立，才是 P3 最干净的 interaction evidence。

---

# 18. Low-rank 初始化

推荐：

```text
U,V: Xavier
a,b: zeros
```

step0：

\[
T_{fk}=W_0.
\]

第一步：

- \(a,b\) 可收到 gradient；
- U/V 初始 residual coefficient 为 0，早期 gradient 可能较小，这是预期现象。

如果真实测试显示长期无梯度，再单独调整 tiny init；不要提前改。

---

# 19. Low-rank 参数量

rank=16：

\[
2dr+(F+K)r
=
2\times128\times16+7\times16
=
4208.
\]

加 shared \(W_0\) 总 operator params 约：

```text
20.6K
```

显著低于 full OFR。

---

# 20. P3-B 正式实验：30 个新增 runs

如果 interaction branch GO：

比较：

```text
LR-ADD
LR-INT
```

5 datasets × 3 seeds：

\[
5\times2\times3=30
\]

新增 runs。

O0 / OFR 使用 P3-A 已有 3-seed 结果。

最终表：

| Dataset | O0 | OADD | OFR | LR-ADD | LR-INT |
|---|---:|---:|---:|---:|---:|

重点：

\[
LR\text{-}INT-LR\text{-}ADD
\]

和：

\[
LR\text{-}INT-OFR.
\]

---

# 21. P3-B GO

## Interaction GO

如果：

```text
LR-INT > LR-ADD
```

在至少 3/5 datasets mean 为正，

并且：

```text
至少 2 datasets >= +0.20~0.30 pp
或 3/3 seeds 同号
```

同时没有系统性 F1 下降：

\[
\boxed{
Factor×Relation interaction GO
}
\]

---

## Additive GO

若：

```text
LR-ADD ≈ LR-INT
```

但两者明显优于 O0：

保留 LR-ADD，删除 interaction。

---

## Full-only GO

若：

```text
OFR 明显优于 LR variants
```

需判断：

- low-rank rank 是否过低；
- full operator gain 是否主要来自容量。

只做一次 rank sensitivity 后决定，不无限调参。

---

# 22. Optional FiLM baseline

只有 Low-rank specialization 已明确有价值时才加。

\[
T_{fk}^{FiLM}(x)
=
(1+\gamma_{fk})
\odot
W_0x
\]

\[
\gamma_{fk}\in\mathbb R^d,
\quad
\gamma_{fk}=0
\]

不加 bias。

参数：

\[
FKd=12\times128=1536.
\]

如果：

```text
FiLM ≈ LR-INT
```

说明 channel-wise modulation 已足够。

如果：

```text
LR-INT > FiLM
```

说明需要低秩 cross-channel mixing。

正式报告 FiLM 时同样必须 3 seeds：

\[
5\times3=15
\]

runs。

---

# 23. Rank sensitivity

只有 LR-INT 是主候选后才做。

测试：

\[
r\in\{8,16,32\}.
\]

先选 3 个代表数据集：

```text
strong positive
neutral
hard/low-S_R
```

若只是内部 sanity：

```text
seed42
```

可以用于快速排查，但不能写成正式结论。

如果要进入论文 sensitivity：

```text
3 datasets × 3 ranks × 3 seeds
```

并报告 mean±std。

默认 rank16 不做 dataset-specific tuning。

---

# 24. Composition-UOT 与 P3

P3 主结构搜索**固定 NullSoftmax**。

final operator 冻结后，可以少量 compatibility check：

```text
final operator + NullSoftmax
final operator + Composition-UOT
```

优先：

```text
ele-fashion
```

以及 P3 最强正例。

这个实验不是用来重新选择 P3 operator。

---

# 25. Macro-F1 与 Accuracy

所有 P3 正式结果必须同时保存：

```text
Best Val Accuracy
Val Macro-F1 at best-val-Acc epoch
Test Accuracy
Test Macro-F1
```

结构去留仍由：

\[
\boxed{
Validation Accuracy
}
\]

决定。

Macro-F1 用于：

- 检查 minority-class representation；
- 防止小 Acc gain 伴随明显 F1 regression；
- Movies / ele-fashion 特别重要。

---

# 26. 推荐代码结构

新增：

```text
src/models/biaxis_p3_components.py
src/models/biaxis_p3.py
configs/model/biaxis_p3.yaml

tests/test_biaxis_p3.py

scripts/analyze_p3_checkpoint.py
scripts/run_p3_operator_screen.py
scripts/run_p3_lowrank_screen.py
scripts/run_p3_optional.py
scripts/summarize_p3.py
```

原则：

```text
不修改 biaxis_p0.py
不修改 biaxis_p1.py
不修改 biaxis_p2.py
```

---

# 27. P3 Model 继承 P2

推荐：

```python
from .biaxis_p2 import Model as P2Model

class Model(P2Model):
    ...
```

P3 必须显式强制：

```text
self.p2_mode == "null_softmax"
```

或者在 config validation 中直接拒绝其它 mode 作为主 P3 实验。

---

# 28. P3 `_graph_update()` 实现关键

P2 当前：

```text
relation decomposition
g_perm
score
Gamma
g_mix = sum_k Gamma*g
m = W0(g_mix)
```

P3 不能等 relation sum 后再加 pair operator。

P3 应：

```text
relation decomposition
g_perm
score
NullSoftmax Gamma
gamma_graph = Gamma[...,1:]
operator(g_perm, gamma_graph, W0)
```

得到：

\[
m_i^f
=
\sum_k
\Gamma_{ifk}
T_{fk}(g_{ik}^f).
\]

Local \(\Gamma_{if0}\) 仍只通过 residual local factor：

\[
f_i
\]

保留。

---

# 29. Unit Tests

新增：

```text
tests/test_biaxis_p3.py
```

至少包括：

## 29.1 Zero-residual equivalence

所有 residual 为 0：

```text
O0 / OF / OR / OADD / OFR
```

输出都应等价 P2 shared W0 path。

---

## 29.2 Shared linear identity

验证：

\[
W_0
\left(
\sum_k\Gamma g_k
\right)
=
\sum_k\Gamma W_0g_k.
\]

---

## 29.3 Mode isolation

```text
OF  only A
OR  only B
OADD A+B
OFR A+B+C
```

---

## 29.4 Pair cell routing

构造：

```text
只有 Gamma[f=1,k=2] 非零
```

确认只有对应 cell residual 影响输出。

---

## 29.5 Gradient

真实 nonconstant loss 下：

```text
W0
A
B
C
```

对应 mode 获得 finite nonzero gradient。

关闭参数不参与计算。

---

## 29.6 Low-rank zero equivalence

`a=b=0`：

```text
LR-ADD == O0
LR-INT == O0
```

---

## 29.7 LR parameter matching

断言：

```text
trainable_params(LR-ADD)
==
trainable_params(LR-INT)
```

---

## 29.8 LR formula

手算：

```text
a+b
a+b+a*b
```

与实现一致。

---

## 29.9 FiLM zero equivalence

\[
\gamma=0\Rightarrow FiLM=Shared.
\]

---

## 29.10 Memory discipline

禁止：

```text
[N,F,K,d,d]
```

长期 transformed `[N,F,K,d]` 副本。

---

## 29.11 Fast path policy

测试配置必须确认：

```text
p2.deterministic == false
```

P3 batch scripts 不得自动打开 deterministic mode。

---

# 30. 默认 P3 Config

```yaml
name: biaxis_p3

hidden_dim: 256
factor_dim: 128
dropout: 0.2
activation: gelu
norm: layernorm

lambda_common: 0.02
lambda_orth: 0.01
lambda_recon: 0.3

lr: 0.001
weight_decay: 0.0001
full_graph_training: true

p1:
  factor_aware: true
  num_relations: 4
  relation_dim: 32
  relation_temperature: 0.5
  edge_chunk_size: 500000

p2:
  mode: null_softmax
  score_hidden_dim: 64
  epsilon: 0.2
  null_score_init: 0.0
  deterministic: false

p3:
  operator_mode: shared

  lowrank_rank: 16

  operator_reg_weight: 0.0
  interaction_reg_weight: 0.0
```

---

# 31. P3 推进漏斗

```text
P3-D0  Repository Audit
        ↓
P3-D1  Full Operator Components
        ↓
P3-D2  Model Integration
        ↓
P3-D3  Movies Smoke（seed42，仅 debug）
        ↓
P3-A   75-run Multi-seed Full Operator Study
        ↓
        根据 3-seed 结果分支
        ↓
P3-B   30-run Multi-seed Low-rank Study
        ↓
P3-C   Optional FiLM / Rank
        ↓
P3 FINAL
```

不再设置：

```text
deterministic reproducibility gate
bitwise repeat gate
deterministic 25-run screen
```

---

# 32. AI/Codex Prompt 1：P3 Repository Audit

```text
你现在协助我在 CrisRipper777/0901 中正式进入 Bi-Axis P3。

先不要修改代码。

P0/P1/P2 已冻结。
P3 只做 NC。

重要实验规则：
- 不使用 deterministic experiment mode。
- 当前仓库虽然有 p2.deterministic=true 的严格验证路径，但太慢，仅保留调试用途。
- P3 正式实验必须 p2.deterministic=false。
- 科学比较统一使用 seeds 42/43/44，汇报 mean±population std。
- 单 seed 仅用于 smoke/debug，不用于 GO/NO-GO。
- 关键比较使用 paired-seed delta。
- 单次 <0.2pp 差异不单独解释。

P2 最终冻结：
- Unified Null-Augmented Factor–Relation Plan
- P3 默认 coupler = NullSoftmax
- Composition-UOT 仅 optional
- 不再研究 routing/transport solver

P3 唯一科学问题：
Should the same structural relation transform Common/Text-private/Visual-private differently?

主公式：
m_i^f = sum_k Gamma_ifk T_fk(g_ik^f)

P3-A：
T_fk = W0 + A_f + B_k + C_fk

variants：
O0   = W0
OF   = W0 + A_f
OR   = W0 + B_k
OADD = W0 + A_f + B_k
OFR  = W0 + A_f + B_k + C_fk

请审查：
1. 当前 main branch 中 deterministic mode 加在什么位置？P3 如何确保它始终 false？
2. P3 如何最小侵入继承 P2？
3. P2 哪些逻辑可复用，哪些必须在 relation sum 前改写？
4. P3 config 如何显式固定 p2.mode=null_softmax？
5. 如何实现五 operator modes 而不构造 [N,F,K,d,d]？
6. zero residual 如何保证等价 P2？
7. 为什么 O0 必须作为 P3 内部 3-seed control 重新跑？
8. P3-A 哪些比较存在参数量/可辨识性 confound？
9. batch runner 如何支持 5 datasets × 5 variants × 3 seeds 并 resume/skip？
10. 给出最小侵入 implementation audit。

不要写代码。
```

---

# 33. Prompt 2：Full Operator Components

```text
基于 P3 audit，只实现 operator components，不接主模型。

新增：
src/models/biaxis_p3_components.py
tests/test_biaxis_p3.py

实现：
FullResidualFactorRelationOperator

modes：
shared
factor
relation
additive
full_interaction

数学：
shared: W0
factor: W0+A_f
relation: W0+B_k
additive: W0+A_f+B_k
full_interaction: W0+A_f+B_k+C_fk

要求：
- A[F,d,d], B[K,d,d], C[F,K,d,d] 全部 zero init
- bias=False
- 不构造 [N,F,K,d,d]
- shared/factor 尽可能 aggregate-first
- relation/pair 用 K/F×K 小循环即时累加
- input g_perm[N,F,K,d], gamma_graph[N,F,K], shared_w0
- output message[N,F,d]
- no extra regularizer
- 不启用 deterministic mode

tests：
- zero residual equivalence
- shared linear identity
- mode isolation
- pair routing
- gradients
- shape/no NaN
- no giant tensor

不要实现 low-rank。
不要接 biaxis_p3。
```

---

# 34. Prompt 3：集成 `biaxis_p3.py`

```text
Full operator components 已通过测试。

新增：
src/models/biaxis_p3.py
configs/model/biaxis_p3.yaml

要求：
1. Model 继承 biaxis_p2.Model。
2. config 显式：
   p2.mode=null_softmax
   p2.deterministic=false
3. P0 M1 / P1 M2 / P2 score+Gamma 数学保持不变。
4. override _graph_update。
5. 唯一变化：
   P2:
      g_mix=sum Gamma*g
      m=W0(g_mix)
   P3:
      m=sum Gamma*T_fk(g_k)
6. Local Gamma0 不经过 graph operator。
7. update 仍 LayerNorm(f+m)。
8. no new aux loss。
9. forward/inference 兼容现有 NC runner。
10. 不允许 P3 batch script 自动覆盖 p2.deterministic=true。

operator_mode：
shared
factor
relation
additive
full_interaction

tests：
- five modes forward
- O0 与 P2 NullSoftmax 相同 weights 下输出近似一致
- zero-init equivalence
- inference equivalence
- gradient
- params
- p2.deterministic=false
```

---

# 35. Prompt 4：P3 Diagnostics

```text
实现：
compute_p3_diagnostics
scripts/analyze_p3_checkpoint.py

输出：

P2 frozen：
K_eff
S_R
null/graph mass
Gamma usage matrix

Operator：
||W0||
||A_f||/||W0||
||B_k||/||W0||
||C_fk||/||W0||

usage-weighted pair strength

operator pair distances

usage-weighted message-level deviation：
||T_fk(g)-W0g||/(||W0g||+eps)

params/runtime/peak memory

best checkpoint 后 no_grad。
不使用 labels。
不修改训练路径。
不启用 deterministic mode。
```

---

# 36. Prompt 5：Smoke

```text
现在只做 P3 smoke，不做科学判断。

先运行全部 P0/P1/P2/P3 tests。

Movies seed42：

O0 / OF / OR / OADD / OFR

各跑 5 epochs。

检查：
- no NaN
- residual gradient
- zero-init 合理
- params
- runtime
- peak memory
- checkpoint diagnostics 可读取
- p2.deterministic=false

Smoke 成功后停止。

不要因为单 seed 指标高低选择 variant。
不要跑 deterministic repeat。
```

---

# 37. Prompt 6：P3-A 75-run Multi-seed Study

```text
P3 smoke 已通过。

正式执行 P3-A：

datasets:
Movies
Toys
Grocery
ele-fashion
Reddit-S

variants:
O0
OF
OR
OADD
OFR

seeds:
42
43
44

总计 75 runs。

统一：
p2.mode=null_softmax
p2.epsilon=0.2
p2.deterministic=false
no operator regularizer
zero residual init
其它协议全部冻结。

要求：
1. 支持 resume / skip completed。
2. 继续使用快速默认 aggregation。
3. 不开启 deterministic mode。
4. 每 run 保存 Val Acc / Val F1 / Test Acc / Test F1。
5. 汇报 mean±population std。
6. 额外计算 paired-seed delta：
   OF-O0
   OR-O0
   OADD-O0
   OFR-OADD
   OFR-O0
7. 每个 delta 输出：
   mean
   std
   positive_seed_count / 3
8. 生成：
   p3_operator_results.csv
   p3_operator_deltas.csv
   p3_operator_mechanism.csv
   P3_OPERATOR_REPORT.md

报告回答：
- factor transformation 是否必要？
- relation transformation 是否必要？
- additive two-axis 是否必要？
- pair-specific upper-bound 是否比 additive 更好？
- 是否存在参数量/identifiability confound？
- P3-B 应进入 interaction / additive / single-axis / no-go 哪个 branch？

不要实现 low-rank，先分析。
```

---

# 38. Prompt 7：Low-rank Study

只有 P3-A GO 后执行。

```text
实现 LowRankFactorRelationOperator：

U[d,r]
V[d,r]
a[F,r]
b[K,r]

rank=16。

lowrank_add:
c=a+b

lowrank_interaction:
c=a+b+a*b

T=W0x + U[c*(V^T x)]

初始化：
U,V Xavier
a,b zero

要求：
- step0 == Shared
- LR-ADD / LR-INT 参数量完全相同
- 不显式构造 T_fk
- latent rank-r compute
- p2.deterministic=false

tests：
zero equivalence
formula
param matching
gradient
no giant tensor

正式实验：
datasets = 5 NC datasets
modes = LR-ADD, LR-INT
seeds = 42/43/44

新增 30 runs。

O0/OADD/OFR 使用 P3-A 已有 multi-seed reference。

必须汇报：
LR-INT - LR-ADD paired delta
mean±std
positive seed count
Acc/F1
params/runtime/memory

这是 interaction 最关键的参数匹配证据。
```

---

# 39. Prompt 8：Optional FiLM / Rank

```text
只有 Low-rank specialization 已经明确 GO 才执行。

A. FiLM：
T_fk(x)=(1+gamma_fk)*W0x
gamma[F,K,d], zero init, no bias。

若要作为正式 baseline：
5 datasets × seeds42/43/44。

B. Rank sensitivity：
r=8/16/32。

内部快速 sanity 可以只 seed42，
但如果写入正式论文 sensitivity，
必须补到 3 seeds 并汇报 mean±std。

不要打开 deterministic mode。
不要新增其它 operator family。
```

---

# 40. P3 最终决策

## Outcome A — Interaction GO

\[
LR\text{-}INT>LR\text{-}ADD
\]

多 seed 稳定。

最终：

\[
\boxed{
Unified\ Allocation
+
Low\text{-}rank\ Factor\text{-}Relation\ Operator
}
\]

---

## Outcome B — Additive GO

\[
OADD>O0,\quad
LR\text{-}ADD\approx LR\text{-}INT.
\]

最终 low-rank additive。

---

## Outcome C — Single-axis GO

只 OF 或 OR 有明确 multi-seed 收益。

最终保留最简单单轴 operator。

---

## Outcome D — Shared Wins

所有 specialization 无稳定收益。

\[
\boxed{
T_{fk}=W_0
}
\]

P3 NO-GO，接受结果。

---

# 41. Definition of Done

- [ ] `p2.deterministic=false` throughout P3
- [ ] 不使用 deterministic batch training
- [ ] 5 full operator variants 正确
- [ ] P3-A 75 runs 完成
- [ ] mean±population std 完成
- [ ] paired-seed deltas 完成
- [ ] Acc + Macro-F1 完成
- [ ] factor / relation / additive / pair effects 区分
- [ ] mechanism diagnostics 完成
- [ ] 若 GO，LR-ADD/LR-INT 30 runs 完成
- [ ] LR variants 参数完全匹配
- [ ] 边界关键结果必要时扩展到 5 seeds
- [ ] 必要时 FiLM / rank sensitivity
- [ ] Params/runtime/memory 报告
- [ ] 不重新打开 P2 routing
- [ ] 不提前跑 LP
- [ ] 输出 `P3_REPORT.md`
- [ ] 明确 P3 GO / Partial GO / NO-GO
- [ ] final operator frozen

---

# 42. 创新表述边界

不能声称以下 primitive 本身新：

```text
relation-specific linear transform
edge-conditioned filter
FiLM
low-rank matrix/tensor factorization
```

P3 的潜在创新来自：

\[
\boxed{
\textbf{
semantic-factor × latent-structural-relation indexed graph operator
}
}
\]

即：

> 在 P0–P2 已建立的 Semantic Ownership × Relational Function 二维空间上，
> 进一步让 transformation 定义在 factor–relation cell 上。

low-rank 只是参数高效实现。

---

# 43. 如果 P3 成功，整篇方法主公式

\[
\boxed{
m_i^f
=
\sum_k
\Gamma_{ifk}
T_{fk}(g_{ik}^f)
}
\]

其中：

```text
Semantic Factorization -> who owns the semantics?
Structural Relation     -> what relational pattern is present?
Gamma                    -> how much evidence should be used?
T_fk                     -> how should that evidence be interpreted?
```

---

# 44. 当前阶段最重要的纪律

1. **P3 不使用 deterministic mode。**
2. **正式结论一律基于多 Seed mean±std。**
3. **关键 comparison 使用 paired-seed delta。**
4. **单 seed 只做 smoke/debug。**
5. **小于 0.2 pp 的单次差异不强解释。**
6. **边界关键结论优先补 seed45/46，而不是开启慢速 deterministic。**
7. **P3 只研究 transformation，不重新打开 P2 routing。**
8. **先 full operator 做 axis attribution，再 low-rank 做参数匹配。**
9. **OADD 必须存在。**
10. **LR-ADD vs LR-INT 必须参数完全匹配。**
11. **所有 residual zero init。**
12. **Test 不用于架构选择。**
13. **Macro-F1 与 Accuracy 同时保存。**
14. **Shared W0 最好就接受 P3 NO-GO。**
