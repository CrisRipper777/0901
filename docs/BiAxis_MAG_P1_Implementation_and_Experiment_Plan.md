# Bi-Axis MAG — P1 阶段实现与实验推进文档
## 阶段目标：验证 Semantic Factor × Structural Relation 二维建模是否真正有效

> 代码仓库：`CrisRipper777/0901`  
> 前置阶段：P0 已完成并判定 **STRONG GO**  
> P1 任务范围：**仅 Node Classification**  
> 数据集：Movies / Toys / Grocery / ele-fashion / Reddit-S  
> seeds：42 / 43 / 44  
> P1 暂不实现：UOT / Sinkhorn / Low-rank Factor–Relation Operator / Pseudo Nodes / Diffusion / MoE / Rewiring

---

# 0. P1 的唯一研究问题

P0 已经验证：

\[
\boxed{
\text{同一 graph neighborhood 对 } C,P_t,P_v \text{ 的作用并不相同}
}
\]

因此 P1 不再回答“Common/Private 是否存在”，而是进一步验证：

\[
\boxed{
\text{Semantic Factor}\times\text{Structural Relation}
}
\]

是否比只建模 Semantic Factor 或只建模 Structural Relation 更有效。

核心问题：

> 在 Common / Text-private / Visual-private 已经被解耦以后，图拓扑中是否存在可独立于多模态语义识别的潜在结构关系，并且不同 semantic factors 是否应该以不同强度、不同偏好使用这些 structural relations？

P1 的目标不是 SOTA，而是建立后续 P2/P3 的**因果和实验基础**。

---

# 1. 实验协议：从 P1 开始正式冻结

从本阶段开始，**禁止修改主实验协议**。

固定：

```text
NC training      = full-graph transductive training
optimizer        = AdamW
max epochs       = 300
patience         = 30
checkpoint       = validation Accuracy
final test       = checkpoint frozen 后一次性测试
metrics          = Accuracy + Macro-F1
std              = population std (ddof=0)
seeds            = 42 / 43 / 44
```

必须继续复用仓库现有：

```text
src/tasks/nc.py
configs/task/nc.yaml
```

## 1.1 P1 禁止修改

AI/Codex 不得修改：

```text
src/tasks/nc.py
configs/task/nc.yaml
src/data/loaders.py
src/data/splits.py
configs/dataset/*.yaml
现有 baseline model configs
P0 diagnostics 的统计定义
```

除非发现明确 bug，并先单独报告、停止实现、等待人工确认。

## 1.2 模型超参数允许调整的边界

P1 可配置：

```text
num_relations K
relation_dim
relation_temperature
selector_hidden_dim
model dropout
model learning rate（如确有必要）
```

但 P1-screen 第一轮必须使用统一默认设置，不做 dataset-specific tuning。

---

# 2. P0 模块冻结：M1 Semantic Factorization 不再修改

必须复用现有：

```text
src/models/biaxis_components.py
src/models/biaxis_p0.py
```

当前 Semantic Factorization：

\[
x_t,x_v
\rightarrow
h_t,h_v
\rightarrow
(c_t,c_v,p_t,p_v)
\]

\[
c=\frac{c_t+c_v}{2}
\]

保持：

```text
hidden_dim = 256
factor_dim = 128
lambda_common = 0.02
lambda_orth = 0.01
lambda_recon = 0.3
```

禁止在 P1 中重新引入：

```text
DecAlign OT
GMM prototypes for semantics
MMD
cross-modal Transformer
新的 contrastive objective
新的 factor loss
```

P1 的性能增益必须主要来自 graph-side modeling，而不是重新调 M1。

---

# 3. P1 模型结构：只增加三个 graph-side 对象

P1 完整流程：

```text
Text / Visual
      ↓
P0 Semantic Factorizer
      ↓
C / Pt / Pv
      │
      │                  Graph topology A
      │                        ↓
      │              Structural Signature
      │                        ↓
      │              Relation Decomposition
      │                  R1 ... RK
      │                        │
      └──────────────┬─────────┘
                     ↓
           Factor Graph Budget β
                     +
          Factor–Relation Selection α
                     ↓
           Shared Graph Transformation W0
                     ↓
             C' / Pt' / Pv'
                     ↓
                 P0 Fusion
                     ↓
                    z
                     ↓
                 NC head
```

P1 中只有：

1. **Structural Relation Decomposition**
2. **Factor Graph Budget**
3. **Factor–Relation Selection**

图变换算子只使用一个共享 \(W_0\)。

\[
\boxed{
P1\text{ 不允许实现 }W_{f,k}
}
\]

因为 `how to transform` 是 P3 的问题。

---

# 4. P1 推荐代码组织

新增文件：

```text
src/models/biaxis_p1_components.py
src/models/biaxis_p1.py
configs/model/biaxis_p1.yaml
tests/test_biaxis_p1.py
scripts/run_p1_screen.py
scripts/run_p1_confirm.py
scripts/analyze_p1_checkpoint.py
scripts/summarize_p1.py
```

**不要修改 `biaxis_p0.py`。**

推荐：

```python
from .biaxis_p0 import Model as P0Model

class Model(P0Model):
    ...
```

这样 P1 继承同一套：

- modality projectors
- Common/Private encoders
- reconstruction heads
- P0 auxiliary objectives
- factor dimensions

P1 只覆盖 graph-aware `forward()` / `inference()`。

注意：P0 的 chunked `inference()` 对 P1 不再有效，因为 P1 使用完整 topology；必须在 P1 重写 inference，执行完整 full-graph forward。

---

# 5. M2：Topology-only Structural Relation Decomposition

## 5.1 设计原则

Relation Axis 在进入 Factor–Relation Coupling 前必须满足：

\[
\boxed{
R = f(A), \qquad R\not\leftarrow X_t,X_v,C,P_t,P_v
}
\]

也就是说：

> structural relation 本身由 topology 决定；semantic factors 只在后续决定怎样“消费” relation。

这是二维解耦最重要的边界。

---

# 6. Topology Diffusion Signature

P1 第一版不使用 LapPE/RWSE 大型位置编码，也不使用语义特征。

从 observed graph 构造三个 topology-only scalars。

首先：

\[
u_i^{(0)}=\log(1+d_i)
\]

其中 \(d_i\) 为节点度。

定义 row-normalized transition：

\[
P=D^{-1}A.
\]

然后：

\[
u^{(1)}=Pu^{(0)}
\]

\[
u^{(2)}=Pu^{(1)}.
\]

最终：

\[
\boxed{
s_i^{raw}=[u_i^{(0)},u_i^{(1)},u_i^{(2)}]
}
\]

分别表示：

- own structural scale
- 1-hop neighborhood structural profile
- 2-hop structural profile

对三个维度在全图做 z-score normalization：

\[
\bar s=(s-\mu)/(\sigma+\epsilon).
\]

再：

\[
s_i=MLP_S(\bar s_i)\in\mathbb R^{d_r}.
\]

默认：

```text
relation_dim d_r = 32
```

---

# 7. Edge Structural Token 必须对无向边方向不敏感

对 edge \((j,i)\)：

\[
e_{ij}=MLP_E[
(s_i+s_j)
\Vert
|s_i-s_j|
\Vert
(s_i\odot s_j)
].
\]

不要直接使用：

\[
[s_i\Vert s_j]
\]

作为唯一输入，否则同一条无向边的两个方向可能被赋予不同 relation。

输出：

```text
e_ij: [E, relation_dim]
```

---

# 8. Relation Prototypes

维护：

\[
\rho_1,\ldots,\rho_K\in\mathbb R^{d_r}.
\]

默认：

```text
K = 4
relation_temperature = 0.5
```

relation assignment：

\[
\boxed{
r_{ij,k}
=
\frac{
\exp(\cos(e_{ij},\rho_k)/\tau_R)
}{
\sum_l\exp(\cos(e_{ij},\rho_l)/\tau_R)
}
}
\]

因此：

\[
\sum_k r_{ij,k}=1.
\]

解释：

> 一个 observed edge 可以 soft-belong to 多个 latent structural relation bases。

不得人为命名：

```text
shared edge
private edge
heterophilous edge
complementary edge
```

Relation IDs 只叫：

```text
R1 ... RK
```

并注意 prototype permutation：不同 seed 下 `R1` 不一定对应同一种 structural pattern。

---

# 9. 不生成 K 张稠密邻接矩阵

严禁：

```python
A_rel = torch.zeros(K, N, N)
```

也不要 materialize：

```text
[E, K, factor_dim]
```

第一版只允许保存：

```text
edge_index: [2,E]
r:          [E,K]
```

关系传播使用 sparse/scatter 运算。

## 9.1 推荐 aggregation 实现

对于每个 relation \(k\)，计算 incoming relation mass：

\[
m_{i,k}=\sum_{j\in N(i)}r_{ji,k}.
\]

relation availability：

\[
\boxed{
a_{i,k}=\frac{m_{i,k}}{d_i+\epsilon}
}
\]

因此非孤立节点上近似：

\[
\sum_k a_{i,k}=1.
\]

对 factor \(f\)：

\[
\tilde g_{i,k}^{f}
=
\sum_{j\in N(i)}r_{ji,k}f_j
\]

然后使用 weighted mean：

\[
\boxed{
g_{i,k}^{f}
=
\frac{\tilde g_{i,k}^{f}}{m_{i,k}+\epsilon}
}
\]

这样 relation occupancy 与 relation semantic content 被分开：

- \(a_{ik}\)：这个 relation 在邻域里有多少
- \(g_{ik}^f\)：这个 relation 提供什么 factor content

避免“大 relation 因为边多而天然 message norm 更大”。

## 9.2 推荐计算优化

可以把三个 factors concatenate：

\[
F_{cat}=[C\Vert P_t\Vert P_v]\in\mathbb R^{N\times3d_f}
\]

每个 relation 仅做一次 sparse aggregation，再 split 为三个 factor message。

这样每层只需约 \(K\) 次 sparse aggregation，而不是 \(3K\) 次。

---

# 10. Structural Signature 缓存

NC 使用固定 full graph，因此 raw topology signature：

\[
s^{raw}=f(A)
\]

无需每 epoch 重算。

推荐在 `Model` 中：

```python
self.register_buffer(
    "_cached_raw_struct_signature",
    torch.empty(0),
    persistent=False,
)
```

第一次 full-graph forward：

1. 根据 `edge_index` 计算 raw signature
2. detach
3. 缓存

以后只让：

```text
MLP_S
MLP_E
relation prototypes
```

参与学习。

缓存不得依赖 semantic features。

测试中必须验证：改变 `x` 但保持 `edge_index` 不变时，relation decomposition 输入保持相同。

---

# 11. M3a：Factor Graph Budget

P0 显示 Common / Private 的 graph propagation gain 强度不同，因此 P1 不应该强制每个 factor 100% 使用 graph message。

定义：

\[
\beta_i^f\in[0,1].
\]

先构造 relation-averaged context：

\[
\bar g_i^f
=
\sum_k a_{i,k}g_{i,k}^f.
\]

Graph Budget：

\[
\boxed{
\beta_i^f
=
\sigma\left(
MLP_B[f_i\Vert\bar g_i^f]
\right)
}
\]

语义：

```text
beta ≈ 0 : retain node-local factor evidence
beta ≈ 1 : strongly use graph relational evidence
```

初始化建议：

- budget final layer small/zero initialization
- bias = 0

使模型初始：

\[
\beta\approx0.5.
\]

不要初始化成接近 0 或 1。

---

# 12. M3b：Factor–Relation Selection

relation score：

\[
s_{i,f,k}
=
MLP_R[
 f_i
 \Vert g_{i,k}^f
 \Vert(f_i\odot g_{i,k}^f)
 \Vert a_{i,k}
].
\]

然后：

\[
\boxed{
\alpha_{i,f,k}
=Softmax_k(s_{i,f,k})
}
\]

其中：

\[
\sum_k\alpha_{i,f,k}=1.
\]

解释：

- \(\beta_i^f\)：**how much graph evidence**
- \(\alpha_{i,f,k}\)：**which structural relations**

这两个量必须分开。

---

# 13. P1 只使用共享 Graph Operator

所有 factor / relation 共用：

\[
W_0\in\mathbb R^{d_f\times d_f}.
\]

先：

\[
g_i^f
=
\sum_k\alpha_{i,f,k}g_{i,k}^f.
\]

再：

\[
m_i^f=W_0g_i^f.
\]

更新：

\[
\boxed{
\tilde f_i
=
LayerNorm(
 f_i+\beta_i^f m_i^f
)
}
\]

最终：

\[
z_i
=
Fusion[
\tilde c_i\Vert
\tilde p_i^t\Vert
\tilde p_i^v
].
\]

P1 不允许：

```text
W_C
W_Pt
W_Pv
W_R1 ... W_RK
W_{f,k}
low-rank adapter
MoE expert
```

这些全部留给 P3。

---

# 14. P1 的主实验不是“模块消融”，而是 2×2 factorial design

为了避免参数/encoder 差异污染结论，**四个 variant 全部使用同一个 P0 Semantic Factorizer**。

这里的 Factor ON/OFF 定义为：

> graph module 是否保留并识别 `C/Pt/Pv` 的 factor identity。

不是重新训练另一个“无解耦 encoder”。

---

# 15. 四个 Variant 的精确定义

## F0R0 — Factor-blind + Single Relation

P0 先得到：

\[
C,P_t,P_v.
\]

先融合成一个 factor-blind state：

\[
q_i
=
Proj_q(Fusion_{P0}[C_i\Vert P_{t,i}\Vert P_{v,i}])
\in\mathbb R^{d_f}.
\]

图中只有一个 relation：

\[
K=1,
\qquad r_{ij,1}=1.
\]

只对 \(q\) 做 budget + graph update。

这个 variant 表示：

> 语义已被编码，但 graph propagation 看不到 Common/Private identity，也看不到多 relation。

---

## F1R0 — Factor-aware + Single Relation

保持：

\[
C,P_t,P_v
\]

分开传播。

但：

\[
K=1.
\]

所以：

\[
\alpha_{if1}=1.
\]

每个 factor 可以有独立 graph budget \(\beta_i^f\)，但所有邻居属于同一结构关系。

这个 variant 回答：

> 只做 factor-specific graph usage 是否已经足够？

---

## F0R1 — Factor-blind + Latent Relations

先融合成：

\[
q_i.
\]

使用：

\[
K=4
\]

个 topology-only latent relations。

只对单一 \(q\) 学习：

\[
\beta_i^q,
\quad\alpha_{i,q,k}.
\]

这个 variant 回答：

> 只做 latent relation modeling、但不保留 semantic factor identity 是否足够？

---

## F1R1 — Full Bi-Axis P1

\[
\boxed{
\{C,P_t,P_v\}\times\{R_1,...,R_K\}
}
\]

每个 factor 独立学习：

\[
\beta_i^f
\]

以及：

\[
\alpha_{i,f,k}.
\]

但所有 cell 仍共享 \(W_0\)。

这是 P1 的完整模型。

---

# 16. 额外 Local Reference

P0 topology-free：

```text
biaxis_p0
```

作为 Local Reference，不属于 2×2 interaction effect。

用于观察：

\[
P1\ graph\ module
\]

是否真的比 topology-free factorization 有增益。

---

# 17. Interaction Effect

对同一 dataset / seed，定义性能函数 \(P(F,R)\)。

Factor 主效应：

\[
\Delta_F=P(F1R0)-P(F0R0).
\]

Relation 主效应：

\[
\Delta_R=P(F0R1)-P(F0R0).
\]

最重要：

\[
\boxed{
\Delta_{FR}
=
P(F1R1)-P(F1R0)-P(F0R1)+P(F0R0)
}
\]

若：

\[
\Delta_{FR}>0,
\]

说明 Factor 与 Relation 的联合建模产生正 interaction，而不是两个模块简单相加。

P1-screen / confirm 都要分别用：

```text
Val Accuracy
Test Accuracy
Test Macro-F1
```

计算。

**模型去留主要根据 validation 结果；test 只做冻结后的确认。**

---

# 18. Graph Budget 的必要性要单独验证

因为 P0 明确观察到不同 factor 的 graph gain 强度不同，所以 budget 不是任意模块。

但仍需实验确认。

在完整 F1R1 上追加三个小 ablation：

```text
B0: beta = 1                       # 强制所有 factor 完全使用 graph
B1: shared beta_i                 # C/Pt/Pv 共用同一个 budget
B2: factor-specific beta_i^f      # P1 默认
```

第一轮只跑：

```text
Movies
Grocery
ele-fashion
seed=42
```

如果：

```text
B2 ≈ B0
```

且 learned beta 无明显差异，则 P1 后续可删除 Graph Budget，避免多余模块。

如果 B2 稳定更好且：

\[
\beta_C,\beta_{Pt},\beta_{Pv}
\]

存在非平凡差异，则保留。

---

# 19. P1 必须记录的机制诊断

## 19.1 Relation Occupancy

\[
Occ_k
=
\frac{1}{E}\sum_{ij}r_{ij,k}.
\]

记录：

```text
rel_occ_0 ... rel_occ_K-1
```

Relation effective number：

\[
K_{eff}
=
\exp\left(-\sum_kOcc_k\log Occ_k\right).
\]

记录：

```text
rel_effective_num
rel_assignment_entropy
```

---

## 19.2 Graph Budget

对：

```text
C / Pt / Pv
```

记录：

```text
beta_mean
beta_std
beta_p10
beta_p50
beta_p90
beta_low_frac   # beta < 0.05
beta_high_frac  # beta > 0.95
```

特别观察：

\[
\bar\beta_C,
\bar\beta_{Pt},
\bar\beta_{Pv}.
\]

P0 暗示 Private 可能在部分数据集有更强 graph demand，但 P1 **不得人为强制这个顺序**。

---

## 19.3 Factor–Relation Selection

对每个 factor：

\[
H(\alpha^f_i)
=-\sum_k\alpha_{ifk}\log\alpha_{ifk}.
\]

记录：

```text
alpha_entropy_c
alpha_entropy_pt
alpha_entropy_pv
```

以及 node-wise JS divergence：

\[
JS(C,Pt)
=
\frac1N\sum_iJS(\alpha_i^C,\alpha_i^{Pt}).
\]

记录：

```text
alpha_js_c_pt
alpha_js_c_pv
alpha_js_pt_pv
```

不要只比较全局平均 alpha，因为平均后可能掩盖 node-wise 差异。

---

## 19.4 Factor–Relation Usage Matrix

定义：

\[
U_{f,k}
=
\frac1N\sum_i
\beta_i^f\alpha_{i,f,k}.
\]

最终保存：

```text
3 × K matrix
```

这是 P1 最重要的解释性对象之一。

可用于未来论文 heatmap：

```text
             R1   R2   R3   R4
Common
Text-private
Visual-private
```

---

# 20. Relation Collapse：第一版只诊断，不提前正则

默认：

```text
lambda_relation_balance = 0
lambda_alpha_entropy = 0
lambda_budget_reg = 0
```

只有出现明确 collapse，例如：

```text
某一个 Occ_k > 0.85
K_eff 接近 1
```

才允许尝试非常弱的 occupancy regularization。

例如：

\[
L_{bal}=KL(\bar r\Vert Uniform).
\]

初始：

```text
lambda_relation_balance = 1e-3
```

任何新增正则必须作为独立实验，不得偷偷并入默认配置。

---

# 21. P1 默认配置建议

新增：

```text
configs/model/biaxis_p1.yaml
```

建议：

```yaml
name: biaxis_p1

# Frozen P0 semantic factorization
hidden_dim: 256
factor_dim: 128
dropout: 0.2
activation: gelu
norm: layernorm

lambda_common: 0.02
lambda_orth: 0.01
lambda_recon: 0.3
orth_fallback_batch: 16

# Keep current unified NC protocol.
lr: 0.001
weight_decay: 0.0001

p1:
  factor_aware: true
  num_relations: 4

  relation_dim: 32
  relation_temperature: 0.5

  selector_hidden_dim: 64
  budget_hidden_dim: 64
  use_graph_budget: true

  eps: 1.0e-8

  # diagnostic-only by default
  relation_balance_weight: 0.0
  alpha_entropy_weight: 0.0
  budget_reg_weight: 0.0

  # optional engineering knob for large E
  edge_chunk_size: 500000
```

P1-screen 不做 dataset-specific override。

---

# 22. Variant 通过 Hydra override 实现，不写四份模型

统一模型：

```text
biaxis_p1.py
```

通过：

### F0R0

```bash
model.p1.factor_aware=false \
model.p1.num_relations=1
```

### F1R0

```bash
model.p1.factor_aware=true \
model.p1.num_relations=1
```

### F0R1

```bash
model.p1.factor_aware=false \
model.p1.num_relations=4
```

### F1R1

```bash
model.p1.factor_aware=true \
model.p1.num_relations=4
```

`num_relations=1` 时必须强制：

\[
r_{ij,1}=1,
\quad
\alpha_{i,f,1}=1.
\]

不应该浪费计算去跑 prototype softmax。

---

# 23. P1 Model API 要求

继续满足仓库契约：

```python
forward(x, edge_index)
    -> z, None, None, aux_loss, aux_info
```

其中 `aux_loss` 至少包含冻结的 P0：

```text
common consistency
common-private orthogonality
reconstruction
```

P1 默认不增加 graph regularization loss。

必须重写：

```python
@torch.no_grad()
def inference(...):
```

P1 inference 执行完整 graph forward，不得像 P0 那样按 node chunk 独立编码 graph message。

可以保留继承的：

```python
encode_factors(...)
```

用于检查 topology-free semantic factors。

新增：

```python
@torch.no_grad()
def compute_p1_diagnostics(x, edge_index):
    ...
```

返回 best-checkpoint 的：

```text
relation occupancy
relation effective number
beta statistics
alpha entropy
node-wise JS
usage matrix
```

---

# 24. P1 单元测试必须覆盖

新增：

```text
tests/test_biaxis_p1.py
```

至少包括：

## 24.1 Shape

随机小图：

```text
N=17
text_dim=13
visual_dim=19
K=4
```

检查：

```text
z          [17,256]
r          [E,4]
beta       [17,3]       # factor-aware
alpha      [17,3,4]     # factor-aware
```

## 24.2 Relation simplex

\[
\sum_kr_{ij,k}=1
\]

数值误差范围内成立。

## 24.3 Selection simplex

\[
\sum_k\alpha_{ifk}=1.
\]

## 24.4 Budget bounds

\[
0\le\beta_i^f\le1.
\]

## 24.5 K=1 special case

确认：

```text
r = 1
alpha = 1
```

并且 relation weighted mean 等于普通 neighbor mean。

## 24.6 Topology-only relation

同一个 `edge_index`：

```python
x1 != x2
```

relation decomposition 输出必须完全一致（eval mode）。

这是二维解耦核心测试。

## 24.7 Edge permutation invariance

随机打乱 edge ordering 后：

```text
final aggregated messages
```

应相同。

## 24.8 Reverse-edge consistency

对无向 edge：

```text
relation(j,i) == relation(i,j)
```

在相同 structural signatures 下应成立。

## 24.9 Gradient

确认 gradient 到达：

```text
P0 factorizer
structural MLP
edge structural MLP
relation prototypes
budget MLP
selector MLP
shared W0
fusion
```

## 24.10 No dense adjacency

代码审查确认没有：

```text
[N,N]
[K,N,N]
[E,K,d]
```

级别 tensor。

## 24.11 Inference-forward equivalence

小图 eval：

```text
model.forward(full graph)
```

与：

```text
model.inference(full graph)
```

输出一致。

---

# 25. P1 训练流程：漏斗推进

## Phase P1-D0：Implementation Smoke

只跑：

```text
Movies
seed=42
F1R1
5 epochs
```

目标：

- 无 NaN
- 显存合理
- relation assignment 非异常
- beta/alpha 有梯度
- train loss 可下降
- model forward/inference 正常

此阶段不判断科学结果。

---

# 26. Phase P1-D1：四 Variant Smoke

Movies / seed=42 / 10 epochs：

```text
F0R0
F1R0
F0R1
F1R1
```

检查：

- 四种 config switch 真正改变计算图
- K=1 特例正常
- factor-aware=false 不产生 3-factor selector
- output metrics 可比较
- checkpoint 可保存/读取
- diagnostics 可在 best checkpoint 后运行

全部通过才允许 P1-screen。

---

# 27. Phase P1-Screen：20 runs

数据：

```text
Movies
Toys
Grocery
ele-fashion
Reddit-S
```

seed：

```text
42
```

Variants：

```text
F0R0
F1R0
F0R1
F1R1
```

总计：

\[
4\times5=20\text{ runs}.
\]

每个 run 使用完整 frozen NC protocol。

**Screen 阶段主要依据：best validation Accuracy + mechanism diagnostics。**

Test 指标仍保存，但不得据此修改模型。

---

# 28. P1-Screen 输出表

```text
outputs/p1/screen/p1_screen_results.csv
```

格式：

| Dataset | Variant | Seed | Best Val Acc | Test Acc | Test F1 | Params | Epoch Time | Peak Mem |
|---|---|---:|---:|---:|---:|---:|---:|---:|

另输出：

```text
p1_screen_interaction.csv
```

| Dataset | ΔF | ΔR | ΔFR | F1R1-F1R0 | F1R1-F0R1 |
|---|---:|---:|---:|---:|---:|

以及：

```text
p1_screen_mechanism.csv
```

至少包括：

```text
rel_effective_num
beta_C
beta_Pt
beta_Pv
JS_C_Pt
JS_C_Pv
JS_Pt_Pv
```

---

# 29. P1-Screen GO / REVISE / NO-GO

## Strong GO

满足大部分：

1. \(F1R1>F1R0\) 于至少 3/5 datasets
2. \(F1R1>F0R1\) 于至少 3/5 datasets
3. \(\Delta_{FR}>0\) 于至少 3/5 datasets
4. relation 未 collapse
5. node-wise factor relation JS 非零且有明显差异
6. F1R1 至少达到 GCN/MMGCN 附近的健康性能区间

则进入 P1-confirm。

## Revise

### A. Relation collapse

```text
K_eff ≈ 1
```

但 F1R0 已有效。

说明 Relation Decomposition 本身需要修，而不是核心 hypothesis 失败。

可尝试：

```text
temperature 0.5 -> 1.0
weak balance loss 1e-3
prototype initialization improvement
```

一次只改一个因素。

### B. Alpha identical across factors

```text
JS ≈ 0
```

但 P0 明确存在 factor-dependent utility。

说明 selector 容量/输入不足。

优先检查实现与 normalization，不直接加复杂模块。

### C. Beta saturation

```text
beta ≈ 0 或 1
```

大量节点发生。

先检查 message scale、initialization、LayerNorm，而不是加正则。

## Scientific NO-GO

如果：

- relation decomposition 正常、有多个 relation；
- alpha/beta 正常；
- 但 F1R1 在 5 个数据集均无法超过 F1R0 / F0R1；
- interaction effect 接近 0 或为负；

则说明：

> P0 中的 factor-dependent neighborhood utility 并不需要显式 Structural Relation Axis 来解释。

此时不能进入 P2 强行上 OT。

---

# 30. Phase P1-Budget Ablation

只有 F1R1 screen 有信号后再做。

Datasets：

```text
Movies
Grocery
ele-fashion
```

Seed：42。

比较：

```text
B0 beta=1
B1 shared beta
B2 factor-specific beta
```

如果 B2 无贡献，可在 P1-final 删除 budget，保持方法简洁。

---

# 31. Phase P1-Confirm：60 runs

如果 screen GO：

```text
5 datasets
× 4 variants
× 3 seeds
= 60 runs
```

seeds：

```text
42 43 44
```

正式报告：

```text
mean ± population std
```

指标：

```text
Val Accuracy
Test Accuracy
Test Macro-F1
```

并对每个 seed 保存 best-checkpoint mechanism diagnostics。

---

# 32. 当前 NC baseline 参照线

P1 期间不重新调 baseline。

当前统一协议 reference：

| Model | Movies Acc/F1 | Toys | Grocery | ele-fashion | Reddit-S |
|---|---|---|---|---|---|
| MLP | 50.91 / 31.18 | 75.03 / 71.31 | 77.30 / 65.52 | 85.82 / 63.22 | 92.01 / 85.03 |
| GCN | 53.91 / 45.58 | 79.16 / 75.72 | 79.97 / 70.01 | 85.16 / 71.74 | 93.82 / 88.81 |
| SAGE | 49.28 / 27.23 | 78.92 / 75.44 | 80.29 / 70.28 | 87.25 / 74.95 | 94.69 / 90.20 |
| MMGCN | 53.64 / 46.69 | 77.63 / 74.40 | 80.41 / 70.65 | 87.62 / 76.80 | 96.12 / 92.16 |
| MGAT | 53.78 / 38.72 | 79.05 / 76.43 | 81.82 / 72.89 | 86.86 / 72.03 | 96.47 / 92.26 |
| DMGC | 47.33 / 25.59 | 71.27 / 67.19 | 71.90 / 59.79 | 86.75 / 71.74 | 93.35 / 87.93 |
| DGF | 52.10 / 31.19 | 78.71 / 75.10 | 81.17 / 69.61 | 86.83 / 72.29 | 96.26 / 90.95 |
| DiP | 55.01 / 46.86 | 79.84 / 76.89 | 83.15 / 75.24 | 87.73 / 76.61 | 96.37 / 91.84 |

P1 机制 GO 不要求立刻超过 DiP，但存在三条性能线：

```text
Scientific line: F1R1 > F1R0 / F0R1
Health line:     F1R1 ≈ or > GCN/MMGCN region
Final target:    P2/P3 后与 DiP / strongest baseline 竞争
```

特别是 ele-fashion 必须同时看 Macro-F1，不能只看 Accuracy。

---

# 33. P1 输出目录建议

```text
outputs/p1/
├── smoke/
├── screen/
│   ├── Movies/
│   ├── Toys/
│   ├── Grocery/
│   ├── ele-fashion/
│   ├── Reddit-S/
│   ├── p1_screen_results.csv
│   ├── p1_screen_interaction.csv
│   ├── p1_screen_mechanism.csv
│   └── P1_SCREEN_REPORT.md
├── budget_ablation/
└── confirm/
    ├── p1_confirm_results.csv
    ├── p1_confirm_interaction.csv
    ├── p1_confirm_mechanism.csv
    └── P1_REPORT.md
```

每个 run 保存：

```text
config snapshot
seed
variant
best checkpoint
training log
best val acc
test acc/f1
mechanism diagnostics
runtime
peak memory
```

---

# 34. AI/Codex 推进原则

**不要一次给 AI 一个 prompt 让它把 P1 全写完。**

建议 6 个阶段：

```text
Prompt 1  P1 repository audit + protocol freeze
Prompt 2  Structural Relation components + tests
Prompt 3  Budget/Selector + biaxis_p1 integration
Prompt 4  Variant + diagnostics + checkpoint analysis
Prompt 5  Smoke tests
Prompt 6  Screen batch + summary
```

每一步完成后先 code review，再进入下一步。

---

# 35. Prompt 1：仓库审查 + 冻结协议

```text
你现在协助我在仓库 CrisRipper777/0901 中进入 Bi-Axis 项目的 P1 阶段。

先不要修改任何代码。

背景：
P0 已完成，核心结论是 Common/Text-private/Visual-private 对同一 graph neighborhood 的 utility 存在稳定差异。P1 只验证 Semantic Factor × Structural Relation 二维建模，不实现 OT、Sinkhorn、Low-rank operator、pseudo node、diffusion 或 MoE。

实验协议已经正式冻结：
- 只做 NC
- datasets = Movies, Toys, Grocery, ele-fashion, Reddit-S
- full-graph training
- AdamW
- 300 epochs
- patience 30
- val Accuracy checkpoint
- final test 一次
- Acc + Macro-F1
- seeds 42/43/44

严格禁止修改：
- src/tasks/nc.py
- configs/task/nc.yaml
- src/data/loaders.py
- src/data/splits.py
- configs/dataset/*.yaml
- 现有 baseline configs
- P0 统计定义

请仔细审查：
1. src/models/biaxis_p0.py
2. src/models/biaxis_components.py
3. src/models/factory.py
4. src/tasks/nc.py
5. configs/model/biaxis_p0.yaml
6. P0 diagnostics/probes
7. 当前 tests

然后输出 P1 implementation audit，重点回答：
- 如何继承/复用 P0 Model 才能冻结 M1 实现而不复制逻辑？
- P1 为什么必须重写 inference？
- full-graph 下 topology signature 在哪里缓存最安全？
- 怎样实现 relation weighted aggregation 而不生成 [N,N] / [K,N,N] / [E,K,d]？
- F0R0/F1R0/F0R1/F1R1 如何通过一个 model config switch 实现？
- 哪些地方最容易造成 semantic leakage into relation decomposition？
- 哪些地方最容易造成显存爆炸？
- 当前代码哪些接口不能动？

不要写代码，不要修改协议。
```

---

# 36. Prompt 2：只实现 Structural Relation Decomposition

```text
基于已完成的 P1 implementation audit，现在只实现 M2：Topology-only Structural Relation Decomposition。

不要实现 graph budget、factor-relation selector、最终 P1 model。

新增：
- src/models/biaxis_p1_components.py
- tests/test_biaxis_p1.py（先加入 relation 部分测试）

要求：

1. 实现 TopologyDiffusionSignature：
   u0 = log(1+degree)
   u1 = D^{-1} A u0
   u2 = D^{-1} A u1
   raw=[u0,u1,u2]
   全图 z-score 后 MLP 到 relation_dim=32。

2. 输入必须只有 edge_index / num_nodes，不允许接收 x/text/image/C/Pt/Pv。

3. 实现 symmetric edge token：
   [s_i+s_j, abs(s_i-s_j), s_i*s_j]
   -> edge MLP -> e_ij。

4. 实现 K 个 learnable relation prototypes，并通过 cosine + temperature softmax 得到 r:[E,K]。

5. K=1 时走严格 fast path：r=ones，不运行 prototype softmax。

6. raw topology signature 可以 cache；cache 必须 persistent=False，且不进入 state_dict。

7. 不创建 [N,N]、[K,N,N] 或 [E,K,d] tensor。

8. 实现 relation weighted aggregation helper：
   - relation mass m_ik
   - availability a_ik=m_ik/degree_i
   - relation weighted mean g_ik
   - 支持将 C/Pt/Pv concat 后每个 relation 一次 sparse/scatter aggregation。

9. 默认不加 relation balance loss。

单元测试必须覆盖：
- r rows sum to 1
- K=1 r=1
- topology-only：改变 x 不改变 r
- reverse-edge consistency
- edge permutation invariance
- weighted mean correctness（手算小图）
- gradient 到 structural MLP / edge MLP / prototypes
- 无 dense adjacency

实现后运行 tests，报告：
- 修改文件
- tensor shape
- 时间/空间复杂度
- tests 结果

不要进入 Budget/Selector。
```

---

# 37. Prompt 3：Budget + Selector + P1 Model Integration

```text
Structural Relation Decomposition 已通过测试。现在实现 P1 的 M3 和主 Model。

请新增：
- FactorGraphBudget
- FactorRelationSelector
- src/models/biaxis_p1.py
- configs/model/biaxis_p1.yaml
并扩展 tests/test_biaxis_p1.py。

科学约束：
- M1 必须复用/继承现有 biaxis_p0，保持 hidden_dim=256, factor_dim=128, lambda_common=0.02, lambda_orth=0.01, lambda_recon=0.3。
- relation decomposition 只能来自 topology。
- P1 只有 shared graph operator W0，不允许 W_f/W_k/W_fk。
- 不实现 OT/Sinkhorn/low-rank/MoE/pseudo node/diffusion。

完整公式：
1. beta_if = sigmoid(MLP_B([f_i, gbar_i^f]))
2. score_ifk = MLP_R([f_i, g_ik^f, f_i*g_ik^f, availability_ik])
3. alpha_if = softmax_k(score_ifk)
4. g_i^f = sum_k alpha_ifk g_ik^f
5. m_i^f = W0 g_i^f
6. f'_i = LayerNorm(f_i + beta_if * m_i^f)
7. z = P0 fusion([C',Pt',Pv'])

要求：
- W0 对所有 factor/relation 共享。
- budget 初始化约为 0.5，不要初始化饱和。
- K=1 时 alpha=1 fast path。
- isolated node 不产生 NaN。
- forward 返回框架 5 元组。
- 保留 P0 aux loss。
- 重写 inference 为 full-graph exact forward。
- encode_factors 仍保持 topology-free。

实现四种 config variant：
F0R0 factor_aware=false K=1
F1R0 factor_aware=true  K=1
F0R1 factor_aware=false K=4
F1R1 factor_aware=true  K=4

Factor OFF 的定义：仍先使用同一 P0 factorizer，但 graph module 看不到 factor identity；用 P0 fusion 后投影得到单一 q∈R^128，再进行 graph update。

测试：
- 四 variants shape/forward
- alpha simplex
- beta range
- K=1 fast path
- gradient flow
- inference-forward equivalence
- factor_aware=false 确认只产生一个 graph state
- factor_aware=true 确认产生 3-factor beta/alpha

不要写 batch experiment scripts。
```

---

# 38. Prompt 4：机制 Diagnostics + Variant 汇总

```text
P1 主模型已经通过单元测试。现在实现 best-checkpoint 机制 diagnostics，不修改模型公式。

新增：
- model.compute_p1_diagnostics(...)
- scripts/analyze_p1_checkpoint.py
- scripts/summarize_p1.py 的基础结构

best checkpoint 后需要输出：

Relation:
- occ_k
- relation effective number
- mean relation assignment entropy

Budget（factor-aware 时）：
- mean/std/p10/p50/p90
- beta<0.05 fraction
- beta>0.95 fraction
for C/Pt/Pv

Selector：
- alpha entropy C/Pt/Pv
- node-wise JS(C,Pt), JS(C,Pv), JS(Pt,Pv)

Usage matrix：
U_fk = mean_i beta_if * alpha_ifk
保存为 JSON + CSV。

注意：
- relation IDs 在不同 seeds 间有 permutation，不直接平均 R1 对应的具体语义。
- 可以平均 K_eff/entropy/JS/beta；usage matrix 原始版本 per-run 保存。
- diagnostics 必须 no_grad，不修改模型状态。
- 不使用 test labels。

所有文件写入 outputs/p1/...，不得覆盖 P0 输出。
```

---

# 39. Prompt 5：Smoke Test

```text
现在只做 P1 smoke，不跑正式 screen。

先运行全部 tests。

然后：
A. Movies seed=42 F1R1，5 epochs。
B. Movies seed=42 四 variants，各 10 epochs。

必须继续调用现有 src.main / src.tasks.nc frozen protocol，不复制训练器。

检查并报告：
- 每个 variant 是否成功训练
- best val/test 只作 smoke 参考
- 参数量
- epoch time
- peak GPU memory（能测则测）
- relation K_eff / occupancy
- beta mean/range
- alpha entropy / JS
- 是否有 NaN
- 是否 relation collapse
- 是否 beta saturation
- 四 variant 是否真的走不同逻辑分支

如果出现问题，只定位原因，不自动增加正则、不进入 OT、不改协议。
```

---

# 40. Prompt 6：P1-Screen 20 Runs

```text
P1 smoke 已通过。现在生成并执行 P1-screen。

Datasets：
Movies
Toys
Grocery
ele-fashion
Reddit-S

Seed：42

Variants：
F0R0
F1R0
F0R1
F1R1

总计 20 runs。

要求：
1. 每个 run 必须调用现有 frozen NC trainer。
2. 不修改任何超参数。
3. 每个 run 保存 best checkpoint 和 config snapshot。
4. checkpoint 后调用 P1 diagnostics。
5. 单个 run 失败不得覆盖其它结果。
6. 支持 resume / skip completed。
7. 生成：
   outputs/p1/screen/p1_screen_results.csv
   outputs/p1/screen/p1_screen_interaction.csv
   outputs/p1/screen/p1_screen_mechanism.csv
   outputs/p1/screen/P1_SCREEN_REPORT.md
8. interaction effect 同时用 best val Acc、test Acc、test Macro-F1 计算；模型是否进入 confirm 主要看 validation。
9. 报告对比当前 GCN/MMGCN/DiP reference，但不要因没超过 DiP 判 P1 失败。
10. 只给出 GO / REVISE / NO-GO 分析，不实现 P2。
```

---

# 41. P1 代码审查清单

每次 AI 实现后人工确认：

- [ ] P0 文件没有被偷偷改动
- [ ] nc.py / nc.yaml 没改
- [ ] split/dataset configs 没改
- [ ] relation module 没读 x/C/Pt/Pv
- [ ] relation edge token symmetric
- [ ] K=1 有 fast path
- [ ] 没有 dense adjacency
- [ ] 没有 E×K×d 大 tensor
- [ ] graph budget 与 relation selector 是两个量
- [ ] W0 唯一且共享
- [ ] F0/F1 定义符合文档
- [ ] P0 aux loss 仍存在
- [ ] eval/inference 使用 full graph
- [ ] diagnostics 来自 best checkpoint
- [ ] screen 不看 test 调模型

---

# 42. P1 完成 Definition of Done

只有满足以下条件，P1 才算结束：

- [ ] 结构关系模块通过全部单元测试
- [ ] relation axis 严格 topology-only
- [ ] F0R0/F1R0/F0R1/F1R1 四 variants 统一模型实现
- [ ] P1-screen 20 runs 完成
- [ ] interaction effect 汇总完成
- [ ] relation occupancy/K_eff 完成
- [ ] factor beta statistics 完成
- [ ] factor relation JS 完成
- [ ] usage matrix 完成
- [ ] 完成 budget ablation（仅 screen GO 后）
- [ ] P1-confirm 60 runs 完成（仅 screen GO 后）
- [ ] 生成 `P1_REPORT.md`
- [ ] 对 P1 给出 GO / REVISE / NO-GO 决策
- [ ] 未提前实现 UOT
- [ ] 未提前实现 low-rank operator

---

# 43. P1 成功后才进入 P2

若 P1 最终证明：

\[
F1R1>F1R0,
\quad
F1R1>F0R1,
\]

并且：

\[
\Delta_{FR}>0
\]

跨多个数据集稳定成立，同时 factor budgets / relation preferences 确实不同，则下一阶段才问：

> 当前 \(\beta_i^f\alpha_{ifk}\) 是两个独立预测器拼成的 heuristic coupling，能否用 UOT/transport plan 统一建模 factor demand 与 relation supply？

即 P2：

\[
\boxed{
\beta_i^f\alpha_{ifk}
\rightarrow
\Gamma_{ifk}^{UOT}
}
\]

如果 P1 本身不成立，则禁止进入 P2。

---

# 44. 当前阶段最重要的科研纪律

1. **P1 验证二维建模，不追求一次到位的最终模型。**
2. **M1 冻结，避免将 graph-side 增益与 factorizer 调参混淆。**
3. **Relation 必须 topology-only。**
4. **Factor identity 只在 coupling 时进入。**
5. **Budget 回答 how much，Selection 回答 from where。**
6. **P1 只有共享 W0；how to transform 留给 P3。**
7. **Screen 用 validation 决策，不用 test 做架构搜索。**
8. **不因单 dataset 失败立刻改模型。**
9. **不因模型图看起来简单而加模块。**
10. **只有 P1 实验证明 Factor×Relation 有真实 interaction，后续 OT/low-rank 才有意义。**
