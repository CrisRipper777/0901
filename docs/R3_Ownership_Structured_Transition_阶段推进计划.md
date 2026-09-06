# R3 阶段推进计划：基于 Semantic Ownership 的图计算架构重构

> 项目仓库：`https://github.com/CrisRipper777/0901`  
> 当前阶段：结束 P0–R2D2.9 的诊断式推进，进入第二代架构重构（R3）  
> 总目标：形成一篇具有清晰统一故事线、方法创新明确、且在 NC/LP 下游任务上达到或超过 DiP 等强基线的 MAG 表示学习论文。  
> 工作原则：**不再继续修补 A0；不再无边界增加模块；把既有诊断结果转化为架构约束，集中增强“解耦后的 graph computation”。**

---

# 0. 本阶段最终决策

## 0.1 论文主线

本阶段采用类似 GMoPE 的研究组织方式，但不机械复制其模块数量：

\[
\text{成熟思想 A}+\text{成熟思想 B}
\rightarrow
\text{组合后暴露的矛盾}
\rightarrow
\text{MAG-specific 定制修补}
\rightarrow
\text{统一框架}
\]

对应到本工作：

### 成熟思想 A：Semantic Ownership

Common/Private disentanglement 将节点内部的多模态语义划分为：

\[
C_i,\quad P_{t,i},\quad P_{v,i}.
\]

其回答：

> **同一节点内部，不同模态信息“属于谁”？**

第一版 R3 直接复用现有 P0 Semantic Factorizer，不重新引入 OT/MMD，不同时重做 common/private 形成过程。

### 成熟思想 B：Conditional / Dynamic Message Computation

借鉴 ECC、GNN-FiLM、Dynamic Operator 等成熟思想的核心机制：

> **一条 graph message 的 transformation 应根据 source / target semantic state 动态变化，而不应永远是固定的 \(Wh_j\)。**

它回答：

> **source semantic state 应以什么功能方式作用于 target semantic state？**

### A+B 暴露的问题：Semantic Ownership–Propagation Mismatch

Common/Private 解耦后，一节点不再是单一 homogeneous hidden state，而是多个具有不同 semantic ownership 的 states。

此时：

1. **独立 factor-wise propagation**
   \[
   C\to C,\quad P_t\to P_t,\quad P_v\to P_v
   \]
   能保持 ownership，但遗漏已被 R2-0A 证明存在的 off-diagonal cross-factor graph information。

2. **Early fusion 后统一传播**
   会重新抹平 \(C/P_t/P_v\) 的 semantic ownership。

3. **邻居聚合后再 cross-factor interaction**
   会失去 neighbor-level source-target conditional structure；历史上也反复出现“信息存在但 E2E 增益很小”。

因此核心矛盾是：

\[
\boxed{
\text{Semantic Ownership} \not\Rightarrow \text{Semantic Isolation}
}
\]

以及：

\[
\boxed{
\text{Graph Connectivity} \not\Rightarrow \text{Uniform Semantic Propagation}
}
\]

### MAG-specific 定制修补：Ownership-Structured Semantic Transition

不再把新模型描述成：

> P0 + 三路 GNN + Cross-Factor Residual

而是：

> **将 Semantic Ownership 直接定义为 graph message passing 的状态空间，并把每一条 graph edge 重新定义为 ownership states 之间的结构化 functional transition。**

暂定工作名：

**R3 / Ownership-Structured Semantic Transition Network**

论文名称后续再定。

---

# 1. 从 P0–R2D2.9 继承的设计约束

R3 不继承 A0 的具体模块，只继承实验事实。

## 1.1 必须保留

### A. Semantic Ownership

\[
X_t,X_v\rightarrow C,P_t,P_v.
\]

P0 已经证明这些 factors 不是完全任意的切分，并表现出不同的图效用。

### B. Cross-factor graph information

R2-0A 已发现大量稳定的 off-diagonal 信息，例如：

\[
C_j\rightarrow P_{t,i},
\qquad
P_{v,j}\rightarrow P_{t,i}.
\]

因此 R3 不能只允许 diagonal propagation。

### C. Pre-aggregation conditional computation

历史最重要的 architecture clue：

\[
\text{source-target semantic computation}
\]

应发生在：

\[
\text{MESSAGE before AGGREGATE}.
\]

R3 核心计算必须在 edge message 被邻居聚合之前执行。

### D. Structured multi-scale information

保留：

\[
H^{(0)},H^{(1)},H^{(2)}
\]

等 intermediate states，但不再固定某 factor 必须走 2-hop，也不再增加 hop router。

### E. Co-adaptation / gradient coupling 风险

D1.5/G1 等已经说明：

- 新分支可有 forward value，但 joint training 可能破坏收益；
- zero-init 整个 branch 会造成 gradient starvation；
- residual / normalization 必须保证 exact identity 语义清楚。

因此 R3 从第一天就做 gradient / activation audit。

---

## 1.2 第一版明确不进入核心模型

以下机制当前均不作为 R3-v1 核心：

- A0 的 K=4 topology relation prototypes；
- complex \(\Gamma\)；
- OFR；
- explicit homophily / heterophily edge taxonomy；
- similarity / difference 双图；
- pseudo nodes；
- MoE；
- spectral branch；
- global Transformer；
- neighbor Attention；
- source-factor Attention；
- hop Attention；
- Transformer final fusion；
- 多层 scalar gates；
- 9 套互相独立的 \(a\to b\) MLP。

理由：它们当前都不是核心科学问题的必要条件。

---

# 2. 最终冻结的 R3-v1 模型架构

---

## 2.1 Stage I — Semantic Ownership Formation

复用 P0：

\[
X_i^t,X_i^v
\rightarrow
H_i^{C,(0)},H_i^{P_t,(0)},H_i^{P_v,(0)}.
\]

记：

\[
\mathbf H_i^{(0)}
=
[
H_i^{C,(0)},
H_i^{P_t,(0)},
H_i^{P_v,(0)}
].
\]

保留 P0 原有辅助目标：

\[
L_{\text{common}},
\quad
L_{\text{orth}},
\quad
L_{\text{recon}}.
\]

第一轮不修改 common averaging / P0 factorizer；Common refinement 留到 R3 architecture family 已经站住后再单独判断。

---

## 2.2 Semantic Ownership Lifting

概念上，把原始节点 \(i\) 从一个 hidden state：

\[
h_i
\]

提升成三个 semantic ownership states：

\[
(i,C),\quad(i,P_t),\quad(i,P_v).
\]

代码中不真的构造三倍节点图，只将 hidden state 组织为：

```text
H: [num_nodes, 3, hidden_dim]
factor index:
0 = C
1 = Pt
2 = Pv
```

原 graph edge：

\[
j\rightarrow i
\]

提供 interaction support。

Semantic Ownership 决定：

\[
a,b\in\{C,P_t,P_v\}.
\]

于是每条 graph edge 潜在诱导一个 \(3\times3\) semantic transition system：

\[
\mathcal T_{ji}
=
\begin{bmatrix}
T^{C\to C} & T^{P_t\to C} & T^{P_v\to C}\\
T^{C\to P_t} & T^{P_t\to P_t} & T^{P_v\to P_t}\\
T^{C\to P_v} & T^{P_t\to P_v} & T^{P_v\to P_v}
\end{bmatrix}.
\]

这里不是 9 套独立网络。

---

# 3. Ownership-Structured Semantic Transition Layer

这是论文真正的核心计算 primitive。

设第 \(\ell\) 层：

\[
H_i^{a,\ell},
\qquad
a\in\mathcal A=\{C,P_t,P_v\}.
\]

---

## 3.1 Same-node Semantic Context：只做 conditioning

利用历史 R2-0C 的 same-node interaction signal，但不额外建立 Semantic Interaction Branch。

定义：

\[
S_i^\ell
=
\phi_s
[
H_i^{C,\ell}
\Vert
H_i^{P_t,\ell}
\Vert
H_i^{P_v,\ell}
].
\]

约束：

- \(S_i\) 不是第四个 factor；
- \(S_i\) 不直接进入 final fusion；
- \(S_i\) 不替换任何 \(C/P_t/P_v\)；
- 它只 condition 后续 transition operator。

其语义是：

> target node 当前整体 multimodal semantic context。

第一轮提供 config 开关：

```yaml
use_same_node_context: true/false
```

---

## 3.2 Semantic Space → Relational Space

为了避免直接假定 \(C/P_t/P_v\) 处于完全可互换的坐标空间，所有 functional transfer 先进入共享 relational space。

对 source factor \(a\)：

\[
v_j^a = V_a H_j^a.
\]

对 target factor \(b\)：

\[
q_i^b = Q_b H_i^b.
\]

其中：

\[
V_a,Q_b:\mathbb R^{d_h}\rightarrow\mathbb R^{d_r}.
\]

建议初始：

```yaml
hidden_dim: 256
relation_dim: 128
factor_id_dim: 16
```

如果显存压力明显：

```yaml
relation_dim: 64
```

作为首个降配选项。

---

## 3.3 Diagonal Transition：Ownership-Preserving Propagation

对于：

\[
a=b,
\]

使用稳定的 same-ownership structural propagation：

\[
m_{ji}^{b\to b}
=
D_b^{\text{diag}}
W_b^{\text{diag}}
v_j^b.
\]

其作用：

\[
\boxed{
\text{ownership-preserving structural contextualization}
}
\]

第一版 diagonal 不使用 dynamic basis，不引入额外 Attention。

这不是“另外一个 GNN 模块”，而是 semantic transition operator 的 diagonal block。

---

## 3.4 Off-diagonal Transition：Functional Cross-Ownership Transfer

对于：

\[
a\neq b,
\]

需要回答：

> source factor \(a\) 在当前 target factor \(b\) 与 target multimodal context 下，应该执行何种 transformation？

构造 relational descriptor：

\[
r_{ji}^{ab}
=
\phi_r
[
q_i^b
\Vert
v_j^a
\Vert
\hat S_i
\Vert
e_a
\Vert
e_b
],
\]

其中：

\[
\hat S_i=P_sS_i
\]

映射到合适维度，\(e_a,e_b\) 为 factor identity embedding。

第一版不要强制加入：

\[
q\odot k,\quad |q-k|
\]

等额外手工 interaction features。

只有后续证据需要时才升级。

---

# 4. Dynamic Functional Basis Operator

这是 R3 性能版的核心函数族。

---

## 4.1 为什么不是 scalar gate

scalar gate：

\[
g_{ji}x
\]

主要改变“多少”。

R3 要回答的是：

> **信息应该如何发生功能变换。**

因此使用 feature transformation basis。

---

## 4.2 为什么不是 A0 的 K=4 relation

A0 的 K=4 在回答：

> edge 属于什么 topology relation？

R3 basis 回答：

> **这一条 semantic transfer 应执行什么 functional transformation？**

即：

\[
\text{Relation Type}\neq\text{Operator Function}.
\]

---

## 4.3 推荐实现：Low-Rank Dynamic Functional Basis

设置：

```yaml
num_bases: 4
basis_rank: 16
```

共享 basis：

\[
\mathcal B_r(x)
=
A_r(B_r x),
\qquad
r=1,\dots,R.
\]

其中：

\[
B_r:\mathbb R^{d_r}\rightarrow\mathbb R^{d_{\text{rank}}},
\]

\[
A_r:\mathbb R^{d_{\text{rank}}}\rightarrow\mathbb R^{d_r}.
\]

根据 relational descriptor：

\[
\omega_{ji}^{ab}
=
\operatorname{softmax}
(
g_\omega(r_{ji}^{ab})
)
\in\mathbb R^R.
\]

功能 message：

\[
\tilde m_{ji}^{a\to b}
=
\sum_{r=1}^R
\omega_{ji,r}^{ab}
\mathcal B_r(v_j^a).
\]

最后 target-specific decode：

\[
m_{ji}^{a\to b}
=
D_b^{\text{cross}}
(
\tilde m_{ji}^{a\to b}
).
\]

这实现：

\[
H_j^a
\rightarrow
\mathcal R
\rightarrow
H_i^b.
\]

---

## 4.4 训练稳定性：off-diagonal small-but-nonzero initialization

历史上出现过：

- semantic residual 过强，rewrite ownership；
- zero-init branch 导致 gradient starvation。

因此采用：

```yaml
offdiag_init_scale: 0.1
```

原则：

\[
\epsilon_{\text{off}}>0
\]

且不等于 0。

例如：

\[
m_{ji}^{a\to b}
\leftarrow
\epsilon_{\text{off}}
m_{ji}^{a\to b},
\qquad
\epsilon_{\text{off}}=0.1.
\]

第一版可设为：

- 固定 0.1；或
- learnable per-layer scalar，初始化 0.1。

推荐先用 **learnable per-layer scalar**，但它只是 LayerScale/稳定化参数，不作为论文 gate 创新。

必须记录其训练轨迹。

---

# 5. Pre-Aggregation Message → Simple Neighbor Aggregation

R3-v1 不使用 neighbor attention。

对于每个 source-target channel：

\[
\bar m_i^{a\to b}
=
\frac{1}{|\mathcal N(i)|}
\sum_{j\in\mathcal N(i)}
m_{ji}^{a\to b}.
\]

核心条件计算：

\[
\mathcal O_{\theta_{ji}^{ab}}
\]

已经发生在聚合之前。

这一版集中验证：

\[
\boxed{
\text{how-to-transform}
}
\]

而不是：

\[
\boxed{
\text{which-neighbor}
}
\]

如果后续明确发现 neighbor composition 是 bottleneck，再做 weighted aggregation；R3-v1 不做。

---

# 6. Source-Channel Preservation

对于每个 target factor \(b\)，不要提前将三个 source channels 平均。

保留：

\[
\bar m_i^{C\to b},
\quad
\bar m_i^{P_t\to b},
\quad
\bar m_i^{P_v\to b}.
\]

构造：

\[
M_i^b
=
[
\bar m_i^{C\to b}
\Vert
\bar m_i^{P_t\to b}
\Vert
\bar m_i^{P_v\to b}
].
\]

然后：

\[
\Delta H_i^b
=
U_b
\left(
\operatorname{LN}
[
H_i^b
\Vert
M_i^b
]
\right).
\]

这里 source identity 一直保留到 target-specific update。

不开 source attention，不做 source gate。

---

# 7. Ownership-Preserving Target-State Update

使用 pre-LN residual：

\[
H_i^{b,\ell+1}
=
H_i^{b,\ell}
+
\eta_\ell
\Delta H_i^b.
\]

建议：

```yaml
layer_scale_init: 0.1
```

关键约束：

- \(\eta_\ell\neq0\)；
- residual 禁止 post-add 新 LayerNorm 导致 “scale=0 但输出仍改变”；
- scale=0 的测试模式必须产生 exact identity（用于单元测试）；
- off-diagonal disable 后必须严格退化为 diagonal semantic-state propagation。

---

# 8. 两层 Iterative Semantic-State Evolution

第一版：

```yaml
num_transition_layers: 2
```

数据流：

\[
\mathbf H^{(0)}
\rightarrow
\mathbf H^{(1)}
\rightarrow
\mathbf H^{(2)}.
\]

每一层重新计算：

- same-node context；
- source/target relational descriptor；
- off-diagonal dynamic functional transition；
- source-wise aggregation；
- target ownership update。

因此 graph computation 是持续的 semantic-state evolution，而不是 side branch。

---

# 9. Multi-Scale State Retention

保留：

\[
H^{b,(0)},
\quad
H^{b,(1)},
\quad
H^{b,(2)}.
\]

对于每个 factor：

\[
\bar H_i^b
=
M_b
[
H_i^{b,(0)}
\Vert
H_i^{b,(1)}
\Vert
H_i^{b,(2)}
].
\]

第一版：

```yaml
multi_scale: concat
```

使用 Linear / 2-layer MLP 压回 `hidden_dim`。

不使用：

- hop Attention；
- hop gate；
- Pt-specific 2-hop rule。

它是 supporting architecture property，不包装为主要创新。

---

# 10. Final Fusion

得到：

\[
\bar C_i,\quad
\bar P_{t,i},\quad
\bar P_{v,i}.
\]

直接：

\[
z_i
=
MLP_f
[
\bar C_i
\Vert
\bar P_{t,i}
\Vert
\bar P_{v,i}
].
\]

第一版不使用 Transformer / Attention fusion。

原则：

> 真正强的 computation 应发生在 graph propagation 中，而不是依赖 final fusion 修补。

---

# 11. R3-v1 完整数据流

```text
Text / Image
    ↓
P0 Semantic Ownership Factorization
    ↓
C⁰ / Pt⁰ / Pv⁰
    ↓
┌─────────────────────────────────────────────────┐
│ Ownership-Structured Semantic Transition Layer │ × 2
│                                                 │
│ Same-node context (conditioning only)           │
│     ↓                                           │
│ Semantic → Relational projections               │
│     ↓                                           │
│ 3×3 semantic-state transition                   │
│   diagonal: ownership-preserving propagation    │
│   off-diagonal: dynamic functional basis        │
│     ↓                                           │
│ Pre-aggregation functional messages             │
│     ↓                                           │
│ Mean neighbor aggregation                       │
│     ↓                                           │
│ Preserve C→b / Pt→b / Pv→b source channels     │
│     ↓                                           │
│ Target-specific Pre-LN residual update           │
└─────────────────────────────────────────────────┘
    ↓
C¹/Pt¹/Pv¹ → C²/Pt²/Pv²
    ↓
Factor-wise 0/1/2-state retention
    ↓
C̄ / P̄t / P̄v
    ↓
Simple concat + MLP
    ↓
z
    ↓
NC / LP
```

---

# 12. 暂不加入 Exposure Gate

附件建议保留一个 Local/Graph Exposure：

\[
\rho_i^b.
\]

R3-v1 **暂不默认加入**。

原因：

1. 过去 exposure 相关结果存在高值/接近饱和现象；
2. 其独立收益未达到 Semantic Ownership / cross-factor / pre-aggregation 那样的证据等级；
3. 当前主线需要集中于 transition operator，而不是再次回到 gate 系统。

仅当后续出现以下诊断时再进入 R3-2：

- graph/off-diagonal update norm 长期过大；
- factors 明显 re-entangle；
- 某些数据集需要大量 Local/No-Transport；
- learned operator 明显把所有邻居都强制写入 target state。

到时再添加唯一的 target graph exposure：

\[
\rho_i^b
=
\sigma g_\rho(H_i^b,M_i^b).
\]

不增加其他 gate。

---

# 13. 论文故事线冻结版

## 13.1 Problem

Common/Private disentanglement 解决了：

> What semantic information does a multimodal node own?

但没有解决：

> How should ownership-specific semantics interact across graph connections?

---

## 13.2 Gap：Semantic Ownership–Propagation Mismatch

传统 GNN 默认：

\[
\text{one node}=\text{one homogeneous hidden state}.
\]

而 disentanglement 产生：

\[
\text{one node}=\{C,P_t,P_v\}.
\]

于是：

- independent propagation → semantic isolation；
- early fusion → ownership destruction；
- post-aggregation interaction → neighbor-level conditional information loss。

---

## 13.3 Key Insight

\[
\boxed{
\text{Semantic ownership should be the state space of graph message passing itself.}
}
\]

Topology 只定义：

\[
\textbf{WHERE}
\]

interaction 可以发生；

Semantic Ownership 定义：

\[
\textbf{WHAT}
\]

信息在传；

Functional Transition 定义：

\[
\textbf{HOW}
\]

source semantic state 作用于 target semantic state。

---

## 13.4 Method

提出：

\[
\boxed{
\text{Ownership-Structured Semantic Transition}
}
\]

每条 graph edge 对 \(C/P_t/P_v\) 诱导一个结构化 semantic-state transition：

- diagonal transitions：保持 ownership-specific graph contextualization；
- off-diagonal transitions：在 aggregation 前通过 dynamic functional basis 执行 target-conditioned cross-ownership transfer；
- relational bridge：在 relational space 中发生交互，再写回 target ownership space；
- target residual state update：保持 semantic state identity。

---

## 13.5 预期 Contributions

### Contribution 1 — Problem formulation

提出并系统研究 MAG 中的：

**Semantic Ownership–Propagation Mismatch**。

### Contribution 2 — Ownership-Structured Graph Operator

将 conventional single-state message passing 扩展为：

\[
3\times3
\]

common/private ownership-state transition system。

### Contribution 3 — Functional Relational Transfer

通过 shared low-rank dynamic functional bases，在 aggregation 前根据 source/target semantics 动态改变 message transformation，而不是只给 edge 一个 scalar weight。

### Contribution 4 — Empirical evidence

通过：

- P0/R2 diagnostic motivation；
- main benchmark；
- operator ablation；
- transition heatmaps；
- basis utilization；
- pre/post comparison；
- NC/LP generalization；

验证方法的有效性。

最终论文贡献条数可以压缩为 3 条；这里保留 4 条作为实验组织草案。

---

# 14. R3 配置设计

推荐新配置：

`configs/model/biaxis_r3.yaml`

核心字段：

```yaml
name: biaxis_r3

# P0
hidden_dim: 256
lambda_common: 0.02
lambda_orth: 0.01
lambda_recon: 0.30

# transition
num_transition_layers: 2
relation_dim: 128
factor_id_dim: 16
transition_mode: basis          # diagonal | static | film | basis
cross_factor: true
use_dual_space: true
use_same_node_context: true
preserve_source_channels: true

# dynamic basis
num_bases: 4
basis_rank: 16
router_hidden_dim: 128
offdiag_init_scale: 0.1
layer_scale_init: 0.1

# aggregation
neighbor_aggregation: mean      # v1 only mean

# readout
multi_scale: concat             # last | concat
fusion: concat_mlp

# optional, default OFF
use_exposure: false

# logging / debug
log_transition_stats: true
log_basis_stats: true
log_grad_stats: false
edge_chunk_size: 200000
```

需要允许命令行覆盖所有字段。

---

# 15. 代码结构建议

仓库当前已经存在：

- `src/models/biaxis_p0.py`
- `src/models/biaxis_final.py`
- `src/models/biaxis_cort.py`
- 对应 components；
- Hydra configs；
- `src/tasks/nc.py` / `lp.py`；
- `forward -> (z, None, None, aux_loss, aux_info)` 接口。

R3 不应破坏旧模型。

推荐新增：

```text
src/models/
├── biaxis_r3.py
└── biaxis_r3_components.py

configs/model/
└── biaxis_r3.yaml

tests/
├── test_biaxis_r3_shapes.py
├── test_biaxis_r3_identity.py
├── test_biaxis_r3_gradients.py
└── test_biaxis_r3_inference.py

scripts/
├── run_r3_challenge.py
└── summarize_r3.py
```

若现有 factory 是动态 import，则尽量不修改 factory。

---

# 16. 工程实现硬约束

## 16.1 不复制七个模型文件

所有 R3 variants 必须共用一个代码路径，通过 config 开关控制。

## 16.2 不修改历史模型行为

以下模型结果必须完全不受影响：

- biaxis_final/A0；
- biaxis_cort；
- biaxis_r2_*；
- existing baselines。

## 16.3 Forward 接口必须保持项目 contract

```python
forward(x, edge_index)
    -> (z, None, None, aux_loss, aux_info)
```

`out_dim` 必须正确。

## 16.4 NC/LP 都必须可用

R3 作为 representation encoder，不在模型内部绑定 NC classifier。

## 16.5 inference 必须与 eval forward 逻辑一致

在 tiny graph 上：

```text
eval forward == inference
```

允许浮点微差。

## 16.6 禁止 zero-init starvation

任何 residual scale / offdiag scale：

```text
init > 0
```

## 16.7 禁止 hidden LayerNorm identity confound

如果设置：

```text
layer_scale = 0
```

则 transition update 必须是 exact identity。

不要使用：

```text
LN(H + 0 * Delta)
```

作为“identity”，因为它不是 identity。

---

# 17. Dynamic Basis 的高效实现要求

严禁为每条 edge 显式 materialize：

```text
[E, d_rel, d_rel]
```

矩阵。

推荐：

1. 对 edge chunk 取 source relational value：
   ```python
   v_src = v_nodes[src]
   ```

2. 对每个 basis 做 low-rank transform：
   ```python
   z_r = A_r(B_r(v_src))
   ```

3. router 得到：
   ```python
   omega: [E_chunk, R]
   ```

4. 混合：
   ```python
   message = sum_r omega[:, r:r+1] * z_r
   ```

5. 通过 target decoder；
6. `index_add_` / scatter 到 target；
7. chunk 结束立即释放 edge-level tensors。

第一版建议：

```yaml
edge_chunk_size: 100000 ~ 200000
```

可自动调节。

---

# 18. 必须记录的 aux_info

在 P0 原有日志基础上新增。

## 18.1 Transition magnitude

每层、每 target factor：

```text
r3_l1_diag_norm_c
r3_l1_diag_norm_pt
r3_l1_diag_norm_pv

r3_l1_offdiag_norm_c
r3_l1_offdiag_norm_pt
r3_l1_offdiag_norm_pv

r3_l1_offdiag_diag_ratio_c
...
```

第二层同理。

## 18.2 Source-target channel strength

至少记录 9 个 channel 的平均 aggregated norm：

```text
C->C
Pt->C
Pv->C
C->Pt
Pt->Pt
Pv->Pt
C->Pv
Pt->Pv
Pv->Pv
```

用于后续 transition heatmap。

## 18.3 Basis utilization

每层记录：

```text
basis_mean_weight_r0...rR
basis_entropy
basis_top1_occupancy_r0...rR
```

最好可按 `a->b` 细分，但注意日志规模。

## 18.4 Ownership preservation

每层记录：

```text
cos(C, Pt)
cos(C, Pv)
cos(Pt, Pv)

norm(C)
norm(Pt)
norm(Pv)
```

以及相对 P0：

```text
delta_norm / state_norm
```

## 18.5 Stability

记录：

```text
layer_scale
offdiag_scale
NaN/Inf
max activation norm
```

## 18.6 Gradient audit（debug 模式）

只在 smoke / diagnostic run 打开：

```text
grad_factorizer
grad_diag
grad_basis
grad_router
grad_target_update
grad_fusion
```

不得每个正式 run 全量保存巨大梯度。

---

# 19. R3-0：实现与正确性阶段

目标：

> 不看性能，先证明新的 computation graph 实现正确、可训练、没有历史 G1 类结构性 bug。

---

## R3-0A — 代码实现

实现：

- P0 factorizer 复用；
- Ownership state tensor；
- same-node conditioner；
- relational projections；
- diagonal path；
- static offdiag；
- FiLM offdiag；
- dynamic basis offdiag；
- source-channel-preserving aggregation；
- target pre-LN residual update；
- 2-layer state evolution；
- multi-scale concat；
- final fusion；
- aux stats。

---

## R3-0B — 单元测试

必须通过：

### T1 Shape

随机 tiny graph：

```text
N = 32
E ≈ 100
```

检查每层 shape。

### T2 Diagonal-only degeneration

```yaml
cross_factor=false
```

必须完全没有 off-diagonal contribution。

### T3 Exact identity

手动：

```text
layer_scale = 0
```

则 layer 输出必须：

```text
H_out == H_in
```

容差 `1e-6`。

### T4 Edge-order invariance

打乱 `edge_index` 顺序，Mean aggregation 输出应一致。

### T5 Gradient audit

一次 backward 后：

- basis params nonzero grad；
- router nonzero grad；
- target update nonzero grad；
- diagonal path nonzero grad；
- factorizer（若未 freeze）nonzero grad。

禁止“has_grad=true 但 grad_norm=0”的整支路现象。

### T6 Offdiag initialization

初始：

```text
0 < offdiag/diag ratio << 1
```

建议约 0.02–0.30，而不是 0 或 >1。

### T7 Forward / inference parity

eval 模式 tiny graph 上一致。

### T8 Existing tests

```bash
pytest tests/
```

全部通过。

---

## R3-0C — Smoke run

Movies：

```bash
python -m src.main \
  dataset=Movies task=nc model=biaxis_r3 \
  num_runs=1 seed=42 \
  task.epochs=3 \
  task.evaluate_test=false
```

目标：

- 无 OOM；
- loss 正常；
- aux stats 正常；
- basis 非全 uniform / 非瞬间单 basis collapse；
- gradients 正常；
- 输出目录和 results.json 正常。

---

# 20. R3-1：Challenge Set 架构生命力实验

只跑：

\[
\boxed{
Movies,\ Toys,\ Grocery
}
\]

因为这三个正是当前 A0 仍明显落后 strongest baseline 的关键数据集。

Seeds：

\[
42,43,44.
\]

**R3-1 全部 Val-only，不访问 Test。**

---

# 21. R3-1 核心 Variant Matrix

本阶段不是局部 GO/NO-GO，而是完整跑完 nested system variants。

---

## V0 — SEP / Block-Diagonal Ownership Propagation

```yaml
cross_factor: false
transition_mode: diagonal
use_same_node_context: false
multi_scale: last
```

含义：

\[
C\to C,\quad Pt\to Pt,\quad Pv\to Pv.
\]

回答：

> Semantic Ownership + 独立 graph propagation 能达到什么水平？

---

## V1 — STATIC / Static Structured Transition

```yaml
cross_factor: true
transition_mode: static
use_dual_space: true
use_same_node_context: false
preserve_source_channels: true
multi_scale: last
```

允许 off-diagonal，但 transformation 不根据 edge/source-target state 动态变化。

回答：

> **仅仅开放 cross-factor connectivity 是否足够？**

---

## V2 — DIRECT-DYN / Naive A+B

```yaml
cross_factor: true
transition_mode: basis
use_dual_space: false
use_same_node_context: false
preserve_source_channels: false
multi_scale: last
```

语义：

> Common/Private + Dynamic Message Passing 的直接组合。

直接在 semantic hidden space 上做 conditional dynamic operator，并简单整合 source channels。

这是论文中最重要的 “Naive A+B” control。

---

## V3 — OST / Custom Repair Core

```yaml
cross_factor: true
transition_mode: basis
use_dual_space: true
use_same_node_context: false
preserve_source_channels: true
multi_scale: last
```

加入：

- Semantic → Relational → Target dual-space bridge；
- source-channel preservation；
- target-specific ownership update；
- small nonzero offdiag init。

这是：

\[
\boxed{
\text{Ownership-Structured Semantic Transition Core}
}
\]

回答：

> 针对 A+B 暴露的 ownership–propagation mismatch，结构性修补是否有效？

---

## V4 — OST+C / Context-Conditioned

V3 +

```yaml
use_same_node_context: true
```

回答：

> 历史 same-node interaction information 能否通过 conditioning cross-node computation 转化为 E2E utility？

---

## V5 — FULL

V4 +

```yaml
multi_scale: concat
```

即完整候选模型。

---

## V6 — FILM（Operator control）

在 V4/V5 外围架构完全相同的情况下：

```yaml
transition_mode: film
```

只需先在 challenge set 做。

回答：

> dynamic basis 是否真正优于轻量 feature-wise conditional transform？

V6 可以在 R3-1 与 V0–V5 一起跑，也可以在 V5 有明显生命力后紧接着跑；但正式进入 R3-2 前必须完成。

---

# 22. R3-1 实验规模

核心：

```text
V0–V5
× Movies/Toys/Grocery
× seeds 42/43/44
= 54 runs
```

FiLM control：

```text
1 variant
× 3 datasets
× 3 seeds
= 9 runs
```

总计最多：

```text
63 critical runs
```

允许多卡/AI 自动并行，但每个结果必须独立保存配置快照。

---

# 23. R3-1 评价指标

NC：

- Val Accuracy；
- Val Macro-F1；
- best epoch；
- 3-seed mean ± std。

同时汇总：

- params；
- peak GPU memory；
- train sec/epoch；
- transition statistics。

第一阶段禁止以 Test 结果进行 architecture selection。

---

# 24. R3-1 GO / NO-GO

不要再被 +0.1～0.2pp 的局部增量满足。

---

## Strong GO

满足大部分：

1. FULL 在 M/T/G 平均 Acc 相比 A0：
   \[
   \gtrsim +0.8\sim1.0\text{pp}
   \]
2. 至少 2/3 challenge datasets 追平或超过当前 DiP；
3. Macro-F1 不以明显下降换 Acc；
4. seed 方差可控；
5. V3/V4/V5 至少一个明显优于 V2 Naive A+B；
6. off-diagonal messages 确实被使用且 ownership 没有快速 collapse。

直接进入 R3-2/3。

---

## GO

1. FULL 相对 A0 平均：
   \[
   \gtrsim +0.5\sim0.7\text{pp}
   \]
2. 与 DiP gap 明显缩小；
3. 3 个 challenge dataset 至少 2 个为正；
4. V3/V4 对 V2 有稳定机制性改善。

允许做一次核心 operator capacity 调优，然后重新确认。

---

## Weak / Review

如果：

\[
+0.2\sim0.5\text{pp}
\]

但仍明显低于 DiP：

不要增加外围模块。

先诊断：

- V0 是否本身过弱；
- offdiag norm 是否过小；
- basis 是否 collapse；
- direct vs dual-space；
- gradient conflict；
- target update 是否吞掉 source channels。

---

## NO-GO

FULL：

\[
<+0.2\sim0.3\text{pp vs A0}
\]

且仍明显输 DiP。

此时不要：

- 加 neighbor attention；
- 加 pseudo node；
- 加 MoE；
- 加 heterophily；
- 调 20 个 gate 温度。

直接回到：

\[
\boxed{
\mathcal O_\theta
}
\]

检查核心 transition operator 是否选错。

---

# 25. R3-1 结果解释树

## Case A：V0 已接近/超过 A0

说明：

> A0 的 K4/\(\Gamma\)/OFR 可以彻底退休。

这是好结果。

## Case B：V1 > V0

说明 cross-factor connectivity 本身有 E2E utility。

## Case C：V2 > V1

说明：

\[
\boxed{
\text{conditional functional transformation > static cross-factor connection}
}
\]

这是非常理想的论文证据。

## Case D：V3 > V2

这是最重要结果：

> Naive “A+B” 不足；Dual-space + ownership-preserving structured transition 的 MAG-specific repair 有实际价值。

这会使 GMoPE 式故事真正成立。

## Case E：V4 > V3

说明 same-node multimodal information 的正确用途是：

\[
\text{condition cross-node graph computation}
\]

而不是独立 semantic refiner。

## Case F：V5 > V4

说明 multi-scale state retention 能在新的 architecture family 中兑现历史 probe signal。

## Case G：FiLM ≈ Basis

优先保留 FiLM，模型更简单。

## Case H：Basis 明显 > FiLM

Basis 成为最终论文 operator。

---

# 26. R3-2：核心函数族与有限调优

只有 R3-1 至少达到 GO 才进入。

目标：

> 提升 architecture ceiling，不扩展科学主线。

---

## 26.1 允许调的参数

只调：

```text
relation_dim: 64 / 128 / 256
num_bases: 2 / 4 / 8
basis_rank: 8 / 16 / 32
num_transition_layers: 2 / 3
offdiag_init_scale: 0.05 / 0.1 / 0.2
hidden_dim: 256 / 384
dropout
lr
weight_decay
```

不允许同时全网格。

使用小型 sequential search。

---

## 26.2 推荐顺序

### Step 1 — Operator capacity

固定其他配置：

```text
R = 2 / 4 / 8
rank = 8 / 16 / 32
```

先 Movies/Toys/Grocery seed42 screen。

最优 2 个配置再跑 3 seeds。

### Step 2 — Relation dimension

```text
64 / 128 / 256
```

### Step 3 — Depth

```text
L=2 vs L=3
```

只有 L=2 已经 strong 才测试 3。

### Step 4 — Hidden width

只有性能仍受限且 operator utilization 正常时：

```text
256 -> 384
```

过去 generic width 失败，因此 width 不是第一优先级。

---

# 27. R3-2 可选 Exposure 进入条件

只有出现以下情况之一才测试：

1. offdiag/diag ratio 长期 >1；
2. factor cosine 在 layer 2 大幅逼近；
3. graph update norm > state norm；
4. 某 challenge dataset 因 graph update 明显退化；
5. Local/No-Transport 诊断重新显示强 dataset-dependent graph demand。

测试：

```yaml
use_exposure: false / true
```

只允许一个 target-level exposure。

若平均收益 <0.2pp 或不稳定，删除。

---

# 28. R3-2 训练 protocol：Co-adaptation control

先以 joint training 为基线。

若观察到：

- factorizer 与 transition gradient cosine 强负；
- factorizer loss 明显恶化；
- frozen-forward transition 有值、joint 后掉；
- seed 方差异常大；

再启用 staged training。

---

## 28.1 Staged protocol

### Warm-up

前：

```text
10–20 epochs
```

冻结 P0 factorizer，仅训练：

- transition layers；
- readout；
- task head。

### Joint fine-tuning

解冻 P0。

理想情况：

```text
lr_factorizer = 0.1 × lr_transition
```

如果当前 task runner 不支持 parameter-group scale，则：

1. 先只做 freeze/unfreeze；
2. 后续再增加 generic optional optimizer param-group hook；
3. 必须保证所有历史模型默认行为不变。

不在第一版直接引入 PCGrad。

---

# 29. R3-3：5 个 NC 数据集正式确认

架构与主要超参数冻结后跑：

\[
\boxed{
Movies,\ Toys,\ Grocery,\ ele-fashion,\ Reddit-S
}
\]

× seeds：

\[
42,43,44.
\]

此阶段允许 Test，因为 architecture 已经被冻结，开始做正式 benchmark。

主要比较：

- GCN；
- GraphSAGE；
- MMGCN；
- MGAT；
- DGF；
- DiP；
- LGMRec；
- A0；
- R3。

视项目已有 baseline 结果决定是否补其他模型。

---

# 30. R3-3 性能目标

目标不是“超过 A0”。

目标：

\[
\boxed{
\text{直接进入 strongest baseline 区间并尽可能超过 DiP}
}
\]

建议论文级 gate：

- 至少 3/5 NC datasets Top-1；
- 5 datasets average rank \(\le1.5\sim2.0\)；
- M/T/G 至少 2/3 超过 DiP；
- ele-fashion / Reddit-S 不显著倒退；
- Acc、Macro-F1 同时报告；
- 3-seed std 不显著劣于 strongest baseline。

若仅个别数据集落后 <0.2–0.3pp，但 average rank 更强，可以继续。

---

# 31. R3-4：LP 泛化验证

R3-3 通过后，冻结 architecture，不再根据 NC 结果重构模型。

开始 LP。

优先顺序：

### Phase LP-A

MAGB：

- Movies-LP；
- Toys-LP；
- Grocery-LP；
- Reddit-S-LP。

### Phase LP-B

MM-Graph：

- sports-copurchase；
- cloth-copurchase；
- books-lp。

指标：

- MRR；
- Hits@1；
- Hits@3；
- Hits@10。

所有正式比较使用同一 LP protocol、同一 filtered negative sampling、公平 message-passing graph。

大图必要时使用 layerwise inference / sampling，不改变 operator 定义。

---

# 32. R3-5：论文级完整实验

性能站住以后再补。

---

## 32.1 Main Benchmark

NC + LP 全任务。

## 32.2 Core Ablation

至少：

- w/o Semantic Ownership（early fused counterpart）；
- diagonal only；
- static transition；
- naive direct dynamic；
- w/o dual-space；
- source-channel mean vs preserve；
- w/o same-node context；
- last layer vs multi-scale；
- FiLM vs Basis。

## 32.3 Timing Ablation

参数尽量 matched：

\[
\text{Pre-Aggregation}
\quad vs\quad
\text{Post-Aggregation}.
\]

这是历史实验和论文主张的重要验证。

## 32.4 Transition Structure Analysis

画：

\[
3\times3
\]

semantic transition heatmap。

不要再画 K=4 relation prototype。

## 32.5 Basis Analysis

- basis usage；
- entropy；
- dataset/factor-conditioned occupancy；
- static vs dynamic。

## 32.6 Ownership Preservation

比较：

- P0；
- Naive A+B；
- R3 repair；

随 layer 变化的：

\[
sim(C,P_t),sim(C,P_v),sim(P_t,P_v).
\]

## 32.7 Efficiency

报告：

- Params；
- FLOPs/近似 complexity；
- peak GPU memory；
- training time/epoch；
- inference time。

目标不要求比 DiP 小，但应在单卡 RTX 3090 24GB 可稳定运行。

---

# 33. 结果文件与报告规范

每一批实验结束，生成：

```text
docs/r3/
├── R3_0_implementation_report.md
├── R3_1_challenge_results.csv
├── R3_1_challenge_report.md
├── R3_2_operator_tuning_report.md
├── R3_3_nc_benchmark.csv
├── R3_3_nc_report.md
├── R3_4_lp_benchmark.csv
└── R3_final_handoff.md
```

不要覆盖旧 R2 文档。

---

# 34. R3-1 Challenge Report 必须包含

## A. Environment

- git commit；
- branch；
- GPU；
- torch/PyG；
- command；
- config。

## B. 每个 variant × dataset × seed

- best val acc；
- best val F1；
- best epoch；
- params；
- peak memory；
- runtime。

## C. 3-seed summary

mean ± std。

## D. 对比

- vs A0；
- vs DiP；
- vs strongest current baseline。

## E. Mechanism stats

至少：

- diag/offdiag norm；
- 9-channel transition strength；
- basis entropy；
- layer 0/1/2 factor cosine；
- offdiag scale。

## F. Anomalies

- NaN；
- OOM；
- gradient zero；
- basis collapse；
- early stopping异常；
- seed instability。

## G. 不要写最终论文结论

报告只给：

- raw evidence；
- matched comparison；
- implementation facts。

将诊断解释留给下一轮我们共同分析。

---

# 35. 返给我分析时需要的材料

完成 R3-1 后请至少提供：

1. `R3_1_challenge_report.md`
2. `R3_1_challenge_results.csv`
3. 关键 `results.json`
4. 关键 3 个 seed 的 history CSV
5. FULL 模型的 aux transition stats
6. 当前 commit hash
7. 若有异常，附对应 log
8. R3 相关代码 diff 或仓库链接

届时重点分析：

- architecture family 是否 GO；
- A+B 是否成立；
- custom repair 是否真正贡献；
- Dynamic Basis 是否有用；
- same-node conditioner 是否留下；
- multi-scale 是否留下；
- 是否进入 5-dataset benchmark；
- 下一步是 operator capacity 升级还是训练优化。

---

# 36. 可直接交给 Codex / Claude Code 的完整 Prompt

下面 Prompt 建议在项目根目录、R3 新分支上执行。

---

## Prompt A：R3-0 架构实现 + 正确性审查

```text
你现在作为本项目的资深图学习/多模态图工程研究员工作。

仓库：
https://github.com/CrisRipper777/0901

目标：
不要继续修改 A0/CORT，也不要把新机制作为 A0 side branch。
请在现有统一 MAG baseline 框架中，从 P0 Semantic Ownership 出发实现全新的第二代 R3 模型：

Ownership-Structured Semantic Transition Network

核心研究逻辑：
1. P0 Common/Private factorization 已经成立，得到 C / Pt / Pv。
2. 传统 factor-wise graph propagation 只允许 diagonal C->C / Pt->Pt / Pv->Pv，会遗漏真实存在的 cross-factor graph information。
3. early fusion 会破坏 semantic ownership。
4. post-aggregation interaction 会丢失 neighbor-level source-target conditional structure。
5. 因此 R3 要把 C/Pt/Pv 作为 graph message passing 的原生 state space，并把每条 graph edge 建模为一个 3x3 semantic-state transition operator：
   - diagonal = ownership-preserving propagation
   - off-diagonal = pre-aggregation target-conditioned functional transfer
6. 不要实现 neighbor Attention、MoE、pseudo node、homophily/heterophily branch、K=4 relation、Gamma、OFR、Transformer fusion。

请先完整阅读：
- CLAUDE.md
- src/models/biaxis_p0.py
- src/models/biaxis_components.py
- src/models/biaxis_final.py
- src/models/biaxis_cort.py / components（只用于吸取历史接口/bug教训，不继承架构）
- src/models/factory.py
- src/tasks/nc.py
- src/tasks/lp.py
- src/tasks/inference.py
- configs/model/biaxis_p0.yaml
- configs/model/biaxis_final.yaml
- 现有 R2/R3 相关 analysis/docs，如果仓库中存在

先输出一份 1～2 页 implementation audit：
- 当前 model interface
- P0 factorizer 如何复用
- forward/inference/sampled training 的约束
- 你计划新增/最小修改哪些文件
- 任何可能的内存风险

然后实现。

====================
一、必须新增
====================

src/models/biaxis_r3.py
src/models/biaxis_r3_components.py
configs/model/biaxis_r3.yaml

必要时新增 tests 和 scripts，但不要复制多个 variant model 文件。

所有 variants 必须由同一个 R3 code path + config 开关产生。

保持项目接口：
Model(cfg, data_info)
forward(x, edge_index) -> (z, None, None, aux_loss, aux_info)
out_dim 正确
NC/LP 都能复用。

====================
二、P0 Semantic Ownership
====================

第一版直接复用现有 P0 factorizer，保持：
C / Pt / Pv
以及 common / orth / recon loss。

不要修改 P0 common averaging。
不要引入 OT/MMD。
不要重新调 factorizer。

内部建议统一成：
H: [N, 3, hidden_dim]
factor order = C, Pt, Pv

====================
三、Same-node context
====================

实现：
S_i = MLP([C_i || Pt_i || Pv_i])

但 S_i：
- 不是第四个 factor
- 不进入 final output
- 只 condition relational transition
- config: use_same_node_context

====================
四、Relational Space
====================

factor-specific source/target projection：
v_j^a = V_a H_j^a
q_i^b = Q_b H_i^b

hidden_dim 默认 256
relation_dim 默认 128
factor id embedding 默认 16

cross-factor functional interaction 在 relational space 中发生，再由 target decoder 写回 target ownership state。

====================
五、Diagonal transition
====================

对于 a=b：
使用稳定 static same-factor relational transform + mean graph aggregation。
不要 dynamic basis。
不要 attention。

这是 transition operator 的 diagonal block，而不是单独 GNN module。

====================
六、Off-diagonal transition
====================

对于 a!=b：
构造 descriptor：
r_ji^{ab} = MLP([q_i^b || v_j^a || projected S_i(optional) || e_a || e_b])

实现 3 种 operator backend：
1. static
2. film
3. basis

其中最终重点是 basis。

basis 使用低秩共享 functional bases：
R=4
rank=16

对于每个 edge/source-target pair：
omega = softmax(router(r_ji_ab))  # over functional bases, NOT neighbors

basis transform 不能显式生成 [E,d,d] matrix。
使用低秩：
B_r(x) = A_r(C_r(x))
然后：
m = sum_r omega_r * B_r(v_src)

再由 target-specific decoder D_b 写回 target space。

operator bases 在所有 a->b off-diagonal pairs 间共享；
source/target factor identities作为 condition。
禁止 6/9 套完全独立网络。

====================
七、small nonzero off-diagonal init
====================

offdiag contribution 初始 scale = 0.1（configurable）。
必须 >0，禁止 zero-init starvation。

允许 learnable per-layer LayerScale，但不要 node-wise gate。

====================
八、Pre-aggregation
====================

所有 source-target conditional transformation 必须发生在 neighbor aggregation 之前。

aggregation R3-v1 固定 mean。
不要 neighbor attention。

必须支持 edge chunking，避免把所有 6 个 off-diagonal pair 的大 edge tensor同时常驻显存。

建议 edge_chunk_size configurable，默认 200000。

====================
九、Preserve source channels
====================

对 target b：
分别聚合：
m(C->b), m(Pt->b), m(Pv->b)

保持三个 source channels 到 target-specific update。
不要提前 mean。
不要 source attention/gate。

target update:
Delta H_b = U_b(PreLN([H_b || m_Cb || m_Ptb || m_Pvb]))

H_b_next = H_b + eta_l * Delta H_b

eta_l 初始化 0.1，必须非零。

注意 exact identity：
如果 eta_l 人为设为 0，layer output 必须严格等于 input；
不要使用 LN(H + 0*Delta) 造成 identity confound。

====================
十、2-layer state evolution
====================

默认 L=2。
每层重新计算 context / transition / aggregation / update。

保存 H0,H1,H2。

multi_scale:
- last
- concat

concat 模式：
per-factor [H0||H1||H2] -> MLP -> hidden_dim

final:
[Cbar||Ptbar||Pvbar] -> simple MLP -> z

不要 Transformer fusion。

====================
十一、config variants
====================

必须支持：

cross_factor: true/false
transition_mode: diagonal/static/film/basis
use_dual_space: true/false
use_same_node_context: true/false
preserve_source_channels: true/false
multi_scale: last/concat
use_exposure: false  # 先只留接口，不实现或默认完全关闭也可以

====================
十二、aux_info / logging
====================

保留 P0 stats。
新增：
- layer-wise diag norm
- offdiag norm
- offdiag/diag ratio
- 9 source->target channel norms
- basis mean weights
- basis entropy
- basis top1 occupancy
- C/Pt/Pv pairwise cosine after each layer
- update/state norm ratio
- offdiag scale / layer scale

debug 模式记录主要 module gradient norm。

不要让正常正式 run 的 aux_info 巨大到影响性能。

====================
十三、必须新增测试
====================

1. shape test
2. diagonal-only test
3. exact identity test when layer_scale=0
4. edge-order invariance
5. gradient audit:
   basis/router/diag/update 必须 nonzero gradient
6. offdiag init ratio >0 且不要明显 >1
7. eval forward vs inference parity on tiny graph
8. pytest tests/ 全部通过
9. 确认历史模型默认结果路径/接口不受影响

====================
十四、禁止事项
====================

禁止：
- 修改/删除旧 R2/A0 代码
- 用 Test 结果进行 R3-1 model selection
- 自动加入 neighbor attention
- 自动加入 exposure gate
- 自动加入 MoE/pseudo nodes
- 为了涨点偷偷改 dataset split、evaluation、negative sampling、early stopping
- 让不同 variant 走完全不同训练协议
- 在结果不理想时自行继续增加新模块

====================
十五、实现完成后
====================

请生成：
docs/r3/R3_0_implementation_report.md

包含：
- 新增/修改文件
- 精确 architecture
- tensor shapes
- parameter count
- complexity / memory说明
- 所有 tests
- smoke run command
- gradient audit
- 已知限制
- git diff 摘要

先完成 R3-0，不要直接跑几十组正式实验。
```

---

# 37. Prompt B：R3-1 Challenge Set 批量实验

R3-0 报告确认无结构 bug 后再运行。

```text
继续当前 0901 仓库的 R3 工作。

现在不要修改 R3 架构。
任务是执行 R3-1 Challenge Set architecture viability experiment。

严格要求：
- 数据集只用 Movies / Toys / Grocery
- task=nc
- seeds = 42,43,44
- architecture selection 阶段必须 Val-only
- task.evaluate_test=false
- 所有 variants 使用相同 dataset split / task protocol / early stopping
- 不因前一个 variant 表现弱就停止后续 FULL
- 不做额外超参数搜索
- 保存完整 Hydra config snapshot、results、history、日志

运行以下 variants：

V0 SEP:
cross_factor=false
transition_mode=diagonal
use_same_node_context=false
multi_scale=last

V1 STATIC:
cross_factor=true
transition_mode=static
use_dual_space=true
use_same_node_context=false
preserve_source_channels=true
multi_scale=last

V2 DIRECT-DYN:
cross_factor=true
transition_mode=basis
use_dual_space=false
use_same_node_context=false
preserve_source_channels=false
multi_scale=last

V3 OST:
cross_factor=true
transition_mode=basis
use_dual_space=true
use_same_node_context=false
preserve_source_channels=true
multi_scale=last

V4 OST+C:
与 V3 相同，但 use_same_node_context=true

V5 FULL:
与 V4 相同，但 multi_scale=concat

V6 FILM:
与 V5 外围完全相同，但 transition_mode=film

每个 variant × dataset × seed 都执行。

另外加载/汇总当前 commit 上：
- A0 / biaxis_final
- DiP
- 当前 strongest baseline
的匹配结果；如果已有结果不要重复跑，必须验证协议/commit可比。

输出：
docs/r3/R3_1_challenge_results.csv
docs/r3/R3_1_challenge_report.md

CSV 至少包含：
variant,dataset,seed,best_val_acc,best_val_macro_f1,best_epoch,
params,peak_gpu_mem,sec_per_epoch,
diag_norm,offdiag_norm,offdiag_diag_ratio,
basis_entropy,
c_pt_cos,c_pv_cos,pt_pv_cos

Report 必须包含：
1. 3-seed mean±std
2. vs A0 delta
3. vs DiP delta
4. vs strongest delta
5. V1-V0 / V2-V1 / V3-V2 / V4-V3 / V5-V4 的 paired delta
6. 9-channel transition strength heatmap 数据表
7. basis usage
8. ownership cosine layer evolution
9. gradient/NaN/OOM/seed instability anomalies
10. 参数、显存、运行时间
11. 原始 evidence，不要擅自编论文故事

不要：
- 使用 Test
- 修改模型
- 因结果差再调参
- 加新模块

完成后停止，把报告和原始文件交给我进行下一轮诊断。
```

---

# 38. Prompt C：R3-1 结果整理要求

若代码智能体只负责实验执行，可额外给：

```text
请不要继续改模型。
基于 R3-1 所有原始 outputs，做纯结果汇总。

必须从 results.json / history.csv / logs 自动读取，不手抄数值。
所有 mean/std 用实际 seeds 42/43/44 计算。
输出：
- per-run raw table
- per-dataset mean±std
- macro average across M/T/G
- paired delta table
- rank table
- mechanism table
- efficiency table

检查：
- 是否每个 cell 3 seeds 齐全
- 是否任何 run 使用 Test
- 是否存在不同 config/split/epoch protocol
- 是否有 NaN/OOM/early-stop异常
- 是否存在 missing run

不要解释成因，只做 evidence report。
```

---

# 39. 当前阶段最重要的纪律

1. **不要再继续 architecture brainstorming。**
2. **R3-v1 先证明新 computation family 是否进入 DiP 性能区间。**
3. **FULL 必须整体跑完，不能再用“单模块不涨就停止 joint”规则。**
4. **若失败，优先升级 transition operator，不扩展外围模块。**
5. **所有架构选择先 Val-only。**
6. **正式 benchmark 才使用 Test。**
7. **P0 factorizer 第一轮保持固定设计，避免同时改变两个研究对象。**
8. **不为了论文故事保留无效模块。**
9. **Dynamic Basis 如果不胜 FiLM，就删。**
10. **Same-node context / multi-scale 如果没有稳定收益，也删。**
11. **最终论文只保留实验支持的机制。**

---

# 40. 当前预计的最终论文方法形态

如果 R3-1 / R3-2 结果符合预期，最终方法将收敛为：

\[
\boxed{
\text{Semantic Ownership Factorization}
+
\text{Ownership-Structured Functional State Transition}
+
\text{Ownership-Preserving Iterative State Evolution}
}
\]

其中：

- Semantic Ownership 是第一轴；
- Functional Semantic Transition 是第二轴；
- Dual-space + target-state-preserving update 是 A+B 后的 MAG-specific adaptation；
- Multi-scale retention / simple fusion 只是 supporting design；
- 不再回到 topology relation taxonomy。

最终一句话：

> **Instead of applying generic graph propagation after multimodal disentanglement, we treat semantic ownership as the state space of graph computation and model each graph edge as a structured, target-conditioned functional transition among common/private semantic states.**

---

# 41. 下一次返回数据后的诊断问题

完成 R3-1 后，不再重新从头讨论 Idea，只回答以下问题：

1. R3 architecture family 是否真正 GO？
2. V2 Naive A+B 相比 V0/V1 是否已有价值？
3. V3 Dual-space repair 是否稳定优于 V2？
4. Dynamic Basis 是否优于 FiLM？
5. same-node context 是否应保留？
6. multi-scale 是否应保留？
7. off-diagonal transfer 是否真正被模型使用？
8. factors 是否在传播中重新纠缠？
9. 主要瓶颈是 operator capacity、optimization 还是 backbone？
10. 是否已经足以进入 5-dataset NC benchmark？
11. 若未达到 DiP，下一步只允许改哪一个核心部位？

完成这些判断后再进入 R3-2。
