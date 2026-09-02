# Bi-Axis MAG — P0 阶段实现与实验推进文档
## 目标：验证 Factor-Dependent Neighborhood Utility 是否真实存在

> 项目基础：`CrisRipper777/MAG_baseline`  
> 当前阶段：P0（只验证研究问题，不实现 Relation / OT / Low-rank Operator）  
> 核心研究命题：  
> **Common/Private 解耦回答了同节点跨模态信息“属于谁”的问题，但并不决定图邻居对不同语义因子是否有用、以及使用图拓扑是否会产生相同收益。**

---

# 0. P0 的定位：这一阶段不做“最终模型”

P0 不是为了追求 SOTA，也不是为了证明 UOT、Relation Prototype、Tensor Operator 有效。

P0 只回答一个问题：

$$
\boxed{
\text{Do different semantic factors benefit differently from the same graph neighborhood?}
}
$$
即，在完成同节点跨模态语义解耦后：

$$
\mathcal F=\{C,P_t,P_v\}
$$
对于同一个图邻域 $ \mathcal N(i)$，是否存在：

$$
Utility_i^C(\mathcal N(i))
\neq
Utility_i^{P_t}(\mathcal N(i))
\neq
Utility_i^{P_v}(\mathcal N(i)).
$$
如果这个现象不存在，那么后续“Semantic Factor × Structural Relation”的二维框架缺少经验依据，P1 不应继续复杂化。

如果这个现象在多个数据集、多个 seed、NC/LP 两类任务上稳定存在，那么 P0 将直接形成论文中的 **Empirical Motivation / Observation**。

---

# 1. 本阶段数据集

## 1.1 Node Classification

正式 P0-NC 使用：

- Movies
- Toys
- Grocery
- ele-fashion
- Reddit-S

指标：

- Accuracy
- Macro-F1

P0 内部诊断主要使用 validation split；最终确认后再报告 test。

---

## 1.2 Link Prediction

正式 P0-LP 使用：

- sports-copurchase
- cloth-copurchase

指标：

- MRR
- Hits@1
- Hits@3
- Hits@10

LP 必须严格使用现有 task-safe protocol：

- message-passing graph 只使用训练边；
- validation/test positive edges 不得进入 encoder message graph；
- 使用现有 fixed negative / filtered negative 协议；
- 不为了 P0 重新生成数据划分。

---

# 2. P0 总体路线

P0 拆成四个子阶段：

```text
P0-A  Semantic Factorizer
      ↓
P0-B  Factorization Sanity
      ↓
P0-C  Factor-wise Edge/Neighborhood Diagnostics
      ↓
P0-D  Factor-wise Propagation Utility Probe
      ↓
GO / NO-GO
```

核心要求：

> P0 中不得实现 latent relation、OT、pseudo node、MoE、frequency decomposition、rewiring、relation-specific operator。

否则无法判断观察到的现象究竟来自语义解耦本身，还是后加模块制造出来的。

---

# 3. P0-A：实现最小 Semantic Factorizer

## 3.1 输入

现有 `MAG_baseline` 已经保留：

```python
data.x_t   # text feature
data.x_i   # image/visual feature
data.x     # concatenated joint feature
```

但当前 `Model.forward()` 接口仍主要接收：

```python
forward(x, edge_index)
```

因此第一版不要修改整个 task runner 的数据接口。

利用：

```python
data_info["text_dim"]
data_info["visual_dim"]
```

在模型内部对 `x` 切分：

```python
x_t = x[:, :text_dim]
x_v = x[:, text_dim:text_dim + visual_dim]
```

必须加入 assert：

```python
assert x.size(-1) >= text_dim + visual_dim
```

避免模态顺序错误。

---

## 3.2 网络结构

统一 hidden dimension：

```text
hidden_dim = 256
factor_dim = 128
```

### Modality Projection

$$
h_i^t=P_t(x_i^t), \qquad
h_i^v=P_v(x_i^v)
$$

建议：

```python
Linear(input_dim, 256)
LayerNorm(256)
GELU
Dropout
Linear(256, 256)
```

文本、视觉 projection 不共享参数。

---

### Common Encoder

Common encoder 参数共享：

$$
c_i^t=E_C(h_i^t)
$$

$$
c_i^v=E_C(h_i^v)
$$

推荐：

```python
shared_common_encoder = MLP(256 -> 128 -> 128)
```

得到：

```text
c_t: [N, 128]
c_v: [N, 128]
```

Common consensus：

$$
c_i=\frac{c_i^t+c_i^v}{2}
$$
第一版不要加 cross-attention、OT、prototype。

---

### Private Encoders

$$
p_i^t=E_t^P(h_i^t)
$$

$$
p_i^v=E_v^P(h_i^v)
$$

两个 private encoders 参数独立：

```python
private_text_encoder   = MLP(256 -> 128 -> 128)
private_visual_encoder = MLP(256 -> 128 -> 128)
```

输出：

```text
p_t: [N, 128]
p_v: [N, 128]
```

---

## 3.3 Local representation

P0 主任务 representation：

$$
z_i^{local}
=
F_{local}
[c_i\Vert p_i^t\Vert p_i^v].
$$
推荐：

```python
fusion = nn.Sequential(
    nn.Linear(128 * 3, 256),
    nn.LayerNorm(256),
    nn.GELU(),
    nn.Dropout(dropout),
)
```

输出：

```text
z_local: [N, 256]
```

设置：

```python
self.out_dim = 256
```

以兼容现有 NC/LP runner。

---

# 4. P0-A Loss 设计

本阶段建议：

$$
L =
L_{task}
+
\lambda_c L_{common}
+
\lambda_o L_{orth}
+
\lambda_r L_{rec}.
$$
`L_task` 仍由现有 task runner 负责。

模型只通过：

```python
aux_loss
```

返回：

$$
L_{aux}
=
\lambda_cL_{common}
+
\lambda_oL_{orth}
+
\lambda_rL_{rec}.
$$
注意现有 runner 外部还会乘：

```python
cfg.task.loss.aux_weight
```

因此必须避免“双重放大”。

推荐两种方案任选一种：

### 方案 A：模型内部设置各项 lambda，runner 的 `aux_weight=1.0`

推荐。

### 方案 B：模型返回未经缩放的 loss 总和，统一由 task 配置控制

不推荐，因为 common/orth/rec 相对权重仍需内部区分。

---

## 4.1 Common Consistency

$$
L_{common}
=
1-
\frac{1}{N}\sum_i
\cos(c_i^t,c_i^v)
$$

代码注意：

```python
F.normalize(..., dim=-1)
```

先规范化再点积。

---

## 4.2 Common–Private Separation

不要使用单样本 cosine 作为唯一 separation。

优先实现 batch cross-covariance：

对于：

$$
C_t\in\mathbb R^{B\times d},
P_t\in\mathbb R^{B\times d},
$$
中心化：

$$
\bar C_t=C_t-\mu(C_t)
$$

$$
\bar P_t=P_t-\mu(P_t)
$$

然后：

$$
Cov(C_t,P_t)=
\frac{\bar C_t^\top\bar P_t}{B-1}.
$$
损失：

$$
L_{orth}
=
\frac{\|Cov(C_t,P_t)\|_F^2}{d^2}
+
\frac{\|Cov(C_v,P_v)\|_F^2}{d^2}.
$$
如果 batch size 太小，则临时 fallback 到 cosine-overlap loss。

---

## 4.3 Reconstruction

加入轻量 decoder：

$$
\hat h_i^t=D_t[c_i^t\Vert p_i^t]
$$

$$
\hat h_i^v=D_v[c_i^v\Vert p_i^v].
$$

损失：

$$
L_{rec}
=
MSE(\hat h^t,h^t)
+
MSE(\hat h^v,h^v).
$$
作用：

- 防止 Private collapse；
- 防止所有信息全部塞进 Common；
- 保证解耦后的两部分仍能覆盖原始 modality information。

它不是最终论文创新，可以在后续阶段重新判断是否保留。

---

## 4.4 初始权重

第一轮使用：

```yaml
lambda_common: 0.1
lambda_orth: 0.01
lambda_recon: 0.1
```

不要在 P0 一开始大规模调参。

只有出现明显 collapse 再调整。

---

# 5. 建议代码结构

新增：

```text
src/models/
├── biaxis_p0.py
└── biaxis_components.py
```

建议组件：

```python
class ModalityProjector(nn.Module):
    ...

class SemanticFactorizer(nn.Module):
    ...

class ReconstructionHead(nn.Module):
    ...

class Model(nn.Module):
    ...
```

其中 `SemanticFactorizer` 提供统一接口：

```python
def forward(self, x_t, x_v):
    return {
        "h_t": h_t,
        "h_v": h_v,
        "c_t": c_t,
        "c_v": c_v,
        "c": c,
        "p_t": p_t,
        "p_v": p_v,
    }
```

主 `Model` 必须再提供：

```python
@torch.no_grad()
def encode_factors(self, x, edge_index=None):
    ...
```

返回：

```python
{
    "c": ...,
    "c_t": ...,
    "c_v": ...,
    "p_t": ...,
    "p_v": ...,
    "z_local": ...
}
```

这个接口后续 P0 diagnostics 必须复用，避免 diagnostics 再实现一套 factorizer。

---

# 6. Model.forward 接口

必须兼容现有框架：

```python
def forward(self, x, edge_index):
    factors = self._encode(x)
    z = self.fusion(torch.cat([
        factors["c"],
        factors["p_t"],
        factors["p_v"],
    ], dim=-1))

    aux_loss, aux_info = self._compute_aux(factors)

    return z, None, None, aux_loss, aux_info
```

P0 模型**不使用 edge_index**参与 representation learning。

这是刻意设计：

> P0-A 要先得到纯语义 Common/Private，不让 topology 提前污染 semantic factorization。

但 `edge_index` 参数保留以兼容现有 Model API。

---

# 7. aux_info：从第一天开始记录内部状态

至少输出：

```python
aux_info = {
    "p0_common_loss": ...,
    "p0_orth_loss": ...,
    "p0_recon_loss": ...,

    "p0_common_sim": ...,
    "p0_private_sim": ...,

    "p0_c_norm": ...,
    "p0_pt_norm": ...,
    "p0_pv_norm": ...,
}
```

如果计算成本允许：

```text
p0_cp_overlap_t
p0_cp_overlap_v
```

现有 task runner 已经支持汇总 `aux_info`，不要另写大量 `print()`。

---

# 8. P0-B：Factorization Sanity 检查

P0 的第一个判定不是 Accuracy。

要先确认：

$$
C,P_t,P_v
$$
确实学出了有意义且不塌缩的空间。

---

## 8.1 Common / Private Cross-modal Similarity

计算：

$$
S_C=
\frac1N
\sum_i\cos(c_i^t,c_i^v)
$$

$$
S_P=
\frac1N
\sum_i\cos(p_i^t,p_i^v).
$$

希望：

$$
S_C>S_P.
$$
但不要要求：

$$
S_C\rightarrow1
$$
因为过强 Common alignment 可能导致 collapse。

---

## 8.2 Common–Private Overlap

计算：

$$
O_t=
\|Cov(C_t,P_t)\|_F
$$

$$
O_v=
\|Cov(C_v,P_v)\|_F.
$$

训练过程中应该下降或保持较低。

---

## 8.3 Effective Rank

建议离线对：

```text
C
P_t
P_v
```

采样最多 10k 节点。

SVD：

$$
\sigma_1,\dots,\sigma_d.
$$
归一化：

$$
q_l=
\frac{\sigma_l}{\sum_j\sigma_j}.
$$
effective rank：

$$
r_{eff}
=
\exp
\left(
-\sum_lq_l\log q_l
\right).
$$
如果：

```text
effrank ≈ 1
```

说明 collapse。

不要每 epoch 做 SVD，只在 best checkpoint 后离线计算。

---

# 9. P0-C：Factor-dependent Edge Utility

这是第一组真正服务论文 Hypothesis 的诊断。

对所有 observed message-passing edges：

$$
(i,j)\in E
$$
分别计算：

$$
s_{ij}^C=
\cos(c_i,c_j)
$$

$$
s_{ij}^T=
\cos(p_i^t,p_j^t)
$$

$$
s_{ij}^V=
\cos(p_i^v,p_j^v).
$$

注意：

- 对无向图只保留一个 canonical direction 做统计；
- 大图可随机 sample 200k–1M edges；
- sample 固定 seed，确保不同模型可比较。

---

## 9.1 Spearman correlation

输出：

```text
rho_C_T
rho_C_V
rho_T_V
```

如果三者都非常接近 1：

> 不同 factor 对边的排序基本一致，二维假设风险很高。

---

## 9.2 Top-q Edge Jaccard

分别取：

```text
q = 10%, 20%
```

计算：

$$
J(C,T)
=
\frac{|Top_C\cap Top_T|}
{|Top_C\cup Top_T|}.
$$
输出：

```text
jaccard_top10_C_T
jaccard_top10_C_V
jaccard_top10_T_V

jaccard_top20_C_T
...
```

---

## 9.3 Edge preference disagreement

定义：

$$
D_{ij}^{CT}
=
|s_{ij}^C-s_{ij}^T|.
$$
统计：

```text
mean_abs_gap_C_T
mean_abs_gap_C_V
mean_abs_gap_T_V
```

以及高差异边比例：

```text
P(|s_C - s_T| > 0.25)
```

阈值 0.25 仅用于工程观察，不作为论文理论阈值。

---

# 10. P0-D：Factor-wise Graph Propagation Utility

这是 P0 最关键的证据。

同一个 factor 同时构造：

### Local

$$
F^f
$$

### Graph-contextualized

$$
G^f=
\hat A F^f.
$$

推荐加 residual：

$$
\tilde F^f=
LayerNorm(F^f+G^f).
$$
这里 propagation 固定，不学习 relation、不学习 attention。

第一版使用现有 GCN normalization：

$$
\hat A=
D^{-1/2}(A+I)D^{-1/2}.
$$

---

# 11. NC Probe 设计

对每个 factor：

```text
C
P_t
P_v
```

分别得到：

```text
C_local
C_graph

Pt_local
Pt_graph

Pv_local
Pv_graph
```

每一种 representation 单独训练一个 linear classifier：

```python
nn.Linear(factor_dim, num_classes)
```

严格：

- probe 只使用 train nodes；
- hyperparameter 统一；
- validation 用于 early stopping；
- P0 决策看 validation；
- final confirmation 再测 test。

记录：

$$
Acc_{local}^f, F1_{local}^f
$$
\[
Acc_{graph}^f, F1_{graph}^f.
\]

全局传播收益：

\[
\Delta Acc^f
=
Acc_{graph}^f-Acc_{local}^f
\]

\[
\Delta F1^f
=
F1_{graph}^f-F1_{local}^f.
\]

---

## 11.1 NC Node-level Conflict

更重要的是 per-node utility。

分别利用 local probe / graph probe 得到每个 validation node 的真实标签 CE：

\[
\ell_i^{f,local}
\]

\[
\ell_i^{f,graph}.
\]

定义：

\[
\Delta_i^f
=
\ell_i^{f,local}
-
\ell_i^{f,graph}.
\]

所以：

```text
delta > 0 : graph helps
delta < 0 : graph hurts
```

然后计算：

\[
Corr(\Delta^C,\Delta^{P_t})
\]

\[
Corr(\Delta^C,\Delta^{P_v})
\]

\[
Corr(\Delta^{P_t},\Delta^{P_v}).
\]

以及 conflict rate：

\[
Conflict(C,T)
=
P[
sign(\Delta_i^C)
\neq
sign(\Delta_i^{P_t})
].
\]

同理计算：

```text
conflict_C_V
conflict_T_V
```

再计算三因子一致/冲突模式：

```text
all_help
all_hurt
C_help_T_hurt
C_help_V_hurt
T_help_C_hurt
V_help_C_hurt
...
```

这部分非常适合作为未来论文 empirical motivation。

---

# 12. LP Probe 设计

LP 不建议只看 global MRR。

也应该定义 factor-wise local / graph utility。

对：

```text
C
P_t
P_v
```

分别构造：

```text
F_local
F_graph
```

每种 representation 单独训练**相同结构**的 LinkPredictor。

建议直接复用现有 `src.models.LinkPredictor`，避免 predictor 差异成为混淆因素。

---

## 12.1 LP Global Gain

记录：

\[
MRR_{local}^{f}
\]

\[
MRR_{graph}^{f}
\]

\[
\Delta MRR^f
=
MRR_{graph}^{f}
-
MRR_{local}^{f}.
\]

同理 Hits@K。

---

## 12.2 LP Edge-level Reciprocal-Rank Gain

现有 LP validation/test 为：

- positive target
- fixed negative targets

因此对每条正样本边 \(e=(u,v)\) 可以得到：

\[
RR_e^{f,local}
\]

\[
RR_e^{f,graph}.
\]

定义：

\[
\Delta_e^f
=
RR_e^{f,graph}
-
RR_e^{f,local}.
\]

然后与 NC 完全一致地计算：

```text
corr_delta_C_T
corr_delta_C_V
corr_delta_T_V

conflict_C_T
conflict_C_V
conflict_T_V
```

这里：

```text
delta > 0 : graph propagation improves the rank of the true positive target
delta < 0 : graph propagation hurts it
```

这是 LP 中最有说服力的 factor-dependent neighborhood utility 证据。

---

# 13. LP 特别注意：不能产生 edge leakage

P0-LP diagnostics 必须遵守：

1. encoder message graph 只能使用现有 `data.edge_index`；
2. `data.edge_index` 对 LP 已由 loader 构造成 train-edge graph；
3. 不得在 graph propagation 时加入 valid/test positive edges；
4. probe training 只用 train edges；
5. validation/test 使用现有 negative sets；
6. 不得根据 test MRR 调 factorizer / loss 权重。

如果为了计算 factor probe 临时使用：

```python
A_full = train + valid + test
```

则该实验无效。

---

# 14. 建议新增 diagnostics 工具

新增：

```text
src/utils/biaxis_p0_diagnostics.py
```

至少包含：

```python
def compute_factor_sanity(factors, max_nodes=10000):
    ...

def compute_edge_factor_statistics(
    factors,
    edge_index,
    max_edges=500000,
    seed=42,
):
    ...

def propagate_fixed_gcn(
    h,
    edge_index,
):
    ...

def run_nc_factor_probes(...):
    ...

def run_lp_factor_probes(...):
    ...

def compute_conflict_statistics(delta_dict):
    ...

def save_p0_report(...):
    ...
```

不要把 diagnostics 全塞进 `biaxis_p0.py`。

---

# 15. 输出文件规范

每次 P0 run 统一生成：

```text
outputs/p0/
└── <dataset>/
    └── seed_<seed>/
        ├── factor_sanity.json
        ├── edge_statistics.json
        ├── nc_probe.csv          # NC only
        ├── nc_node_delta.pt      # NC only
        ├── lp_probe.csv          # LP only
        ├── lp_edge_delta.pt      # LP only
        ├── conflict_stats.json
        └── summary.json
```

最终汇总：

```text
outputs/p0/
├── p0_nc_summary.csv
├── p0_lp_summary.csv
├── p0_conflict_summary.csv
└── p0_report.md
```

---

# 16. 配置文件

新增：

```text
configs/model/biaxis_p0.yaml
```

建议初始：

```yaml
name: biaxis_p0

hidden_dim: 256
factor_dim: 128
dropout: 0.2
activation: gelu
norm: layernorm

lambda_common: 0.1
lambda_orth: 0.01
lambda_recon: 0.1

full_graph_training: false

p0:
  diagnostics: true
  max_diag_nodes: 10000
  max_diag_edges: 500000
  edge_top_ratios: [0.1, 0.2]
  gap_threshold: 0.25

  probe:
    epochs: 200
    patience: 20
    lr: 0.001
    weight_decay: 0.0001
```

不要在 model config 中硬编码数据集。

---

# 17. 训练方式：不要一次跑完 7 个数据集

采用漏斗。

## Stage D0：代码正确性

```text
Movies-NC
seed=42
1 run
```

检查：

- forward shape；
- aux loss；
- factor norm；
- no NaN；
- training loss 下降；
- `encode_factors()` 可调用；
- diagnostics 能完整输出。

---

## Stage D1：Problem Screening

```text
Movies-NC
ele-fashion-NC
seed=42
```

重点看：

- `S_C > S_P`；
- factor effective rank；
- edge correlation；
- NC propagation conflict。

如果完全没有 signal，先处理 factorizer，而不是跑 7 个数据集。

---

## Stage D2：NC Confirmation

```text
Movies
Toys
Grocery
ele-fashion
Reddit-S

seeds = 42,43,44
```

共：

\[
5\times3=15
\]

个正式 P0-NC runs。

---

## Stage D3：LP Confirmation

```text
sports-copurchase
cloth-copurchase

seeds = 42,43,44
```

共：

\[
2\times3=6
\]

个 P0-LP runs。

LP 放在 NC Problem 已经有明确证据之后。

---

# 18. P0 实验总表

| ID | Task | Experiment | 目的 |
|---|---|---|---|
| P0-1 | NC/LP | Local fused representation | 检查 factorizer 可训练 |
| P0-2 | All | Common vs Private similarity | Common/Private sanity |
| P0-3 | All | Effective rank | 排除 collapse |
| P0-4 | All | Edge similarity Spearman | 邻居排序是否 factor-dependent |
| P0-5 | All | Top-10/20% Jaccard | useful-edge overlap |
| P0-6 | NC | Factor local vs fixed-GCN probe | graph 对各 factor 是否不同收益 |
| P0-7 | NC | Node-level Δ conflict | 最核心 NC 证据 |
| P0-8 | LP | Factor local vs fixed-GCN LinkPredictor | LP global utility |
| P0-9 | LP | Edge-level RR Δ conflict | 最核心 LP 证据 |

---

# 19. P0 GO / NO-GO 判据

这些是**内部研发判据，不是论文硬阈值**。

## GO 强信号

多个数据集重复出现：

### Factorization

\[
S_C>S_P
\]

并且三个 factor 无明显 rank collapse。

### Edge ranking

至少部分数据集：

```text
Spearman(C,T) < 0.7
or
Spearman(C,V) < 0.7
```

或：

```text
Top20 Jaccard < 0.6
```

### Propagation utility

至少两个 NC 数据集存在明显：

```text
ConflictRate(C,T) > 15%
or
ConflictRate(C,V) > 15%
```

并且 seed 间稳定。

### Cross-task support

至少一个 LP 数据集也观察到：

```text
factor-wise ΔMRR 不一致
```

以及 non-trivial RR conflict。

如果 NC+LP 都有信号，二维框架的经验基础很强。

---

## NO-GO / REVISE

以下情况需要停止进入 P1：

### 情况 A

\[
S_C\approx S_P
\]

说明 factorization 没学好。

处理：先修 M1，不判断核心假设。

### 情况 B

Factor collapse。

处理：调 reconstruction / orthogonality / factor dimension。

### 情况 C

factorization 正常，但：

```text
edge rankings highly correlated
propagation deltas highly correlated
conflict ≈ 0
```

说明二维假设在当前表示上缺少证据。

不要上 Relation/OT 硬制造区别。

### 情况 D

只有单一数据集有 signal。

结论暂时不成立，扩大 seed / dataset 后再判断。

---

# 20. P0 不做的事情

AI 编码时明确禁止：

- 不实现 Relation Prototype；
- 不实现 UOT；
- 不实现 Sinkhorn；
- 不实现 Factor–Relation Operator；
- 不实现 relation-specific GNN；
- 不实现 pseudo node；
- 不实现 DiP；
- 不实现 frequency decomposition；
- 不改 graph topology；
- 不做 KNN rewiring；
- 不引入 test labels；
- 不根据 test 调参；
- 不修改现有 baseline 结果；
- 不重写 NC/LP protocol，除非为了通用 factor diagnostics 增加独立工具函数。

---

# 21. 单元测试要求

新增：

```text
tests/test_biaxis_p0.py
```

必须覆盖：

### Shape

随机：

```text
N=17
text_dim=13
visual_dim=19
```

验证：

```text
c_t/c_v/c/p_t/p_v -> [17, factor_dim]
z -> [17, hidden_dim]
```

### Modality split

验证 `[text, visual]` 顺序无误。

### Loss finite

```python
torch.isfinite(aux_loss)
```

### Gradient

确认：

```text
common encoder
private text encoder
private visual encoder
projection
reconstruction head
fusion
```

均有 gradient。

### Common parameter sharing

确认：

```text
c_t/c_v 使用同一 shared_common_encoder 实例
```

### Fixed propagation

小图手工计算 normalized aggregation，与函数输出一致。

### No topology dependency in factorizer

同一个 `x`，传不同 `edge_index`，`encode_factors()` 输出必须完全一致。

这条测试很重要：

> P0 semantic factorization 必须是 topology-free。

---

# 22. AI 编码推进方式：不要给一个超长 Prompt 一次写完

推荐拆成 5 次。

---

## Prompt 1：仓库审查，不改代码

直接给 AI：

```text
你现在协助我在现有 MAG_baseline 仓库中实现一个新的研究阶段 P0。

先不要修改任何代码。

请仔细阅读并总结：
1. src/models/factory.py
2. src/models/gcn.py
3. src/models/dip.py（只理解接口，不复用其方法）
4. src/tasks/nc.py
5. src/tasks/lp.py
6. src/tasks/inference.py
7. src/data/loaders.py
8. src/data/types.py
9. configs/model/
10. Movies、ele-fashion、sports-copurchase 的 dataset config

重点回答：
- Model 必须满足什么 forward/inference 接口？
- text/image 在 data.x 中是什么顺序？
- NC NeighborLoader 下 model.forward 收到什么？
- LP LinkNeighborLoader 下 model.forward 收到什么？
- aux_loss / aux_info 是怎么被 task runner 使用的？
- 现有 inference_mode 对新模型有什么约束？
- 如果我要实现 topology-free SemanticFactorizer + 离线 factor diagnostics，最小侵入的代码改动方案是什么？
- 有哪些 sampling/global-node-id 相关风险？

输出一份 implementation audit。
不要写代码，不要擅自修改训练协议。
```

---

## Prompt 2：只实现 Semantic Factorizer

```text
基于前一步仓库审查，现在只实现 P0-A Semantic Factorizer。

科学目标：
将同一节点文本/图像表示解耦为：
C：跨模态 Common
Pt：Text Private
Pv：Visual Private。

禁止实现 topology/relation/OT/router/pseudo-node。

请新增：
- src/models/biaxis_components.py
- src/models/biaxis_p0.py
- configs/model/biaxis_p0.yaml
- tests/test_biaxis_p0.py

要求：
1. 输入 data.x 的顺序严格沿用仓库现有 [text, image]。
2. text/image 分别 project 到 hidden_dim=256。
3. Common encoder 对两模态共享参数。
4. text-private / visual-private encoder 参数独立。
5. factor_dim=128。
6. c=(c_t+c_v)/2。
7. fusion([c,pt,pv]) -> out_dim=256。
8. 实现 common consistency、cross-covariance orthogonality、reconstruction loss。
9. forward 必须返回仓库要求的：
   z, None, None, aux_loss, aux_info
10. 实现 encode_factors()。
11. factorizer 输出不得依赖 edge_index。
12. 不修改 NC/LP 的主训练协议。
13. tests 必须验证 shape、finite loss、gradient、parameter sharing、topology independence。

实现后请：
- 列出所有修改文件；
- 解释每个 tensor shape；
- 运行相关 tests；
- 报告测试结果；
- 不开始 P0 diagnostics。
```

---

## Prompt 3：实现通用 diagnostics

```text
现在 P0-A 已完成。不要修改模型科学结构。

实现：
src/utils/biaxis_p0_diagnostics.py

要求支持 NC 和 LP 共用的：
1. factor sanity:
   common_sim/private_sim/cross-covariance/effective-rank
2. observed-edge factor similarity:
   cosine for C/Pt/Pv
3. pairwise Spearman
4. Top-10% / Top-20% edge Jaccard
5. mean absolute similarity gap
6. fixed one-hop GCN normalized propagation

注意：
- 大图支持 max_edges 随机采样；
- 固定随机 seed；
- 无向边统计避免双计；
- effective rank 最多采样 max_nodes；
- 所有 diagnostics 必须 torch.no_grad；
- 不使用标签，除 probe 部分外；
- 不改变 model 参数。

先实现通用无监督 diagnostics 和单元测试。
不要实现 NC/LP probe。
```

---

## Prompt 4：实现 NC probe

```text
现在实现 P0-NC factor propagation utility probe。

目标：
分别对 C/Pt/Pv 比较 local representation 与 fixed-GCN one-hop contextualized representation。

要求：
1. 从已训练好的 biaxis_p0 best checkpoint 提取 factors。
2. factorizer 冻结。
3. 对每个 factor 构造：
   local F
   graph LN(F + A_norm F)
4. local 和 graph 各训练独立但结构完全一致的 Linear classifier。
5. 只用 train_idx 训练。
6. val_idx 用于 early stopping。
7. test 只在 final confirm 模式使用。
8. 输出 Acc/Macro-F1。
9. 保存每个 validation node：
   CE_local
   CE_graph
   delta = CE_local - CE_graph
10. 计算 factor-pair delta Spearman/Pearson 和 sign conflict rate。
11. 输出 JSON/CSV/PT 文件。
12. 不能根据 test 结果选择 probe 或超参。

请增加必要脚本和测试，但不要改已有 baseline 训练逻辑。
```

---

## Prompt 5：实现 LP probe

```text
现在实现 P0-LP factor propagation utility probe。

必须严格保持现有 MAG_baseline LP protocol，不允许 edge leakage。

目标：
对 C/Pt/Pv 分别比较 local 与 fixed-GCN graph representation 的 LP utility。

要求：
1. encoder message graph 只能使用现有 data.edge_index（train-edge graph）。
2. factorizer 使用 best training checkpoint，并冻结。
3. local/graph 每种 representation 使用同结构 LinkPredictor。
4. predictor 只用 train edge supervision。
5. validation/test 使用现有 edge_split 和 fixed negatives。
6. 输出 MRR/Hits@1/3/10。
7. 对每条 validation positive edge，计算：
   RR_local
   RR_graph
   delta_RR = RR_graph - RR_local
8. 计算 C/Pt/Pv 的 delta_RR correlation 和 sign conflict rate。
9. 不把 valid/test positive edges加入 propagation adjacency。
10. 增加显式 assertion / test 防止 leakage。
11. 输出 CSV/JSON/PT。
```

---

# 23. 最终批量运行 Prompt

当所有单数据集测试通过后：

```text
P0 代码已经完成并通过 Movies seed=42 debug。

现在不要修改模型结构和超参数。
请生成批量实验脚本：

NC：
Movies
Toys
Grocery
ele-fashion
Reddit-S

LP：
sports-copurchase
cloth-copurchase

seeds:
42 43 44

执行顺序：
1. 每个 dataset/seed 训练 biaxis_p0；
2. 保存 best checkpoint；
3. 运行 factor sanity；
4. 运行 edge factor diagnostics；
5. NC 调用 NC factor probe；
6. LP 调用 LP factor probe；
7. 汇总为：
   outputs/p0/p0_nc_summary.csv
   outputs/p0/p0_lp_summary.csv
   outputs/p0/p0_conflict_summary.csv
   outputs/p0/p0_report.md

要求：
- 单个失败任务不能覆盖已有结果；
- 每个任务记录 config snapshot、seed、checkpoint；
- 汇总均值和标准差；
- 不自动根据 test 结果改配置；
- 不进行 P1 实现。
```

---

# 24. 结果汇总表模板

## NC

| Dataset | Common Sim | Private Sim | rho C/T | rho C/V | Jaccard C/T | ΔAcc C | ΔAcc Pt | ΔAcc Pv | Conflict C/T | Conflict C/V |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | | | | | | | | | | |
| Toys | | | | | | | | | | |
| Grocery | | | | | | | | | | |
| ele-fashion | | | | | | | | | | |
| Reddit-S | | | | | | | | | | |

---

## LP

| Dataset | rho C/T | rho C/V | ΔMRR C | ΔMRR Pt | ΔMRR Pv | RR Conflict C/T | RR Conflict C/V |
|---|---:|---:|---:|---:|---:|---:|---:|
| sports-copurchase | | | | | | | |
| cloth-copurchase | | | | | | | |

---

# 25. P0 完成后的决策逻辑

## Case A：Strong GO

如果：

- factorization sanity 正常；
- edge ranking 在多个数据集明显 factor-dependent；
- NC node-level propagation conflict 稳定存在；
- LP RR conflict 也能观察到；

则进入 P1：

\[
\boxed{
Semantic Factorization
\times
Structural Relation Decomposition
}
\]

P1 的核心任务变成：

> 既然同一邻域对不同 semantic factors 的作用不同，那么能否独立学习 topology-side latent relation，再进行 Factor × Relation coupling？

---

## Case B：Partial GO

如果：

- NC 很强；
- LP 弱；

可以继续 P1，但论文主线先定位 NC / general node representation，LP 作为补充。

不要为了 LP 强行修改 hypothesis。

---

## Case C：Factorizer failure

如果 Common/Private sanity 不成立：

先修 M1。

不允许据此否定“Factor × Relation”假设。

---

## Case D：Scientific NO-GO

如果 factorization 正常，但：

- edge ranking 几乎完全相同；
- graph propagation gain 对三 factors 高度一致；
- conflict rate 接近零；
- 7 个任务多数如此；

则停止 P1。

说明当前数据/表示下：

\[
Semantic\ Ownership
\]

虽然存在，但并没有表现为显著不同的 graph-neighborhood utility。

此时应该修改研究问题，而不是继续堆 Relation/OT。

---

# 26. P0 阶段最重要的科研纪律

1. **先证明现象，再造模块。**
2. **Validation 决策，Test 只做最终确认。**
3. **Factorization 与 topology 在 P0 必须隔离。**
4. **LP 必须严格防泄漏。**
5. **不要用一个数据集的结果决定故事。**
6. **不要为了保住 Idea 强行设计 relation。**
7. **所有失败结果都记录。**
8. **所有 diagnostics 固定 seed 和 sampling strategy。**
9. **所有 probe 使用完全一致的容量和优化协议。**
10. **P0 最终产物不是“一个模型”，而是一组能够回答核心科学问题的证据。**

---

# 27. P0 Definition of Done

只有全部完成才进入 P1：

- [ ] `biaxis_p0.py` 可在现有 MAG_baseline 训练；
- [ ] 5 个 NC + 2 个 LP 数据集接口均正常；
- [ ] Common/Private factorizer 通过 sanity checks；
- [ ] 无 factor collapse；
- [ ] edge factor correlation / Jaccard 输出完成；
- [ ] NC factor-wise fixed propagation probe 完成；
- [ ] NC node-level conflict 完成；
- [ ] LP factor-wise fixed propagation probe 完成；
- [ ] LP edge-level RR conflict 完成；
- [ ] seed=42/43/44 批量实验完成；
- [ ] 所有实验使用统一配置原则；
- [ ] 输出统一 CSV/JSON；
- [ ] 写出 `P0_REPORT.md`；
- [ ] 根据 P0 evidence 给出 GO / REVISE / NO-GO 决策；
- [ ] 未提前实现 P1 模块。

---

# 28. P0 最终要回答的五句话

当 P0 完成后，你应该能够用数据回答：

1. **Common 与 Private 是否真的形成不同语义空间？**
2. **同一 observed graph edge 在不同 semantic factor 空间中的相关性是否不同？**
3. **同一个 graph neighborhood 对 Common / Text-private / Visual-private 的传播收益是否不同？**
4. **这种差异是否具有 node-level / edge-level conflict，而不只是平均性能差异？**
5. **这种现象是否跨数据集、跨 seed、跨 NC/LP 任务稳定存在？**

只有这五个问题得到足够积极的答案，后续 P1 的：

$$
\boxed{
Semantic\ Factor
\times
Structural\ Relation
}
$$


才真正有实验上的研究动机。

