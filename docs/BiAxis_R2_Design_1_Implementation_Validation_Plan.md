# Bi-Axis Performance-R2-Design-1
## Formal Method Specification, Implementation & End-to-End Validation Plan

**Repository:** `CrisRipper777/0901`  
**Current retained reference:** `biaxis_final / A0`  
**R2-0 status:** `CLOSED / DIAGNOSIS COMPLETE`  
**R2-Design-0 status:** `PASS`  
**R2-Design-1 target:** 将 R2-0 的诊断结果收敛成一个最小、可训练、可证伪的正式 R2 架构，并通过分阶段 end-to-end 实验判断它是否真正解决性能瓶颈。

---

# 0. 本阶段只做 Candidate A

R2-Design-0 已冻结首选方向：

\[
\boxed{\textbf{Factor-Context Functional Modulation}}
\]

正式概念：

\[
\boxed{
\textbf{Semantic Ownership}
\times
\textbf{Functional Relational Transfer}
}
\]

R2-Design-1 **不再比较多个大架构 candidate**，也不引入：

```text
OT / UOT
K prototype relations
Gamma router
OFR
Gsim / Gdiff
Graph Transformer
edge-level attention
MoE
new contrastive/synergy loss
global multi-hop
```

本阶段只回答：

> R2-0 frozen probes 所发现的 semantic interaction 与 target-conditioned factor-context interaction，能否被一个简洁的 end-to-end architecture 稳定兑现？

---

# 1. 证据约束：正式 R2 必须忠实于哪些结果？

## 1.1 KEEP

必须保留：

\[
C,\;P_t,\;P_v
\]

作为 semantic ownership states。

当前 P0 已证明：

- factorization 不系统性毁掉原 multimodal information；
- common/private factor graph utility 明显不同；
- factor-specific semantics 对 downstream graph learning 有意义。

所以 R2 不重新发明 factorizer。

---

## 1.2 MODIFY

当前：

\[
C=\frac{c_t+c_v}{2}
\]

存在 information compression evidence。

但 common averaging 只是 secondary weakness，因此只做轻量修正：

\[
\boxed{\text{node-adaptive common consensus}}
\]

而不是重新引入大规模 cross-modal encoder。

---

## 1.3 REMOVE / DE-PRIORITIZE

第一版 R2 不再使用：

\[
[\log d,P\log d,P^2\log d]
\rightarrow
K\text{ prototypes}
\rightarrow
\Gamma
\rightarrow
T_{fk}.
\]

理由：

- R0 relation differentiation 弱；
- R1 router/relation variants 全 NO-GO；
- R2-0A plain contexts ≈ current relation contexts；
- R2-0B richer structural basis只有 WEAK evidence；
- final residual structural headroom 很小。

---

## 1.4 NEW CORE

R2-0C 真正支持的 relation object 是：

\[
\boxed{
q_i^{a\rightarrow b}
=
\psi(F_i^b,N_i^a)
}
\]

而不是第一版就做：

\[
q_{ji}^{a\rightarrow b}
=
\psi(F_i^b,F_j^a,e_{ji}).
\]

其中：

\[
N_i^a
=
\operatorname{Mean}_{j\in\mathcal N(i)}F_j^a.
\]

含义：

> source-factor neighborhood context \(a\) 对 target semantic state \(b\) 的功能兼容性。

---

# 2. R2 正式计算图

完整第一版：

```text
Text / Visual embeddings
        │
        ▼
P0 Semantic Ownership
C_t, C_v, P_t, P_v
        │
        ├──────────── R2-S/J only ──────────────┐
        ▼                                       │
Lightweight Semantic Refinement                 │
Adaptive Common + Factor Interaction Residual   │
        │                                       │
        ▼                                       │
F* = {C*, Pt*, Pv*}                             │
        │                                       │
        ▼                                       │
Simple 1-hop factor-wise aggregation            │
N^C, N^Pt, N^Pv                                 │
        │                                       │
        ├──── baseline diagonal graph path ─────┤
        │                                       │
        └──── R2-F/J Functional Modulation ─────┘
                    target F_b* × source N_a
                    3 × 3 functional cells
                              │
                              ▼
                    Residual factor updates
                              │
                              ▼
                     C', Pt', Pv'
                              │
                              ▼
                       Existing fusion
                              │
                              ▼
                          z_final
```

---

# 3. 四个正式 Variant

必须使用同一个 `biaxis_r2` implementation，通过 config toggles 构造四个版本。

| Variant | Semantic Refiner | Functional Transfer | 作用 |
|---|---:|---:|---|
| **R2-B0** | OFF | OFF | 新 clean parent：P0 + simple diagonal 1-hop |
| **R2-F** | OFF | ON | 单独检验 Functional Relation |
| **R2-S** | ON | OFF | 单独检验 Semantic Refinement |
| **R2-J** | ON | ON | 最终联合候选 |

正式命名建议：

```text
biaxis_r2_b0
biaxis_r2_f
biaxis_r2_s
biaxis_r2_j
```

但四个 YAML 都指向同一个 model implementation。

---

# 4. R2-B0 — Clean Parent

## 4.1 输入

从现有 P0 获得：

\[
F_i^0=
\{C_i,P_i^t,P_i^v\}.
\]

B0 保持 current fixed common：

\[
C_i=\frac12(c_i^t+c_i^v).
\]

不启用 semantic refinement。

---

## 4.2 Simple factor-wise graph contexts

对：

\[
a\in\{C,P_t,P_v\}
\]

计算：

\[
N_i^a
=
P F^a
=
\frac1{d_i}\sum_{j\in\mathcal N(i)}F_j^a.
\]

必须复用当前经过审计的 `neighbor_mean()`。

---

## 4.3 Source transform

每个 source factor 一个轻量 transform：

\[
V_a:\mathbb R^d\to\mathbb R^d.
\]

实现：

```python
nn.ModuleList([
    nn.Linear(d, d, bias=False)
    for _ in range(3)
])
```

不做 9 个 full \(W_{ab}\)。

---

## 4.4 Baseline diagonal update

对 target \(b\)：

\[
M_{i,\text{base}}^b
=
V_b(N_i^b).
\]

使用 target-specific message norm：

\[
\bar M_{i,\text{base}}^b
=
LN_b(M_{i,\text{base}}^b).
\]

Graph residual strength：

\[
\rho_b^{base}
=
\sigma(r_b^{base}).
\]

初始化：

\[
r_b^{base}=0
\Rightarrow
\rho_b^{base}=0.5.
\]

最终：

\[
\boxed{
F_i^{b\prime}
=
F_i^b
+
\rho_b^{base}\bar M_{i,\text{base}}^b
}
\]

isolated node：

\[
N_i^b=0
\]

且 LayerNorm(0)=0，因此：

\[
F_i^{b\prime}=F_i^b.
\]

这一点要通过 unit test。

---

# 5. Semantic Refiner — R2-S / R2-J

目标：

\[
\boxed{
\text{refine usability without destroying ownership}
}
\]

不做 Transformer。

---

# 6. Adaptive Common Consensus

当前：

\[
C=\frac12(c_t+c_v).
\]

构造：

\[
u_i^c
=
[
c_i^t,
c_i^v,
c_i^t\odot c_i^v,
|c_i^t-c_i^v|
].
\]

Common gate：

\[
[\ell_i^t,\ell_i^v]
=
g_c(u_i^c).
\]

\[
[\omega_i^t,\omega_i^v]
=
Softmax([\ell_i^t,\ell_i^v]).
\]

\[
\boxed{
C_i^0
=
\omega_i^t c_i^t
+
\omega_i^v c_i^v.
}
\]

### Architecture

推荐：

```text
Linear(4d, 64)
GELU
Linear(64, 2)
Softmax
```

最后一个 Linear：

```text
weight = 0
bias = 0
```

因此 step 0：

\[
\omega_t=\omega_v=0.5.
\]

模型初始严格退化为 current common average。

这是必须的稳定性设计。

---

# 7. Ownership-Preserving Factor Interaction Residual

先定义：

\[
F_i^{0}
=
\{C_i^0,P_i^t,P_i^v\}.
\]

interaction block：

\[
I_i=
[
C_i^0\odot P_i^t,
C_i^0\odot P_i^v,
P_i^t\odot P_i^v,
|C_i^0-P_i^t|,
|C_i^0-P_i^v|,
|P_i^t-P_i^v|
].
\]

维度：

\[
6d.
\]

共享 trunk：

```text
Linear(6d, d)
LayerNorm(d)
GELU
Dropout(0.2)
```

得到：

\[
r_i^{sem}\in\mathbb R^d.
\]

三个 factor-specific residual heads：

\[
\Delta_i^b=W_b^{sem}r_i^{sem}.
\]

三个 final heads：

```text
Linear(d,d,bias=False)
```

**zero initialization**。

于是训练第 0 步：

\[
\Delta_i^b=0.
\]

输出：

\[
\boxed{
F_i^{b,*}
=
F_i^{b,0}
+
\Delta_i^b.
}
\]

不要在这里额外做 full mixing/Transformer。

---

# 8. P0 Auxiliary Loss 的处理

非常重要：

current：

\[
L_{common},
L_{orth},
L_{recon}
\]

继续只作用于 **base decomposition**：

\[
c_t,c_v,p_t,p_v
\]

而不是 refined factors。

即：

\[
\boxed{
\text{decomposition defines ownership;}
\quad
\text{refinement improves usability.}
}
\]

不要新增：

```text
interaction loss
contrastive loss
OT loss
synergy loss
gate regularization
```

R2 第一版只靠 architecture + 原 P0 aux。

---

# 9. Functional Transfer — R2-F / R2-J

这是 R2 的核心。

输入 refined 或 base factors：

\[
F_i^*=
\{F_i^{C,*},F_i^{Pt,*},F_i^{Pv,*}\}.
\]

R2-F 中：

\[
F^*=F^0
\]

即不启用 Semantic Refiner。

---

# 10. 先聚合，再功能性交互

source contexts：

\[
N_i^a=P F^{a,*}.
\]

只做 ordinary 1-hop。

第一版禁止：

```text
G2
Gsim
Gdiff
K relations
edge attention
```

---

# 11. 3×3 Functional Compatibility

对于：

\[
a,b\in\{C,P_t,P_v\}
\]

构造：

\[
u_i^{a\to b}
=
[
F_i^{b,*},
N_i^a,
F_i^{b,*}\odot N_i^a,
|F_i^{b,*}-N_i^a|,
e_a^{src},
e_b^{tgt}
].
\]

其中：

\[
e_a^{src},e_b^{tgt}\in\mathbb R^{t}
\]

type embedding。

推荐：

```text
type_dim = 8
gate_hidden = 64
```

共享 scorer：

```text
Linear(4d + 2*type_dim, gate_hidden)
GELU
Linear(gate_hidden, 1)
```

得到：

\[
s_i^{a\to b}.
\]

独立 gate：

\[
\boxed{
g_i^{a\to b}
=
\sigma(s_i^{a\to b})
}
\]

禁止使用：

\[
Softmax_a.
\]

理由：

不同 source factors 可以同时有用，R1 已证明 competitive routing 容易出现错误竞争。

---

# 12. Functional Message

source transform 与 B0 共用：

\[
v_i^a
=
V_a(N_i^a).
\]

cell message：

\[
m_i^{a\to b}
=
g_i^{a\to b}
v_i^a.
\]

聚合：

\[
M_{i,\text{func}}^b
=
\frac13
\sum_a
m_i^{a\to b}.
\]

再做：

\[
\bar M_{i,\text{func}}^b
=
LN_b^{func}(M_{i,\text{func}}^b).
\]

---

# 13. Functional Residual 必须是“B0 + 小残差”

为最大限度吸收 R1 的训练教训：

R2-F/J 不用 functional path 替代 B0，而是：

\[
\boxed{
\text{B0 diagonal path}
+
\text{small functional residual}
}
\]

定义：

\[
\rho_b^{func}
\]

为直接 LayerScale parameter：

```text
init = 0.01
```

不使用 sigmoid。

最终：

\[
\boxed{
F_i^{b\prime}
=
F_i^{b,*}
+
\rho_b^{base}\bar M_{i,\text{base}}^b
+
\rho_b^{func}\bar M_{i,\text{func}}^b
}
\]

R2-B0 / R2-S：

\[
\rho^{func}=0
\]

且 functional scorer 根本不实例化或不参与 forward。

这样：

\[
\boxed{
R2-F = B0 + minimal functional residual
}
\]

是严格 clean comparison。

---

# 14. Gate 初始化

scorer 最后层：

推荐：

```text
weight ~ small normal(std=1e-3)
bias = 0
```

所以初始：

\[
g\approx0.5
\]

但 functional effective magnitude 还有：

\[
\rho^{func}=0.01.
\]

因此 step 0 新 path 近似：

\[
0.005\times message.
\]

既不会完全阻断 scorer gradient，也不会扰乱 B0。

---

# 15. Final Fusion

继续使用现有：

```text
Linear(3d, hidden_dim)
LayerNorm
GELU
Dropout
```

不新增 Transformer / MoE。

当前 fusion 未被诊断为瓶颈。

---

# 16. 参数预算

默认：

```text
hidden_dim = 256
factor_dim = 128
type_dim = 8
gate_hidden = 64
semantic_hidden = 128 (= factor_dim)
dropout = 0.2
```

R2 新增参数应尽量控制：

```text
Semantic Refiner        < ~150k
Functional scorer       < ~50k
3 source transforms     ~49k
type embeddings         negligible
norm/scales             negligible
```

目标：

\[
\boxed{
\text{R2 total parameters 不应接近 DiP 8M 级别}
}
\]

建议 AI 在报告中输出每个 variant parameter count。

---

# 17. 显存复杂度

Functional Transfer 在 node-context 层：

\[
O(3Ed+9Nd)
\]

而不是 edge-level：

\[
O(9Ed).
\]

实现时禁止永久 materialize：

```text
[N,3,3,4d]
```

可以：

```text
for target b:
    for source a:
        on-the-fly build interaction
```

或最多一次处理：

```text
[N,9,d]
```

但 ele-fashion memory smoke 前必须记录峰值。

---

# 18. 必须实现的 diagnostics

R2 的 mechanism diagnostics 必须从第一天就写，不要等模型有效后再补。

---

## 18.1 Semantic Refiner diagnostics

记录：

### Common weights

\[
mean(\omega_t),
mean(\omega_v)
\]

\[
std(\omega_t),
std(\omega_v)
\]

以及：

```text
frac(w_t < .05)
frac(w_t > .95)
```

判断 modality collapse。

### Semantic residual ratio

对每 factor：

\[
R_{sem}^b
=
\frac{
\|\Delta^b\|_2
}{
\|F^{b,0}\|_2+\epsilon
}.
\]

输出 mean/std。

### Ownership health

继续输出已有：

```text
common_sim
private_sim
C/P overlap
factor norms
```

避免 refinement 破坏 factorizer。

---

## 18.2 Functional diagnostics

对 3×3 gate：

\[
G^{a\to b}
\]

记录：

```text
mean gate
std gate
p05
p50
p95
frac < .05
frac > .95
```

输出 3×3 matrix。

### Message contribution

\[
C_{a\to b}
=
\frac{
E_i\|m_i^{a\to b}\|
}{
E_i\sum_{a'}\|m_i^{a'\to b}\|+\epsilon
}.
\]

输出 3×3。

### Functional residual magnitude

\[
R_{func}^b
=
\frac{
\|\rho_b^{func}M_{func}^b\|
}{
\|F^{b,*}\|+\epsilon
}.
\]

### Base graph magnitude

\[
R_{base}^b
=
\frac{
\|\rho_b^{base}M_{base}^b\|
}{
\|F^{b,*}\|+\epsilon
}.
\]

### Learned residual scales

输出：

```text
rho_base_C/Pt/Pv
rho_func_C/Pt/Pv
```

---

# 19. 机制异常判据

仅用于诊断，不自动决定性能 GO：

### Common collapse suspicion

```text
>80% nodes w_t > .95
or
>80% nodes w_t < .05
```

### Gate dead

绝大多数 9 cells：

```text
mean gate < .05
```

### Gate saturated

绝大多数：

```text
mean gate > .95
```

### Functional path inactive

\[
R_{func}^b<0.01
\]

全部 factor。

### Functional path dominates

\[
R_{func}^b>1
\]

持续出现。

### Semantic refiner dominates

\[
R_{sem}^b>1
\]

提示 ownership path 被覆盖。

这些都只作机制审查。

---

# 20. 训练协议

继续统一 NC：

```text
full graph
CE train nodes only
AdamW
300 epochs
patience 30
best checkpoint = Val Accuracy
Val Accuracy primary
Val Macro-F1 secondary
```

R2 screen：

```text
evaluate_test = false
```

最终模型冻结前禁止 Test。

---

# 21. Optimizer

第一版不要重新大扫超参。

直接沿用 A0：

```text
lr = 1e-3
weight_decay = 1e-4
```

所有 R2 参数和 backbone 共用 AdamW。

R1.5 已经没有证据支持 group-wise LR 是主瓶颈。

---

# 22. 阶段执行路线

```text
D1-0  Implementation Audit + Unit Tests
 ↓
D1-1  R2-B0 Clean Parent
 ↓
D1-2  R2-F Functional Transfer
 ↓
D1-3  R2-S Semantic Refiner
 ↓
D1-4  R2-J Joint
 ↓
D1-5  Guard datasets + 3-seed confirmation
 ↓
D1-6  Final diagnosis
```

---

# 23. 参考 baseline

每个 R2 variant 同时报告：

### vs R2-B0

用于判断新增机制的净价值。

### vs current A0 / biaxis_final

用于判断是否真正解决原模型性能问题。

不要只看 relative B0。

---

# 24. D1-0 — Implementation Audit

目标：

- 不改 A0；
- 新 R2 模型完全独立；
- 确认 P0 hyperparameters 与 current `biaxis_final` 一致；
- 建立四 variant configs；
- 写完整 unit tests；
- 只做 forward/backward smoke。

---

# 25. D1-1 — R2-B0

第一轮：

```text
Movies/Toys/Grocery
seed42
```

B0 不是创新模型，目的只是回答：

> 删除 K-relation/Gamma/OFR 后，simple factor-wise propagation 是否仍能保持 A0 的主要性能？

### B0 interpretation

如果：

\[
mean_{M/T/G}(B0-A0)\ge-0.50pp
\]

且没有 dataset：

\[
<-1.0pp
\]

则：

```text
B0 = ACCEPTABLE CLEAN PARENT
```

如果 B0 平均明显优于 A0：

这是很强的“旧复杂关系链无必要”证据。

如果：

\[
mean<-0.8pp
\]

或单 dataset：

\[
<-1.5pp
\]

则先停止后续 R2，审计 B0 是否过弱/实现不公平。

---

# 26. D1-2 — R2-F Functional Transfer

先：

```text
M/T/G
seed42
```

Primary：

\[
Score_F
=
mean_{M/T/G}
[
ValAcc(R2F)-ValAcc(B0)
].
\]

### Seed42 GO-to-confirm

要求：

\[
Score_F\ge+0.30pp
\]

且：

```text
>=2/3 datasets positive
```

### Strong seed42

\[
\ge+0.50pp.
\]

同时看：

\[
R2F-A0.
\]

如果相对 B0 有提升但仍明显低于 A0，只能判“机制有效但尚未解决”。

---

# 27. D1-3 — R2-S Semantic Refiner

同样：

```text
M/T/G
seed42
```

Primary：

\[
Score_S
=
mean[
R2S-B0
].
\]

### GO-to-confirm

\[
Score_S\ge+0.20pp
\]

且 2/3 positive。

Semantic refinement 是 enhancement，不要求和 Functional 同样 +0.30 门槛。

### Strong

\[
\ge+0.50pp.
\]

---

# 28. D1-4 — R2-J Joint

只有满足以下任意条件才执行：

### Condition 1

R2-F：

```text
GO
```

且 R2-S：

\[
Score_S\ge-0.10pp.
\]

### Condition 2

R2-S Strong/GO，
而 R2-F 至少不是灾难性：

\[
Score_F\ge-0.10pp.
\]

如果 Functional 明确 NO-GO：

\[
Score_F<-0.30pp
\]

不要靠 Semantic Refiner 把它掩盖进 Joint。

---

# 29. R2-J seed42 GO

相对 B0：

\[
Score_J\ge+0.40pp
\]

且：

```text
>=2/3 target datasets positive
```

同时相对 A0：

\[
mean(R2J-A0)\ge+0.20pp
\]

才进入正式 3-seed。

目标不是仅仅证明比 B0 好，而是开始超过现 A0。

---

# 30. Guard datasets

对通过 seed42 GO 的 candidate：

```text
ele-fashion
Reddit-S
seed42
```

要求：

\[
\Delta ValAcc\ge-0.20pp
\]

相对 A0。

如果一个 candidate 在 M/T/G 有 gain，但 guards 明显掉：

\[
<-0.30pp
\]

需要先审查是否过度依赖 cross-factor graph transfer。

---

# 31. D1-5 — 3-seed Formal Confirmation

只有通过 seed42 + guards 的 candidate：

```text
seeds 42/43/44
5 datasets
Val only
```

A0 使用已有 formal seed42/43/44 结果，不必重新训练，前提是 A0 regression tests PASS。

---

# 32. Formal verdict

最终 candidate 相对 A0：

### STRONG

M/T/G mean：

\[
\ge+0.50pp
\]

且：

```text
>=2/3 target datasets positive
>=2/3 seeds positive on those datasets
guards safe
```

### GO

\[
\ge+0.30pp
\]

且 2/3 target positive，guards safe。

### WEAK

\[
+0.15\sim+0.30pp.
\]

### NO-GO

\[
<+0.15pp
\]

或不稳定。

---

# 33. Test protocol

整个 D1：

```text
evaluate_test=false
```

R2-Design-1 完成时也**不要自动 Test**。

先把 Val + mechanism diagnostics 返回人工审查。

只有下一阶段明确冻结 final model/config 后，再单独做 Test。

---

# 34. 必须保存的训练信息

每个 run：

```text
dataset
seed
variant
parameter_count
peak_memory
epoch_time
best_epoch
stop_epoch
best_val_acc
best_val_macro_f1
train_acc_at_best
train_loss_at_best
```

机制：

```text
common weights
semantic residual ratios
3x3 gate matrix
3x3 contribution matrix
base/functional residual ratios
rho scales
P0 factor health
```

---

# 35. 输出目录

建议：

```text
outputs/perf_r2d1/
  audit/
  b0/
  functional/
  semantic/
  joint/
  guards/
  confirm/
  summary/
```

输出：

```text
R2D1_AUDIT.md
R2D1_B0_REPORT.md
R2D1_FUNCTIONAL_REPORT.md
R2D1_SEMANTIC_REPORT.md
R2D1_JOINT_REPORT.md
R2D1_CONFIRM_REPORT.md
R2D1_FINAL_DIAGNOSIS.md

r2d1_results.csv
r2d1_mechanism.csv
r2d1_resource.csv
```

---

# 36. 需要实现的代码

建议：

```text
src/models/
  biaxis_r2.py
  biaxis_r2_components.py

configs/model/
  biaxis_r2_b0.yaml
  biaxis_r2_f.yaml
  biaxis_r2_s.yaml
  biaxis_r2_j.yaml

scripts/
  run_perf_r2d1.py
  summarize_perf_r2d1.py

tests/
  test_biaxis_r2.py
  test_biaxis_r2_components.py
```

不要修改：

```text
biaxis_p0.py
biaxis_p1.py
biaxis_p2.py
biaxis_p3.py
biaxis_final.yaml
```

除非只是 model registry 需要增加 import/registration。

---

# 37. Prompt 1 — D1-0 Implementation Audit + Code
## 先执行这一条

```text
我们进入 Bi-Axis Performance-R2-Design-1。

R2-0 已关闭，R2-Design-0 已冻结 Candidate A：
Factor-Context Functional Modulation。

本阶段不是继续做 probe，而是实现一个最小、可证伪的 end-to-end R2 architecture。

科学定义：

Semantic Ownership:
C / Pt / Pv

Functional Relation:
q_i^{a->b} = psi(F_i^b, N_i^a)

其中：
N_i^a = ordinary 1-hop mean aggregation of source factor a。

第一版 Relation 是 factor-context level，
不是 edge-level，
不做 K relations / Gamma / OFR / Gsim/Gdiff / multi-hop。

本 Prompt 只做：
implementation audit + code + unit tests + forward/backward smoke。

不要运行正式 M/T/G 训练。

==================================================
A. Repository audit
==================================================

审查：

src/models/biaxis_p0.py
src/models/biaxis_components.py
src/models/biaxis_p1_components.py
src/models/biaxis_p3.py
configs/model/biaxis_final.yaml
src/tasks/nc.py
model loading / Hydra registry

确认：

1. R2 可以直接继承/复用 P0 factorizer 和 _compute_aux；
2. current final A0 的 P0 hyperparameters：
   hidden_dim
   factor_dim
   dropout
   activation
   norm
   lambda_common
   lambda_orth
   lambda_recon
   必须原样复制；
3. neighbor_mean 可直接复用；
4. R2 不依赖 P1/P2/P3；
5. existing A0 行为不能改变。

输出：

docs/R2_Design_1_Implementation_Audit.md

==================================================
B. 新代码
==================================================

新建：

src/models/biaxis_r2_components.py
src/models/biaxis_r2.py

configs/model/biaxis_r2_b0.yaml
configs/model/biaxis_r2_f.yaml
configs/model/biaxis_r2_s.yaml
configs/model/biaxis_r2_j.yaml

tests/test_biaxis_r2_components.py
tests/test_biaxis_r2.py

必要时只增加最小 model registry 支持。

==================================================
C. 四个 variant
==================================================

同一个 Model，通过 config：

semantic_refiner.enabled
functional_transfer.enabled

产生：

B0:
semantic=false
functional=false

F:
semantic=false
functional=true

S:
semantic=true
functional=false

J:
semantic=true
functional=true

所有 variant 都保留 simple diagonal 1-hop B0 path。

==================================================
D. B0 graph path
==================================================

Factors：
[C,Pt,Pv]

N^a = neighbor_mean(F^a)

3 source transforms：

V_a = Linear(d,d,bias=False)

M_base^b = V_b(N^b)

msg_norm_base[b] = LayerNorm(d)

rho_base[b] =
sigmoid(raw_rho_base[b])

raw_rho_base init = 0
=> rho_base=.5

F_graph^b =
F^b
+
rho_base[b] * LN(M_base^b)

isolated N=0 时：
message must be exactly zero after LN
因此 output factor must equal input factor。

写 unit test。

==================================================
E. Semantic Refiner
==================================================

只 S/J 开启。

Adaptive Common：

u_c =
[c_t,c_v,c_t*c_v,abs(c_t-c_v)]

common gate：
Linear(4d,64)
GELU
Linear(64,2)
Softmax

最后 Linear weight/bias 全 0 初始化，
确保 step0:
w_t=w_v=.5

C0 =
w_t*c_t + w_v*c_v

Factor interaction：

I =
[
 C0*Pt,
 C0*Pv,
 Pt*Pv,
 abs(C0-Pt),
 abs(C0-Pv),
 abs(Pt-Pv)
]

trunk：
Linear(6d,d)
LayerNorm
GELU
Dropout(current dropout)

3 heads：
Linear(d,d,bias=False)

3 heads zero-init。

F*_b =
F0_b + head_b(trunk(I))

P0 aux losses 仍然只基于：
c_t,c_v,p_t,p_v
不对 refined output 加任何新 loss。

==================================================
F. Functional Transfer
==================================================

只 F/J 开启。

使用 semantic-refined factors（J）
或 raw ownership factors（F）。

先：

N^a = neighbor_mean(F*_a)

source transforms V_a 与 B0 共用。

source/target type embeddings：

3 src embeddings
3 tgt embeddings
type_dim=8

对每 a,b：

u_ab =
[
 F_b,
 N_a,
 F_b*N_a,
 abs(F_b-N_a),
 e_src[a],
 e_tgt[b]
]

shared scorer：

Linear(4d + 16,64)
GELU
Linear(64,1)

最后 Linear：
weight small normal std=1e-3
bias=0

gate：
g_ab = sigmoid(score)

禁止 Softmax。

message：
m_ab = g_ab * V_a(N_a)

M_func_b =
mean_a(m_ab)

msg_norm_func[b] = LayerNorm(d)

rho_func[b] =
direct learnable scalar
init = 0.01

最终：

F'_b =
F*_b
+
rho_base[b]*LN(M_base_b)
+
rho_func[b]*LN(M_func_b)

注意：
B0 diagonal path 始终存在。

==================================================
G. Final fusion
==================================================

继续使用和 P0/A0 一致的：

Linear(3d,hidden)
Norm
GELU
Dropout

不要 Transformer / MoE。

==================================================
H. Diagnostics
==================================================

实现 no-grad diagnostics：

Semantic：

common_weight_text mean/std
common_weight_visual mean/std
frac wt<.05 / >.95

semantic residual ratio C/Pt/Pv

existing:
common_sim
private_sim
C/P overlap
factor norms

Functional：

3x3 gate mean/std/p05/p50/p95
frac gate<.05
frac gate>.95

3x3 message contribution norm ratio

rho_base C/Pt/Pv
rho_func C/Pt/Pv

base residual ratio
functional residual ratio

==================================================
I. Unit tests
==================================================

必须覆盖：

1. B0/F/S/J forward shapes；
2. aux loss finite；
3. no-edge / isolated graph：
   B0 graph residual exactly zero；
4. semantic=false 时不实例化/不使用 semantic path；
5. functional=false 时 rho_func/scorer 不参与；
6. semantic gate zero-init -> exactly .5/.5；
7. semantic residual heads zero-init -> exact zero residual；
8. functional scorer output finite；
9. gate range [0,1]；
10. rho_func init=.01；
11. source/target factor order = [C,Pt,Pv]；
12. permutation of edge order does not change eval beyond numerical tolerance；
13. chunked neighbor_mean path identical；
14. old biaxis_final regression smoke unchanged；
15. no Test logic inside model。

==================================================
J. Resource smoke
==================================================

只做 synthetic + Movies small/full single forward/backward smoke。

记录：

B0/F/S/J parameter counts
peak forward memory
peak train-step memory

不要正式训练。

输出：

outputs/perf_r2d1/audit/R2D1_AUDIT.md

完成后停止。
不要跑 D1-1。
```

---

# 38. Prompt 2 — D1-1 R2-B0

```text
D1-0 audit 已人工/计划通过。

现在只运行 R2-B0 clean parent。

Datasets：
Movies/Toys/Grocery

Seed：
42

Val only
evaluate_test=false

Protocol：
300 epochs
patience30
best by Val Accuracy
same A0 lr=1e-3
wd=1e-4
full graph

同时读取现有 A0 seed42 Val 作为 reference，
不要重新训练 A0。

保存：
best checkpoint
training history
resource metrics
P0 diagnostics
rho_base

输出：

outputs/perf_r2d1/b0/R2D1_B0_REPORT.md
outputs/perf_r2d1/b0/b0_results.csv

必须给：

B0 - A0
per dataset
M/T/G macro mean

ACCEPTABLE CLEAN PARENT：

mean(B0-A0)>=-0.50pp
且没有 dataset<-1.0pp。

如果：
mean<-0.8pp
或任何 dataset<-1.5pp

标记：
B0 AUDIT REQUIRED

并停止后续，不自动跑 F/S。

如果 acceptable，
报告后停止。
```

---

# 39. Prompt 3 — D1-2 R2-F

```text
R2-B0 已确认 acceptable。

现在只运行 R2-F：
B0 + target-conditioned functional residual。

Datasets：
Movies/Toys/Grocery
Seed42
Val only。

所有训练协议与 B0 完全一致。

输出：

F vs B0
F vs A0

Val Acc
Val Macro-F1
best epoch
resource

必须输出 best checkpoint diagnostics：

3x3 gate matrix
3x3 contribution matrix
rho_base
rho_func
base residual ratio
functional residual ratio
P0 factor health

Primary：

Score_F =
mean_MTG(F-B0)

GO-to-confirm：

Score_F >= +0.30pp
且 >=2/3 datasets positive。

Strong：
>=+0.50pp。

额外记录：
mean_MTG(F-A0)

不要根据 gate pattern 修改模型。
不要调 scorer。
不要改 init。
不要跑 seeds43/44。

输出：

outputs/perf_r2d1/functional/
  functional_results.csv
  functional_mechanism.csv
  R2D1_FUNCTIONAL_REPORT.md

完成后停止。
```

---

# 40. Prompt 4 — D1-3 R2-S

```text
现在只运行 R2-S：
B0 + Semantic Refiner，
Functional Transfer OFF。

Movies/Toys/Grocery
seed42
Val only
same protocol。

输出：

S-B0
S-A0

common weight stats
semantic residual ratios
P0 ownership diagnostics

Primary：

Score_S =
mean_MTG(S-B0)

GO-to-confirm：

Score_S >= +0.20pp
且 >=2/3 datasets positive。

Strong：
>=+0.50pp。

重点检查：

common gate 是否 collapse；
semantic residual ratio 是否 >1；
P0 common/private health 是否明显恶化。

不要调整 gate/hidden size。
不要加 loss。

输出：

outputs/perf_r2d1/semantic/
  semantic_results.csv
  semantic_mechanism.csv
  R2D1_SEMANTIC_REPORT.md

完成后停止。
```

---

# 41. Prompt 5 — D1-4 R2-J

```text
读取已经完成的 R2-F / R2-S seed42 结果。

只有满足预注册条件才运行 R2-J：

Condition 1:
Score_F >= +0.30pp
且
Score_S >= -0.10pp

或：

Condition 2:
Score_S >= +0.20pp
且
Score_F >= -0.10pp

如果 Score_F < -0.30pp：
Functional core 明确失败，
不要用 Semantic Refiner 掩盖，
不运行 J。

若满足：

运行 R2-J
Movies/Toys/Grocery
seed42
Val only。

输出：

J-B0
J-A0
J-F
J-S

全部 semantic + functional diagnostics。

R2-J seed42 GO：

mean_MTG(J-B0)>=+0.40pp
>=2/3 datasets positive

并且：

mean_MTG(J-A0)>=+0.20pp

否则不进入正式 confirm。

输出：

outputs/perf_r2d1/joint/
  joint_results.csv
  joint_mechanism.csv
  R2D1_JOINT_REPORT.md

完成后停止。
```

---

# 42. Prompt 6 — Guards + 3-seed Confirmation

```text
根据 D1-2/3/4 的预注册门槛，
选择所有通过 GO-to-confirm 的 candidates。

不要因为某个 seed42 数字最高而临时选择新 variant。

Step A — Guards

对每个 candidate：

ele-fashion
Reddit-S
seed42
Val only

要求相对 A0：
每个 guard >= -0.20pp

记录 Macro-F1。

Step B — Formal confirm

只有 guards safe 的 candidates：

Movies/Toys/Grocery/ele-fashion/Reddit-S
seeds42/43/44
Val only

已有 seed42 结果复用，不重复。

A0 使用已有 formal 42/43/44 reference。

输出：

3-seed mean
population std ddof=0
positive seed count
paired delta vs B0
paired delta vs A0

Formal verdict relative A0：

STRONG:
M/T/G mean >= +0.50pp
>=2/3 target datasets positive
对应 dataset >=2/3 seeds positive
guards safe

GO:
>=+0.30pp
2/3 target positive
guards safe

WEAK:
+0.15~+0.30

NO-GO:
<+0.15 or unstable

同时汇总 mechanism stability：
gate matrix across seeds
common weights across seeds
residual scales across seeds

输出：

outputs/perf_r2d1/confirm/
  confirm_results.csv
  confirm_mechanism.csv
  R2D1_CONFIRM_REPORT.md

禁止 Test。

完成后停止。
```

---

# 43. Prompt 7 — Final R2-Design-1 Synthesis

```text
R2-Design-1 所有允许执行的实验已完成。

只读取已有结果。
不要跑新实验。
不要 Test。
不要调参。

输出：

outputs/perf_r2d1/summary/
  R2D1_MASTER_TABLE.csv
  R2D1_FINAL_DIAGNOSIS.md

必须回答：

1. R2-B0 是否是 acceptable clean parent？
2. 删除 K-relation/Gamma/OFR 后损失多少？
3. Functional Transfer 是否有 reproducible end-to-end gain？
4. Semantic Refiner 是否有 reproducible gain？
5. Joint 是否出现 synergy / interference？
6. 哪些 dataset 受益？
7. Movies 的 Pv-source conditional interaction 是否在 learned mechanism 中有对应证据？
8. learned gate 是否 collapse/saturate？
9. semantic refiner 是否破坏 ownership？
10. 新模型参数量/显存/时间相对 A0 如何？
11. 最佳 candidate 相对 current A0 是否达到 GO / STRONG？
12. 是否值得进入下一阶段 R2-Design-2 / final benchmark？

必须区分：

mechanism activation
vs
performance evidence

不得因为 gate 非均匀就说模型有效。

不得使用 Test。

最后给状态：

R2-Design-1:
PASS / PARTIAL / NO-GO

Best candidate:
B0 / F / S / J / none

然后停止，等待人工/ChatGPT。
```

---

# 44. 结果回来后需要给我的材料

完整执行允许的阶段后，请把以下材料一起返给我：

```text
docs/R2_Design_1_Implementation_Audit.md

outputs/perf_r2d1/audit/R2D1_AUDIT.md

outputs/perf_r2d1/b0/
outputs/perf_r2d1/functional/
outputs/perf_r2d1/semantic/
outputs/perf_r2d1/joint/       （若满足条件）
outputs/perf_r2d1/confirm/     （若满足条件）
outputs/perf_r2d1/summary/

相关 CSV
完整日志摘要
最新 GitHub 代码
```

特别需要：

```text
r2d1_results.csv
r2d1_mechanism.csv
r2d1_resource.csv
R2D1_FINAL_DIAGNOSIS.md
```

---

# 45. 当前不要做的事情

在 R2-Design-1 完成之前，禁止：

```text
看 Test
换 lr
扫 dropout
改 hidden/factor dim
加 2-hop
加 edge attention
加 OT
加 contrastive loss
加 synergy loss
把 gate 改成 feature-wise
把 scalar gate 改成 multi-head
给 9 cells 独立 full matrices
加 RoleMAG-style edge roles
加 DiP pseudo-nodes
```

如果 minimal R2 都无法带来 end-to-end gain，再讨论这些。

---

# 46. 当前执行方式

严格按：

```text
Prompt 1
→ Prompt 2
→ 检查 B0 gate
→ Prompt 3
→ Prompt 4
→ 根据门槛决定 Prompt 5
→ 根据门槛决定 Prompt 6
→ Prompt 7
```

执行。

不要让 AI 自行跳过 GO/NO-GO gate。

---

# 47. 本阶段最终科学问题

R2-Design-1 最终只需要回答：

\[
\boxed{
\textbf{
Can target-conditioned semantic factor–context interaction
produce reproducible end-to-end gains
once the old topology-prototype routing machinery is removed?
}
}
\]

如果答案是 YES：

正式 R2 的核心方法成立。

如果 Functional NO-GO、Semantic GO：

论文方向需要转为 Semantic Refinement + simple propagation。

如果 Functional/Semantic 都 NO-GO：

说明 R2-0 frozen headroom 不能被当前 end-to-end realization 兑现，需要重新研究 optimization/interaction realization，而不是继续堆模块。
