# Bi-Axis R2-Design-2.0
## Factor-Specific Propagation Horizon Validation Plan

**Repository:** `CrisRipper777/0901`  
**Previous stage:** `R2-Design-1.6 = PARTIAL`  
**Current mainline hypothesis:**  
\[
oxed{
	extbf{Different semantic ownership factors require different propagation horizons.}
}
\]

**Primary evidence entering this stage:**

- A0 graph-control: relation-neutralized ≈ full, local-only markedly worse  
  ⇒ graph propagation is essential, but current K-relation specialization is not.
- Pt-factor 2-hop signal is cross-parent reproducible on Movies/Toys/Grocery.
- Joint/global 2-hop is not supported.
- Reddit-S provides a natural negative-control: Pt 2-hop is harmful there.
- High-pass/diversification is not supported.
- Interaction route is only weak/parent-specific and is no longer the mainline.

**Stage objective:**  
把 frozen-probe 中观察到的 factor-specific propagation-range evidence 转化为最小、可训练、可解释的 end-to-end mechanism，并判断其是否能稳定超过 A0。

---

# 0. 本阶段只验证“传播尺度”这一条主线

R2-Design-2.0 不再并行发散其他方向。

本阶段禁止：

```text
K-prototype relation
Gamma relation router
OFR
PRODDIFF combination
FiLM
Semantic residual
high-pass channel
MoE
node-wise router
edge attention
3-hop/4-hop
PPR/APPNP
new contrastive/aux loss
OT/UOT
Test
large hyperparameter sweep
```

本阶段只允许：

```text
0-hop
1-hop
2-hop
factor-specific scale coefficients
warm-start / controlled unfreezing
```

---

# 1. 核心科学问题

本阶段要回答的不是：

> 2-hop 是否更好？

而是：

\[
oxed{
	extbf{
Do different semantic ownership factors
prefer different graph propagation horizons?
}
}
\]

具体：

\[
f\in\{C,P_t,P_v\}
\]

是否需要不同的：

\[
\gamma^f=[\gamma_0^f,\gamma_1^f,\gamma_2^f].
\]

其中：

\[
H_0^f=F^f
\]

\[
H_1^f=PH_0^f
\]

\[
H_2^f=PH_1^f.
\]

---

# 2. 设计原则

## 2.1 不硬编码 Pt=2-hop

Pt 2-hop 是诊断证据，不是最终设计。

不能写：

```text
Pt 固定用 2-hop
C/Pv 固定用 1-hop
```

因为：

- Reddit-S 中 Pt 2-hop明显有害；
- ele-fashion近似无明显 2-hop需求；
- 不同数据集可能有不同 factor-specific horizon。

正确目标：

\[
oxed{
	extbf{learn propagation profile, not hard-code hop depth}
}
\]

---

## 2.2 第一版先 factor-global，不做 node-wise

第一版系数：

\[
\gamma_k^f
\]

对所有节点共享。

即每个数据集训练出：

```text
C:  [γ0, γ1, γ2]
Pt: [γ0, γ1, γ2]
Pv: [γ0, γ1, γ2]
```

暂时不做：

\[
\gamma_{i,k}^f
\]

因为 node-wise dynamic routing 会重新引入 gate/router instability。

只有 global factor-specific profile formal GO 后，才考虑 node-adaptive。

---

## 2.3 第 0 步必须严格退化到现有 1-hop parent

所有新机制初始化时：

\[
	ext{output}=1	ext{-hop baseline}
\]

避免再出现“新模块初始化就扰乱 parent”的问题。

---

# 3. Parent 选择

## 3.1 Design Parent：B0

R2-Design-2.0 的机制实现先建立在 B0 clean scaffold 上：

\[
P0
ightarrow
	ext{factor-wise graph propagation}
ightarrow
fusion.
\]

理由：

- 不含旧 K/Gamma/OFR；
- graph path简单；
- M/T/G Acc与 A0 同一性能带；
- 适合严格测试 scale mechanism。

---

## 3.2 Performance Reference：A0

最终 formal verdict 必须比较：

\[
oxed{
	ext{candidate vs A0}
}
\]

而不是只比较 candidate vs B0。

因此：

```text
B0 = implementation parent
A0 = final performance reference
```

---

# 4. 三个正式 Variant

本阶段只做三个版本。

---

# 5. M0 — 1-hop Baseline

即当前 B0：

\[
H_{graph}^f=H_1^f.
\]

保留现有：

```text
source transform
message norm
base residual
fusion
```

M0 只作为 matched parent，不重新设计。

已有 formal B0 结果可复用。

---

# 6. M1 — Factor-Specific H1↔H2 Interpolation

这是最重要的第一候选。

对每 factor：

\[
D_2^f=H_2^f-H_1^f.
\]

定义：

\[
oxed{
\widetilde H^f
=
H_1^f
+
lpha_f(H_2^f-H_1^f)
}
\]

等价于：

\[
\widetilde H^f
=
(1-lpha_f)H_1^f+lpha_fH_2^f.
\]

其中：

```text
alpha_C
alpha_Pt
alpha_Pv
```

均为独立 learnable scalar。

---

## 6.1 M1 初始化

要求：

\[
lpha_f=0
\]

初始严格：

\[
\widetilde H^f=H_1^f.
\]

即 step0 与 B0 图上下文一致。

---

## 6.2 alpha 参数化

第一版推荐直接 parameter：

```python
alpha = nn.Parameter(torch.zeros(3))
```

不使用 sigmoid / softmax。

原因：

- 不人为限制到 [0,1]；
- 若某 dataset 需要 suppress 2-hop，可以学负值；
- 若更依赖 2-hop，可学正值；
- 不引入 gate saturation。

但报告必须记录：

```text
alpha magnitude
alpha sign
alpha seed stability
```

如果训练出现极端值：

\[
|lpha|>2
\]

标记 instability warning。

不要自动 clamp。

---

# 7. M1 的 graph message

先计算：

\[
H_0^f=F^f
\]

\[
H_1^f=PH_0^f
\]

\[
H_2^f=PH_1^f.
\]

再：

\[
\widetilde H^f
=
H_1^f+lpha_f(H_2^f-H_1^f).
\]

之后继续复用 B0 source transform：

\[
M^f=V_f(\widetilde H^f).
\]

再走：

```text
LayerNorm
rho_base
residual factor update
fusion
```

不增加额外 MLP。

这样：

\[
oxed{
	ext{M1 只增加 3 个真正关键 scalar parameters}
}
\]

非常适合 falsification。

---

# 8. M2 — Factor-Specific 0/1/2-Hop Mixture

只有 M1 seed42 GO 后才进入。

定义：

\[
oxed{
\widetilde H^f
=
\gamma_0^fH_0^f+
\gamma_1^fH_1^f+
\gamma_2^fH_2^f
}
\]

其中：

\[
\gamma^f
=
Softmax(	heta^f/	au).
\]

第一版：

```text
tau = 1.0
```

不扫 temperature。

---

## 8.1 M2 初始化

要求：

\[
\gamma_1^fpprox1
\]

\[
\gamma_0^f,\gamma_2^fpprox0.
\]

推荐 logits：

```text
theta_f = [-4, 4, -4]
```

这样：

```text
γ1 ≈ 0.9993
```

初始近似 B0 1-hop。

不要用 uniform init。

---

# 9. 为什么 M2 包含 0-hop

A0 graph-control 已经说明：

```text
local-only 很差
```

但这并不意味着：

\[
H_0
\]

完全没必要。

0-hop 在 scale basis 中承担：

\[
oxed{	ext{ego preservation}}
\]

而不是替代 graph propagation。

M2 的目标是允许每个 factor学习：

\[
	ext{local / 1-hop / 2-hop}
\]

的不同组合。

---

# 10. 第一阶段训练方式

M1/M2 都不建议 from-scratch。

采用：

\[
oxed{	extbf{B0 warm-start}}
\]

加载对应：

```text
dataset
seed
B0 best checkpoint
```

然后新增 scale parameters。

---

# 11. Phase A — Frozen Scale Calibration

第一步冻结 B0 parent：

```text
P0 factorizer frozen
source transforms frozen
graph norm frozen
fusion frozen
```

只训练：

```text
scale parameters
fresh classifier
```

M1：

```text
3 alpha scalars + classifier
```

M2：

```text
3x3 hop logits + classifier
```

目的：

\[
oxed{
	ext{先证明 scale basis 本身有 representation value}
}
\]

而不让 backbone co-adapt。

---

# 12. HEAD Control

同一个 B0 checkpoint：

```text
HEAD = frozen B0 + fresh classifier
```

M1/M2 必须复用：

```text
same exact classifier init
same seed
same train/val
same optimizer
```

不能用已有 B0 classifier 直接比较。

---

# 13. Frozen Training Protocol

```text
AdamW
lr = 1e-3
wd = 1e-4
300 epochs
patience = 30
best by Val Accuracy
```

因为 scale params极少：

weight decay 对 scale parameters建议：

```text
0
```

classifier：

```text
wd=1e-4
```

即 optimizer parameter groups：

```text
scale params: wd=0
classifier: wd=1e-4
```

不扫 lr。

---

# 14. M1 Seed42 Screen

Datasets：

```text
Movies
Toys
Grocery
seed42
```

Primary：

\[
Gain_{M1/B0}
=
mean(M1-HEAD).
\]

GO：

\[
oxed{
Gain\ge+0.30pp
}
\]

且：

```text
>=2/3 datasets positive
Macro-F1 无 <-0.50pp warning
```

Strong：

\[
\ge+0.50pp.
\]

---

# 15. M1 Mechanism Consistency

除了性能，必须同时检查：

### Expected probe alignment

Movies/Toys/Grocery：

预期：

\[
lpha_{Pt}>0.
\]

至少：

```text
>=2/3 datasets alpha_Pt > 0
```

并且：

\[
lpha_{Pt}
\]

平均大于：

\[
lpha_C,lpha_{Pv}
\]

则标记：

```text
PROBE-CONSISTENT
```

不是 GO 必要条件，但若性能提升却完全不满足，则机制解释需要谨慎。

---

# 16. Reddit-S Negative-Control

如果 M1 seed42 GO：

必须马上跑：

```text
ele-fashion
Reddit-S
seed42
```

Reddit-S 关键诊断：

frozen probe显示：

\[
Pt: H2-H1<0.
\]

因此期待：

\[
lpha_{Pt}^{Reddit}
\le
lpha_{Pt}^{M/T/G}.
\]

最好接近 0 或为负。

定义：

```text
NEGATIVE-CONTROL PASS
```

如果：

\[
lpha_{Pt}^{Reddit}
\]

仍然显著大正值，而且性能下降：

说明 global parameter 没学到正确 horizon，应停止。

---

# 17. M1 Guards

要求：

```text
ele-fashion Acc vs HEAD >= -0.20pp
Reddit-S Acc vs HEAD >= -0.20pp

Macro-F1 delta >= -0.50pp
```

如果 Reddit-S：

```text
performance safe
AND alpha_Pt 自动回退
```

这是非常强的 mechanism validation。

---

# 18. M1 Formal Confirmation

seed42 + guards 通过后：

运行：

```text
Movies
Toys
Grocery
seeds 42/43/44
```

每 seed：

加载对应 seed B0 checkpoint。

比较：

```text
M1 vs frozen HEAD
M1 representation最终 vs A0 formal Val
```

---

# 19. M1 Formal Verdict

### Mechanism GO vs B0 scaffold

\[
mean_{M/T/G}(M1-HEAD)\ge+0.30pp
\]

且：

```text
>=2/3 datasets positive
对应 dataset >=2/3 seeds positive
guards safe
```

### Final Candidate GO vs A0

更重要：

\[
oxed{
mean_{M/T/G}(M1-A0)\ge+0.30pp
}
\]

且：

```text
>=2/3 dataset mean positive
对应 positive dataset >=2/3 seeds positive
guards >= -0.20pp
Macro-F1 safe
```

只有这一条满足，M1 才能作为最终 architecture candidate。

---

# 20. M2 Entry Gate

只有 M1 满足至少：

```text
Mechanism GO vs B0
```

才执行 M2。

如果 M1 NO-GO：

不要因为 M2 更灵活就直接继续。

这能避免：

```text
simple hypothesis失败
→ 用更大自由度救
```

---

# 21. M2 Seed42 Screen

同样：

```text
Movies/Toys/Grocery seed42
```

Frozen B0。

比较：

```text
M2 vs HEAD
M2 vs M1
```

---

# 22. M2 额外 GO 判据

除了：

\[
Gain_{M2/B0}\ge+0.30pp
\]

还要求：

\[
M2-M1\ge+0.10pp
\]

macro，才说明 full 0/1/2 mixture 比简单 H1-H2 calibration值得。

否则：

\[
oxed{
M1 preferred by parsimony
}
\]

---

# 23. M2 Scale Diagnostics

必须输出：

3×3：

```text
factor x hop
γ0
γ1
γ2
```

以及：

```text
entropy per factor
effective hop depth
```

定义：

\[
Depth^f
=
0\gamma_0^f+
1\gamma_1^f+
2\gamma_2^f.
\]

期待：

M/T/G：

\[
Depth^{Pt}
>
Depth^C
\]

或：

\[
Depth^{Pt}
>
Depth^{Pv}
\]

至少部分稳定。

Reddit-S：

\[
Depth^{Pt}
\]

应回落。

---

# 24. Phase B — Controlled Unfreezing

只有 frozen M1/M2 达到 mechanism GO 才执行。

使用 R2D1.6 已经实现的 matched schedule framework。

---

# 25. 三种 Schedule

对同一：

```text
parent checkpoint
scale init
classifier init
seed
```

比较：

## S0 FROZEN

只：

```text
scale + classifier
```

## S1 GRAPH-UNFREEZE

```text
epoch1-30:
 scale + classifier

epoch31+:
 unfreeze source transforms
 unfreeze graph message norms
 keep P0 frozen
 keep fusion frozen
```

## S2 GRAPH+FUSION-UNFREEZE

```text
epoch1-30 frozen parent

epoch31+:
 unfreeze graph path
 unfreeze fusion
 keep P0 factorizer frozen
```

本阶段不 unfreeze P0。

---

# 26. 为什么不直接 FULL parent unfreeze

当前 ownership decomposition 是论文第一轴。

先冻结 P0：

\[
oxed{
	extbf{preserve semantic ownership anchor}
}
\]

如果 Scale route成立但必须重新改变 P0 才工作，说明解释会变复杂。

所以 Design-2.0 不做 P0 unfreeze。

---

# 27. Schedule GO

相对 frozen：

如果：

\[
S1/S2-FROZEN\ge+0.20pp
\]

且 Macro-F1 safe：

标记：

```text
ADAPTATION BENEFIT
```

若：

\[
<-0.20pp
\]

标：

```text
CO-ADAPTATION HARM
```

最终选择：

```text
best Val mean
+ safety
+ stability
```

而不是默认越多 unfreeze 越好。

---

# 28. Gradient Diagnostics

只在 schedule phase 做。

采样：

```text
epoch1
epoch10
epoch30
epoch31
best
```

对：

```text
source transforms
graph norms
fusion
```

记录：

```text
grad norm
parameter update ratio
```

本模型没有“branch off”复杂 counterfactual的必要性。

重点看：

```text
unfreeze 后是否突发梯度/参数漂移
```

---

# 29. Scale Diagnostics

每个 run 必须保存：

## M1

```text
alpha_C
alpha_Pt
alpha_Pv
```

per epoch / best epoch。

## M2

```text
gamma_C[0:2]
gamma_Pt[0:2]
gamma_Pv[0:2]
```

per epoch / best。

---

# 30. Smoothing Diagnostics

对每 factor：

\[
S_k^f
=
E_i\cos(H_0^f,H_k^f).
\]

记录：

```text
sim(H0,H1)
sim(H0,H2)
```

以及：

\[
\|H_2-H_1\|/\|H_1\|.
\]

帮助解释：

- 2-hop 是否只是更平滑；
- Pt 是否在结构上变化更大；
- 数据集差异。

---

# 31. Class-Conditional Diagnostics

只用于分析，不用于训练。

按 validation labels：

计算每 class：

```text
mean H1/H2 feature norm
M1/M2 prediction F1
```

如果实现成本低，可记录：

```text
per-class scale contribution
```

但不做 class-conditioned router。

---

# 32. Resource Diagnostics

记录：

```text
params
peak GPU memory
epoch time
best epoch
```

但不设置：

```text
must be far smaller than DiP
```

只要求：

```text
24GB 3090 feasible
```

---

# 33. Output Structure

```text
outputs/perf_r2d20/
  audit/
  m0/
  m1_screen/
  m1_guards/
  m1_confirm/
  m2_screen/
  m2_confirm/
  schedule/
  summary/
```

输出：

```text
R2D20_AUDIT.md

m1_results.csv
m1_scale_trajectory.csv
m1_smoothing.csv
R2D20_M1_REPORT.md

m2_results.csv
m2_scale_trajectory.csv
m2_smoothing.csv
R2D20_M2_REPORT.md

schedule_results.csv
schedule_scale.csv
schedule_gradient.csv
R2D20_SCHEDULE_REPORT.md

R2D20_MASTER_TABLE.csv
R2D20_HYPOTHESIS_LEDGER.csv
R2D20_FINAL_DIAGNOSIS.md
```

---

# 34. Hypothesis Ledger

至少包含：

```text
Shared propagation depth
Factor-specific propagation horizon
Pt-specific 2-hop demand
Global factor-specific scale
0/1/2-hop mixture
Node-adaptive scale
High-pass/diversification
Interaction PRODDIFF
K-prototype relation
Scale-based routing
From-scratch training
Warm-start training
Graph unfreezing
P0 ownership preservation
```

状态：

```text
SUPPORTED
OPEN
CONDITIONAL
WEAK
CLOSED
```

---

# 35. 最终 Route 决策

## Route C1 — Simple Factor-Specific Interpolation

如果：

```text
M1 formal GO vs A0
M2 no material improvement
```

则正式方法优先：

\[
oxed{
H_1^f+lpha_f(H_2^f-H_1^f)
}
\]

---

## Route C2 — Factor-Specific 0/1/2 Mixture

如果：

```text
M2 formal GO
AND
M2-M1 >= +0.10pp
```

则：

\[
oxed{
\sum_{k=0}^2\gamma_k^fH_k^f
}
\]

作为正式 axis。

---

## Route C3 — Node-Adaptive Scale

只有：

```text
M1/M2 global factor-specific mechanism formal GO
```

但存在：

```text
dataset内部 scale需求明显 heterogeneous
```

才进入 Design-2.1 考虑：

\[
\gamma_{i,k}^f.
\]

---

## Route E — Reopen Task-Aware Relation Learning

如果：

```text
M1 NO-GO
```

则不继续 M2/node-wise。

这意味着：

\[
oxed{
	ext{probe-level Pt 2-hop preference
cannot be realized by simple factor-specific horizon calibration}
\]

此时才转去：

```text
task-aware semantic-aware relation learning
```

而不是继续堆 scale complexity。

---

# 36. Prompt 1 — D2.0-0 Audit + Implementation

```text
进入 Bi-Axis R2-Design-2.0：
Factor-Specific Propagation Horizon Validation。

当前冻结：

A0 = Performance Reference
B0 = Design/Clean Parent

主假设：
Different semantic ownership factors require different propagation horizons.

关键证据：
Pt 2-hop 在 A0/B0 × Movies/Toys/Grocery × 3 seeds cross-parent 重复；
global/joint 2-hop不成立；
Reddit-S Pt 2-hop显著负；
high-pass关闭；
interaction不再作为主线。

本 Prompt 只做 implementation + audit + unit tests。
不要正式训练。
不要 Test。

实现统一 model：
biaxis_r2_scale

支持：

mode=M0
mode=M1
mode=M2

M0：
exact B0 1-hop。

M1：
H0=F
H1=P F
H2=P H1

Hmix_f =
H1_f + alpha_f*(H2_f-H1_f)

alpha_C/Pt/Pv direct nn.Parameter
init=0

禁止 sigmoid/softmax/clamp。

之后继续复用 B0：
source transform
message norm
rho_base
factor residual
fusion。

M2：
Hmix_f =
sum_k gamma_fk Hk_f

gamma=softmax(theta_f)
k=0,1,2

theta init=[-4,+4,-4]
tau=1 fixed。

M2 只有在后续 M1 GO 后才训练，
但代码/测试现在可实现。

新增：

src/models/biaxis_r2_scale.py
src/models/biaxis_r2_scale_components.py

configs/model/
 biaxis_r2_scale_m0.yaml
 biaxis_r2_scale_m1.yaml
 biaxis_r2_scale_m2.yaml

scripts/
 perf_r2d20_m1.py
 perf_r2d20_m2.py
 perf_r2d20_schedule.py
 summarize_perf_r2d20.py

tests/
 test_biaxis_r2_scale.py

必须保证：

1. M0 与 B0 forward/checkpoint-load 数学一致；
2. M1 alpha=0 -> exact M0；
3. M2 init -> numerically near M0，并报告 max diff；
4. isolated node保持正确；
5. H2 使用 same normalized neighbor_mean sequentially；
6. factor order [C,Pt,Pv]；
7. no high-pass；
8. no K relation/Gamma/OFR；
9. no Test access；
10. scale diagnostics 可返回。

实现 frozen trainer：

load B0 best checkpoint
freeze parent params
train scale + fresh classifier

exact classifier init可保存复用。

输出：
outputs/perf_r2d20/audit/R2D20_AUDIT.md

完成后停止。
```

---

# 37. Prompt 2 — M1 Seed42 Screen

```text
D2.0 audit PASS。

现在只运行 M1 frozen scale calibration。

Datasets：
Movies
Toys
Grocery

seed42
Val only
No Test。

每 dataset：
load对应 B0 best checkpoint
freeze all B0 parent params

train：
alpha_C/Pt/Pv
fresh classifier

HEAD：
same frozen B0
same exact classifier init
alpha fixed 0

optimizer：
alpha params lr=1e-3 wd=0
classifier lr=1e-3 wd=1e-4

300 epochs
patience30
best by Val Accuracy。

输出：

M1 vs HEAD：
Acc
Macro-F1
per-class F1
best epoch

alpha trajectory：
alpha_C
alpha_Pt
alpha_Pv
every epoch

best：
alpha values

smoothing：
cos(H0,H1)
cos(H0,H2)
norm(H2-H1)/norm(H1)

GO：
M/T/G macro M1-HEAD >=+0.30pp
>=2/3 datasets positive
无 Macro-F1<-0.50pp warning。

STRONG>=+0.50。

Mechanism consistency：
记录是否 >=2/3 datasets alpha_Pt>0，
以及 alpha_Pt 是否平均大于 alpha_C/Pv。

不要因为 alpha 不符合预期就修改模型。

输出：

outputs/perf_r2d20/m1_screen/
 m1_results.csv
 m1_scale_trajectory.csv
 m1_smoothing.csv
 R2D20_M1_SCREEN_REPORT.md

完成后停止。
```

---

# 38. Prompt 3 — M1 Guards + Formal Confirm

```text
只有 M1 seed42 GO 才执行。

Step A Guards：

ele-fashion
Reddit-S
seed42

同 frozen B0 protocol。

要求：

Acc vs HEAD >= -0.20pp
Macro-F1 >= -0.50pp

重点输出 Reddit-S：

alpha_Pt
并与 M/T/G alpha_Pt 比较。

标记：
NEGATIVE-CONTROL PASS
如果 Reddit-S alpha_Pt 明显更低/接近0/为负，
且 performance safe。

Step B Formal：

Movies
Toys
Grocery
seeds42/43/44

每 seed对应 B0 checkpoint。

输出：

M1 vs HEAD
M1 final metric vs formal A0

3-seed mean
population std ddof=0
positive seed count

Scale：
best alpha mean±std
alpha sign consistency

Mechanism GO vs B0：

M/T/G M1-HEAD macro >=+0.30pp
>=2/3 dataset mean positive
positive dataset >=2/3 seeds positive
guards safe

Final GO vs A0：

M/T/G M1-A0 macro >=+0.30pp
>=2/3 dataset mean positive
对应 >=2/3 seeds positive
guards >=-0.20pp
Macro-F1 safe

输出：

outputs/perf_r2d20/m1_confirm/
 m1_confirm_results.csv
 m1_confirm_scale.csv
 R2D20_M1_CONFIRM_REPORT.md

如果 Mechanism GO 不成立：
禁止运行 M2。
停止等待 synthesis。

如果 Mechanism GO：
可进入 M2。
```

---

# 39. Prompt 4 — M2 Screen / Confirm

```text
只有 M1 Mechanism GO 才执行 M2。

M2：
factor-specific 0/1/2-hop mixture。

H0
H1
H2

gamma_f=softmax(theta_f)
theta init=[-4,4,-4]
tau=1 fixed。

先：
Movies/Toys/Grocery seed42
frozen B0
same HEAD protocol。

输出：

M2-HEAD
M2-M1

gamma 3x3：
C/Pt/Pv × hop0/1/2

entropy per factor
effective depth：
Depth_f=sum k*gamma_fk

GO：

M2-HEAD >=+0.30pp
>=2/3 positive
F1 safe

且若：
M2-M1 < +0.10pp macro

则标：
NO MATERIAL ADVANTAGE OVER M1
优先 M1。

如果 M2 seed42 有 material advantage：
补 guards + seeds43/44，
使用 M1 同样 formal protocol。

输出：

outputs/perf_r2d20/m2_screen/
outputs/perf_r2d20/m2_confirm/
R2D20_M2_REPORT.md

完成后停止。
```

---

# 40. Prompt 5 — Controlled Schedule

```text
只对 formal Mechanism GO 的最佳 M1/M2 candidate 执行。

使用 exact same：
B0 checkpoint
scale initial state
classifier initial state
seed

比较：

S0 FROZEN：
scale+classifier only

S1 GRAPH-UNFREEZE：
epoch1-30 frozen
epoch31+ unfreeze:
 source transforms
 graph message norms
 keep P0 frozen
 keep fusion frozen

S2 GRAPH+FUSION：
epoch1-30 frozen
epoch31+ unfreeze:
 graph path
 fusion
 keep P0 factorizer frozen

第一轮：
Movies/Toys/Grocery seed42

如果 schedule差异 >=0.20pp，
再补 seeds43/44。

输出：

Acc
Macro-F1
best epoch
scale coefficients
parameter drift
grad norm
update ratio

采样：
epoch1/10/30/31/best

Verdict：

ADAPTATION BENEFIT：
S1/S2-FROZEN >=+0.20pp
F1 safe

CO-ADAPTATION HARM：
<=-0.20pp

JOINT SAFE：
within ±0.10pp。

输出：

outputs/perf_r2d20/schedule/
 schedule_results.csv
 schedule_scale.csv
 schedule_gradient.csv
 R2D20_SCHEDULE_REPORT.md

完成后停止。
```

---

# 41. Prompt 6 — Final Synthesis

```text
R2-Design-2.0 所有允许阶段已完成。

不要跑新实验。
不要 Test。
不要调参。
不要实现 node-wise routing。

读取：

M1
M2
guards/formal
schedule

输出：

outputs/perf_r2d20/summary/
 R2D20_MASTER_TABLE.csv
 R2D20_HYPOTHESIS_LEDGER.csv
 R2D20_FINAL_DIAGNOSIS.md

必须回答：

1. factor-specific propagation horizon 是否 end-to-end成立？
2. M1 是否胜 B0 HEAD？
3. M1 是否 formal 胜 A0？
4. alpha_Pt 是否与 frozen Pt 2-hop诊断一致？
5. Reddit-S 是否自动回退？
6. C/Pt/Pv 是否学到不同 horizon？
7. M2 是否真正优于 M1？
8. 0-hop 是否有价值？
9. schedule 是否改善？
10. 是否存在 co-adaptation harm？
11. Macro-F1 是否安全？
12. 下一阶段选择：
   C1 simple interpolation
   C2 0/1/2 mixture
   C3 node-adaptive scale
   E reopen task-aware relation
13. 是否值得进入 R2-Design-2.1 最终 architecture consolidation？

状态：

R2-Design-2.0:
PASS / PARTIAL / NO-GO

不要跑 Test。
等待人工/ChatGPT。
```

---

# 42. 完成后返给我的材料

请返回：

```text
outputs/perf_r2d20/audit/R2D20_AUDIT.md

outputs/perf_r2d20/m1_screen/
outputs/perf_r2d20/m1_confirm/       # 若进入
outputs/perf_r2d20/m2_screen/        # 若进入
outputs/perf_r2d20/m2_confirm/       # 若进入
outputs/perf_r2d20/schedule/         # 若进入
outputs/perf_r2d20/summary/

R2D20_MASTER_TABLE.csv
R2D20_HYPOTHESIS_LEDGER.csv
R2D20_FINAL_DIAGNOSIS.md

m1_results.csv
m1_scale_trajectory.csv
m1_smoothing.csv
m1_confirm_results.csv
m1_confirm_scale.csv

m2_results.csv
m2_scale_trajectory.csv

schedule_results.csv
schedule_scale.csv
schedule_gradient.csv

最新 GitHub commit
```

---

# 43. 本阶段最重要的纪律

不得：

```text
看到 Pt 2-hop probe 很强 → 硬编码 Pt=2
M1 失败 → 直接用更复杂 M2救
M2 失败 → 直接 node-wise router
scale coefficient 非均匀 → 宣称有效
只看 B0 gain 不看 A0
只看 Accuracy 不看 Macro-F1
```

必须同时满足：

\[
oxed{
	ext{performance}
+
	ext{seed stability}
+
	ext{probe-mechanism consistency}
+
	ext{guard safety}
}
\]

---

# 44. R2-Design-2.0 最终科学判据

如果 M1/M2 formal GO：

\[
oxed{
	extbf{
Semantic ownership factors exhibit distinct,
learnable and task-relevant propagation horizons.
}
}
\]

这才足以进入 R2-Design-2.1 做最终方法整合。

如果 M1 都 NO-GO：

\[
oxed{
	extbf{
The observed Pt 2-hop probe signal is not directly realizable
through simple factor-specific horizon calibration.
}
}
\]

则停止继续堆 scale mechanism，转回：

\[
oxed{
	extbf{Task-aware / semantic-aware relation learning}
}
\]

重新定义第二轴。
