# Bi-Axis Performance-R1.5 — A0 Optimization / Capacity / Training Audit Plan

## 0. 阶段定位

Performance-R1 已结束：A1、A1+reg、A2、BL、BR、BLR、C1SG 均 NO-GO。R1.5 不再发明新 relation/router/multi-hop 机制，也不组合 A/B/C。目标是在进入 R2 前系统排除：

- optimizer / learning rate / weight decay 未调优；
- hidden/factor capacity 不足；
- P0 auxiliary objectives 与下游 CE 冲突；
- dropout / schedule / group-wise LR 不合理；
- R1 中“机制激活但 end-task 下降”是否主要来自 optimization/co-adaptation；
- 当前结构输入带宽是否不足。

所有 R1.5 模型选择只依据 Validation；最终配置冻结前禁止用 Test 选择。

---

## 1. 当前显存优化的处理原则

当前 main 已有两项 implementation optimization：

1. P3 的 relation aggregation → g_perm → transport scorer 使用 `torch.utils.checkpoint(..., use_reentrant=False)`；
2. `relation_weighted_mean()` 将 `features[src]` gather 提出 relation loop，chunk 路径每 chunk 只 gather 一次。

这两项理论上不改变模型函数，但 R1.5 必须先做 regression audit。后续所有 seed42 paired comparison 都以 **当前 main + memory_checkpoint=true 的 fresh A0** 为 anchor，不能直接把旧训练轨迹当 paired baseline。

---

## 2. 总路线

```text
R15-0  Memory Patch + Fresh A0 Regression Audit
  ↓
R15-1  Training / Gradient / Optimization Audit
  ↓
R15-2  LR × Weight Decay Screen
  ↓
R15-3  Capacity Screen
  ↓
R15-4  Aux-loss / Dropout / Schedule / Group-LR（条件执行）
  ↓
R15-D  R2-readiness diagnostics
        ├─ Frozen-A0 2-hop adapter sanity
        └─ Structural-observation headroom probe
  ↓
R15-F  3-seed confirmation
```

---

## 3. 统一纪律

弱项 target：

```text
Movies / Toys / Grocery
```

guards：

```text
ele-fashion / Reddit-S
```

Primary：Val Accuracy  
Secondary：Val Macro-F1

效应等级（相对当前代码 fresh A0）：

- Strong：M/T/G mean ΔVal ≥ +0.50 pp，至少 2/3 为正；
- Useful：≥ +0.30 pp，至少 2/3 为正；
- Weak：+0.15 ~ +0.30 pp；
- No meaningful signal：< +0.15 pp。

guard 默认要求 ele/Reddit ΔVal ≥ −0.20 pp。

R1.5 禁止：
- 重新启用 A1/A2/BL/BR/BLR/C1SG；
- 组合 A+B/A+C/B+C/A+B+C；
- 改 K / relation architecture / operator family；
- 用任何 R1 variant 的 Test “复活”模型。

---

## 4. Test 隔离

建议增加 backward-compatible：

```yaml
task:
  evaluate_test: false
```

默认 `true`，不改变正式 benchmark。

当 false：
- 不访问 `test_idx`；
- 不计算/记录 test metrics；
- 仍保存 best model/head checkpoint；
- R1.5 screen 全部用 false。

只有 R15-F 最终配置完全冻结后再 `evaluate_test=true`。

---

# R15-0 — Memory Patch Regression Audit

## 5. Same-weight forward equivalence

同一 weights，对：

```text
p3.memory_checkpoint=false
p3.memory_checkpoint=true
```

在 Movies / Grocery / ele-fashion 比较：

```text
z_final
g_perm
gamma
total loss
```

要求至少：

```text
allclose(rtol=1e-6, atol=1e-7)
```

GPU atomic 引入极小误差时不强求 bitwise。

## 6. One-step gradient equivalence

固定相同：
- initialization；
- data；
- RNG state；
- dropout RNG。

比较 checkpoint OFF/ON 的：

```text
CE loss
aux loss
total loss
```

及参数组 gradient：

```text
P0 factorizer
P1 structural encoder/prototypes
P2 scorer/null score
P3 operator
fusion
classifier
```

记录：

\[
err_g = ||g_on-g_off|| / (||g_off||+eps)
\]

主要参数组应约 ≤1e-5~1e-4。若明显更大，先修 memory patch。

## 7. Gather-hoist equivalence

写旧 reference：

```text
features[src] inside relation loop
```

比较 new implementation 的：
- g；
- mass；
- grad(features)；
- grad(r)；
- full path；
- chunk path。

## 8. Fresh A0 seed42 anchor

当前 main、memory_checkpoint=true：

```text
Movies/Toys/Grocery/ele-fashion/Reddit-S
seed42
300 epochs
patience30
evaluate_test=false
```

保存：
- Val Acc/F1；
- best epoch；
- stop epoch；
- peak memory；
- epoch time；
- best checkpoint。

后续 seed42 delta 全部相对这一 anchor。

---

# R15-1 — Training / Gradient / Optimization Audit

## 9. 保存完整训练历史

每 epoch：

```text
epoch
train_total_loss
train_ce_loss
train_aux_loss
train_acc
val_acc
val_macro_f1
lr
patience_left
common_loss
orth_loss
recon_loss
common_sim
private_sim
C/Pt/Pv norm
C-P overlap
```

统计：
- best_epoch；
- stop_epoch；
- train-val gap；
- last-20-epoch val slope；
- 是否 hit 300；
- 是否 early plateau；
- 是否明显 overfit。

## 10. Gradient decomposition

只做 Movies/Toys/Grocery。

状态：
1. seed42 initialization；
2. fresh A0 best checkpoint。

参数组：

```text
G0 text_projector
G1 visual_projector
G2 common_encoder
G3 private_text_encoder
G4 private_visual_encoder
G5 local fusion
G6 structural signature + edge token + prototypes
G7 transport scorer / null score
G8 P3 operator
G9 classifier
```

分别计算加权后的：

\[
L_{CE},\quad
L_c=\lambda_cL_{common},\quad
L_o=\lambda_oL_{orth},\quad
L_r=\lambda_rL_{recon}.
\]

记录：

\[
||g_{CE}||,\ ||g_c||,\ ||g_o||,\ ||g_r||,\ ||g_{aux}||,\ ||g_{total}||.
\]

重点指标：

\[
R_{aux/CE}=||g_{aux}||/(||g_{CE}||+eps)
\]

经验解释：
- <0.2：aux 很弱；
- 0.2~1：同量级；
- >1：可能主导；
- >3：强烈怀疑 imbalance。

再计算：

\[
cos(g_{CE},g_c),\quad
cos(g_{CE},g_o),\quad
cos(g_{CE},g_r),\quad
cos(g_{CE},g_{aux}).
\]

重点：
- <−0.2：实质 conflict；
- <−0.4：强 conflict。

## 11. Parameter-group update ratio

一个 optimizer step 后：

\[
U_g=||\Delta	heta_g||/(||	heta_g||+eps)
\]

报告 G0~G9。

若关键 group 间 max/min >10×，则统一 LR 值得怀疑，R15-4 才考虑 group-wise LR。

## 12. R15-1 输出

```text
outputs/perf_r15/audit/MEMORY_PATCH_AUDIT.md
outputs/perf_r15/anchor/fresh_a0_seed42.csv
outputs/perf_r15/audit/training_history_summary.csv
outputs/perf_r15/audit/gradient_norms.csv
outputs/perf_r15/audit/gradient_cosines.csv
outputs/perf_r15/audit/update_ratios.csv
outputs/perf_r15/audit/R15_TRAINING_AUDIT.md
```

报告只给事实性结论：
- aux objective imbalance：evidence / no evidence；
- CE-vs-aux conflict：evidence / no evidence；
- unified LR group imbalance：evidence / no evidence；
- under-training / overfitting：evidence / no evidence。

不要自动调参。

---

# R15-2 — LR × Weight Decay Screen

R15-1 审查 PASS 后执行。

\[
lr\in\{3e-4,1e-3,3e-3\}
\]

\[
wd\in\{0,1e-4,1e-3\}
\]

9 组；baseline `1e-3/1e-4` 已有 fresh anchor，不重复。

新增：

```text
8 configs × Movies/Toys/Grocery × seed42 = 24 runs
```

global score：

\[
Score_{opt}=mean(\Delta M,\Delta T,\Delta G)
\]

只取 Top-2 global configs，再跑 ele/Reddit guards。

最终选 R15-OPT。若所有候选 mean gain < +0.15 pp，则 R15-OPT=baseline。

---

# R15-3 — Capacity Screen

Parent optimizer = R15-OPT。

第一轮不改 relation_dim/K。

候选：

```text
C0 hidden=256 factor=128
C1 hidden=384 factor=128
C2 hidden=512 factor=128
C3 hidden=384 factor=160
C4 hidden=384 factor=192
```

解释：
- C1/C2：增强 modality projector/fusion capacity；
- C3/C4：增强 factor + factor-relation operator capacity。

每个新 capacity 先 ele-fashion 2-epoch memory smoke。

若：
```text
peak >22GB 或 OOM
```
直接 reject。

通过者跑 M/T/G seed42，最多保留 Top-2，再做 guards。

若全部 mean gain < +0.15 pp，则回 256/128，不继续 768/256 暴力扩容。

---

# R15-4 — Objective / Regularization / Schedule

必须由 R15-1 证据驱动，禁止大网格。

### 如果 reconstruction gradient 强且与 CE 冲突

```text
L0 baseline  λc=.02 λo=.01 λr=.30
L1 weak-rec  .02 / .01 / .10
L2 no-rec    .02 / .01 / 0
```

### 如果所有 aux 都明显冲突

加纯诊断：

```text
L3 no-aux = 0/0/0
```

如果性能升但 factor semantics 崩，结论是 factor objective 与 downstream objective 有 tradeoff；不要直接把 no-aux 当最终论文模型。

### 如果 Common 过弱、recon 不冲突

可测：

```text
λc=.05
λo=.01
λr=.10 或 audit 支持值
```

### Dropout

只有训练历史显示 overfit 才测：

```text
0.1 / 0.2 / 0.3
```

### Scheduler

只有 best epoch 很晚或 constant-LR plateau 时测一个：

```text
10-epoch warmup + cosine decay to 1e-5
```

### Group-wise LR

只有 update-ratio audit 明确支持时才测；不要提前展开笛卡尔网格。

每个 conditional family 先 M/T/G seed42；mean gain ≥+0.15 且 2/3 positive 才跑 guards。

最终得到 R15-TUNED。

---

# R15-D1 — Frozen-A0 2-hop Adapter Sanity

这是诊断，不自动成为 final model。

只做 M/T/G seed42。

加载 best A0/tuned-A0 checkpoint，彻底冻结：

```text
P0/P1/P2/P3
```

在 no_grad 中预计算 F0/F1/F2。

Matched comparison：

### DA0
只训练 fresh Linear head on F1。

### DA2
训练 small zero-init 2-hop adapter + fresh Linear head：

\[
F^{out}=F^{(1)}+\lambda W(F^{(2)}-F^{(1)})
\]

backbone 绝对无 gradient。

若 Movies/Toys DA2−DA0 ≥+0.4pp，说明 second-hop 本身有价值，C1SG 失败更可能与 joint optimization/backbone drift 有关。

若仍≈0/negative，彻底关闭 multi-hop hypothesis。

---

# R15-D2 — Structural Observation Headroom Probe

当前 structural relation 的原始输入只有：

\[
S_3=[\log d,P\log d,P^2\log d]
\]

R1 从未扩展 topology observation。

只做 M/T/G。

构造 topology-only：

\[
S_+=[
\log d,
P\log d,
P^2\log d,
P^3\log d,
mean_N(d),
std_N(d),
mean_N(\log d),
std_N(\log d)
]
\]

固定 StandardScaler + Ridge(alpha=1.0)，train fit / val eval。

比较：

```text
Probe(z_final)
Probe([z_final|S3])
Probe([z_final|S+])
```

定义：

\[
\Delta_{struct-headroom}
=
Probe([z_{final}|S_+])-Probe([z_{final}|S_3])
\]

若至少 2/3 weak datasets ≥+0.3~0.5 pp，是 R2 “structural observation bandwidth 不足”的强证据。

---

# R15-F — 3-seed Final Confirmation

只有 R15-TUNED seed42：

\[
mean\Delta_{M/T/G}\ge+0.30pp
\]

才执行。

跑 current-code：

```text
A0 seeds43/44
R15-TUNED seeds43/44
5 datasets
evaluate_test=false
```

与 seed42 合并。

判定：
- Strong：M/T/G mean ≥+0.50 pp；
- Useful：≥+0.30；
- Weak：+0.15~+0.30；
- NO-GO：<+0.15。

最终 config 完全冻结后，才 `evaluate_test=true` 做唯一一次正式 Test。

---

# AI / Codex Prompt 1 — 现在只执行这个

```text
我们进入 Bi-Axis Performance-R1.5。

背景：
- R1-A/B/C 所有新增机制均 NO-GO。
- final parent 仍为 biaxis_final A0。
- R1.5 不是新架构阶段，目标是在进入 R2 前排除：
  optimizer / lr / wd / capacity / aux-objective / training dynamics
  是否才是当前性能瓶颈。
- 禁止实现任何新 relation/router/multi-hop mechanism。
- 禁止读取 Test 做任何选择。

当前 main 刚做两个 memory optimization：
1. biaxis_p3 context+transport-scorer 段用
   torch.utils.checkpoint(use_reentrant=False)
2. relation_weighted_mean 将 features[src] gather 移出 relation loop /
   每 chunk 只 gather 一次。

第一步只做 R15-0 + R15-1，不跑 lr/wd/capacity sweep。

A. Memory patch regression audit

1. 审计：
   src/models/biaxis_p1_components.py
   src/models/biaxis_p3.py
   configs/model/biaxis_final.yaml
   src/tasks/nc.py

2. same weights 比较 p3.memory_checkpoint=false/true：
   Movies/Grocery/ele-fashion。
   比较 z_final/g_perm/gamma/loss，
   allclose(rtol=1e-6,atol=1e-7)，记录 max abs/rel error。

3. one-step gradient equivalence：
   same init/data/RNG；
   控制 dropout RNG；
   比较 OFF/ON 参数组 gradient relative error。

4. 为 relation_weighted_mean 写旧 gather reference，
   对 full/chunk path 比较：
   g/mass/grad(features)/grad(r)。

如果 regression FAIL，停止，不训练。

B. Val-only screen capability

给 nc protocol 增加 backward-compatible：
task.evaluate_test，default=true。

R1.5 override=false。
false 时：
- 禁止访问 test_idx；
- 不计算 test metric；
- 保存 best model/head；
- 只返回 Val；
- 原 benchmark default 行为不变；
- 写 unit test。

C. Fresh current-code A0 anchor

memory audit PASS 后运行：
Movies/Toys/Grocery/ele-fashion/Reddit-S
seed42
model=biaxis_final
memory_checkpoint=true
evaluate_test=false
300ep/patience30。

所有 run 保存 best checkpoint。

每 epoch history：
epoch/train_total_loss/train_ce_loss/train_aux_loss/train_acc/
val_acc/val_macro_f1/lr/patience_left/
P0 common/orth/recon losses + existing aux info。

D. Gradient / optimization audit

只用 Movies/Toys/Grocery。

状态：
1. seed42 initialization
2. fresh A0 best checkpoint

参数组：
text_projector
visual_projector
common_encoder
private_text_encoder
private_visual_encoder
local fusion
structural encoder + edge token + prototypes
transport scorer/null score
P3 operator
classifier

分别计算：
weighted CE/common/orth/reconstruction/aux/total gradient norms；
aux/CE norm ratio；
cos(CE, common/orth/recon/aux)。

再计算一个 optimizer step 的：
||Delta theta_group||/(||theta_group||+eps)。

E. Training history synthesis

输出：
best_epoch
stop_epoch
train-val gap
last20-val slope
early plateau
hit max epoch
overfit evidence

输出：
outputs/perf_r15/audit/MEMORY_PATCH_AUDIT.md
outputs/perf_r15/anchor/fresh_a0_seed42.csv
outputs/perf_r15/audit/training_history_summary.csv
outputs/perf_r15/audit/gradient_norms.csv
outputs/perf_r15/audit/gradient_cosines.csv
outputs/perf_r15/audit/update_ratios.csv
outputs/perf_r15/audit/R15_TRAINING_AUDIT.md

最后只给事实性判定：
- aux objective imbalance: evidence/no evidence
- CE-vs-aux gradient conflict: evidence/no evidence
- unified LR group imbalance: evidence/no evidence
- under-training/overfitting: evidence/no evidence

不要执行 R15-2。
等待人工审查。
```

---

# 后续 Prompt（暂时不要执行）

## Prompt 2 — LR/WD

```text
R15-0/R15-1 audit 已人工通过。
只做：
lr=[3e-4,1e-3,3e-3]
wd=[0,1e-4,1e-3]
M/T/G seed42，Val only。
baseline 1e-3/1e-4 不重复。
按 M/T/G mean delta 选 Top-2，再跑 ele/Reddit guards。
不要跑 capacity。
```

## Prompt 3 — Capacity

```text
Parent optimizer=人工确认 R15-OPT。
候选：
256/128, 384/128, 512/128, 384/160, 384/192。
每个新 variant 先 ele-fashion 2ep memory smoke；
peak>22GB/OOM reject。
通过者 M/T/G seed42，Top-2 guards。
不要继续扩大到 768/256。
```

## Prompt 4 — Objective/Training

```text
只运行人工根据 R15-1 gradient/training audit 指定的 2~3 个 contrasts。
禁止 lambda 大网格。
M/T/G seed42；mean gain>=+0.15 且 2/3 positive 才 guards。
```

## Prompt 5 — R2-readiness

```text
A. Frozen-backbone 2-hop adapter：DA0 vs DA2，M/T/G seed42。
B. Structural headroom：S3 vs Splus fixed Ridge。
均只看 Val，不改主模型。
```

## Prompt 6 — 3-seed confirm

```text
只有 R15-TUNED seed42 M/T/G mean>=+0.30 才跑。
A0 + R15-TUNED seeds43/44，5 datasets，Val only。
先回传人工审查，不读 Test。
```

---

## 当前立即执行

现在只执行 **Prompt 1：R15-0 + R15-1 Audit**。

R1.5 的目的不是再找一个“看起来新”的模块，而是把下面这个问题回答干净：

> 当前 Bi-Axis 的弱项性能差距，主要来自普通训练/容量不足，
> 还是来自架构本身没有获得足够丰富的 semantic / structural evidence？

如果 R1.5 tuning 明显改善，说明 A0 under-tuned；
如果仍没有实质改善，进入 R2 就是由 R0→R1→R1.5 的系统证据推动的重构，而不是盲目换架构。
