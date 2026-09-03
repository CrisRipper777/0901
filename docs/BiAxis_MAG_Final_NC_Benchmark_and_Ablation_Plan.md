# Bi-Axis MAG — Final NC Benchmark & Paper-facing Ablation Plan
## 阶段目标：冻结模型后回到统一 Benchmark，完成 NC 主对比、正式消融、效率与机制证据

> Repository: `CrisRipper777/0901`  
> Final model entry: `model=biaxis_final`  
> 当前范围：**只做 Node Classification**  
> NC datasets: Movies / Toys / Grocery / ele-fashion / Reddit-S  
> Seeds: 42 / 43 / 44（正式主表统一 3 seeds；已有 5-seed P3 结果仅作为补充稳定性证据）  
> Primary architecture-selection metric: Validation Accuracy  
> Paper metrics: Test Accuracy + Test Macro-F1  
> Std: population std (`ddof=0`)  
> **模型架构已 Frozen；本阶段禁止再设计 P4 或根据 Test 结果修改主体模型。**

---

# 0. 这一阶段最重要的观念转换

P0 / P1 / P2 / P3 是**内部研发漏斗**，不是最终论文的模块命名。

论文不应该写成：

```text
P0 -> P1 -> P2 -> P3
```

也不应该把正式消融简单写成：

```text
w/o P1
w/o P2
w/o P3
```

正式论文应回到最终故事：

\[
\boxed{\text{Semantic Ownership} \times \text{Relational Function}}
\]

最终方法可以面向论文抽象为四个概念步骤：

1. **Semantic Ownership Modeling**
2. **Structural Relation Discovery**
3. **Unified Factor–Relation Allocation**
4. **Hierarchical Factor–Relation Transformation**

因此，正式消融必须回答：

> 去掉某一个“论文 claim”后会怎样？

而不是回答：

> 去掉某个历史开发阶段会怎样？

---

# 1. 当前最终模型（Frozen）

最终消息：

\[
m_i^f
=
\sum_{k=1}^{K}
\Gamma_{ifk}
T_{fk}(g_{ik}^{f})
\]

其中：

\[
T_{fk}=W_0+A_f+B_k+C_{fk}.
\]

完整数据流：

```text
Text / Image
    ↓
Semantic Ownership
C / Pt / Pv
    ↓
Topology-only Structural Relation Discovery
R1 ... RK
    ↓
Factor × Relation contexts g_ifk
    ↓
Unified Null-Augmented Plan Γ_ifk
    ↓
Hierarchical Factor–Relation Operator T_fk
    ↓
Weighted graph message
    ↓
Residual update + Fusion
    ↓
NC classifier
```

Frozen choices：

```text
K = 4
P2 coupler = NullSoftmax
epsilon = 0.2
P3 = Full Cell-conditioned / hierarchical A+B+C decomposition
deterministic = false
```

---

# 2. 阶段总路线

```text
B0  Benchmark / Protocol Provenance Audit
 ↓
B1  Final Model Main Benchmark
 ↓
B2  Paper-facing Main Ablation
 ↓
B3  Design-choice Ablation（复用 P1/P2/P3，少量新增）
 ↓
B4  Efficiency + Mechanism + Sensitivity
 ↓
B5  Paper Baseline Expansion（如需要）
 ↓
NC FINAL
```

**先做 B0 → B1。不要一开始同时跑所有消融。**

如果 Final Model 在主 Benchmark 上存在明显实现/协议问题，先解决 Benchmark，不要让消融结果一起污染。

---

# 3. B0 — Benchmark / Protocol Provenance Audit

## 3.1 冻结统一协议

必须确认所有模型使用同一个 NC task protocol：

```text
full-graph training
AdamW
300 epochs
patience = 30
early_stop_min_epoch = 1
CE only on train nodes
checkpoint by Val Accuracy
test exactly once after best checkpoint
seeds = 42 / 43 / 44
population std
```

Models 自己如果有官方/冻结的 `lr` / `weight_decay` preset，可以保留，但必须记录。

---

## 3.2 当前 Benchmark 模型集合

现有统一 baseline：

```text
MLP
GCN
GraphSAGE
MMGCN
MGAT
DMGC
DGF
DiP
```

加：

```text
Bi-Axis Final
```

第一轮主表：

\[
8\ baselines + 1\ ours = 9\ models.
\]

---

## 3.3 现有 baseline 输出是否可以复用？

**不能凭“以前跑过”直接复用。**

AI 先审计本地：

```text
outputs/baseline_nc/
```

逐个检查：

- dataset
- model
- seed
- task config
- split / split seed
- epochs
- patience
- checkpoint criterion
- optimizer
- inference mode
- model config
- git / hydra config（若输出中有）
- results.json 完整性

如果与当前 frozen protocol 完全一致：

\[
\boxed{\text{复用 baseline，不重跑。}}
\]

如果某个 baseline 的协议/配置改变：

\[
\boxed{\text{只重跑受影响的 model-dataset-seed。}}
\]

不要无意义重跑全部 120 baseline runs。

---

## 3.4 Final Model 建议重新用 final entry 跑一遍

即使 OFR 历史 checkpoint 与 final model 数学相同，主论文结果建议用：

```text
model=biaxis_final
```

重新生成一套干净目录：

```text
outputs/final_nc_benchmark/
```

5 datasets × 3 seeds：

\[
\boxed{15\ runs}
\]

这样论文最终主表有清晰、独立的 reproduction provenance。

---

# 4. B1 — Final Model Main Benchmark

## 4.1 第一轮先 Smoke

```text
5 datasets × seed42 × biaxis_final
```

检查：

- 无 NaN
- final config 确实 `NullSoftmax + full_interaction`
- `p2.deterministic=false`
- train/val/test split 正确
- val checkpoint / test-once 正确
- parameter count 合理
- ele-fashion 不 OOM
- output paths 清楚

Smoke 不是论文结果。

---

## 4.2 正式 Final Runs

```text
datasets:
Movies
Toys
Grocery
ele-fashion
Reddit-S

model:
biaxis_final

seeds:
42
43
44
```

15 runs。

---

## 4.3 主结果必须生成两层表

### Table 1 — Test Accuracy

| Model | Movies | Toys | Grocery | ele-fashion | Reddit-S | Avg Rank |
|---|---:|---:|---:|---:|---:|---:|

每格：

```text
mean ± population std
```

标：

```text
Best
Second Best
```

---

### Table 2 — Test Macro-F1

同样格式。

---

## 4.4 同时生成辅助统计

每 dataset：

```text
best baseline
ours
ours - best baseline
rank
```

整体：

```text
average rank
# top-1
# top-2
mean delta vs strongest baseline
```

如果使用相同 seeds/splits，还输出：

```text
paired-seed Δ vs DiP
paired-seed Δ vs strongest baseline
positive seeds / 3
```

但主论文不要用 paired test delta 做结构选择。

---

## 4.5 Benchmark 后的判断

### Strong competitive

例如：

```text
多数数据集 Top-1 / Top-2
Avg Rank 接近 1
Acc 与 F1 均有竞争力
```

→ 直接进入正式消融。

### Mixed but competitive

不同数据集有输有赢，但：

```text
平均排名好
有明确强项
没有系统性 F1 崩塌
```

→ 仍进入消融，并分析图特性。

### Weak

如果 final 在 4–5/5 数据集明显落后强 baseline：

**不要立刻 unfreeze 模型。**

依次检查：

1. final benchmark implementation
2. protocol
3. dataset split
4. learning-rate / weight-decay 是否意外被 task/model override
5. final alias 是否确实加载 frozen structure
6. baseline 是否有不公平 protocol

只有排除实验问题后，才讨论研究风险。

---

# 5. 正式消融应该怎样划分？

## 核心原则

正式消融按：

\[
\boxed{\textbf{论文故事 / Contribution}}
\]

划分，而不是 P0/P1/P2/P3。

我建议分成三层：

```text
A. Main Story Ablation
B. Design-choice Ablation
C. Parameterization / Mechanism Analysis
```

---

# 6. A — Main Story Ablation（论文主文优先）

最终论文 claim：

```text
Semantic Factor Axis
Structural Relation Axis
Adaptive Unified Allocation
Hierarchical Transformation
```

因此主消融建议如下。

---

## FULL — Bi-Axis Final

\[
F=3,\quad K=4
\]

\[
\Gamma=\text{NullSoftmax}
\]

\[
T_{fk}=W_0+A_f+B_k+C_{fk}.
\]

---

## A1 — w/o Semantic Factor Axis

论文名称建议：

```text
w/o Semantic Factor Axis
```

不要写 `w/o P0`。

### 定义

保留原 multimodal encoders / semantic factorizer 的 local representation，但**graph side 不再看到 C/Pt/Pv identity**。

使用 P1 已存在的 factor-blind 思路：

\[
q_i=Proj_q(z_i^{local})
\]

Graph side：

\[
F=1.
\]

仍保留：

```text
K=4 topology-only relations
Null-Augmented allocation
relation-aware transformation
```

为了得到最小可辨识 collapse：

\[
T_k=W_0+B_k.
\]

为什么不是 A/C？

当 \(F=1\) 时：

- factor main effect 可吸收到 \(W_0\)
- cell correction 与 relation effect 可合并

所以 \(W_0+B_k\) 是最干净的一维关系轴模型。

### 它回答

> 如果不显式区分 Common / Text-private / Visual-private，Factor–Relation Space 是否仍然有价值？

---

## A2 — w/o Structural Relation Axis

论文名称：

```text
w/o Structural Relation Axis
```

不要写 `R0` 或 `w/o P1`。

### 定义

保留：

\[
C,P_t,P_v
\]

但：

\[
K=1,\qquad r_{ij,1}=1.
\]

因此：

\[
g_{i1}^{f}
=
\text{plain neighbor mean of factor }f.
\]

NullSoftmax 变成：

```text
Local vs Graph
```

即：

\[
\Gamma_{if0},\Gamma_{if1}.
\]

最小可辨识 operator：

\[
T_f=W_0+A_f.
\]

当 \(K=1\)：

- relation main effect 可吸收到 W0
- cell correction可吸收到 factor effect

### 它回答

> 如果不区分 latent structural relations，只对不同 semantic factors 聚合普通邻居，性能如何？

---

## A3 — w/o Adaptive Factor–Relation Allocation

论文名称：

```text
w/o Adaptive Allocation
```

### 保留

```text
F=3 semantic factors
K=4 structural relations
Full hierarchical T_fk
```

### 删除

所有 learned factor-dependent allocation。

使用 topology-only relation availability：

\[
a_{ik}
=
\frac{m_{ik}}{d_i+\epsilon}.
\]

设置：

\[
\Gamma_{if0}=0
\]

\[
\Gamma_{ifk}=a_{ik},
\quad
\forall f.
\]

即：

- 不学习 Local/Graph demand
- 不学习 factor-specific relation preference
- 只保留结构本身的 relation occupancy

### 它回答

> Factor–Relation cell 已经存在时，是否真的需要 semantic-factor-dependent adaptive allocation？

---

## A4 — w/o Hierarchical Transformation

论文名称：

```text
w/o Hierarchical Operator
```

即 P3 O0：

\[
T_{fk}=W_0.
\]

完整保留：

```text
semantic factors
relations
Γ
```

只删除 transformation specialization。

这是最干净的 operator ablation。

已有 P3 结果可以复用。

---

## A5 — w/o Cell-specific Correction

论文名称：

```text
w/o Cell-specific Correction
```

即 P3 OADD：

\[
T_{fk}=W_0+A_f+B_k.
\]

删除：

\[
C_{fk}.
\]

### 它回答

> global/factor/relation main effects 已存在后，cell-specific correction 是否仍有额外价值？

已有 P3 结果可以复用。

---

# 7. Main Ablation 主表建议

| Variant | Semantic Factor Axis | Relation Axis | Adaptive Γ | Hierarchical Op | Cell Correction |
|---|---|---|---|---|---|
| Full | ✓ | ✓ | ✓ | ✓ | ✓ |
| w/o Semantic Factor Axis | ✗ | ✓ | ✓ | reduced | ✗ |
| w/o Structural Relation Axis | ✓ | ✗ | ✓ | reduced | ✗ |
| w/o Adaptive Allocation | ✓ | ✓ | ✗ | ✓ | ✓ |
| w/o Hierarchical Operator | ✓ | ✓ | ✓ | ✗ | ✗ |
| w/o Cell-specific Correction | ✓ | ✓ | ✓ | ✓ | ✗ |

然后报：

```text
Test Accuracy
Test Macro-F1
Params
```

正式结构选择依旧来自 Val Accuracy，主文表展示 Test。

---

# 8. Main Ablation 需要跑多少新实验？

已有可复用：

```text
Full          -> Final benchmark 15 runs
w/o Hierarchical Operator -> P3 O0
w/o Cell Correction       -> P3 OADD
```

真正新增：

```text
A1 w/o Semantic Factor Axis
A2 w/o Structural Relation Axis
A3 w/o Adaptive Allocation
```

5 datasets × 3 modes × 3 seeds：

\[
\boxed{45\ new\ runs}
\]

这是非常合理的正式消融成本。

---

# 9. B — Design-choice Ablation

不要把所有探索 variant 都塞进主消融表。

Design-choice 是为了回答：

> 为什么最后选择这个实现？

---

## B1 Allocation Design

推荐比较：

```text
Separate Budget + Selector   (P1 β×α)
Null-Augmented Softmax       (Final)
Composition-UOT              (optional)
```

如果愿意再补一个最干净的 Null ablation：

### No-Null Softmax

只对：

\[
R_1,\dots,R_K
\]

做 softmax：

\[
\Gamma_{if0}=0,
\qquad
\sum_{k=1}^K\Gamma_{ifk}=1.
\]

它专门回答：

> Local / No-Transport state 是否必要？

这个实验推荐，但不是进入 Benchmark 的前置条件。

成本：

\[
5\times3=15\ runs.
\]

---

## B2 Operator Design

直接复用 P3：

```text
Shared       O0
Factor       OF
Relation     OR
Additive     OADD
Full         OFR
```

不用重跑。

正式论文可放一张：

| Operator | Factor cond. | Relation cond. | Cell correction | Acc/F1 |
|---|---|---|---|---|

它支撑：

```text
Factor identity > Relation-only
A+B useful
C_fk conditional
```

---

# 10. C — Parameterization / Mechanism Analysis

这些不要放在主消融大表里。

放 Supplement / Analysis section。

已有：

```text
LR-ADD
LR-INT
Basis16v2
DirectCell
OFR
```

用于回答：

> 为什么最终使用 hierarchical A+B+C parameterization？

核心结论：

```text
same function class DirectCell < OFR
expressively complete Basis16 不能稳定恢复
parameter count alone cannot explain
hierarchical parameterization has optimization/regularization advantage
```

不需要再跑。

---

# 11. Mechanism Analysis 应怎样整理成论文证据？

建议只保留最有解释力的指标。

## 11.1 Semantic Factor Demand

每 factor：

```text
mean graph mass:
C
Pt
Pv
```

展示：

\[
1-\Gamma_{if0}.
\]

---

## 11.2 Factor-specific Relation Selectivity

报告 conditional relation JS：

```text
JS(C,Pt)
JS(C,Pv)
JS(Pt,Pv)
```

重点：

```text
Grocery
ele-fashion
Reddit-S
```

形成正例 / 正例 / 退化例。

---

## 11.3 Relation specialization

同时报告：

```text
K_eff
S_R
```

但明确：

\[
S_R
\]

不是 cell interaction 的充分条件。

---

## 11.4 Effective Operator Interaction

最终使用 double-centered：

\[
I_{fk}
=
T_{fk}
-\bar T_{f\cdot}
-\bar T_{\cdot k}
+\bar T_{\cdot\cdot}.
\]

主指标：

\[
\bar S_I
=
\frac{
\sum_{fk}u_{fk}\|I_{fk}\|
}{
(\sum_{fk}u_{fk})\|W_0\|+\epsilon
}.
\]

建议论文图：

```text
ele-fashion
Grocery
Movies
Toys
Reddit-S
```

按 normalized interaction strength 排序。

---

# 12. B4 — Efficiency

主 Benchmark 完成、模型有竞争力后再做。

指标：

```text
# Parameters
Training time / epoch
Peak GPU memory
Inference time
```

优先比较：

```text
GCN
GraphSAGE
MMGCN
MGAT
DGF
DiP
Bi-Axis Final
```

不要用 total wall-clock to early stopping 作为唯一时间指标，因为不同模型 best epoch 不同。

推荐标准化：

```text
固定 20 个 training epochs
丢弃前 5 个 warmup epochs
报告后 15 epoch 平均时间
同一 GPU
同一 dataset
同一 process concurrency = 1
```

Peak memory：

```text
torch.cuda.max_memory_allocated()
```

选择：

```text
Grocery
ele-fashion
```

或者全部 5 datasets。

---

# 13. B4 — Sensitivity

只做真正影响模型定义的参数。

## 优先级 1：Number of Relations K

\[
K\in\{2,4,8\}
\]

推荐 3 datasets：

```text
Grocery        relation/cell strong
ele-fashion    interaction strong
Reddit-S       relation-degenerate
```

3 seeds：

\[
3\times3\times3=27\ runs.
\]

注意：

```text
K=1
```

已经作为 `w/o Structural Relation Axis`，不用再放 sensitivity。

---

## 优先级 2：epsilon

之前已有 sanity：

\[
0.1,0.2,0.4
\]

不建议当前立刻重新正式跑。

如果投稿前需要 sensitivity 图，再补：

```text
3 datasets × 3 epsilon × 3 seeds
```

否则 supplementary 说明已有 sanity 即可。

---

# 14. Paper Baseline Expansion

当前 8 个 baseline 足够做**第一轮内部统一 Benchmark**。

但最终投稿前，要重新审视：

> 是否缺少近两年与 MAG representation learning 最相关且有公开实现的强 baseline？

不要现在阻塞 Final Benchmark。

流程：

```text
先看 Bi-Axis Final 在当前 8 baseline 上是否有竞争力
↓
再挑 1–3 个最相关强方法
↓
统一到同数据/同 split/同 metric protocol
```

不要为了“baseline 数量多”堆很多不相关模型。

---

# 15. 禁止事项

从现在开始：

1. 不根据 Test 表重新修改主体模型。
2. 不设计 P4。
3. 不重新打开 UOT/Router 搜索。
4. 不为每个 dataset 单独调一套 hyperparameters。
5. 不用 deterministic 慢路径做正式 batch。
6. 不把 P0/P1/P2/P3 直接作为论文正式模块名。
7. 不把所有探索负结果都塞进主文。
8. 不只报告 Accuracy，不报 Macro-F1。
9. 不混用不同 seed 数的 mean 进行 paired comparison。
10. 不用历史结果，除非 protocol provenance audit 通过。

---

# 16. 推荐输出目录

```text
outputs/final_nc_benchmark/
  main/
    <dataset>/
      biaxis_final/
        seed_42/
        seed_43/
        seed_44/
  tables/
    nc_main_per_seed.csv
    nc_main_table_acc.csv
    nc_main_table_f1.csv
    nc_main_rank.csv
    NC_MAIN_REPORT.md

outputs/final_nc_ablation/
  main_story/
    ...
  design/
    ...
  tables/
    nc_ablation_per_seed.csv
    nc_ablation_table.csv
    NC_ABLATION_REPORT.md

outputs/final_nc_analysis/
  efficiency/
  sensitivity/
  mechanism/
```

---

# 17. AI Prompt 1 — Benchmark Provenance Audit（先做这个）

```text
你现在协助我进入 Bi-Axis Frozen Model 的最终 NC Benchmark 阶段。

先不要修改代码，不要跑训练。

Repository: CrisRipper777/0901

最终模型：
model=biaxis_final

Frozen：
- P0 semantic ownership factorization
- P1 topology-only K=4 structural relations
- P2 Null-Augmented Plan + NullSoftmax, epsilon=0.2
- P3 hierarchical Full Cell-conditioned Operator
  T_fk = W0 + A_f + B_k + C_fk
- p2.deterministic=false
- 不再修改模型结构

当前只做 NC：
Movies / Toys / Grocery / ele-fashion / Reddit-S
seeds = 42,43,44

请审查：
1. configs/task/nc.yaml 的当前统一协议。
2. configs/model/biaxis_final.yaml。
3. scripts/run_nc_baseline_table.py。
4. 8 个 baseline 的 model configs：
   mlp/gcn/sage/mmgcn/mgat/dmgc/dgf/dip。
5. 本地 outputs/baseline_nc/ 中现有结果的 hydra/config/results。
6. dataset split 是否 seed-aligned。
7. 哪些已有 baseline results 可以严格复用。
8. 哪些必须重跑以及原因。
9. 是否存在 model-specific lr/wd override；列成表。
10. 给出 protocol/provenance audit。

输出：
- PASS / FAIL checklist
- reusable baseline runs
- rerun list
- 任何 fairness 风险

不要写代码。
不要跑训练。
```

---

# 18. AI Prompt 2 — 实现 Final Benchmark Runner

```text
Benchmark provenance audit 已通过。

现在实现一个独立 final NC benchmark runner。

优先新增：
scripts/run_nc_final_benchmark.py
scripts/summarize_nc_final_benchmark.py

不要破坏 scripts/run_nc_baseline_table.py 的历史输出。

功能：

1. datasets:
Movies,Toys,Grocery,ele-fashion,Reddit-S

2. models:
已有 8 baselines + biaxis_final

3. seeds:
42,43,44

4. baseline：
如果 audit 判定 outputs/baseline_nc 中结果可复用，直接读取；
只跑 missing / invalid runs。

5. biaxis_final：
统一运行 model=biaxis_final，
保存到 outputs/final_nc_benchmark/main/...

6. 协议：
task=nc
full graph
p2.deterministic=false
禁止 override final model 的结构参数。

7. 输出：
nc_main_per_seed.csv
nc_main_table_acc.csv
nc_main_table_f1.csv
nc_main_rank.csv
NC_MAIN_REPORT.md

8. 每 dataset/model 汇报：
Val Acc mean±population std
Test Acc mean±population std
Test Macro-F1 mean±population std
num_seeds

9. 计算：
best baseline
ours - best baseline
rank
average rank
top1 count
top2 count

10. 如果同 seed 对应同 split：
额外输出 ours-vs-DiP / ours-vs-best-baseline paired delta；
但不要用 test delta 做架构选择。

11. resume/skip。
12. 不开启 deterministic mode。
13. 不修改 frozen model files。

先实现和测试 runner，不立刻全跑。
```

---

# 19. AI Prompt 3 — Final Model Smoke

```text
Final benchmark runner 已通过静态测试。

只跑 biaxis_final：

Movies
Toys
Grocery
ele-fashion
Reddit-S

seed42

正常完整训练协议（不是 5 epoch），作为 main benchmark 的第一批 run。

检查：
- config 最终解析
- NullSoftmax
- full_interaction
- deterministic=false
- results.json
- test once
- no NaN
- no OOM
- metrics
- params

先给 smoke audit。
不要根据结果修改模型。
```

---

# 20. AI Prompt 4 — 完成 Main Benchmark

```text
Smoke 已通过。

完成：
biaxis_final × 5 datasets × seeds42/43/44。

然后聚合：
8 baselines + biaxis_final。

生成最终 NC main comparison report。

必须分析：
1. 每 dataset strongest baseline。
2. Bi-Axis Final 的 Acc / Macro-F1 rank。
3. vs DiP。
4. vs strongest baseline。
5. average rank。
6. top1/top2 count。
7. 是否存在 Acc 提升但 F1 下降。
8. seed variance。
9. 哪些 dataset 是 strongest positive / neutral / negative cases。

不要做消融。
不要改模型。
把完整结果给我审查。
```

---

# 21. AI Prompt 5 — Formal Ablation Repository Audit

Main Benchmark 审查通过后执行。

```text
现在开始设计论文正式消融。

先不要写代码。

注意：
P0/P1/P2/P3 是内部研发阶段，不作为论文正式消融名。

Paper-facing modules：
A. Semantic Factor Axis
B. Structural Relation Axis
C. Adaptive Unified Factor–Relation Allocation
D. Hierarchical Factor–Relation Operator
E. Cell-specific Correction

计划 Main Ablations：

FULL
T_fk = W0 + A_f + B_k + C_fk

A1 w/o Semantic Factor Axis:
graph side 使用 factor-blind q，F=1；
K=4；
NullSoftmax；
最小 operator T_k=W0+B_k。

A2 w/o Structural Relation Axis:
F=3；
K=1，r=1，plain neighbor context；
NullSoftmax Local/Graph；
最小 operator T_f=W0+A_f。

A3 w/o Adaptive Allocation:
F=3,K=4；
保留 full T_fk；
Gamma_if0=0；
Gamma_ifk=a_ik（topology relation availability，对 factors 相同）。

A4 w/o Hierarchical Operator:
P3 O0，T=W0。

A5 w/o Cell-specific Correction:
P3 OADD，T=W0+A_f+B_k。

请仔细审查当前：
biaxis_p0/p1/p2/p3/final
相关 components/configs/tests。

回答：
1. A1 是否可以安全复用 P1 factor-blind q path？
2. A2 如何用现有 K=1 fast path？
3. A3 如何最小侵入固定 Gamma=a？
4. 是否应新增独立 biaxis_ablation model/config，避免修改 frozen files？
5. A1/A2 的 minimal identifiable operator 定义是否正确？
6. 哪些已有 P3 results 可以复用？
7. 参数量 confound 如何在论文中披露？
8. 给 implementation plan。

不要修改代码。
```

---

# 22. AI Prompt 6 — 实现 Paper-facing Ablation Model

```text
根据 audit，实现独立 ablation layer。

新增：
src/models/biaxis_ablation.py
configs/model/biaxis_ablation.yaml
tests/test_biaxis_ablation.py

不要修改：
biaxis_final.py/yaml
biaxis_p0/p1/p2/p3 frozen files

支持 modes：

full_reference
no_factor_axis
no_relation_axis
no_adaptive_allocation
shared_operator
no_cell_correction

要求：

full_reference：
必须与 biaxis_final 数学/输出一致（相同 weights）。

no_factor_axis：
- graph side F=1
- q=Proj_q(z_local)
- K=4 topology relations
- NullSoftmax over Local+K
- operator W0+B_k
- fusion_q output
- semantic aux objective 是否仍训练，沿用 audit 决定并明确记录

no_relation_axis：
- F=3 C/Pt/Pv
- K=1 strict r=1
- NullSoftmax over Local+Graph
- operator W0+A_f

no_adaptive_allocation：
- F=3,K=4
- relation decomposition unchanged
- gamma_local=0
- gamma_graph=a_ik expanded across factors
- full hierarchical operator unchanged
- no transport scorer influence on messages

shared_operator：
复用 final factor/relation/Gamma；
T=W0

no_cell_correction：
T=W0+A_f+B_k

tests：
- full_reference == biaxis_final
- no_factor shape / no factor identity
- K=1 strict neighbor relation
- fixed gamma sums to 1
- fixed gamma factor-independent
- correct operator collapse
- gradients
- no NaN
- inference
- frozen model files untouched
```

---

# 23. AI Prompt 7 — Ablation Smoke

```text
只跑 Movies seed42：

full_reference
no_factor_axis
no_relation_axis
no_adaptive_allocation
shared_operator
no_cell_correction

5 epochs smoke 即可。

检查：
- forward
- params
- loss
- gradient
- memory
- output
- no NaN

重点验证 full_reference 与 final implementation 等价。

不要根据 smoke metric 判断模块价值。
```

---

# 24. AI Prompt 8 — Main Story Ablation

```text
Ablation smoke 已通过。

正式运行新 ablations：

datasets:
Movies,Toys,Grocery,ele-fashion,Reddit-S

modes:
no_factor_axis
no_relation_axis
no_adaptive_allocation

seeds:
42,43,44

45 new runs。

Full：
复用 final benchmark。

shared_operator：
复用 P3 O0 matched seeds42/43/44。

no_cell_correction：
复用 P3 OADD matched seeds42/43/44。

生成：
NC_ABLATION_MAIN.csv
NC_ABLATION_PER_SEED.csv
NC_ABLATION_REPORT.md

表中统一使用相同 3 seeds。

报告：
Val Acc
Test Acc
Test Macro-F1
Params

对 Full 做 paired-seed delta：
Full - each ablation

输出：
mean Δ
population std Δ
positive seeds / 3

分析必须按论文 claims：
Semantic Factor Axis
Structural Relation Axis
Adaptive Allocation
Hierarchical Operator
Cell-specific Correction

不要写 P0/P1/P2/P3 作为正式模块名。
```

---

# 25. AI Prompt 9 — Allocation Design Ablation（可选但推荐）

```text
Main story ablation 完成后。

新增一个 no_null mode：

保持：
F=3
K=4
same scorer
same hierarchical operator

只删除 Local state：
对 K relations 做 row-wise softmax，
Gamma_if0=0，
sum_k Gamma_ifk=1。

5 datasets × 3 seeds = 15 runs。

与：
P1 separate beta×alpha
Final NullSoftmax
Composition-UOT（已有）
一起整理成 Allocation Design Table。

不要重新调 epsilon/tau。
```

---

# 26. AI Prompt 10 — Efficiency

```text
Main Benchmark + Ablation 完成后做 efficiency。

models:
gcn
sage
mmgcn
mgat
dgf
dip
biaxis_final

datasets:
Grocery
ele-fashion

seed42

每模型单独占 GPU，不并发。

记录：
trainable params
peak GPU allocated memory
epoch time

epoch time：
固定训练20 epochs，
前5 epochs warmup不计，
报告后15 epochs mean/std。

不使用 total early-stopping wall time 比模型。

生成：
NC_EFFICIENCY.csv
NC_EFFICIENCY_REPORT.md
```

---

# 27. AI Prompt 11 — K Sensitivity

```text
只在全部主结果稳定后执行。

Final architecture不改，只做 sensitivity：

K=2,4,8

datasets:
Grocery
ele-fashion
Reddit-S

seeds:
42,43,44

27 runs。

K=4 是 frozen default，不做 dataset-specific selection。

报告：
Val Acc
Test Acc
Test F1
K_eff
S_R
Params
Runtime

目的：
证明模型不依赖某个极窄 K 设置，
并分析 relation-degenerate Reddit-S。
```

---

# 28. Definition of Done — NC Benchmark Stage

## Main benchmark

- [ ] provenance audit
- [ ] final 15 clean runs
- [ ] 8 baselines + ours
- [ ] Test Acc main table
- [ ] Test Macro-F1 main table
- [ ] average rank / top1 / top2
- [ ] strongest baseline deltas

## Formal ablation

- [ ] w/o Semantic Factor Axis
- [ ] w/o Structural Relation Axis
- [ ] w/o Adaptive Allocation
- [ ] w/o Hierarchical Operator
- [ ] w/o Cell-specific Correction
- [ ] same 3 seeds
- [ ] paired deltas
- [ ] params

## Design analysis

- [ ] allocation design table
- [ ] operator design table
- [ ] parameterization analysis reuse
- [ ] mechanism figures

## Efficiency / sensitivity

- [ ] params
- [ ] epoch time
- [ ] peak memory
- [ ] K sensitivity

## Research discipline

- [ ] no P4
- [ ] no test-based tuning
- [ ] no dataset-specific final hyperparameters
- [ ] final architecture unchanged
- [ ] paper terminology no longer uses P0/P1/P2/P3 as modules

---

# 29. 最终论文实验结构建议

如果所有结果完成，实验章节可以直接变成：

### 5.1 Experimental Setup
datasets / baselines / protocol / metrics

### 5.2 Main Results
NC Accuracy + Macro-F1

### 5.3 Ablation Study
story-facing five ablations

### 5.4 Model Analysis
Factor demand / relation selectivity / effective interaction

### 5.5 Design Choices
Null plan / hierarchical operator

### 5.6 Efficiency and Sensitivity
params / time / memory / K

而 P0/P1/P2/P3 的大量研发实验：

```text
保留为内部证据
挑最关键部分进入 supplementary
不按研发时间线写进论文
```

---

# 30. 当前下一步

现在不要直接实现 ablation。

**第一步：只执行 Prompt 1 — Benchmark Provenance Audit。**

把 AI 的 audit 报告发回来，再决定：

- 哪些 baseline 可以直接复用；
- 是否需要重跑任何 baseline；
- final benchmark runner 应怎样接现有 outputs；
- 是否存在公平性风险。

然后再进入 15 个 Final Model 主实验。
