# Bi-Axis MAG 下一阶段性能推进计划
## R2-Design-2.9：System-Level Coordinated Relational Architecture Search

> **总目标**：不再以局部机制是否单独 GO 作为路线筛选标准，而以整体模型的全局性能为第一目标；在基本保持 `Semantic Ownership × Functional Relational Transfer` 故事主线的前提下，构建、搜索并验证能够稳定超越当前最强基线的系统级架构。
>
> **代码仓库**：`https://github.com/CrisRipper777/0901`
>
> **主任务**：先完成 Node Classification（NC）架构定型；NC 成功后再扩展 LP。
>
> **主数据集**：Movies / Toys / Grocery / ele-fashion / Reddit-S。
>
> **默认 seeds**：42 / 43 / 44。所有正式架构候选均直接 3 seeds，不再用单 seed 作为正式 GO/NO-GO 依据。
>
> **实验纪律**：架构搜索阶段只使用 Validation；最终架构、训练策略、超参数全部冻结前，不使用 Test 做任何模型选择。

---

# 1. 为什么下一阶段必须改变推进方式

过去 P0→R2-D2.8 已经反复出现：

```text
信息/机制在 frozen probe、counterfactual 或 causal intervention 中真实存在
                ↓
trainable module 也能学到“正确方向”或显著依赖该信息
                ↓
但最终 end-to-end task gain 很弱、为 0，甚至下降
```

因此，下一阶段不再默认：

```text
单模块弱 → 模块无价值 → 不允许进入联合模型
```

而正式检验系统级假设：

\[
\boxed{
\text{Useful relational information}
\not\Rightarrow
\text{useful downstream gain}
}
\]

原因可能不是“信息不存在”，而是：

\[
\boxed{
\text{information acquisition}
\rightarrow
\text{representation preservation}
\rightarrow
\text{conditional computation}
\rightarrow
\text{state update}
\rightarrow
\text{fusion/readout}
}
\]

这一整条计算链没有被协同设计。

因此，本阶段从 **local mechanism optimization** 转向 **system-level coordinated architecture search**。

---

# 2. 本阶段总原则

## 2.1 Global optimum > local optimum

A0 / `biaxis_final` 不再视为必须保留的最终父模型，只保留三种角色：

1. **reference baseline**：所有新模型都必须与其比较；
2. **warm-start source**：可加载 P0 factorizer / A0 参数；
3. **optional strong path**：在 augmentation / hybrid 架构中可作为一条强路径。

允许最终模型：

- 在 A0 前加入新的 relational block；
- 在 A0 后加入新的 relational block；
- 前后同时加入；
- 完全删除旧 K=4 topology relation + Γ + OFR；
- 使用 A0 与新 relational path 并行，再在 factor space 融合。

**禁止为了保护历史结构而牺牲最终性能。**

## 2.2 不再实行“单机制先 GO，联合模型才允许跑”

允许：

\[
\Delta_A \approx 0,
\quad
\Delta_B \approx 0,
\quad
\Delta_{A\times B} \gg 0
\]

因此，任何 system-level factorial / coordinated combination 都必须完整执行，不得因为某一个 main effect 弱而提前停止。

## 2.3 3 seeds 是最低正式单位

所有 performance-relevant 架构候选：

```text
seed = 42 / 43 / 44
```

禁止用 seed42 单跑结果淘汰架构。

若最终模型与 strongest baseline 差距在 ±0.3pp 内，可补 seeds 45/46。

## 2.4 Performance first，但保留可解释记录

参数量不是硬约束。允许扩大 factor dimension、interaction width、fusion width、block depth、source attention、parallel/hybrid paths。

每次实验必须记录：

- 参数量；
- peak GPU memory；
- training time；
- Val Accuracy / Macro-F1；
- 与 A0 的 paired delta；
- 与 strongest external baseline 的 gap；
- 每个 seed 的 raw result；
- git commit / config snapshot。

---

# 3. 新的核心方法假设：CORT

下一版方法暂称：

## Coordinated Ownership-Relational Transfer（CORT）

论文主线仍保持：

\[
\boxed{
\text{Semantic Ownership}
\times
\text{Functional Relational Transfer}
}
\]

但第二轴不再被拆成若干彼此独立的小 gate，而定义为一条完整 computation pathway：

\[
\boxed{
\text{Relational Allocation}
\rightarrow
\text{Source Preservation}
\rightarrow
\text{Target-conditioned Interaction}
\rightarrow
\text{Ownership-state Update}
}
\]

最终再进入：

\[
\boxed{
\text{Ownership Interaction Fusion}
}
\]

---

# 4. CORT Block 推荐形式

设第 \(l\) 层三个 ownership states 为：

\[
H_i^{(l)}=\{H_i^C,H_i^{P_t},H_i^{P_v}\}
\]

其中 \(a\) 为 source factor，\(b\) 为 target factor。

## 4.1 Coupled Null-Augmented Relational Allocation

优先保留 D2.7 已经出现正性能信号的 coupled formulation，不再先强拆 Exposure / Composition：

\[
s_{ji}^{a\rightarrow b}
=
\psi(
H_i^b,
H_j^a,
H_i^b\odot H_j^a,
|H_i^b-H_j^a|,
e_a,e_b
)
\]

加入 null state：

\[
\{\alpha_{\emptyset i}^{ab},\alpha_{ji}^{ab}\}
=
\operatorname{Softmax}_{\{\emptyset\}\cup\mathcal N(i)}
(\{s_{\emptyset i}^{ab},s_{ji}^{ab}\})
\]

消息：

\[
m_i^{a\rightarrow b}
=
\sum_{j\in\mathcal N(i)}
\alpha_{ji}^{ab}U_aH_j^a
\]

第一版不要加入 Edge-FiLM / Dynamic Basis，先保证完整 computation interface。

## 4.2 Source-channel Preservation

不能在 target interaction 前直接：

\[
\frac13\sum_a m_i^{a\rightarrow b}
\]

而是保留：

\[
M_i^b=[m_i^{C\rightarrow b},m_i^{P_t\rightarrow b},m_i^{P_v\rightarrow b}]
\]

至少到 target-conditioned interaction 后再组合。

## 4.3 Target-conditioned Vector Interaction

不再只输出 scalar gate。

对每个 \(a\to b\)：

\[
h_i^{ab}
=
\phi(
[H_i^b\Vert m_i^{ab}\Vert H_i^b\odot m_i^{ab}\Vert |H_i^b-m_i^{ab}|\Vert e_a\Vert e_b]
)
\]

然后针对 target factor：

\[
\Delta_i^b
=
\Phi_b([h_i^{Cb}\Vert h_i^{P_tb}\Vert h_i^{P_vb}])
\]

第一版推荐：

```text
phi_ab: 2-layer MLP + GELU + LayerNorm + Dropout
Phi_b : 2-layer MLP
```

参数共享不是硬要求，性能优先。

## 4.4 Factor-space Write-back

关键变化：新 relational information 不再只在 classifier 前 concat。

\[
\widetilde H_i^b
=
\operatorname{LN}(H_i^b+\rho_b\Delta_i^b)
\]

\(\rho_b\) 可使用 ReZero-style learnable residual scale，初始化 0 或小正值。

## 4.5 Ownership Interaction Fusion（OIF）

旧 fusion 是：

```text
[C | Pt | Pv] -> Linear -> Norm -> GELU
```

新 relational information 写回 factor 后，为避免再次在 fusion 阶段丢失，增加 interaction-aware fusion：

\[
q_i=[C,P_t,P_v,C\odot P_t,C\odot P_v,P_t\odot P_v,
|C-P_t|,|C-P_v|,|P_t-P_v|]
\]

\[
z_i=\operatorname{MLP}_{fusion}(q_i)
\]

同时保留 `legacy_fusion` 作为 factorial control。后续如果仍有空间，再测试 3-token factor Transformer / attention fusion。

---

# 5. Stage G0 — 重建当前性能参考线

## 5.1 目的

不把历史 benchmark 数字当作不可变事实。在当前代码版本和统一协议下，重新建立：

```text
A0 / strongest-baseline Validation reference
```

## 5.2 实验

五个 NC 数据集 × seeds 42/43/44。

优先重跑当前 benchmark 的正式模型：

```text
MLP
GCN
GraphSAGE
MMGCN
MGAT
DMGC
DGF
DiP
LGMRec
biaxis_final
```

如果某模型当前已无法稳定运行，记录原因，不要静默删除。

## 5.3 输出

```text
outputs/r2d29/g0_reference/
  runs.csv
  aggregate.csv
  strongest_baseline_by_dataset.csv
  resources.csv
  G0_REFERENCE_REPORT.md
```

`runs.csv` 至少包含：

```text
model,dataset,seed,best_epoch,val_acc,val_f1,param_count,peak_mem_mb,train_seconds,git_commit
```

`strongest_baseline_by_dataset.csv`：

```text
dataset,strongest_model,mean_acc,std_acc,mean_f1,std_f1
```

同时输出：

- strongest external baseline（不含 biaxis_final）；
- strongest overall reference（可含 biaxis_final）。

后续所有候选自动计算：

\[
\Delta_{A0},\qquad \Delta_{StrongestExternal}
\]

---

# 6. Stage G1 — 实现统一可配置 CORT 架构

## 6.1 代码原则

不要修改：

```text
biaxis_p0.py
biaxis_p1.py
biaxis_p2.py
biaxis_p3.py
biaxis_final.py
biaxis_r2_relfunc.py
```

新建：

```text
src/models/biaxis_cort.py
src/models/biaxis_cort_components.py
configs/model/biaxis_cort.yaml
src/analysis/perf_r2d29_utils.py
scripts/perf_r2d29_g1_audit.py
scripts/perf_r2d29_g2_synergy.py
scripts/perf_r2d29_g3_topology.py
scripts/perf_r2d29_g4_capacity.py
scripts/perf_r2d29_g5_optimization.py
scripts/perf_r2d29_g6_confirm.py
tests/test_biaxis_cort.py
```

## 6.2 必须预留的 config knobs

```yaml
name: biaxis_cort
factor_dim: 128
hidden_dim: 256

backbone_mode: a0_augment
# a0_augment | pre_a0 | sandwich | replace | hybrid

router_mode: pair_null
# uniform | target_null | pair_null

source_mode: preserve_concat
# mean | preserve_concat | preserve_attn

writeback_mode: factor
# late | factor

fusion_mode: legacy
# legacy | oif | factor_attn

num_blocks: 1
share_blocks: false
interaction_hidden_mult: 2.0
fusion_hidden_mult: 2.0
residual_init: 0.0
pre_norm: true
dropout: 0.2
edge_chunk_size: 50000
```

## 6.3 Backbone modes 严格定义

### `a0_augment`

```text
P0 factors -> A0 graph_update -> CORT -> fusion
```

### `pre_a0`

```text
P0 factors -> CORT -> A0 graph_update -> fusion
```

直接测试“新信息产生后，让后续强 graph machinery 消化”的假设。

### `sandwich`

```text
P0 factors -> CORT_pre -> A0 graph_update -> CORT_post -> fusion
```

### `replace`

```text
P0 factors -> CORT × L -> fusion
```

完全删除旧 K=4 / Γ / OFR 主传播链。

### `hybrid`

```text
                  -> A0 graph path ------┐
P0 factors -------|                       |-> factor-space merger -> fusion
                  -> CORT × L path -------┘
```

禁止只在 z/classifier 前裸 concat。应在 factor space 合并：

\[
H_i^{b,hybrid}=H_i^{b,A0}+G_b([H_i^{b,A0},H_i^{b,CORT}])
\]

## 6.4 G1 Audit 必须通过

1. `router_mode=uniform` 下邻居顺序 permutation invariant；
2. null-softmax 每个 target/pair 的 mass 总和为 1；
3. isolated nodes 不产生 NaN；
4. `residual_init=0` 时 factor-writeback 与 base path 数值差接近 0；
5. 每个新模块都有非零梯度；
6. preserve 模式的 C/Pt/Pv source channels 在写回前保持独立；
7. no Test access；
8. forward/inference API 与现有框架兼容；
9. 大图使用 chunk/checkpoint 防 OOM；
10. 输出参数量与 peak memory。

输出：

```text
outputs/r2d29/g1_audit/
  audit.json
  grad_audit.csv
  equivalence.csv
  G1_AUDIT_REPORT.md
```

G1 只验证实现正确，不做科学 GO/NO-GO。

---

# 7. Stage G2 — Full System Synergy Matrix

这是本阶段第一组核心性能实验。

## 7.1 核心问题

不是：

> Routing 单独有没有 +0.3pp？

而是：

> Routing、source preservation、factor write-back、interaction fusion 是否只有在形成完整 computation path 后才产生显著性能？

## 7.2 2×2×2×2 完整 factorial

固定：

```text
backbone_mode = a0_augment
num_blocks = 1
```

四个因素：

### R — Relational Allocation

```text
R0 = uniform
R1 = pair_null
```

### S — Source Handling

```text
S0 = mean
S1 = preserve_concat
```

### W — Integration Location

```text
W0 = late
W1 = factor write-back
```

`late` 不使用维度膨胀的裸 concat；统一投影到 hidden_dim 后做 z-space residual，确保输出维度一致。

### F — Fusion

```text
F0 = legacy
F1 = OIF
```

总计：

\[
2^4=16\text{ architectures}
\]

每个都运行：

```text
5 datasets × 3 seeds
```

总计 240 runs。

**必须全部跑完，不得因为任意 main effect 弱而停止。**

## 7.3 G2 统计

对 Accuracy 和 Macro-F1 分别计算：

### Main effects

\[
\Delta_R,\Delta_S,\Delta_W,\Delta_F
\]

### Two-way interactions

\[
I_{RS},I_{RW},I_{RF},I_{SW},I_{SF},I_{WF}
\]

### Higher-order interactions

至少输出：

```text
R×S×W
R×S×F
R×W×F
S×W×F
R×S×W×F
```

## 7.4 Global ranking

不要只按 M/T/G macro 排名。每个 variant 输出：

```text
mean_delta_vs_A0_5datasets
mean_delta_vs_strongest_external_5datasets
num_dataset_wins_vs_strongest
worst_dataset_delta_vs_strongest
mean_rank_5datasets
mean_acc
mean_f1
```

候选采用 Pareto + overall performance，不使用单一局部门槛。

至少保留 Top-4 variants 进入 G3。

## 7.5 Matched control

对 `S1=preserve_concat` 的 Top variants，额外跑：

```text
MEAN_DUP
```

即把 mean message 复制 3 份，再走完全相同的 concat MLP，以区分 source identity 与 larger MLP capacity。

Matched control 只用于解释，不参与提前淘汰。

## 7.6 输出

```text
outputs/r2d29/g2_synergy/
  runs.csv
  aggregate.csv
  factorial_cells.csv
  main_effects.csv
  two_way_interactions.csv
  higher_order_interactions.csv
  strongest_gap.csv
  matched_controls.csv
  resources.csv
  G2_SYSTEM_SYNERGY_REPORT.md
```

完成后把 CSV + report 返回给 ChatGPT 独立分析。

---

# 8. Stage G3 — Global Architecture Topology Search

G2 解决“协调链是否存在 synergy”。G3 开始真正搜索整体 model topology。

## 8.1 使用 G2 Top-2 component combinations

不要只保留 Top-1，防止 component × topology interaction 被忽略。

每套 component combination 测：

```text
T1: a0_augment
T2: pre_a0
T3: sandwich
T4: replace, L=1
T5: replace, L=2
T6: replace, L=3
T7: hybrid, L=1
T8: hybrid, L=2
```

全部：

```text
5 datasets × seeds 42/43/44
```

## 8.2 Router granularity 不提前定死

对 G3 最好的 Top-3 topology 再比较：

```text
pair_null
target_null
```

`target_null` 推荐语义：

- real-edge ranking 主要由 target factor b 决定；
- source factor identity 保留在 payload / interaction channel；
- null/exposure 可保留 pair-conditioned 子变体。

不要默认 9-pair ranking 一定更好，也不要因为 D2.8 isolated result 提前删除 pair-specific routing。

## 8.3 Recurrent CORT

对于 `replace/hybrid`：

\[
H^{(l)}\rightarrow Routing^{(l)}\rightarrow Transfer^{(l)}\rightarrow H^{(l+1)}
\]

下一层必须使用更新后的 \(H^{(l+1)}\) 重新计算 routing。

禁止只预计算一次 edge weights 后重复传播。

## 8.4 输出

```text
outputs/r2d29/g3_topology/
  runs.csv
  aggregate.csv
  topology_comparison.csv
  router_granularity.csv
  depth_response.csv
  strongest_gap.csv
  resources.csv
  G3_TOPOLOGY_REPORT.md
```

保留全局性能最好的 Top-3 architectures 进入 G4。

---

# 9. Stage G4 — Structured Capacity Search

过去只证明 generic width 不是答案，不代表适配 relational computation 的 structured capacity 已充分。

## 9.1 对 Top-3 architecture 搜索

### Factor dimension

```text
128
192
256
```

### Interaction width

```text
1×d
2×d
4×d
```

### Source integration

```text
preserve_concat
preserve_attn
```

### Block sharing

```text
share_blocks = true
share_blocks = false
```

### Fusion

```text
OIF-MLP
Factor-Attention (3 factor tokens)
```

不需要完全 Cartesian product。由 AI 生成 12–24 个覆盖主要 axes 的结构化组合，要求每个 axis 至少有清晰对照。

全部组合：

```text
5 datasets × seeds 42/43/44
```

## 9.2 目标

不要求参数越少越好。主要判断：

```text
增加的是不是与 relational computation 对应的 capacity？
能不能继续缩小 / 反超 strongest-baseline gap？
```

同时记录 parameter-performance frontier。

## 9.3 输出

```text
outputs/r2d29/g4_capacity/
  runs.csv
  aggregate.csv
  capacity_grid.csv
  param_performance.csv
  strongest_gap.csv
  resources.csv
  G4_CAPACITY_REPORT.md
```

保留 Top-3 configs 进入 G5。

---

# 10. Stage G5 — Optimization / Co-adaptation Search

架构基本确定后，再系统处理过去反复出现的 optimization accessibility 问题。

## 10.1 至少测试以下 schedules

### O1 — Joint training

```text
load P0/A0 factorizer warm start
all modules trainable from epoch 1
single LR
```

### O2 — Factorizer freeze → gradual unfreeze

```text
Stage A:
factorizer frozen
CORT + fusion + classifier train

Stage B:
unfreeze factorizer
factorizer LR = base LR × 0.1
new modules LR = base LR
```

### O3 — Strong-parent staged adaptation（A0/hybrid）

```text
Stage A:
A0 frozen
CORT + merger/fusion + classifier train

Stage B:
unfreeze A0 fusion / final graph block
LR_parent = LR_new × 0.1

Stage C:
optional unfreeze P0 factorizer
LR_factorizer = LR_new × 0.05~0.1
```

### O4 — Residual curriculum

```text
rho initialized 0
first K epochs keep relational residual small
then release
```

### O5 — Differential LR + scheduler

### O6 — Combined best schedule

由 O1–O5 结果组合形成合理 schedule，不要求某个单优化 trick 先 GO。

## 10.2 训练动态记录

每个 epoch 或固定 interval 输出：

```text
train loss
val acc/f1
||delta_factor|| / ||base_factor||
routing entropy
null mass
source-channel contribution
residual scale rho
grad norm by module
(optional) cosine between base-path and new-path gradients
```

这些指标用于诊断，不直接作为模型选择目标。

## 10.3 输出

```text
outputs/r2d29/g5_optimization/
  runs.csv
  aggregate.csv
  schedule_comparison.csv
  training_dynamics.csv
  grad_stats.csv
  strongest_gap.csv
  G5_OPTIMIZATION_REPORT.md
```

G5 结束后冻结 architecture / capacity / router / depth / fusion / schedule / hyperparameters。

---

# 11. Stage G6 — Final Validation Confirmation

只允许 G5 冻结后的 1–2 个模型进入。

重新独立运行：

```text
5 NC datasets × seeds 42/43/44
```

不得复用 architecture search 中“最好的一次随机运行”作为最终结果。

## 11.1 正式报告

```text
每数据集 mean±std Accuracy
每数据集 mean±std Macro-F1
vs A0 delta
vs strongest external baseline delta
5-dataset average rank
#Wins / #Ties / #Losses
worst regression
parameter count
peak memory
training time
```

理想目标：

```text
Movies/Toys/Grocery 历史约 1pp gap 被基本消除或反超；
ele-fashion / Reddit-S 不出现明显灾难性回退；
overall average rank 优于现有 strongest external baseline。
```

如果关键数据集与 strongest baseline 差距绝对值 <=0.3pp，补 seeds 45/46。

输出：

```text
outputs/r2d29/g6_confirm/
  runs.csv
  aggregate.csv
  final_vs_baselines.csv
  resources.csv
  G6_FINAL_CONFIRM_REPORT.md
```

此时仍先停在 Validation confirmation。人工审查后才允许 Test。

---

# 12. Stage G7 — 架构冻结后再做论文机制归因

只有 final performance candidate 成功后，再回头建立论文证据链。

新的方法论顺序：

```text
先找到整体性能有效的 computation system
→ 再做 controlled ablation / causal attribution
```

而不是：

```text
先要求每个局部组件独立 GO
→ 再决定是否允许组成系统
```

## 12.1 最终核心 ablations

1. w/o Semantic Ownership / common-private collapse control；
2. uniform routing；
3. w/o null state；
4. source mean；
5. MEAN_DUP matched control；
6. w/o target interaction；
7. late integration 替代 factor write-back；
8. legacy fusion 替代 OIF；
9. L=1 / L=2 / L=3；
10. pair vs target routing；
11. within-target shuffle；
12. source-channel shuffle；
13. residual branch off；
14. A0-only / CORT-only / Hybrid decomposition（若最终为 hybrid）。

核心 ablation 建议仍做 3 seeds。

## 12.2 最终论文故事候选

### Problem

Common/Private disentanglement solves semantic ownership within a node, but MAG propagation still lacks a coordinated computation pathway for deciding which ownership-aware relational evidence should enter a target semantic state and how that evidence should be preserved, transformed, and integrated during state evolution.

### Core formulation

\[
\boxed{
\text{Semantic Ownership}
\times
\text{Coordinated Functional Relational Transfer}
}
\]

### Method flow

```text
Semantic Ownership Factorization
        ↓
Ownership-aware Relational Allocation
        ↓
Source-preserving Cross-factor Transfer
        ↓
Target-conditioned Functional Interaction
        ↓
Ownership-state Residual Update
        ↓
Ownership Interaction Fusion
```

若 recurrent CORT 有效，再加入：

```text
Iterative Relational Refinement
```

---

# 13. 给 AI/Codex 的全局 Master Prompt

建议在新的编码会话第一条直接粘贴：

```text
你现在接手一个多模态属性图（MAG）表示学习研究项目。

仓库：
https://github.com/CrisRipper777/0901

研究核心：
Semantic Ownership × Functional Relational Transfer。

当前目标已经从“局部机制归因”切换为“系统级整体性能提升”。最终目标是让新模型在 Movies / Toys / Grocery / ele-fashion / Reddit-S 五个 NC 数据集上整体超过当前 strongest external baselines，而不是追求某个局部 module 的独立 GO。

重要研究纪律：
1. 不要再执行 D2.8 的“单机制先 GO 才允许组合”的 stopping rule。
2. 允许单模块 main effect 很弱，但联合架构通过 interaction/synergy 获得大增益。
3. 所有正式 performance candidates 都必须 seeds=42,43,44。
4. 架构搜索阶段 Val-only，禁止读取/使用 Test 做选择。
5. 参数量不是硬约束，性能优先；但必须记录参数量、显存和训练时间。
6. 不要修改已有 frozen baseline/model files；新代码单独新增。
7. 保持现有 benchmark data split、训练协议和模型接口兼容。
8. 所有对比使用 same-code-path / matched protocol，输出 per-run raw CSV 和 aggregate CSV。
9. 不要只输出总结报告，必须保留原始 run-level 数据。
10. 如果代码/旧报告的结论与实际 CSV/实现不一致，以代码和原始数据为准，并显式指出。
11. 不要为了保护 A0 的历史结构而限制新模型。A0 只是 reference / warm start / optional strong path。
12. 不允许因为某个局部因素单独 NO-GO 而擅自停止联合架构实验。

先仔细阅读：
- src/models/biaxis_p0.py
- src/models/biaxis_components.py
- src/models/biaxis_p1.py
- src/models/biaxis_p2.py
- src/models/biaxis_p3.py
- src/models/biaxis_final.py
- src/models/biaxis_r2_relfunc.py
- src/models/biaxis_r2_relfunc_components.py
- src/tasks/nc.py
- configs/model/biaxis_final.yaml
- configs/model/dip.yaml

重点理解：
- P0 如何产生 C/Pt/Pv factors；
- A0 graph_update 如何得到 f_tilde；
- A0 fusion 如何得到 z；
- D2.7/D2.8 pre-aggregation pair scoring / null routing / source-channel computation；
- 当前 NC runner 如何训练和选择 checkpoint。

后续工作严格按照我给你的《R2-Design-2.9 System-Level Coordinated Relational Architecture Search》阶段文档推进。

每完成一个阶段：
A. 自动执行该阶段全部实验；
B. 自动汇总 CSV；
C. 自动生成 markdown report；
D. 给出 git diff / 新增文件清单 / commit hash；
E. 明确列出失败 run、OOM、NaN 或协议偏差；
F. 不要只告诉我“结果很好/不好”，必须给完整数字；
G. 到阶段边界必须停下，不要自行进入下一阶段。
```

---

# 14. Stage G0 给 AI 的 Prompt

```text
执行 R2-D2.9 Stage G0：重建当前 Validation performance reference。

目标：
在当前仓库/当前 commit 下，以统一协议重新运行五个 NC 数据集 × seeds 42/43/44 的正式 baseline reference。运行：MLP、GCN、GraphSAGE、MMGCN、MGAT、DMGC、DGF、DiP、LGMRec、biaxis_final。

要求：
1. Val-only，不读取 Test 做任何选择。
2. 保持各模型正式配置，不为新模型修改 baseline 超参数。
3. 记录 model,dataset,seed,best_epoch,val_acc,val_f1,param_count,peak_mem_mb,train_seconds,git_commit。
4. 生成每数据集 strongest external baseline（排除 biaxis_final）以及 strongest overall reference。
5. 自动计算 biaxis_final 与 strongest external baseline 的 gap。
6. 任何失败 run 记录 exception 和原因，不静默跳过。

输出：
outputs/r2d29/g0_reference/
runs.csv
aggregate.csv
strongest_baseline_by_dataset.csv
resources.csv
G0_REFERENCE_REPORT.md

完成后不要继续 G1，先汇报完整结果与文件路径。
```

---

# 15. Stage G1 给 AI 的 Prompt

```text
执行 R2-D2.9 Stage G1：实现并审计统一 CORT 模型。

先阅读阶段文档第 3~6 节。

新建：
src/models/biaxis_cort.py
src/models/biaxis_cort_components.py
configs/model/biaxis_cort.yaml
src/analysis/perf_r2d29_utils.py
tests/test_biaxis_cort.py
scripts/perf_r2d29_g1_audit.py

不要修改 biaxis_p0/p1/p2/p3/final/r2_relfunc 的历史实现。

实现 knobs：
backbone_mode = a0_augment | pre_a0 | sandwich | replace | hybrid
router_mode = uniform | target_null | pair_null
source_mode = mean | preserve_concat | preserve_attn
writeback_mode = late | factor
fusion_mode = legacy | oif | factor_attn
num_blocks
share_blocks
interaction_hidden_mult
fusion_hidden_mult
residual_init
pre_norm
edge_chunk_size

CORT block 必须实现：
1. null-augmented relational allocation；
2. source C/Pt/Pv messages 保留到 target interaction；
3. target-conditioned vector interaction；
4. factor-space residual write-back；
5. optional OIF；
6. recurrent blocks 每层基于更新后的 factors 重新计算 routing。

特别注意：
- a0_augment/pre_a0/sandwich 直接复用 A0 factorizer、_graph_update、fusion，不复制可能漂移的历史逻辑；
- replace 模式只保留 semantic factorizer，不使用 K=4/Γ/OFR；
- hybrid 在 factor space merge，不允许只在 classifier 前裸 concat；
- residual_init=0 验证 identity/near-identity；
- 大图需要 chunk/checkpoint；
- 保持现有 forward/inference contract。

完成：mass normalization、permutation invariance、isolated nodes、gradient flow、residual identity、source-channel independence、no NaN、API compatibility、no Test access、memory audit。

输出：
outputs/r2d29/g1_audit/
audit.json
grad_audit.csv
equivalence.csv
G1_AUDIT_REPORT.md

audit 全部 PASS 后停止，不自动开始 G2。
```

---

# 16. Stage G2 给 AI 的 Prompt

```text
执行 R2-D2.9 Stage G2：Full System Synergy Matrix。

前提：G0 reference 和 G1 audit 已完成。

固定：
backbone_mode=a0_augment
num_blocks=1

执行完整 2×2×2×2 factorial：
R: uniform vs pair_null
S: mean vs preserve_concat
W: late vs factor
F: legacy vs oif

共 16 variants。

每个 variant 全部运行：
Movies/Toys/Grocery/ele-fashion/Reddit-S × seeds 42/43/44。

禁止：
- 单 seed 筛选；
- 因某个 main effect 弱提前终止；
- 使用 Test；
- 只汇总 M/T/G 而忽略 ele-fashion/Reddit-S。

额外：
对 Top source-preserving variants 运行 MEAN_DUP matched control。

统计：
- per-dataset mean/std Acc/F1；
- delta vs A0；
- delta vs strongest external baseline；
- 5-dataset mean；
- average rank；
- wins/ties/losses；
- worst dataset regression；
- R/S/W/F main effects；
- 所有 two-way interactions；
- R×S×W、R×S×F、R×W×F、S×W×F、R×S×W×F higher-order interactions。

输出：
outputs/r2d29/g2_synergy/
runs.csv
aggregate.csv
factorial_cells.csv
main_effects.csv
two_way_interactions.csv
higher_order_interactions.csv
strongest_gap.csv
matched_controls.csv
resources.csv
G2_SYSTEM_SYNERGY_REPORT.md

报告必须区分 main effect、interaction effect、global model performance。
不要因为 main effect 不显著就把 component 判死。
按 global performance + Pareto 规则给出 Top-4 variants，但不要开始 G3。
```

---

# 17. Stage G3 给 AI 的 Prompt

```text
执行 R2-D2.9 Stage G3：Global Architecture Topology Search。

读取 G2 Top-4，选择其中 Top-2 component combinations 作为 topology search 基础，保留两个以避免 component×topology interaction 被忽略。

每个 combination 测：
T1 a0_augment
T2 pre_a0
T3 sandwich
T4 replace L=1
T5 replace L=2
T6 replace L=3
T7 hybrid L=1
T8 hybrid L=2

全部：5 datasets × seeds 42/43/44。

随后对 Top-3 topology 比较：
router_mode=pair_null vs target_null。

replace/hybrid 多层 CORT 每层必须基于更新后的 factor states 重新计算 edge routing，不允许重复固定 edge weights。

输出：
outputs/r2d29/g3_topology/
runs.csv
aggregate.csv
topology_comparison.csv
router_granularity.csv
depth_response.csv
strongest_gap.csv
resources.csv
G3_TOPOLOGY_REPORT.md

按五数据集 global performance 选 Top-3 architecture/configs。
不要开始 G4。
```

---

# 18. Stage G4 给 AI 的 Prompt

```text
执行 R2-D2.9 Stage G4：Structured Capacity Search。

输入：G3 Top-3 architectures。

性能优先，不限制参数量。不要做无目标 generic hidden sweep；围绕 CORT 的 structured computation capacity 搜索。

Axes：
factor_dim: 128 / 192 / 256
interaction_hidden_mult: 1 / 2 / 4
source_mode: preserve_concat / preserve_attn
share_blocks: true / false
fusion_mode: oif / factor_attn

生成 12~24 个具有清晰覆盖与对照的组合，不必全 Cartesian product。
所有组合全部：5 datasets × seeds 42/43/44。

记录 performance、params、peak memory、training time、strongest-baseline gap。

输出：
outputs/r2d29/g4_capacity/
runs.csv
aggregate.csv
capacity_grid.csv
param_performance.csv
strongest_gap.csv
resources.csv
G4_CAPACITY_REPORT.md

选 Top-3 进入 G5，但不要自动开始 G5。
```

---

# 19. Stage G5 给 AI 的 Prompt

```text
执行 R2-D2.9 Stage G5：Optimization / Co-adaptation Search。

输入：G4 Top-3 architecture/configs。

比较至少：
O1 joint training
O2 factorizer freeze -> gradual unfreeze
O3 strong-parent staged adaptation（若架构含 A0/hybrid）
O4 residual curriculum
O5 differential LR + scheduler
O6 基于前五项结果形成的 combined best schedule

全部 5 datasets × seeds 42/43/44。

必须记录训练动态：
val acc/f1
factor residual ratio
routing entropy
null mass
source-channel contribution
rho
per-module grad norm
必要时 gradient cosine

不要通过 Test 选择 schedule。

输出：
outputs/r2d29/g5_optimization/
runs.csv
aggregate.csv
schedule_comparison.csv
training_dynamics.csv
grad_stats.csv
strongest_gap.csv
G5_OPTIMIZATION_REPORT.md

选出最终 1~2 个 frozen candidates，并导出完整 final config。
不要开始 G6。
```

---

# 20. Stage G6 给 AI 的 Prompt

```text
执行 R2-D2.9 Stage G6：Final Validation Confirmation。

只使用 G5 已冻结的 1~2 个 final candidates。
不要再改 architecture、training schedule 或超参数。

重新独立运行：5 NC datasets × seeds 42/43/44。

输出：
- mean±std Accuracy/F1；
- delta vs A0；
- delta vs strongest external baseline；
- average rank；
- wins/ties/losses；
- worst regression；
- params/memory/time。

如果 candidate 与 strongest baseline 在某关键数据集差距绝对值 <=0.3pp，则补 seeds 45/46。

输出：
outputs/r2d29/g6_confirm/
runs.csv
aggregate.csv
final_vs_baselines.csv
resources.csv
G6_FINAL_CONFIRM_REPORT.md

这一阶段仍是 Validation confirmation。
先把结果返回给我审查，在我明确允许前不要进行 Test benchmark。
```

---

# 21. 每个阶段完成后返给 ChatGPT 的材料

最低要求：

```text
阶段 report.md
runs.csv
aggregate.csv
阶段专属 analysis CSV
最终 config yaml
git commit hash
```

如果发现异常，再附：

```text
stderr/log
失败 run list
OOM/NaN 记录
相关 diagnostics
```

不要只返 markdown 结论。

---

# 22. 当前最重要的决策规则

本阶段不再问：

> 哪个局部 gate 被证明最优？

而问：

\[
\boxed{
\text{哪一套完整 computation system 能让五个数据集的整体性能达到全局更优？}
}
\]

优先级：

```text
Global downstream performance
        >
Cross-dataset robustness
        >
Architecture coherence
        >
Mechanism attribution
        >
Parameter minimalism
```

机制归因不是取消，而是移动到**最终有效架构找到之后**。

---

# 23. 当前阶段流

```text
G0  重建 baseline reference
 ↓
G1  CORT implementation + audit
 ↓
G2  16-cell full system synergy matrix
 ↓
G3  A0/pre/sandwich/replace/hybrid topology search
 ↓
G4  structured capacity search
 ↓
G5  optimization/co-adaptation search
 ↓
G6  final validation confirmation
 ↓
人工审查
 ↓
Test 一次 + NC final benchmark
 ↓
G7  paper ablation / causal attribution
 ↓
LP extension
```

---

# 24. 何时暂停并返回分析

AI 在以下节点必须暂停：

1. G0 完成；
2. G1 audit 完成；
3. G2 完成；
4. G3 完成；
5. G4 完成；
6. G5 完成；
7. G6 完成。

每次暂停后，将输出文件返给 ChatGPT，由 ChatGPT 独立重新计算/诊断，再决定下一阶段是否修改。

这样避免编码 AI 根据旧报告或局部数字一路自动做错误决策。
