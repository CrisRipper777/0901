# Performance-R1 Repository Audit（Prompt 1 输出）

> 2026-09-03。对应计划 `BiAxis_Performance_R1_Enhancement_Plan.md` §35 Prompt 1。
> 本审计只读不改：未修改任何 frozen 文件，未训练，未实现模块。

---

## 0. 审计对象与关键事实

| 对象 | 文件 | 关键结论 |
|---|---|---|
| P0 factorizer / fusion / aux losses | `src/models/biaxis_p0.py`, `biaxis_components.py` | `_encode(x)` → `{c,p_t,p_v,h_t,h_v,...}` + `z_local`；R1 完全继承，零改动 |
| P1 M2 关系分解 | `src/models/biaxis_p1.py` `_decompose_relations` (L141-162), `biaxis_p1_components.py` | `r [E,K]`、`availability [N,K]`、`deg [N]`；拓扑-only；raw signature 有 input-keyed cache |
| P1 聚合 | `biaxis_p1_components.relation_weighted_mean` (L185-224) | `g [N,K,F*d]` + `mass [N,K]`；K 循环 index_add；`edge_chunk_size` 分块分支存在 |
| P2 计分/plan | `biaxis_p2_components.py` | `FactorRelationScore`（无 availability 输入）、`build_augmented_scores`、`null_augmented_softmax`、`build_reference_capacity`（detach availability） |
| P2 model | `src/models/biaxis_p2.py` `_graph_update` (L138-268) | 删 P1 budget/selector；硬断言 `factor_aware`/`num_relations=4`；`p2.deterministic` 分支 |
| P3 model | `src/models/biaxis_p3.py` | 继承 P2Model；硬断言 `p2.mode∈{null_softmax,composition_uot}`、`deterministic=false`；`_graph_update` (L110-200) 只改 message interpretation（per-cell operator） |
| P3 算子 | `biaxis_p3_components.py` | OFR 全残差 `T=W0+A_f+B_k+C_fk`；zero-init；aggregate-first；无 `[N,F,K,d,d]` |
| final | `src/models/biaxis_final.py` | **P3Model 的薄别名**，无任何新代码 |
| registry | `src/models/factory.py` | `importlib` 按 `cfg.model.name` 动态导入 `module.Model`，**无需注册**——新增 `biaxis_perf_r1.py` 即自动可用 |
| config | `configs/model/biaxis_final.yaml` | 冻结结构：p2.mode=null_softmax、p3.operator_mode=full_interaction |
| trainer/ckpt | `src/tasks/nc.py` | ckpt = `{model_state, head_state, data_info}`（L301-303）；Val Acc 早停 patience30；`task.save_ckpt_path` |
| R0 分析层 | `src/analysis/perf_r0_utils.py` + `scripts/perf_r0_*.py` | 冻结权重 counterfactual 范式；**硬编码 `relation_weighted_mean`** → R1 需要自己的 pipeline helper |
| 测试范式 | `tests/test_biaxis_p{0,1,2,3}.py`（合计 167+61） | zero-init bitwise 等价、参数量、mode 隔离、GPU device 守卫 |
| 已有 mode-switch 先例 | `src/models/biaxis_ablation.py` | `Model(P3Model)` + passthrough 映射 + 8 个等价测试——**R1 应完全照搬此模式** |

### 数据集实际规模（框架 loader 实测，2026-09-03）

| dataset | N | E（无向+双向，预处理后） | x dim | g_perm [N,F,K,d] |
|---|---:|---:|---:|---:|
| Movies | 16,672 | 160,802 | 1536 | 328 MB |
| Toys | 20,695 | 113,402 | 1536 | 407 MB |
| Grocery | 17,074 | 142,262 | 1536 | 336 MB |
| Reddit-S | 15,894 | 283,080 | 1536 | 312 MB |
| ele-fashion | 97,766 | 399,172 | 1024 | **1.92 GB**（f32，含 permute 后视图） |

**重要发现 A**：当前 `edge_chunk_size=500000`，而 5 个数据集中最大 E = ele-fashion 399,172 < 500,000 → **现有 NC 协议下分块分支从未真正触发**。ele-fashion 的显存压力来自节点侧张量（g_perm ≈ 600 MB 实际存储 + `f_cat[src]` 全量 gather `[E,384]` ≈ 613 MB 瞬态），而非边分块。R1-A 的 reliability 路径仍**必须实现分块**（纪律 + 未来更大图），但可以给 reliability 单独一个更小的 chunk（如 200K），把 ele-fashion 的 token 瞬态从 ~155 MB 压到 ~78 MB。

---

## 1. 12 个问题的回答

### Q1. 新 `biaxis_perf_r1` 最安全的继承层次？

```
biaxis_perf_r1.Model(P3Model)        # 与 biaxis_ablation.Model 完全同款模式
    ├── mode=baseline               → 不构造任何新模块，_graph_update 走 super()
    └── mode=semantic_reliability   → 只重写 _graph_update 的 context 聚合块（Q3），
                                      另加 compute_r1_diagnostics
```

理由：
- `biaxis_final` 是 `P3Model` 的薄别名（`biaxis_final.py:13`），继承 P3Model 即继承 OFR 算子、NullSoftmax、P2 硬断言（null_softmax + deterministic=false，`biaxis_p3.py:58-65`），天然满足 R1 的冻结纪律。
- `biaxis_ablation.py` 已用同一模式做过 6-mode + bitwise 等价测试，风险最低。
- P3 的 `_graph_update` 返回 dict（`f_tilde,beta,alpha,r,availability,gamma,null_mass,relation_confidence,theta,g_perm`），R1 只需在该函数体内替换一处代码块（Q3），返回值结构不变 → 下游 P2 diagnostics、P3 operator diagnostics 全部白拿。

### Q2. 如何实现 same-code-path baseline 且保证 same-weights == biaxis_final？

三层保证：

1. **结构**：`mode=baseline` 时 `__init__` 不创建 reliability 模块（`FactorConditionedEdgeReliability` 只在 `semantic_reliability` 模式下实例化）→ **state_dict keys 与 biaxis_final 完全一致** → 可直接 `load_state_dict(biaxis_final 的 model_state)`。
2. **代码路径**：`_graph_update` 开头 `if self.r1_mode == "baseline": return super()._graph_update(...)` → 与 biaxis_final 执行完全相同的算子序列 → **bitwise 等价**（torch.equal）。
3. **配置**：`biaxis_perf_r1.yaml` 逐字段镜像 `biaxis_final.yaml`（hidden_dim=256/factor_dim=128/dropout=0.2/lambda 0.02/0.01/0.3/p1/p2/p3 全同），另加 `r1:` 节；模型内再硬断言 `p3.operator_mode=="full_interaction"`（防御性，防止误用 ablation 式覆盖）。

等价测试（`tests/test_biaxis_perf_r1.py`）：
- 用同一 `data_info` + 同一 cfg 构建 `biaxis_final.Model` 与 `biaxis_perf_r1.Model(mode=baseline)`，把前者 state_dict 载入两者，随机输入 forward → `torch.equal(z)`。
- 训练一步后（或随机 step 后）再次断言 bitwise 相等（排除只在 step-0 成立的假等价）。
- `semantic_reliability` 模式 fresh 模型（η zero-init）与 baseline 输出 bitwise 相等（η≡1 严格成立时的推论，见 Q7）。

### Q3. R1-A reliability 应插入 `relation_weighted_mean` 的哪个位置？

插入 `biaxis_p3._graph_update` 的 **L126-130 这一个块**：

```python
# 现状（P3，biaxis_p3.py:125-130）
f_cat = f_block.reshape(num_nodes, num_factors * factor_dim)
g_cat, _mass = relation_weighted_mean(edge_index, r, f_cat, num_nodes, ...)
g_perm = g_cat.reshape(...).permute(0, 2, 1, 3)
```

替换为 reliability 版聚合（`biaxis_perf_r1_components.reliable_relation_weighted_mean`）：

```python
# R1-A1：r_str 不变、availability 不变，只有 context 聚合带 η
g_perm, effective_mass = reliable_relation_weighted_mean(
    edge_index, r, f_block, self.reliability, num_nodes,
    edge_chunk_size=self.edge_chunk_size,
    rel_chunk_size=self.r1_rel_chunk_size,
)   # g_perm [N,F,K,d]，effective_mass_ifk = Σ_j r_ji,k η_ji^f
```

下游全部原样：`s_rel = transport_scorer(f_block, g_perm)`、`build_augmented_scores`、`build_reference_capacity(availability,...)`、`gamma`、`operator(g_perm, gamma[...,1:], graph_w0)`——**只有 g_perm 的内容变了**，正好是计划 §6 的公式。

注意：不要改动 `relation_weighted_mean` 本体（frozen P1 文件）。新 helper 放 `biaxis_perf_r1_components.py`，η≡1 时与 `relation_weighted_mean` 数值等价（Q7/测试）。

### Q4. 如何避免存 E×F reliability tensor？

**chunk × factor 即时消费**，η 只以 `[chunk, F]`（或 `[chunk]` 单 factor）形式短暂存在：

```text
for start in range(0, E, rel_chunk):          # 边分块
    f_src_c = f_block[src[start:end]]          # [C, F, d]
    f_dst_c = f_block[dst[start:end]]          # [C, F, d]
    eta_c = reliability(f_src_c, f_dst_c)      # [C, F]，即时计算
    for f in range(F):
        for k in range(K):
            w = r_c[:, k] * eta_c[:, f]        # [C]
            acc[:, f, k].index_add_(0, dst_c, w.unsqueeze(-1) * f_src_c[:, f])
            eff_mass[:, f, k].index_add_(0, dst_c, w)
```

- 块内瞬态：token `[C, 97]`、MLP hidden `[C, 64]`、η `[C, F]`、每 (f,k) 的 `[C, d]` —— C=200K 时约 78 MB（token）+ 102 MB（weighted）级别，循环即弃。
- 新增持久张量只有 `effective_mass [N,F,K]`（ele-fashion 4.7 MB）——计划 §9 白名单内。
- `g_perm` 用 `acc` 除 `effective_mass` 得到；`acc [N,F,K,d]` 就是 g_perm 本体，不额外留副本。
- η 统计诊断（§10）在 `compute_r1_diagnostics` 的 `no_grad` 分块重算中完成，训练路径不保存 `[E,F]`。

### Q5. g_perm / effective_mass 应怎样计算？

沿用 P1 的 scatter 语义，权重换成 `r_ji,k · η_ji^f`：

```text
acc_ifk        = Σ_{j∈N(i)}  r_ji,k · η_ji^f · f_j          [N,F,K,d]   （chunk index_add）
effective_mass_ifk = Σ_{j∈N(i)}  r_ji,k · η_ji^f            [N,F,K]
g_ifk          = acc_ifk / (effective_mass_ifk + eps)       [N,F,K,d]
```

实现要点：
- `r` 由 `_decompose_relations` 照常产出（[E,K] f32），不因 reliability 变化。
- 聚合顺序与 `relation_weighted_mean` 不同（chunk×f×k 而非全量×k）→ 浮点舍入不同 → η≡1 等价测试用 `allclose(rtol/atol=1e-5)`，不用 `torch.equal`（P3 的 zero-residual 用 equal 是因为零贡献精确为 0，此处不适用）。
- 隔离节点：无入边 → acc=0、effective_mass=0 → g=0/eps=0，无 NaN；Gamma 的 isolated fast path 不受影响（deg 照常传入）。

### Q6. structural availability 是否继续保持原 a_ik 不变？

**是，严格不变。** `a_ik = mass_ik / deg_i`，其中 `mass` 只由 `r` 算（`relation_mass`，`biaxis_p1_components.py:165-174`）。R1-A1 的 η 只影响 context（g），**不进入 availability**：

- `build_reference_capacity(availability,...)`（P2 容量参考）不动；
- R0 证明 `Γ ∝ a` 不可靠，但那是 routing 的事（R1-B）；R1-A 阶段 availability 的语义（结构供给先验）原样保留；
- `effective_mass` 是**独立新量**，只用于 context 归一化分母、§10 诊断、以及未来 R1-B 的 evidence feature（`log1p(m̃_ifk)`）——不回流到 `nu`/`a`。

### Q7. Reliability gate zero-init η=1 如何严格实现？

标准做法，与 P3/LowRank/Basis 的 zero-init 纪律一致：

```python
self.mlp = nn.Sequential(
    nn.Linear(3 * proj_dim + 1, hidden), get_activation(act),
    nn.Linear(hidden, 1),
)
nn.init.zeros_(self.mlp[-1].weight)
nn.init.zeros_(self.mlp[-1].bias)
eta = 2.0 * torch.sigmoid(delta)     # delta≡0 ⇒ eta≡1.0
```

- f32 下 `sigmoid(0)=0.5` 精确可表示 → `2*0.5=1.0` **精确等于 1** → fresh A1 模型与 baseline 输出 bitwise 相同（Q2 测试 3）。
- η∈(0,2) 由 `2σ(·)` 保证（σ∈(0,1) 严格开区间）。
- 投影 `P_f: 128→32` **bias=True**（防止 f 落在核空间时 u=0 导致 cosine 退化）；cosine 项分母 `‖u_i‖‖u_j‖+eps`，双零向量时定义 cos=0（clamp）。
- 梯度动力学注意：step-0 只有最后一层有非零梯度（hidden 激活非零）；P_f 与 hidden 层梯度在 step-0 精确为 0、在最后一层移动一步后激活——与 LowRank U/V 的注释相同（`biaxis_p3_components.py:307-311`），**预期行为，不是 bug**；梯度测试要按两步设计。

### Q8. factor-specific 128→32 projection + shared MLP 是否有实现/显存问题？

无问题：

- 参数：P_f 合计 3×(128×32+32) = 12,384；shared MLP（97→64→1）= 6,336；η 侧总计 ≈ **18.7K**，相对 OFR 311K 可忽略。
- 瞬态（chunk=200K，见 Q4）：token `[C,97]` ≈ 78 MB、hidden `[C,64]` ≈ 51 MB、η `[C,F]`、每 (f,k) weighted `[C,128]` ≈ 102 MB —— 与现 P3 路径的 `[E,384]` gather（613 MB）同级或更低，循环回收。
- 建议：reliability 单独 `rel_chunk_size`（默认 200,000，config 可调），与 `edge_chunk_size` 解耦；ele-fashion（E=399K）自动分 2 块。
- dtype 纪律：η 与 r 同 f32；`u=torch.nn.functional.normalize` 或直接用原始 P_f 输出均可（cosine 用点积/范数比即可，不必 normalize 存储）。

### Q9. R1-B future dynamic Local score 最安全插在哪里？

在 R1 的 `_graph_update` 内、**score 组装处**（对应 `biaxis_p3.py:132-134` 之后）：

```text
s_rel = transport_scorer(f_block, g_perm)                       # 不变
g_bar = neighbor_mean(edge_index, f_cat, ...)                    # 新增（当前 P2/P3 不计算 g_bar！）
delta_local = MLP_0([f ‖ g_bar ‖ |f-g_bar| ‖ f⊙g_bar])           # zero-init，[N,F,1]
s_aug = build_augmented_scores_with_local(s_rel, null_score + delta_local)
s_rel = s_rel + delta_evidence([log1p(effective_mass) ‖ availability])   # zero-init residual
gamma = null_augmented_softmax(s_aug, eps)                      # 之后全部不变
```

要点：
- **重要发现 B**：P2/P3 已删掉 P1 的 budget/selector，`_graph_update` 中**不计算 g_bar**（P1 算它是给 budget 用的）。R1-B 的 dynamic local 需要新增一次 `neighbor_mean` 调用——函数现成（`biaxis_p1_components.neighbor_mean`，支持 edge_chunk_size），成本 ≈ 一次 `[E,384]` gather（与 relation 聚合同级）。
- 扩展方式：给 `build_augmented_scores` 加可选 `local_score` 参数（不改 frozen 文件——可以在 R1 自己的文件里写一个 4 行 wrapper，不动 P2 组件）。
- `s_ifk^base` 后加 zero-init `δ_ifk^evidence`，输入只含 support 特征（log1p(m̃)/a），**不得**把 availability 变回 hard prior（计划 §20 禁令）。
- 两处都从 `s_aug`/`s_rel` 之后、softmax 之前插入 → 天然满足 zero-init == parent Γ。

### Q10. 共享第二 hop 时，当前 raw topology cache 是否可安全复用？

**安全，可直接复用**，依据：

- cache 语义（`biaxis_p1.py:126-139`）：keyed on `(num_nodes, edge_index.data_ptr())`，内容 = `compute_raw_struct_signature(edge_index).detach()` —— **只依赖拓扑、与 f_block 无关**。R1-C 第二跳传入同一 edge_index（full-graph runner 复用同一 hosted tensor）→ cache hit，且无任何状态被 f 污染。
- `_decompose_relations` 每次调用重算 `s=MLP_S(...)`、`e=MLP_E(...)`、`r=prototypes(e)`：权重相同 → 两跳的 r 数值相同（同 op 序列）；梯度正常回传到 M2 模块（第一跳已建立 autograd 路径，第二跳再贡献一次）。
- P2 deterministic 分支有自己的 cache（`biaxis_p2.py:110-132`）——R1 被 P3 硬断言锁在 `deterministic=false`，不涉及。
- 唯一注意：`_graph_update` 内含 `graph_norm` 与 isolated fast path，R1-C 的 "GraphBlock" = 完整 `_graph_update`（含 norm）。计划 §26 公式的外层 LN 见下方"风险点 3"。

### Q11. 代码文件计划 / 测试计划 / 显存风险 → 见 §2-§4。

### Q12. 不修改任何 frozen model file

确认：所有新增独立文件；`biaxis_p0/p1/p2/p3/final/ablation` 及各自 components/yaml/tests 零改动。等价性全部通过"新代码调用旧代码"（super() 委托 / 导入 frozen 组件）达成，与 P3 阶段对 P2 的处理方式一致（P3 审计先例：8 新文件、零 frozen 改动）。

---

## 2. 代码文件计划

```text
src/models/biaxis_perf_r1_components.py      # FactorConditionedEdgeReliability + reliable_relation_weighted_mean
src/models/biaxis_perf_r1.py                 # Model(P3Model)，r1.mode ∈ {baseline, semantic_reliability}
configs/model/biaxis_perf_r1.yaml            # 镜像 biaxis_final.yaml + r1: 节
tests/test_biaxis_perf_r1.py                 # 组件 + 模型全部单测（§3）
src/analysis/perf_r1_utils.py                # R1 分析层：load_r1_setup / extract_forward_r1 / reliable_pipeline（含 CF override）
scripts/run_perf_r1_screen.py                # 驱动（照抄 run_p3_operator_screen 的加权信号量框架）
scripts/analyze_perf_r1_checkpoint.py        # best-ckpt 诊断：η 统计 + D_ctx + weighted coherence + Δ_relctx + CF0/CF1/CF2
scripts/summarize_perf_r1.py                 # paired delta 汇总 + GO 判定
outputs/perf_r1/{baseline,reliability,relation_calibration,routing,multihop,final,summary}/
```

组件接口设计：

```python
class FactorConditionedEdgeReliability(nn.Module):
    """P_f: 128->32（三投影，bias=True）；shared MLP: [u_i+u_j | |u_i-u_j| | u_i⊙u_j | cos(u_i,u_j)] -> δ；
    η = 2σ(δ)；最后一层 zero-init。forward(f_src_c, f_dst_c) -> η [C, F]。"""

def reliable_relation_weighted_mean(edge_index, r, f_block, reliability, num_nodes,
                                    edge_chunk_size=None, rel_chunk_size=None,
                                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (g_perm [N,F,K,d], effective_mass [N,F,K])。chunk×factor×relation 循环，见 §1 Q4/Q5。"""
```

模型侧：

```python
class Model(P3Model):
    def __init__(...):
        super().__init__(cfg, data_info)          # 全部冻结结构照常
        assert self.p3_operator_mode == "full_interaction"
        self.r1_mode = str(cfg.model.r1.mode)
        if self.r1_mode == "semantic_reliability":
            self.reliability = FactorConditionedEdgeReliability(...)   # baseline 不构造 → state_dict 一致
    def _graph_update(self, f_block, edge_index, num_nodes):
        if self.r1_mode == "baseline":
            return super()._graph_update(f_block, edge_index, num_nodes)
        # 复制 P3 函数体，仅把 L126-130 换成 reliable_relation_weighted_mean；
        # 返回 dict 增加 "effective_mass" 键
    @torch.no_grad()
    def compute_r1_diagnostics(self, x, edge_index) -> dict:
        # P3 diagnostics + η mean/std/p10/p50/p90、frac<0.5、frac>1.5（per factor）
        # + corr(η, semantic cosine)（分块）+ effective_mass 统计
        # + D_ctx（R0 同定义，mask 用 structural mass≥0.5 保持可比）
        # + weighted semantic coherence（r·η 加权 Sim_{f,k}）
```

分析层（`perf_r1_utils.py`）关键差异 vs `perf_r0_utils`：

- `resolve_cfg` 用 `model=biaxis_perf_r1` + `model.r1.mode=...` override（hydra `config_path="../../configs"` 相对 src/analysis 的坑照抄 R0 做法）；
- `extract_forward_r1` 调用模型的 `_graph_update`（它会自动走 reliability 路径），不必在脚本里手写聚合；
- `reliable_pipeline(setup, gamma_override)`：复制 `perf_r0_routing_counterfactual._pipeline` 的结构，但 g_perm/effective_mass 从 checkpoint 模型的 `_graph_update` 取，CF0/CF1/CF2 复用 R0 的 `_cf1/_cf2`（uniform / availability，η 不变、只换 Γ）；
- Δ_relctx 的 fixed Ridge probe 直接复用 `perf_r0_utils.ridge_probe`（Ridge α=1、StandardScaler、train fit / val eval、不读 test）。

驱动（`run_perf_r1_screen.py`）：

```text
MODE_OVERRIDES = {"A0": ["model.r1.mode=baseline"], "A1": ["model.r1.mode=semantic_reliability"]}
# 照抄 run_p3_operator_screen：CUDA_VISIBLE_DEVICES + 每 GPU 加权信号量（ele-fashion 占满卡）、
# summary.json skip/resume、_poll_peak_mem、train.log 解析 Val F1、best-ckpt 后接 analyze 脚本。
```

输出布局（计划 §33）：

```text
outputs/perf_r1/
  baseline/<dataset>/A0/seed_<s>/        # 训练产物 + diagnostics
  reliability/<dataset>/A1/seed_<s>/
  summary/                                # paired CSV + 阶段报告
```

---

## 3. 测试计划（计划 §34 + Prompt 2 清单）

### 组件级（`FactorConditionedEdgeReliability` / `reliable_relation_weighted_mean`）

| # | 测试 | 断言 |
|---|---|---|
| T1 | η neutral | fresh 模块任意输入 → `torch.equal(eta, ones)`（精确 1） |
| T2 | η range | 随机扰动权重后 → `0 < η < 2` 全元素严格成立 |
| T3 | 对称性 | 无向反向边 (j,i)/(i,j) 的 η **精确相等**（token 逐元素对称 + 相同 op 序） |
| T4 | η≡1 聚合等价 | reliable 聚合 vs `relation_weighted_mean`：`allclose(1e-5)`（累加顺序不同，非 bitwise） |
| T5 | chunk/full 等价 | rel_chunk=∞ vs rel_chunk=4096 输出一致（allclose）；chunk≥E 与无 chunk 走同一路径应 equal |
| T6 | 梯度 | 一步 backward：η 侧 loss 有回传；step-0 仅最后一层 grad 非零、P_f/hidden 为零（预期）；两步后 P_f grad 非零；全部 finite |
| T7 | 隔离节点 | 无入边节点 g=0、effective_mass=0、无 NaN |
| T8 | 显存纪律 | 大图 smoke（N=4096、E=50K、d=32、chunk=8K）正常运行 + 断言函数体内无 `[E,F,d]`/`[E,F,K,d]` 形状的持久分配（代码审查 + peak 上界断言，仿 P3 `test_no_giant_tensor_on_large_batch` 的 smoke 风格） |

### 模型级（`biaxis_perf_r1.py`）

| # | 测试 | 断言 |
|---|---|---|
| M1 | baseline same-weights | `biaxis_final.Model` 与 `biaxis_perf_r1(mode=baseline)` 载入同一 state_dict，同输入 `torch.equal(z)`（含随机 step 后复测） |
| M2 | baseline state_dict keys | 与 biaxis_final keys 完全一致（baseline 不构造新模块的直接后果） |
| M3 | A1 zero-init 等价 | fresh `semantic_reliability` 与 fresh baseline 输出 `torch.equal` |
| M4 | mode 校验 | 未知 mode raise；semantic_reliability 下 reliability 模块存在 |
| M5 | inference 等价 | `inference(x, ei)` == eval-mode `forward`（CPU 路径） |
| M6 | A1 梯度 | 训练模式 forward+backward 全部 finite；η 相关参数进入 optimizer 参数集 |
| M7 | 隔离节点 | 全隔离图上 A1 forward 无 NaN、gamma 行和=1 |
| M8 | 诊断键 | `compute_r1_diagnostics` 返回键齐全且 JSON-safe（η stats/D_ctx/effective_mass/weighted coherence + P3 原有键） |

计数预估：组件 ~8 + 模型 ~8 ≈ 16 个新测试，全部 CPU 可跑（GPU 守卫照 P3 惯例）。

---

## 4. 显存风险

| 项 | 量级 | 结论 |
|---|---|---|
| baseline 模式 | == biaxis_final | 零新增（ele-fashion 峰值 ~18 GB 不变） |
| A1 持久新增 | effective_mass `[N,F,K]`：ele-fashion 4.7 MB | 可忽略 |
| A1 瞬态 | chunk 200K：token 78 MB + hidden 51 MB + weighted `[C,128]` 102 MB（循环回收）；较现状 `[E,384]` gather 613 MB **同级或更低** | 安全 |
| A1 参数 | ≈ 18.7K（vs OFR 311K） | 可忽略 |
| R1-B（后续） | 新增 g_bar gather `[E,384]` ≈ 613 MB 瞬态 + 两个小 MLP | 与现状同级，安全 |
| R1-C（后续） | 第二跳 +1× g_perm（ele-fashion ~600 MB）+ `[N,3d]` gate 输入 + F(2) | +~1.3 GB 峰值；继续沿用 ele-fashion 独占整卡策略（driver 加权信号量已支持） |

风险点清单：

1. **chunk 累加顺序**：reliable 聚合顺序与 P1 不同 → 任何"等价"断言必须 allclose 而非 equal；但 A1-vs-baseline 的 fresh 等价是 η≡1 的乘法精确性 + 相同实现路径，仍可 equal。
2. **dtype**：η 与 r 保持 f32；不要引入 double 到聚合主路径（诊断侧可用 f64 累加，照 R0 惯例）。
3. **R1-C 公式外层 LN 与 zero-init 等价冲突**（预先标记，R1-C 阶段定稿）：计划 §26 `F^out = LN(F(1) + λ W(F(2)-F(1)))`，λ=0 时 `F^out = LN(F(1)) ≠ F(1)` —— 违反 §34 测试 1 "depth gate zero-init => parent 1-hop output"。建议二选一：(a) 去掉外层 LN（F(1) 已过 graph_norm）；(b) LN 移到残差路径 `F^out = F(1) + λ·LN(W(F(2)-F(1)))`。推荐 (b)，保留归一化且严格满足 λ=0 等价。**在 R1-C 开工前与审查方确认**。
4. **g_bar 缺失**（Q9）：R1-B 前需在 R1 的 `_graph_update` 里补 `neighbor_mean`，不要在 frozen P1 文件里动。
5. **cfg 漂移**：`biaxis_perf_r1.yaml` 逐字段镜像 final；模型内硬断言 `full_interaction` + 继承的 P2/P3 断言兜底。
6. **`_sig_cache` 与第二跳**：input-keyed、detached，安全；但 R1 不开启 deterministic 模式（继承硬断言）。

---

## 5. 审计结论（可否开工）

- R1-A 全部设计点在现有代码中都有明确落点：插入点唯一（P3 `_graph_update` L126-130）、继承链有已验证先例（biaxis_ablation）、零 frozen 改动可行（P3 审计同款模式）。
- 无阻塞性问题；4 个风险点均已给出处置方案（其中风险点 3 需 R1-C 前确认）。
- **建议按计划 §36 Prompt 2 开工：先实现 `biaxis_perf_r1_components.py` + 组件测试。**
