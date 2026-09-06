# R3-0 Implementation Report — Ownership-Structured Semantic Transition Network

> 阶段：R3-0（实现与正确性，计划 §19）
> 日期：2026-09-06
> 范围：R3-0A 代码实现 / R3-0B 单元测试 / R3-0C Smoke run

---

## 1. 新增 / 修改文件

### 新增

| 文件 | 内容 |
|---|---|
| `src/models/biaxis_r3_components.py` | `OwnershipTransitionLayer`：同一代码路径承载 diagonal/static/film/basis 全部 operator backend + edge chunking + aux stats（计划 §16.1：不复制 variant 文件） |
| `src/models/biaxis_r3.py` | `Model(P0Model)`：继承 P0 factorizer / recon heads / aux 目标 / final fusion，新增 Stage II transition stack + Stage III multi-scale readout + `inference`（整图单次 forward，同 P1 约定） |
| `configs/model/biaxis_r3.yaml` | 全部 knobs（`transition:` 子组），V0-V6 均由此配置 + CLI overrides 产生 |
| `tests/test_biaxis_r3.py` | T1-T7 + 3 个支撑测试（14 tests） |

### 修改（均向后兼容，历史模型行为不变）

| 文件 | 修改 |
|---|---|
| `src/tasks/common.py` | `update_aux_info_stats` / `summarize_aux_info_stats` / `format_aux_info_stats` 的迭代键表从固定 `AUX_INFO_KEYS` 扩展为 "AUX_INFO_KEYS + 实际出现的 `r3_*` 键"。历史模型不产生 `r3_*` 键 → 行为完全不变（全库 582 tests 验证）。 |

## 2. 精确架构（计划 §2-§11 逐条落实）

```
Text / Image
    ↓
P0 SemanticFactorizer（复用 biaxis_components，不改动）
    ↓
H^(0) = stack[C, Pt, Pv]        [N, 3, 128]   factor 顺序 0=C 1=Pt 2=Pv
    ↓
OwnershipTransitionLayer × 2    (L=2)
    1) same-node context   S_i = φ_s([C|Pt|Pv]) → [N, 64]   (仅 conditioning，非第四 factor)
    2) relational 投影      v_j^a = V_a H_j^a, q_i^b = Q_b H_i^b   (dual-space；direct 模式恒等)
    3) diagonal            m_ji^{b→b} = Linear_b^diag(v_j^b)      (static，无 basis/attention)
    4) off-diagonal        r_ji^{ab} = φ_r([q_i^b|v_j^a|S_i|e_a|e_b])
       - static: m = D_b^cross(W_ab^static v_j^a)        (6 个固定 channel 变换)
       - film:   (γ,β) = router(r) → m = D_b^cross(γ⊙v + β)
       - basis:  z_r = A_r(gelu(C_r v)), ω = softmax(router(r)) over R=4 个共享低秩 basis
                 m = D_b^cross(Σ_r ω_r z_r)              (basis 全 a→b 共享；factor identity 只进 router)
    5) offdiag scale        m ← ε_ℓ m   (ε_ℓ 每层可学标量，init 0.1)
    6) pre-aggregation 全部变换发生在聚合之前；edge chunk (200000) 内逐 channel 计算
       → scatter_add 累加 9 个 [N,128] 通道累加器 → 除以 deg（mean，计划 §5）
    7) source preservation M_i^b = [m̄^{C→b} ‖ m̄^{Pt→b} ‖ m̄^{Pv→b}]  (preserve=true)
       ΔH_i^b = U_b( LN( [H_i^b ‖ M_i^b] ) )           (pre-LN，无 post-LN)
    H_i^{b,ℓ+1} = H_i^{b,ℓ} + η_ℓ ΔH_i^b               (η_ℓ 每层可学标量，init 0.1)
    ↓
H^(0), H^(1), H^(2) 保留
    ↓
multi_scale=concat: H̄^b = Linear_b([H^{b,0}‖H^{b,1}‖H^{b,2}])   / last: H̄ = H^(2)
    ↓
z = P0 fusion([C̄‖P̄t‖P̄v]) → 256   (= out_dim)
    ↓
NC / LP head（runner 侧，模型不绑定）
```

### 计划 §1.2 排除项确认

未实现：neighbor attention、MoE、pseudo node、hop router、Transformer fusion、exposure gate（`use_exposure: true` 直接 raise NotImplementedError）、A0 的 K=4 relation / Γ / OFR。

## 3. Tensor shapes（Movies, N=16672, E=160802, d=128, d_r=128, R=4）

| Tensor | Shape | 常驻/瞬态 |
|---|---|---|
| H^(ℓ) | [N, 3, 128] | 常驻（每层一份 + states 列表） |
| v_a / q_b | [N, 128] × 6 | 常驻（层内） |
| S_i | [N, 64] | 常驻（层内，可选） |
| chunk 内 src/dst | [200000] × 2 | 瞬态 |
| chunk 内 v_s / q_dst | [200000, 128] | 瞬态 |
| basis z_r | [200000, 4, 128] | 瞬态（单 chunk 内，立即消费） |
| 消息 m | [200000, 128] | 瞬态（逐 channel 释放） |
| 通道累加器 acc | 9 × [N, 128] | 层内常驻 |
| z | [N, 256] | 输出 |

## 4. 参数量 / 复杂度 / 显存

- **params（Movies，含 head）= 2,170,752**（对比 biaxis_final/A0 ≈ 4.66M，更小）。构成：P0 factorizer+recon ≈ 1.0M；每 transition 层 ≈ 0.53M（V/Q 6×16K + diag 3×16.5K + basis 8×2K + router 46K + D_b^cross 3×16.5K + context 57K + update 3×16.5K）；multi-scale 148K；fusion 98.6K。
- **复杂度**：每层每条边 O(d_r² + R·d_r·rank + R·d_r)（basis 模式）；router 为每条 off-diagonal 边 O(d_r²)。相比 materialize [E, d_r, d_r] 的 O(E·d_r²) 显存，本实现只有 O(E·d_r) 瞬态显存（计划 §17 要求满足）。
- **显存**：ele-fashion（N=97,766, E≈200K）实测：**初版在 FULL/V5 训练 OOM（23.6GB）**——根因不是 chunk 瞬态，而是 autograd 对 6 个 off-diagonal pair 的逐边中间激活（router 输入 [E,352]、激活 [E,128]、basis zs [E,4,128]、decoder 输入）在 forward 期间全部保留。修复（与 p3/CORT 同款）：`transition.memory_checkpoint=true`（默认），diag/offdiag 消息计算段按 chunk 走 `torch.utils.checkpoint(use_reentrant=False)`，反向时重算 [E,d]-级中间量；为保证重算 bitwise 一致，router/context/basis 段内**无 dropout**（P0 factorizer/fusion 的 dropout 0.2 保留）。修复后 ele-fashion V5 smoke：rc=0、~1.5s/epoch、无 NaN；checkpoint-parity 测试（forward bitwise 相等 + grad allclose 1e-6）。edge_chunk_size 200000→100000 仍为降配杠杆（计划 §14）。

## 5. 单元测试（R3-0B，14 tests，全过）

| 计划编号 | 测试 | 结果 |
|---|---|---|
| T1 | layer/model shape、no-edge 安全 | PASS |
| T2 | cross_factor=false 不构造 offdiag 模块；offdiag_scale 冻结 0 时输出对 offdiag 权重扰动 bitwise 不变、梯度为 None、diag 仍有梯度 | PASS |
| T3 | layer_scale=0 → layer 输出与输入 torch.equal（无 LN confound）；模型级：L 层零 scale + multi_scale=last 时 z == fusion(raw factors) bitwise | PASS |
| T4 | edge_index 随机置换 → mean 聚合输出 allclose(1e-5)（repo 累加顺序约定） | PASS |
| T5 | 9 个模块组梯度范数全 >0；无 requires_grad 参数 grad 为 None | PASS |
| T6 | 随机初始化 offdiag/diag ratio = 0.10±0.02 ∈ (0,1)（Movies smoke 实测 0.08-0.15） | PASS |
| T7 | eval forward == inference（CPU 返回）；eval 无 aux 输出；training aux 含 r3_ 键且有限 | PASS |
| T8 | `pytest tests/` 全库 **582 passed**（568 历史 + 14 新） | PASS |

## 6. Smoke run（R3-0C）

```bash
python -m src.main dataset=Movies task=nc model=biaxis_r3 \
  num_runs=1 seed=42 task.epochs=3 task.evaluate_test=false
```

结果：无 OOM；Train Loss 3.19→2.59→2.48；~1.0 s/epoch；全部 P0 + r3 aux stats 正常有限；offdiag/diag ratio 0.08-0.15；basis 未瞬间 collapse（epoch 1 entropy 1.33/1.35，epoch 3 层 1 top1 上升至 0.998——训练早期现象，正式 run 观察）；layer_scale/offdiag_scale 轨迹可学习（0.100→0.098-0.101）；results.json 正常（val-only，无 test 字段）。

## 7. Gradient audit（T5 详情）

loss = Σz² + aux 一次 backward：factorizer / diag / basis_down / basis_up / router / target_decode / update / ms_proj / fusion 九组范数全部 >0；无 "has_grad 但 grad_norm=0" 整支路（计划 §16.6）。

## 8. 已知限制 / 设计点（供 R3-2 参考，未擅自升级）

1. **ΔH 无激活**：按计划 §7 公式字面实现（U_b(LN(·)) 线性写回，非线性只来自 LN / router / context / final fusion）。若 R3-1 ceiling 不足，这是 R3-2 的合法候选（"operator capacity 调优"）。
2. **静态 diagonal 未加 activation**（计划 §3.3 字面）。
3. **router/context 无 dropout**（memory-checkpoint 段必须无随机性，2026-09-06 ele-fashion OOM 修复时决定；P0 factorizer/fusion dropout 0.2 不受影响）。
4. `log_grad_stats=true` 时 grad norm 为上一 step 的值（backward hook 时机），仅 debug 用，默认关闭。
5. sampled training 可用（forward 无全局 buffer 依赖），但 full_graph_training=true 为默认协议（与 A0 一致）。
6. exposure gate 未实现（计划 §12 明确暂缓）。

## 9. Git 状态

- 分支：main；未修改任何历史模型文件（biaxis_p0/p1/p2/p3/final/cort/r2_* 零 diff）。
- 本阶段提交内容：biaxis_r3.py / biaxis_r3_components.py / biaxis_r3.yaml / test_biaxis_r3.py / tasks/common.py（白名单扩展）/ 本报告 / R3 计划文档。
