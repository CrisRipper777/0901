# B0 — Benchmark / Protocol Provenance Audit（2026-09-03）

> 对应计划 §3/§17 Prompt 1。审计范围：8 baseline 的 120 个历史 runs、统一 NC 协议、
> biaxis_final 冻结结构、split 对齐。**结论：全部 PASS，120 baseline runs 全部可
> 复用，零重跑。**

## 1. 统一协议一致性（120/120 runs 全量扫描）

每个 run 的 `.hydra/config.yaml` 快照中 task 配置**全部**为：

```text
training_mode=full_graph, epochs=300, patience=30,
lr=1e-3, weight_decay=1e-4, optimizer=adamw, grad_clip=None,
inference_mode=full, early_stop_min_epoch=1
```

与当前 `configs/task/nc.yaml` 逐项一致（唯一协议组，120/120 runs 无一例外）。
results.json 完整性：120/120 含 val_acc/test_acc/test_macro_f1 {mean,std}。

## 2. Model preset 一致性

8 个 model 的当前 yaml 与快照**语义一致**（差异仅为 YAML 字符串 vs 浮点表示）。
lr/wd preset 表（model 级 override task 级，nc.py:48-49）：

| model | lr | weight_decay | 备注 |
|---|---|---|---|
| mlp | 1e-3 | 1e-4 | = task 默认 |
| gcn | 5e-3 | 1e-5 | preset |
| sage | 5e-3 | 1e-5 | preset |
| mmgcn | 5e-3 | 1e-5 | preset |
| mgat | 5e-3 | 1e-5 | preset（MGAT 坍塌修复后版本：Movies s42 val 0.5429 正常） |
| dmgc | 5e-3 | 1e-5 | preset |
| dgf | 5e-3 | 1e-5 | preset |
| dip | 1e-3 | 1e-5 | preset |
| **biaxis_final** | **1e-3** | **1e-4** | P3 冻结协议（非调参） |

## 3. Split 对齐

- MAGB（Movies/Toys/Grocery/Reddit-S）：per-seed 磁盘 split 文件
  `MAGB_split/<Ds>_nc_seed<S>_train0.6_val0.2.pt` 12/12 齐全（seed 42/43/44），
  加载自磁盘、benchmark 期间不会重新生成 → 与 baseline 完全同 split；
- ele-fashion：官方 `split.pt`（所有 seed 共用，seed 差异仅来自初始化——对所有
  模型公平）。

## 4. biaxis_final 冻结结构核实

- `configs/model/biaxis_final.yaml`：p2.mode=null_softmax、deterministic=false、
  p3.operator_mode=full_interaction、无实验 knobs（测试
  `test_biaxis_final_config_frozen_structure` /
  `test_biaxis_final_config_has_no_unused_knobs` 守护）；
- 模型结构 = M1(C/Pt/Pv) + M2(K=4 topology relations) + NullSoftmax Γ(ε=0.2) +
  T_fk = W0+A_f+B_k+C_fk（full cell-conditioned）；213 tests 全绿。

## 5. 审计结论

| 项目 | 判定 |
|---|---|
| task 协议 120/120 一致 | **PASS** |
| model preset 与当前 yaml 一致 | **PASS** |
| results.json 完整性 | **PASS**（120/120） |
| split seed-aligned | **PASS** |
| biaxis_final 冻结结构 | **PASS** |
| 可复用 baseline runs | **120/120 全部复用** |
| 必须重跑 | **无** |

**Fairness 风险披露（论文方法段需记录）**：
1. baselines 保留各自官方/model preset lr/wd（上表）；biaxis_final 用 P3 冻结
   协议 lr=1e-3/wd=1e-4 —— 均无 dataset-specific 调参；
2. ele-fashion 全 seed 共用官方 split（所有模型一致）；
3. baseline runs 未保存 ckpt/params 日志 → 主表 params 由 model config 计算
   （biaxis_final 实测 1.40M，另记）。

## 6. LGMRec 新增 baseline（2026-09-03，用户指示）

- 移植自 `OpenMAG/src/model/models.py`（LGMRec + HGNNLayer）与
  `OpenMAG/configs/model/lgmrec.yaml`（hidden 128 / 3×LGConv / hyper_num 64 /
  alpha 0.1 / dropout 0.2）；lr/wd preset 5e-3/1e-5（OpenMAG NC task 默认）；
- **协议统一（用户决定 2026-09-03）**：**不采用** OpenMAG 的 InfoNCE aux
  （λ_v=λ_t=0.5）与其 vision/text heads/decoders —— LGMRec 与其他 baseline
  完全一致地跑 plain-CE 统一协议（aux_loss≡0）；OpenMAG 的
  label_smoothing=0.1 也未采用；eval 用 softmax(score/τ) 代替 gumbel 噪声；
  LightGCN 自环在模型内添加（OpenMAG self_loop=True 语义）；
- 第一轮（含 InfoNCE aux）的 15 runs 已归档至
  `outputs/baseline_nc/_lgmrec_infonce_archive/`（不计入主表）；当前主表使用
  纯 CE 版本的 15/15 runs，tests 6/6。
