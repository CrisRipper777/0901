# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MAG_baseline is a PyTorch/PyG/Hydra framework for benchmarking graph neural networks on **multimodal attributed graphs** (nodes with both image and text features). Two tasks: **Node Classification (NC)** and **Link Prediction (LP)**. Features are frozen — no large encoder training.

## Commands

**Run from project root** (`MAG_baseline/`). Environment: `conda activate yhf_env`.

```bash
# Single experiment
python -m src.main dataset=Movies task=nc model=mlp num_runs=3
python -m src.main dataset=sports-copurchase task=lp model=sage num_runs=3

# Smoke test (fast sanity check)
python -m src.main dataset=Movies task=nc model=mlp num_runs=1 task.epochs=1 task.max_train_batches=2 device=cpu

# Tests
pytest tests/

# Batch runs
python scripts/run_all_nc.py    # 6 NC datasets x 3 models
python scripts/run_all_lp.py    # 7 LP datasets x 3 models

# Generate MAGB splits
python scripts/make_magb_splits.py
```

## Architecture

**Entry point:** `src/main.py` (Hydra `@hydra.main`) → loads data → dispatches to `run_nc()` or `run_lp()` → saves `results.json`.

**Data layer** (`src/data/`):
- `loaders.py`: Two loaders — `_load_magb()` (DGL graph + separate `.npy` features) and `_load_mmgraph()` (joint CLIP `.pt` features). Both return `MAGData` dataclass.
- `splits.py`: Auto-generates splits for MAGB datasets; MM-Graph uses shipped split files.
- `graph_utils.py`: Edge index manipulation (canonicalize, preprocess).

**Models** (`src/models/`):
- `factory.py`: Dynamic import by name (`mlp`, `gcn`, `sage`).
- All encoders implement `forward()` → `(z, None, None, aux_loss, {})` and `inference()` for full-graph eval.
- `predictor.py`: `LinkPredictor` MLP — scores (src, dst) via element-wise product.
- GCN adds self-loops internally via `gcn_norm`; others don't. Dataset configs set `add_self_loops: false`.

**Task runners** (`src/tasks/`):
- `nc.py`: Model + linear classifier. Metrics: Accuracy, Macro-F1. Early stopping on val accuracy.
- `lp.py`: Model + `LinkPredictor`. Metrics: MRR, Hits@1/3/10. Early stopping on val MRR. Filtered negative sampling (negatives excluded from all positive edges). Positive label edges excluded from message-passing graph.
- Both run `num_runs` independent runs with seeds `seed + run_id`, select best by val metric, evaluate on test.

## Configuration (Hydra)

Root: `configs/config.yaml` with config groups `dataset/`, `task/`, `model/`. Key overrides:
- `num_runs`, `seed`, `device`
- `task.epochs`, `task.lr`, `task.batch_size`, `task.patience`
- `model.hidden_dim`, `model.num_layers`, `model.dropout`
- Paths: `paths.data_root`, `paths.split_root`

Output: `outputs/YYYY-MM-DD/HH-MM-SS/` containing `results.json`, `main.log`, `.hydra/` config snapshot.

## Datasets

- **MAGB** (4): Movies, Toys, Grocery, Reddit-S — both NC and LP. DGL `.pt` graph + `.npy` features. Splits auto-generated under `../data/MAGB_split/`.
- **MM-Graph** (5): sports-copurchase, cloth-copurchase, books-lp (LP); ele-fashion, books-nc (NC). Joint CLIP features `.pt`, official splits.

Data root: `/hdd1/DataInHere/YHF/data` (set via `paths.data_root` in config).

## Key Dependencies

`torch==2.4.0+cu121`, `torch-geometric==2.7.0`, `dgl==2.4.0+cu121`, `hydra-core==1.3.2`, `numpy==2.4.3`, `scikit-learn`
