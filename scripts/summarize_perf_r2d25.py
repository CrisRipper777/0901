"""R2-Design-2.5 summarizer
(docs/BiAxis_R2_Design_2_5_Structured_Capacity_Utilization_Audit.md).

Stages:
    audit         : D2.5-0 infrastructure audit report (R2D25_AUDIT.md)
    landscape     : D2.5-A alpha landscape CSVs + R2D25_LANDSCAPE_REPORT.md
    transmission  : D2.5-B CSVs + R2D25_TRANSMISSION_REPORT.md
    capacity      : D2.5-C CSVs + R2D25_CAPACITY_REPORT.md
    optimization  : D2.5-D CSVs + R2D25_OPTIMIZATION_REPORT.md
    final         : R2D25_MASTER_TABLE.csv + R2D25_HYPOTHESIS_LEDGER.csv
                    + R2D25_FINAL_DIAGNOSIS.md (17 questions + verdict)

Usage:
    python scripts/summarize_perf_r2d25.py --stage audit
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d25_utils import (  # noqa: E402
    ALPHA_GRAD_VALUES,
    ALPHA_PT_VALUES,
    CAPACITY_MODES,
    DATASETS,
    GUARD_DATASETS,
    R2D25_ROOT,
    SEEDS,
    TARGET_DATASETS,
)

AUDIT_ROOT = R2D25_ROOT / "audit"
LANDSCAPE_ROOT = R2D25_ROOT / "landscape"
TRANSMISSION_ROOT = R2D25_ROOT / "transmission"
CAPACITY_ROOT = R2D25_ROOT / "capacity"
OPTIMIZATION_ROOT = R2D25_ROOT / "optimization"
SUMMARY_ROOT = R2D25_ROOT / "summary"

# ---------------------------------------------------------------------------
# Audit stage (Prompt 1)
# ---------------------------------------------------------------------------


def _collect_audit_facts() -> dict:
    """Live checks of the D2.5 infrastructure (used by --stage audit)."""
    import subprocess

    facts: dict = {"modes": {}, "files": {}}
    # per-mode model instantiation smoke (CPU, tiny cfg)
    import torch
    from omegaconf import OmegaConf

    from src.models.biaxis_r2_capacity import EXPERT_KEYS, MODES, Model

    d, h = 16, 32
    for mode in MODES:
        cfg = OmegaConf.create({
            "model": {
                "name": "biaxis_r2_capacity", "capacity_mode": mode,
                "hidden_dim": h, "factor_dim": d, "dropout": 0.2,
                "activation": "gelu", "norm": "layernorm",
                "lambda_common": 0.02, "lambda_orth": 0.01, "lambda_recon": 0.3,
                "orth_fallback_batch": 16, "full_graph_training": True,
                "edge_chunk_size": None,
                "deep_supervision": {"enabled": False, "lambda": 0.1},
                "path_dropout_p": 0.0,
                "semantic_refiner": {"enabled": False},
                "functional_transfer": {"enabled": False},
            },
        })
        info = {"input_dim": 20, "num_nodes": 30, "num_classes": 5,
                "text_dim": 9, "visual_dim": 11}
        m = Model(cfg, info)
        facts["modes"][mode] = {
            "params": int(m.parameter_count),
            "expert_keys": list(EXPERT_KEYS[mode]),
            "has_h2_path": mode in ("early_mix", "sep_sum", "sep_concat", "inception_012"),
        }
    # reference C2/C4/C5 parameter parity on the tiny cfg
    facts["param_match"] = {
        "sep_concat": facts["modes"]["sep_concat"]["params"],
        "cap_h1_dup": facts["modes"]["cap_h1_dup"]["params"],
        "wide_b0": facts["modes"]["wide_b0"]["params"],
    }
    # pytest count for the new test files
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_biaxis_r2_capacity.py",
         "tests/test_perf_r2d25_utils.py", "-q", "--no-header", "--tb=no"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    facts["pytest"] = (proc.stdout + proc.stderr).strip().splitlines()
    return facts


def _write_audit_report(facts: dict) -> Path:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    report = AUDIT_ROOT / "R2D25_AUDIT.md"
    lines = [
        "# R2-Design-2.5 — D2.5-0 Audit & Infrastructure",
        "",
        "Protocol: seeds 42/43/44; Val only; No Test. No lightweight constraint:",
        "every added block has a defined function and a parameter-matched control.",
        "",
        "## 1. Frozen-fact review (current code, verified by reading)",
        "",
        "- **B0 graph path** (`src/models/biaxis_r2.py`): F* -> 1-hop",
        "  `neighbor_mean` -> per-factor source transform `V_f`",
        "  (Linear(d,d), bias=False) -> per-factor LayerNorm (bias=False, so",
        "  LN(0)=0 keeps isolated nodes exact) -> `rho=sigma(raw)` residual",
        "  (init 0.5) -> P0 one-layer fusion",
        "  `Linear(3d,h) -> LN -> GELU -> Dropout`.",
        "- **M1 EarlyMix** (`biaxis_r2_scale.py`): `Hmix = H1 + alpha_f (H2-H1)`",
        "  with direct scalars (init 0). B0 checkpoints load with strict key",
        "  verification (admissible missing = mixer.alpha).",
        "- **PRODDIFF source-mean compression** (`biaxis_r2d15_adapters.py`):",
        "  9 cells `D^{a->b} = MLP([F_b*N_a, |F_b-N_a|, types])` are compressed",
        "  into 3 target corrections `Delta^b = (1/3) sum_a D^{a->b}` (source",
        "  mean), then `Fhat = F_out + Delta` -> fusion. This is the",
        "  premature-compression suspect: 9 independent cells collapse to a",
        "  3-way mean BEFORE any target-side readout.",
        "- **D2.0.5 frozen facts**: fixed-alpha response curve peaked at",
        "  alpha_Pt in [0.5, 0.75] (mean +0.53pp Val on M/T/G), while SGD",
        "  (M1) only reaches alpha ~ 0.04-0.07 and all trainable schedules",
        "  max out at +0.126pp (optimization realization gap).",
        "",
        "## 2. New infrastructure",
        "",
        "### Models",
        "- `src/models/biaxis_r2_capacity.py` +",
        "  `src/models/biaxis_r2_capacity_components.py`; one class, seven",
        "  modes (config `configs/model/biaxis_r2_capacity.yaml`,",
        "  `model.capacity_mode`).",
        "- `extract_capacity_states`: H0/H1/H2, hop expert outputs, before/after",
        "  LN (`msg_pre_ln`/`msg_post_ln`), before/after residual",
        "  (`pre_residual`/`post_residual`), pre/post fusion, per-expert",
        "  ablation (`forward(..., off_hops=...)`, mode-validated).",
        "- D2.5-D hooks (default OFF): `deep_supervision` aux heads on expert",
        "  outputs (inference-free) and `path_dropout_h1` (train-only).",
        "",
        "### Scripts",
        "- `perf_r2d25_landscape.py` — D2.5-A; frozen B0 + fixed alpha_Pt,",
        "  per-(dataset, seed) shared classifier init; linear V-precompute",
        "  pipeline; diagnostic-only dTrainCE/dValCE vs alpha.",
        "- `perf_r2d25_transmission.py` — D2.5-B; S0-S4 stage probes",
        "  (StandardScaler + Ridge(1.0), TRAIN fit / VAL eval) + PRODDIFF",
        "  9-cells -> source-mean -> factor-add -> fusion retrace (M/T/G).",
        "- `perf_r2d25_capacity_train.py` — D2.5-C; unified schedule",
        "  (P0 frozen 1-20 -> lr 1e-4; graph lr 1e-3; AdamW wd 1e-4;",
        "  warmup10+cosine; 300ep/patience30/best ValAcc); ablation + expert",
        "  diagnostics at best checkpoint.",
        "- `perf_r2d25_optimization.py` — D2.5-D; single-factor interventions",
        "  (expert LR / deep supervision / path dropout), per-expert grad",
        "  norm, update ratio, output norm, classifier sensitivity.",
        "- `summarize_perf_r2d25.py` — this file.",
        "",
        "## 3. Verification (live)",
        "",
    ]
    modes = facts["modes"]
    lines.append("| mode | params (tiny cfg) | expert keys | H2 path |")
    lines.append("|---|---|---|---|")
    for mode in CAPACITY_MODES:
        m = modes[mode]
        lines.append(f"| {mode} | {m['params']} | {','.join(m['expert_keys']) or '-'} "
                     f"| {'yes' if m['has_h2_path'] else 'no'} |")
    lines.append("")
    lines.append("Parameter parity (tiny cfg): "
                 f"sep_concat={facts['param_match']['sep_concat']}, "
                 f"cap_h1_dup={facts['param_match']['cap_h1_dup']}, "
                 f"wide_b0={facts['param_match']['wide_b0']}.")
    lines.append("")
    lines.append("### Tests")
    lines.append("")
    lines.append("```")
    lines += facts["pytest"][-6:]
    lines.append("```")
    lines.append("")
    lines.append("## 4. Pre-registered interpretation matrix")
    lines.append("")
    lines.append("1. C2/C3 > A0 and > WIDE_B0/CAP_H1_DUP -> premature compression")
    lines.append("   was a real bottleneck -> Ownership-Aware Multi-Scale Expert Fusion.")
    lines.append("2. WIDE_B0 ~= C2/C3 and both improve -> generic backbone capacity ->")
    lines.append("   build a stronger backbone first.")
    lines.append("3. Representation utility remains, main CE underuses it, deep")
    lines.append("   supervision helps -> Gradient Starvation SUPPORTED.")
    lines.append("4. All fail -> Semantic-Ownership-Aware Neighbor Utility Learning")
    lines.append("   (learn which neighbor is useful for which factor BEFORE")
    lines.append("   aggregation).")
    lines.append("")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Later stages (implemented with their prompts)
# ---------------------------------------------------------------------------


def _write_landscape_report() -> Path:
    rows = []
    for ds in DATASETS:
        for seed in SEEDS:
            p = LANDSCAPE_ROOT / ds / f"seed_{seed}" / "summary.json"
            if not p.exists():
                continue
            data = json.loads(p.read_text())
            rows.extend(data["rows"])
    if not rows:
        raise RuntimeError("no landscape summaries found — run perf_r2d25_landscape.py first")
    fieldnames = ["dataset", "seed", "alpha_pt", "best_val_acc", "best_val_macro_f1",
                  "best_epoch", "best_train_ce", "best_train_acc", "best_val_ce",
                  "d_train_ce", "d_val_ce"]
    csv_path = LANDSCAPE_ROOT / "alpha_landscape.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    grads_path = LANDSCAPE_ROOT / "alpha_gradients.csv"
    with grads_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "seed", "alpha_pt", "d_train_ce", "d_val_ce"],
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            if r.get("alpha_pt") in ALPHA_GRAD_VALUES:
                writer.writerow(r)
    # ---- verdict computation (pre-registered in the plan) -----------------
    report = LANDSCAPE_ROOT / "R2D25_LANDSCAPE_REPORT.md"
    lines = ["# R2-D2.5-A — alpha_Pt formal objective landscape", ""]
    per_ds = {}
    for ds in DATASETS:
        per_ds[ds] = {"val": {}, "train_ce": {}, "grads": {}}
        for a in ALPHA_PT_VALUES:
            per_ds[ds]["val"][a] = []
            per_ds[ds]["train_ce"][a] = []
        for r in rows:
            if r["dataset"] != ds:
                continue
            a = r["alpha_pt"]
            per_ds[ds]["val"][a].append(r["best_val_acc"])
            per_ds[ds]["train_ce"][a].append(r["best_train_ce"])
    import statistics

    verdicts = {}
    for ds in DATASETS:
        means = {a: statistics.fmean(per_ds[ds]["val"][a]) for a in ALPHA_PT_VALUES}
        base = means[0.0]
        peak = max(ALPHA_PT_VALUES, key=lambda a: means[a])
        gain = means[peak] - base
        n_pos = sum(1 for s in SEEDS if
                    max(per_ds[ds]["val"][a][SEEDS.index(s)] for a in ALPHA_PT_VALUES) >
                    per_ds[ds]["val"][0.0][SEEDS.index(s)] + 1e-9)
        train_ces = {a: statistics.fmean(per_ds[ds]["train_ce"][a]) for a in ALPHA_PT_VALUES}
        train_favors_large = train_ces[1.0] < train_ces[0.0]
        val_favors_large = peak in (0.5, 0.75, 1.0)
        if val_favors_large and train_favors_large and gain >= 0.15 and n_pos >= 2:
            verdicts[ds] = "Optimization Accessibility Failure"
        elif val_favors_large and not train_favors_large and gain >= 0.15 and n_pos >= 2:
            verdicts[ds] = "Objective-Generalization Mismatch"
        elif gain < 0.15 or n_pos < 2:
            verdicts[ds] = "3-seed headroom unstable"
        else:
            verdicts[ds] = "mixed"
        lines.append(
            f"- **{ds}**: peak alpha={peak:g} (gain +{gain:.3f}pp, {n_pos}/3 seeds "
            f"positive); train CE at 0/1.0 = {train_ces[0.0]:.4f}/{train_ces[1.0]:.4f} "
            f"-> **{verdicts[ds]}**"
        )
    lines.append("")
    lines.append("| dataset | alpha | val acc (3-seed mean) | train CE (3-seed mean) |")
    lines.append("|---|---|---|---|")
    for ds in DATASETS:
        for a in ALPHA_PT_VALUES:
            lines.append(
                f"| {ds} | {a:g} | {statistics.fmean(per_ds[ds]['val'][a]):.5f} | "
                f"{statistics.fmean(per_ds[ds]['train_ce'][a]):.4f} |"
            )
    lines.append("")
    lines.append("Aggregate verdict: see per-dataset rows. A dataset-level")
    lines.append("**Optimization Accessibility Failure** means the landscape has")
    lines.append("value that SGD cannot reach; **Objective-Generalization")
    lines.append("Mismatch** means train and val optima disagree.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-Design-2.5 summarizer")
    parser.add_argument("--stage", default="audit",
                        choices=["audit", "landscape", "transmission", "capacity",
                                 "optimization", "final"])
    args = parser.parse_args()
    if args.stage == "audit":
        report = _write_audit_report(_collect_audit_facts())
        print(f"[audit] wrote {report}")
        return
    if args.stage == "landscape":
        print(f"[landscape] wrote {_write_landscape_report()}")
        return
    raise NotImplementedError(
        f"stage={args.stage} is implemented with its own prompt "
        "(Prompt 3: transmission / Prompt 4: capacity / Prompt 5: optimization / "
        "Prompt 7: final)"
    )


if __name__ == "__main__":
    main()
