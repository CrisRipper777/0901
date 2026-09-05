"""R2-Design-2.7 summarizer
(docs/BiAxis_R2_Design_2_7_PreAggregation_Neighbor_Utility_Audit.md).

Stages:
    audit       : D2.7-0 infrastructure audit (R2D27_AUDIT.md)
    matrix      : D2.7-A CSVs + R2D27_MATRIX_REPORT.md
    edge_audit  : D2.7-B CSVs + R2D27_EDGE_AUDIT_REPORT.md
    prepost     : D2.7-C CSV + R2D27_PREPOST_REPORT.md
    ownership   : D2.7-D CSV + R2D27_OWNERSHIP_REPORT.md
    transfer    : D2.7-E CSVs + R2D27_TRANSFER_REPORT.md
    final       : master table + hypothesis ledger + diagnosis

Usage:
    python scripts/summarize_perf_r2d27.py --stage audit
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d27_utils import (  # noqa: E402
    DATASETS,
    GUARD_DATASETS,
    R2D27_ROOT,
    SEEDS,
    TARGET_DATASETS,
    VARIANTS,
)

AUDIT_ROOT = R2D27_ROOT / "audit"
MATRIX_ROOT = R2D27_ROOT / "matrix"
EDGE_AUDIT_ROOT = R2D27_ROOT / "edge_audit"
PREPOST_ROOT = R2D27_ROOT / "prepost"
OWNERSHIP_ROOT = R2D27_ROOT / "ownership"
TRANSFER_ROOT = R2D27_ROOT / "transfer"
NOISE_ROOT = R2D27_ROOT / "noise_optional"
SUMMARY_ROOT = R2D27_ROOT / "summary"


def _collect(root: Path) -> list[dict]:
    rows = []
    if not root.exists():
        return rows
    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name == "head_init":
            continue
        for v_dir in sorted(ds_dir.iterdir()):
            if not v_dir.is_dir():
                continue
            for s_dir in sorted(v_dir.iterdir()):
                sp = s_dir / "summary.json"
                if not sp.exists():
                    continue
                d = json.loads(sp.read_text())
                rows.append({
                    "dataset": ds_dir.name,
                    "variant": d.get("variant", v_dir.name),
                    "seed": int(s_dir.name.split("_")[-1]),
                    "val_acc": d["best_val_acc"],
                    "val_macro_f1": d["best_val_macro_f1"],
                    "best_epoch": d.get("best_epoch"),
                    "stop_epoch": d.get("stop_epoch"),
                    "side_params": d.get("side_params"),
                    "out_dim": d.get("out_dim"),
                })
    return rows


def _load_a0_formal() -> dict[tuple[str, int], dict]:
    from src.analysis.perf_r2_utils import load_a0_reference

    acc = load_a0_reference()
    out = {(ds, s): {"val_acc": v, "val_macro_f1": None} for (ds, s), v in acc.items()}
    for ds in DATASETS:
        for seed in SEEDS:
            p = PROJECT_ROOT / "outputs" / "perf_r1" / "baseline" / ds / "A0" / f"seed_{seed}" / "summary.json"
            if p.exists():
                d = json.loads(p.read_text())
                if d.get("best_val_macro_f1") is not None:
                    out[(ds, seed)]["val_macro_f1"] = float(d["best_val_macro_f1"]) / 100.0
    return out


A0_FORMAL = _load_a0_formal()


def _by_key(rows, variant):
    return {(r["dataset"], r["seed"]): r for r in rows if r["variant"] == variant}


def _paired(cand_rows, base_rows, metric, ds_list, seeds):
    per_ds = []
    for ds in ds_list:
        deltas = []
        for s in seeds:
            c = cand_rows.get((ds, s))
            b = base_rows.get((ds, s))
            if c and b:
                deltas.append(100.0 * (c[metric] - b[metric]))
        if deltas:
            per_ds.append((ds, statistics.fmean(deltas),
                           sum(1 for d in deltas if d > 0), len(deltas)))
    mean = statistics.fmean([m for _, m, _, _ in per_ds]) if per_ds else None
    n_pos = sum(1 for _, m, _, _ in per_ds if m > 0)
    return {"per_ds": per_ds, "mean": mean, "n_pos": n_pos}


# ---------------------------------------------------------------------------
# D2.7-A matrix
# ---------------------------------------------------------------------------


def _write_matrix_report(rows: list[dict]) -> Path:
    MATRIX_ROOT.mkdir(parents=True, exist_ok=True)
    with (MATRIX_ROOT / "matrix_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "seed", "val_acc",
                                               "val_macro_f1", "best_epoch", "stop_epoch",
                                               "side_params", "out_dim"],
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    pair = _by_key(rows, "PAIR_EDGE")
    a0m = _by_key(rows, "A0_BASE")
    controls = {}
    for name in ("UNIFORM", "TARGET_NULL_ONLY", "GENERIC_EDGE", "DIAG_EDGE",
                 "SEMANTIC_SIM"):
        controls[name] = _paired(pair, _by_key(rows, name), "val_acc",
                                 TARGET_DATASETS, SEEDS)
        controls[name]["f1"] = _paired(pair, _by_key(rows, name), "val_macro_f1",
                                       TARGET_DATASETS, SEEDS)
    a0_delta_acc = _paired(pair, a0m, "val_acc", TARGET_DATASETS, SEEDS)
    a0_delta_f1 = _paired(pair, a0m, "val_macro_f1", TARGET_DATASETS, SEEDS)
    with (MATRIX_ROOT / "matrix_controls.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["control", "delta_acc_mtg", "delta_f1_mtg",
                                               "n_pos_ds"], extrasaction="ignore")
        writer.writeheader()
        for name, c in controls.items():
            writer.writerow({"control": name, "delta_acc_mtg": c["mean"],
                             "delta_f1_mtg": c["f1"]["mean"],
                             "n_pos_ds": c["n_pos"]})
    with (MATRIX_ROOT / "matrix_resources.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "seed", "side_params",
                                               "out_dim", "runtime_sec", "peak_allocated_mb"],
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # guards for any candidate >= A0_MATCHED + 0.20pp on M/T/G acc
    guard_rows = []
    for v in ("PAIR_EDGE",):
        cand = _by_key(rows, v)
        for g in GUARD_DATASETS:
            deltas = [100.0 * (cand[(g, s)]["val_acc"] - a0m[(g, s)]["val_acc"])
                      for s in SEEDS if (g, s) in cand and (g, s) in a0m]
            if deltas:
                guard_rows.append((g, statistics.fmean(deltas)))

    selection_go = (controls["TARGET_NULL_ONLY"]["mean"] is not None
                    and controls["TARGET_NULL_ONLY"]["mean"] >= 0.30
                    and controls["TARGET_NULL_ONLY"]["f1"]["mean"] is not None
                    and controls["TARGET_NULL_ONLY"]["f1"]["mean"] >= 0.20
                    and controls["TARGET_NULL_ONLY"]["n_pos"] >= 2)
    incremental_go = (a0_delta_acc["mean"] is not None and a0_delta_acc["mean"] >= 0.30
                      and a0_delta_f1["mean"] is not None and a0_delta_f1["mean"] >= 0.20
                      and a0_delta_acc["n_pos"] >= 2)
    ownership_prelim = False
    if controls["GENERIC_EDGE"]["mean"] is not None:
        ownership_prelim = (controls["GENERIC_EDGE"]["mean"] >= 0.20
                            and controls["GENERIC_EDGE"]["f1"]["mean"] is not None
                            and controls["GENERIC_EDGE"]["f1"]["mean"] >= 0.0) or (
            controls["GENERIC_EDGE"]["f1"]["mean"] is not None
            and controls["GENERIC_EDGE"]["f1"]["mean"] >= 0.30
            and controls["GENERIC_EDGE"]["mean"] >= 0.0)

    report = MATRIX_ROOT / "R2D27_MATRIX_REPORT.md"
    lines = ["# R2-D2.7-A — Pre-aggregation neighbor-utility matrix", "",
             "PAIR_EDGE paired deltas vs controls (M/T/G, 3-seed means, pp):", ""]
    for name in ("UNIFORM", "TARGET_NULL_ONLY", "GENERIC_EDGE", "DIAG_EDGE",
                 "SEMANTIC_SIM"):
        c = controls[name]
        lines.append(
            f"- vs {name}: Acc {c['mean'] if c['mean'] is None else f'{c['mean']:+.3f}'} "
            f"/ F1 {c['f1']['mean'] if c['f1']['mean'] is None else f'{c['f1']['mean']:+.3f}'} "
            f"({c['n_pos']}/3 datasets positive)")
    lines.append(f"- vs A0_MATCHED: Acc {a0_delta_acc['mean'] if a0_delta_acc['mean'] is None else f'{a0_delta_acc['mean']:+.3f}'} "
                 f"/ F1 {a0_delta_f1['mean'] if a0_delta_f1['mean'] is None else f'{a0_delta_f1['mean']:+.3f}'} "
                 f"({a0_delta_acc['n_pos']}/3)")
    lines.append("")
    lines.append(f"- **SELECTION GO: {'PASS' if selection_go else 'not met'}** "
                 "(PAIR - TARGET_NULL_ONLY >= +0.30pp Acc, +0.20pp F1)")
    lines.append(f"- **A0 INCREMENTAL GO: {'PASS' if incremental_go else 'not met'}** "
                 "(PAIR - A0_MATCHED >= +0.30pp Acc, +0.20pp F1)")
    lines.append(f"- **OWNERSHIP preliminary: "
                 f"{'SUPPORTED' if ownership_prelim else 'not met'}** "
                 "(PAIR - GENERIC_EDGE >= +0.20pp Acc with F1>=0, or F1>=+0.30 with Acc>=0)")
    if guard_rows:
        lines.append("")
        lines.append("Guards (vs A0_MATCHED, Acc pp, threshold -0.20pp):")
        for g, m in guard_rows:
            lines.append(f"- {g}: {m:+.3f}pp")
    lines.append("")
    lines.append("| dataset | variant | acc (3-seed mean) | F1 (3-seed mean) |")
    lines.append("|---|---|---|---|")
    for v in ("A0_BASE", "UNIFORM", "TARGET_NULL_ONLY", "GENERIC_EDGE", "DIAG_EDGE",
              "PAIR_EDGE", "SEMANTIC_SIM"):
        for ds in DATASETS:
            sel = [r for r in rows if r["variant"] == v and r["dataset"] == ds]
            if sel:
                lines.append(
                    f"| {ds} | {v} | {statistics.fmean(r['val_acc'] for r in sel):.5f} | "
                    f"{statistics.fmean(r['val_macro_f1'] for r in sel):.5f} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def _write_audit_report() -> Path:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    report = AUDIT_ROOT / "R2D27_AUDIT.md"
    lines = [
        "# R2-Design-2.7 — D2.7-0 Audit & Infrastructure", "",
        "Protocol: seeds 42/43/44; Val only; No Test. A0 (R1-baseline, disclosed)",
        "is the only primary parent; the utility branch may only ADD.", "",
        "## 1. Collision contract (plan §1)", "",
        "- No RoleMAG-style predefined roles (heterophilous/complementary/"
        "  shared channels): continuous task-conditioned utility u_ji^{a->b}.",
        "- No TMTE-style topology evolution: observed graph as support only;",
        "  edge_index is only READ (structural test).",
        "- No CoMAG-style semantic-consistency scorer as the mechanism;",
        "  SEMANTIC_SIM is a control baseline only.", "",
        "## 2. Core formulation (plan §2-§5)", "",
        "- s_ji^{a->b} = psi([F_i^b, F_j^a, F_i^b*F_j^a, |F_i^b-F_j^a|, e_a, e_b])",
        "  (shared psi, t=16, Linear(4d+2t,2d)->LN->GELU->Drop->Linear(2d,d)->",
        "  GELU->Linear(d,1)).",
        "- Null-augmented softmax with tau=1.0 fixed; no top-k in training.",
        "- Selection-only payload U_a (Linear(d,d,bias=False)), m_i^b =",
        "  (1/3) sum_a m_i^{a->b}; side = [z_base | m_C | m_Pt | m_Pv], no",
        "  projection back to h.",
        "- Edge chunking + stable segment softmax: [E,3,3,d] never",
        "  materialized; chunked == unchunked (tested).", "",
        "## 3. Variants (plan §11-§17)", "",
        "A0_BASE / UNIFORM / TARGET_NULL_ONLY / GENERIC_EDGE (scorer width",
        "solved to match PAIR params +/-5%) / DIAG_EDGE / PAIR_EDGE /",
        "SEMANTIC_SIM; plus POST_PAIR / SOURCE_FACTOR_ONLY /",
        "TARGET_FACTOR_ONLY / PAIR_TRANSFORM_UNIFORM / PAIR_TRANSFORM_PRE",
        "for later stages.", "",
        "## 4. Causal machinery (plan §27-§28)", "",
        "remove-top/random/bottom 10/25/50% (per-pair score masks),",
        "keep-top 25/50%, within-target shuffle (two stable sorts, per-target",
        "histograms preserved — tested), source-node shuffle, factor-id",
        "shuffle, noise-10/25% edge injection hooks, side_off (bitwise",
        "z_base). All eval-time; weights never modified.", "",
        "## 5. Verification", "",
        "18 tests in tests/test_biaxis_r2_neighbor_utility.py: A0 untouched",
        "in frozen training; UNIFORM == neighbor_mean of payloads;",
        "TARGET_NULL_ONLY null-0 => weight 1/(deg+1) everywhere; 9 pair",
        "scores; GENERIC single ranking; DIAG off-diagonal zero;",
        "null+neighbor sums to 1; isolated all-null finite; shuffle",
        "histogram preservation; POST aggregates before scoring; no-test",
        "guarded loop; C/Pt/Pv order; chunked==unchunked; collision",
        "guardrails; side_off bitwise == parent.",
        "",
        "GPU smoke (Movies s42, 5 ep): all 7 matrix variants train;",
        "PAIR_EDGE remove_top_10 (-0.69pp) > remove_random_10 (-0.33pp)",
        "at init-like weights — the causal ranking machinery is live.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-Design-2.7 summarizer")
    parser.add_argument("--stage", default="audit",
                        choices=["audit", "matrix", "edge_audit", "prepost",
                                 "ownership", "transfer", "final"])
    args = parser.parse_args()
    if args.stage == "audit":
        print(f"[audit] wrote {_write_audit_report()}")
        return
    if args.stage == "matrix":
        rows = _collect(MATRIX_ROOT)
        if not rows:
            raise RuntimeError("no matrix summaries — run perf_r2d27_matrix.py")
        print(f"[matrix] wrote {_write_matrix_report(rows)}")
        return
    raise NotImplementedError(
        f"stage={args.stage} implemented with its prompt "
        "(edge_audit / prepost / ownership / transfer / final)")


if __name__ == "__main__":
    main()
