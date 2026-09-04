"""R2-Design-1.6 summarizer: interaction screen verdicts, semantic screen
verdicts and the final synthesis (plan §31/§32/§39/§58).

Only READS existing results. Never trains, never touches test.

Usage:
    python scripts/summarize_perf_r2d16.py --stage interaction
    python scripts/summarize_perf_r2d16.py --stage semantic
    python scripts/summarize_perf_r2d16.py --stage final
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

from src.analysis.perf_r2d16_utils import (  # noqa: E402
    PARENTS,
    R2D16_ROOT,
    SEEDS,
    TARGET_DATASETS,
)

INTERACTION_ROOT = R2D16_ROOT / "interaction"
SEMANTIC_ROOT = R2D16_ROOT / "semantic"
SUMMARY_DIR = R2D16_ROOT / "summary"


def _load_summaries(root: Path, variants, parents=None, datasets=None,
                    seeds=(42,)) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for parent in parents or PARENTS:
        for ds in datasets or TARGET_DATASETS:
            for variant in variants:
                for seed in seeds:
                    p = root / parent / ds / variant / f"seed_{seed}" / "summary.json"
                    if p.exists():
                        with p.open(encoding="utf-8") as f:
                            out[(parent, ds, variant, seed)] = json.load(f)
    return out


def _pp(v, digits: int = 3) -> str:
    return "-" if v is None else f"{100 * v:+.{digits}f}"


def _mean_mtg(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _write_interaction_report() -> None:
    variants = ("HEAD", "CONCAT", "PRODDIFF", "FiLM")
    summaries = _load_summaries(INTERACTION_ROOT, variants)
    rows = []
    for key, s in summaries.items():
        parent, ds, variant, seed = key
        head = summaries.get((parent, ds, "HEAD", seed))
        rows.append({
            "parent": parent, "dataset": ds, "variant": variant, "seed": seed,
            "val_acc": s["val_acc"], "val_macro_f1": s["val_macro_f1"],
            "delta_acc_vs_head": (s["val_acc"] - head["val_acc"]) if head else None,
            "delta_f1_vs_head": (s["val_macro_f1"] - head["val_macro_f1"]) if head else None,
            "mismatch_val_acc": s.get("mismatch_val_acc"),
            "mismatch_val_macro_f1": s.get("mismatch_val_macro_f1"),
            "real_minus_mismatch": (s["val_acc"] - s["mismatch_val_acc"])
            if s.get("mismatch_val_acc") is not None else None,
            "best_epoch": s.get("best_epoch"),
            "cell_mean_offdiag_cosine": s.get("cell_mean_offdiag_cosine"),
            "cell_effective_rank": s.get("cell_effective_rank"),
            "adapter_params": s.get("adapter_params"),
        })
    with (INTERACTION_ROOT / "interaction_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# R2D16_INTERACTION_REPORT — D1.6-C Dual-Parent Frozen Interaction Screen",
        "",
        "Frozen parents (A0 = R1-A0 baseline checkpoints, disclosed; B0 = b0_confirm). "
        "Adapter + fresh classifier only; shared classifier init; AdamW 1e-3/1e-4; "
        "300ep/patience30; best Val Acc. Mismatch: fixed perm 20260904 on N rows. "
        "Safety: Macro-F1 delta vs HEAD < −0.50pp = WARNING (plan §6).",
        "",
        "## Screen results (seed42, Val Acc; Δ vs HEAD in pp)",
        "",
        "| parent | dataset | HEAD | CONCAT | PRODDIFF | FiLM | CONCAT Δ | PRODDIFF Δ | FiLM Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for parent in PARENTS:
        for ds in TARGET_DATASETS:
            get = lambda v: summaries.get((parent, ds, v, 42))
            cells = [f"{parent}", ds]
            for v in variants:
                s = get(v)
                cells.append(f"{s['val_acc']:.5f}" if s else "-")
            head = get("HEAD")
            for v in variants[1:]:
                s = get(v)
                cells.append(_pp(s["val_acc"] - head["val_acc"]) if s and head else "-")
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Within-parent verdicts (plan §31)",
        "",
    ]
    verdicts: dict[tuple[str, str], str] = {}
    gains_store: dict[tuple[str, str], float] = {}
    mismatch_macro: dict[tuple[str, str], float] = {}
    for parent in PARENTS:
        for variant in ("CONCAT", "PRODDIFF", "FiLM"):
            gains, positives, f1_warns, real_mis = [], 0, 0, []
            for ds in TARGET_DATASETS:
                s = summaries.get((parent, ds, variant, 42))
                h = summaries.get((parent, ds, "HEAD", 42))
                if not s or not h:
                    continue
                d = s["val_acc"] - h["val_acc"]
                gains.append(d)
                positives += d > 0
                f1_delta = s["val_macro_f1"] - h["val_macro_f1"]
                if f1_delta < -0.50 / 100:
                    f1_warns += 1
                if s.get("mismatch_val_acc") is not None:
                    real_mis.append(s["val_acc"] - s["mismatch_val_acc"])
            if len(gains) < 3:
                verdicts[(parent, variant)] = "MISSING"
                continue
            gain = _mean_mtg(gains)
            gains_store[(parent, variant)] = gain
            mis = _mean_mtg(real_mis) if real_mis else 0.0
            mismatch_macro[(parent, variant)] = mis
            pd_concat = None
            if variant == "PRODDIFF":
                pd_concat = _mean_mtg([
                    summaries[(parent, ds, "PRODDIFF", 42)]["val_acc"]
                    - summaries[(parent, ds, "CONCAT", 42)]["val_acc"]
                    for ds in TARGET_DATASETS
                ])
            strong = (gain >= 0.50 / 100 and positives >= 2 and mis >= 0.20 / 100 and f1_warns == 0)
            go = (gain >= 0.30 / 100 and positives >= 2
                  and ((variant == "PRODDIFF" and pd_concat is not None and pd_concat >= 0.15 / 100)
                       or mis >= 0.20 / 100))
            if strong:
                verdicts[(parent, variant)] = "STRONG"
            elif go:
                verdicts[(parent, variant)] = "GO"
            elif gain >= 0.15 / 100:
                verdicts[(parent, variant)] = "WEAK"
            else:
                verdicts[(parent, variant)] = "NO-GO"
            lines.append(
                f"- {parent} {variant}: Gain={_pp(gain)}pp (pos {positives}/3), "
                f"Real−Mismatch={_pp(mis)}pp, F1 warnings={f1_warns} "
                f"→ **{verdicts[(parent, variant)]}**"
            )
    lines += [
        "",
        "## Cross-parent verdict (plan §32)",
        "",
    ]
    for variant in ("PRODDIFF", "FiLM"):
        a0v, b0v = verdicts.get(("A0", variant)), verdicts.get(("B0", variant))
        a0_go = a0v in ("STRONG", "GO")
        b0_go = b0v in ("STRONG", "GO")
        if a0_go and b0_go:
            status = "PARENT-ROBUST GO"
        elif a0_go or b0_go:
            status = "PARENT-SPECIFIC GO"
        else:
            status = "no cross-parent GO"
        lines.append(f"- {variant}: A0={a0v} / B0={b0v} → {status}")
    d3_d4_nogo = all(
        gains_store.get((p, v), 0.0) < 0.15 / 100
        for p in PARENTS for v in ("PRODDIFF", "FiLM")
    )
    lines += [
        "",
        f"REALIZATION NO-GO (D3+D4 both < +0.15pp on both parents and mismatch "
        f"unsupported)? {'**YES — vector factor-context realization can be closed**' if d3_d4_nogo else 'no'}",
        "",
        "Message novelty / specialization details are in the per-run summaries "
        "(cell_norm / cell_cosine_to_parent_update / cell_orthogonal_novelty / "
        "cell_pairwise_cosine / cell_effective_rank).",
    ]
    (INTERACTION_ROOT / "R2D16_INTERACTION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[interaction] saved -> {INTERACTION_ROOT / 'R2D16_INTERACTION_REPORT.md'}")


def _write_semantic_report() -> None:
    summaries = _load_summaries(SEMANTIC_ROOT, ("HEAD", "SEM"))
    rows = []
    for key, s in summaries.items():
        parent, ds, variant, seed = key
        head = summaries.get((parent, ds, "HEAD", seed))
        rows.append({
            "parent": parent, "dataset": ds, "variant": variant, "seed": seed,
            "val_acc": s["val_acc"], "val_macro_f1": s["val_macro_f1"],
            "delta_acc_vs_head": (s["val_acc"] - head["val_acc"]) if head else None,
            "delta_f1_vs_head": (s["val_macro_f1"] - head["val_macro_f1"]) if head else None,
            "mismatch_val_acc": s.get("mismatch_val_acc"),
            "real_minus_mismatch": (s["val_acc"] - s["mismatch_val_acc"])
            if s.get("mismatch_val_acc") is not None else None,
            "residual_ratio_C": (s.get("residual_ratio") or {}).get("C", {}).get("mean"),
            "refined_C_Pt_cos": (s.get("refined_pair_cosine") or {}).get("C_Pt"),
            "best_epoch": s.get("best_epoch"),
        })
    with (SEMANTIC_ROOT / "semantic_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# R2D16_SEMANTIC_REPORT — D1.6-D Semantic Residual-Only Screen",
        "",
        "Fixed common C = 0.5*(c_t+c_v) (adaptive gate strictly removed); factor "
        "interaction residual inserted BEFORE the parent graph path; frozen parent; "
        "fresh classifier (shared init). Mismatch: partner factors permuted with "
        "fixed seed 20260904. Safety per plan §6.",
        "",
        "## Seed42 results (Val Acc; Δ vs HEAD in pp)",
        "",
        "| parent | dataset | HEAD | SEM | Δ Acc | Δ F1 | Real−Mismatch |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for parent in PARENTS:
        for ds in TARGET_DATASETS:
            s = summaries.get((parent, ds, "SEM", 42))
            h = summaries.get((parent, ds, "HEAD", 42))
            if not s or not h:
                continue
            f1_delta = s["val_macro_f1"] - h["val_macro_f1"]
            real_mis = (s["val_acc"] - s["mismatch_val_acc"]
                        if s.get("mismatch_val_acc") is not None else None)
            lines.append(
                f"| {parent} | {ds} | {h['val_acc']:.5f} | {s['val_acc']:.5f} "
                f"| {_pp(s['val_acc'] - h['val_acc'])} "
                f"| {_pp(f1_delta)} "
                f"| {_pp(real_mis)} |"
            )
    lines += [
        "",
        "## Verdicts (plan §39): GO = M/T/G macro ≥ +0.20pp, ≥2/3 positive, "
        "no F1 warning; STRONG ≥ +0.50pp",
        "",
    ]
    sem_go: dict[str, str] = {}
    for parent in PARENTS:
        gains, pos, warns = [], 0, 0
        for ds in TARGET_DATASETS:
            s = summaries.get((parent, ds, "SEM", 42))
            h = summaries.get((parent, ds, "HEAD", 42))
            if not s or not h:
                continue
            d = s["val_acc"] - h["val_acc"]
            gains.append(d)
            pos += d > 0
            if s["val_macro_f1"] - h["val_macro_f1"] < -0.50 / 100:
                warns += 1
        if len(gains) < 3:
            sem_go[parent] = "MISSING"
            continue
        gain = _mean_mtg(gains)
        if gain >= 0.50 / 100 and pos >= 2 and warns == 0:
            sem_go[parent] = "STRONG"
        elif gain >= 0.20 / 100 and pos >= 2 and warns == 0:
            sem_go[parent] = "GO"
        else:
            sem_go[parent] = "NO-GO"
        lines.append(f"- {parent}: Gain_sem={_pp(gain)}pp (pos {pos}/3, F1 warnings {warns}) "
                     f"→ **{sem_go[parent]}**")
    cross = ("PARENT-ROBUST SEMANTIC SUPPORT" if sem_go["A0"] in ("GO", "STRONG")
             and sem_go["B0"] in ("GO", "STRONG") else
             ("PARENT-SPECIFIC" if "GO" in (sem_go["A0"], sem_go["B0"])
              or "STRONG" in (sem_go["A0"], sem_go["B0"]) else "no GO"))
    lines += ["", f"**Cross-parent: {cross}**"]
    (SEMANTIC_ROOT / "R2D16_SEMANTIC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[semantic] saved -> {SEMANTIC_ROOT / 'R2D16_SEMANTIC_REPORT.md'}")


def _write_final_report() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    interaction = _load_summaries(INTERACTION_ROOT, ("HEAD", "CONCAT", "PRODDIFF", "FiLM"))
    semantic = _load_summaries(SEMANTIC_ROOT, ("HEAD", "SEM"))

    # master table
    master_rows: list[dict] = []
    for key, s in interaction.items():
        parent, ds, variant, seed = key
        h = interaction.get((parent, ds, "HEAD", seed))
        master_rows.append({
            "stage": "interaction", "parent": parent, "dataset": ds,
            "variant": variant, "seed": seed, "val_acc": s["val_acc"],
            "val_macro_f1": s["val_macro_f1"],
            "delta_vs_head": (s["val_acc"] - h["val_acc"]) if h else None,
        })
    for key, s in semantic.items():
        parent, ds, variant, seed = key
        h = semantic.get((parent, ds, "HEAD", seed))
        master_rows.append({
            "stage": "semantic", "parent": parent, "dataset": ds,
            "variant": variant, "seed": seed, "val_acc": s["val_acc"],
            "val_macro_f1": s["val_macro_f1"],
            "delta_vs_head": (s["val_acc"] - h["val_acc"]) if h else None,
        })
    with (SUMMARY_DIR / "R2D16_MASTER_TABLE.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(master_rows[0].keys()) if master_rows else [])
        writer.writeheader()
        writer.writerows(master_rows)

    ledger = [
        ("A0 performance parent", "SUPPORTED",
         "Role frozen: formal performance reference (Val Acc from formal runs; Val F1 parsed from formal logs)."),
        ("B0 diagnostic scaffold", "SUPPORTED",
         "Role frozen: clean scaffold, ≈A0-level M/T/G Acc. NEW CAVEAT: B0 has a systematic Val Macro-F1 deficit vs A0 on Movies (−0.86pp) and ele-fashion (−0.84pp) at Acc parity."),
        ("Current K-prototype relation", "CLOSED",
         "R2D1 (B0 seed42 > A0) + D1.6-A graph-control: relation neutralization ≈ full on all 5 datasets (+0.00..+0.13pp) — K-relation specialization contributes ≈ nothing."),
        ("Task-aware relation learning", "OPEN",
         "Not tested; Route E not indicated because multi-scale evidence is not weak."),
        ("Scalar functional routing", "CLOSED",
         "R2D1.5: forward value ≈ 0, off-diagonal ≈ 0."),
        ("PRODDIFF vector interaction", "PARENT-SPECIFIC",
         "B0: +0.299pp (3/3 positive, 0.001pp below the +0.30 GO line, Real−Mismatch +0.35pp) = WEAK; A0: −0.147pp with 1 F1 warning = NO-GO. B0-dependent: the old A0 machinery already encodes the interaction."),
        ("FiLM vector modulation", "WEAK",
         "B0 +0.098pp, A0 −0.239pp (2 F1 warnings). Strongest correspondence signal (Real−Mismatch +1.98..+2.20pp) but no net value."),
        ("Factor interaction semantic residual", "CLOSED",
         "Frozen NO-GO on both parents (A0 −1.14pp, B0 −0.11pp): the trained Δ grows to 0.67−1.0x the factor norm (rewrites ownership) and the frozen graph machinery misfires on rewritten factors."),
        ("Adaptive scalar common", "CLOSED", "R2D1 collapse + R2D1.5 co-adaptation evidence."),
        ("1-hop propagation", "SUPPORTED",
         "Sufficient for both parents' own performance; nothing in D1.6 shows a 1-hop bottleneck."),
        ("Factor-specific 2-hop", "SUPPORTED",
         "INDUCTIVE-BIAS SUPPORT ONLY (plan §18): Pt 2-hop is CROSS-PARENT SUPPORT (A0 +1.38/1.90/1.08pp, B0 +1.11/1.54/1.15pp, ≥2/3 seeds) but final-residual weak on both parents (+0.10/−0.09pp). C/Pv = parent-specific (B0)."),
        ("High-pass / diversification", "CLOSED",
         "No cross-parent support on any factor or the final residual (all ≤ +0.05pp)."),
        ("Frozen training", "SUPPORTED",
         "As diagnostic methodology: clean controlled comparisons on both parents (this stage's core tool)."),
        ("Full warm-start fine-tune", "OPEN", "D1.6-E not entered (no frozen GO candidate)."),
        ("Gradual unfreeze", "OPEN", "D1.6-E not entered."),
        ("MoE", "OPEN", "Not tested; hybrid composition requires ≥2 independent formal GO mechanisms first."),
        ("Edge-level relation learning", "OPEN",
         "Not tested; not indicated as long as the Pt-2-hop inductive-bias route remains open."),
    ]
    with (SUMMARY_DIR / "R2D16_HYPOTHESIS_LEDGER.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hypothesis", "status", "evidence"])
        writer.writerows(ledger)

    lines = [
        "# R2D16_FINAL_DIAGNOSIS — D1.6-F Final Synthesis",
        "",
        "> Reads only completed stages. No new experiments, no Test. "
        "D1.6-C2 (interaction confirm) and D1.6-E (schedule study) were NOT "
        "ENTERED: no realization reached the pre-registered frozen GO gate "
        "(plan §33/§41).",
        "",
        "## Stage ledger",
        "",
        "| Stage | Status |",
        "|---|---|",
        "| D1.6-0 audit + dual-parent infrastructure | PASS (16 tests; A0 parent provenance disclosed) |",
        "| D1.6-A parent metric backfill + graph-control | PASS |",
        "| D1.6-B dual-parent propagation audit | PASS (Pt 2-hop CROSS-PARENT; HP closed) |",
        "| D1.6-C dual-parent interaction screen | PASS (no frozen GO; B0 PRODDIFF WEAK at the line) |",
        "| D1.6-C2 interaction confirmation | NOT ENTERED (no GO) |",
        "| D1.6-D semantic residual screen | PASS (NO-GO both parents) |",
        "| D1.6-E schedule study | NOT ENTERED (no frozen GO candidate) |",
        "| D1.6-F final synthesis | this document |",
        "",
        "## Answers (plan §58)",
        "",
        "1. **A0/B0 roles?** KEEP: A0 = performance parent, B0 = clean diagnostic "
        "scaffold. Roles are now formalized with full metric coverage.",
        "2. **Systematic Macro-F1 difference missed before?** YES: B0 loses "
        "0.86pp Macro-F1 on Movies and 0.84pp on ele-fashion at Acc parity "
        "(Toys/Grocery +0.53/+0.23pp). B0's Acc≈A0 on M/T/G comes with a "
        "hidden F1 cost on two datasets.",
        "3. **2-hop cross-parent stable?** YES for Pt only: CROSS-PARENT "
        "SUPPORT (A0 +1.38/1.90/1.08, B0 +1.11/1.54/1.15pp; 3 datasets, "
        "≥2/3 seeds). C and Pv are B0-specific.",
        "4. **High-pass cross-parent stable?** NO — no factor, no parent "
        "(final-residual ≤ −0.02pp). Diversification basis CLOSED.",
        "5. **PRODDIFF real value?** B0: +0.299pp, 3/3 positive, Real−Mismatch "
        "+0.353pp — WEAK, 0.001pp below the +0.30 GO line. A0: −0.147pp with "
        "F1 warning — NO-GO. B0-DEPENDENT, never parent-robust.",
        "6. **PRODDIFF over parameter-matched CONCAT?** +0.106pp — positive "
        "but below the +0.15pp specificity bar.",
        "7. **FiLM over PRODDIFF/CONCAT?** NO (+0.098pp B0 / −0.239pp A0); "
        "FiLM has the strongest correspondence signal but no net value.",
        "8. **Mismatch correspondence?** For interaction adapters: real > "
        "mismatch in all 12 runs (+0.16..+2.20pp) — input correspondence is "
        "real, but it does NOT convert into gains on A0. The semantic-screen "
        "mismatch numbers (+7.6..+15.5pp) additionally reflect classifier "
        "co-adaptation to factor rewriting (Δ ≈ 0.7−1.0x factor norm), so "
        "they are not clean correspondence evidence.",
        "9. **Message novelty / effective rank?** Weak specialization: "
        "effective rank ≈ 2 of 9 cells, off-diagonal cosine high — the 3x3 "
        "cells are largely redundant with each other and with the parent "
        "factor update.",
        "10. **Semantic residual-only effective?** NO: NO-GO on both parents "
        "(A0 −1.14pp, B0 −0.11pp). The trained residual grows to 0.67−1.0x "
        "the factor norm — it REWRITES ownership instead of refining, and "
        "the frozen graph machinery (esp. A0's plan/operator tuned to the "
        "original factors) misfires.",
        "11. **Semantic residual parent-robust?** NO (both parents NO-GO).",
        "12. **Frozen vs full vs gradual?** NOT TESTED — D1.6-E gate "
        "(requires a frozen GO candidate) was not met. The matched-init "
        "schedule runner is built and tested; it carries over to the "
        "Design-2 candidate validation.",
        "13. **Optimization coupling proven?** NOT DIRECTLY in this stage "
        "(E not entered). The D1.5-B gradient-conflict diagnosis stands as "
        "the current evidence.",
        "14. **Macro-F1 safety?** B0 interaction adapters: safe (0 warnings). "
        "A0 adapters: UNSAFE (CONCAT 3, PRODDIFF 1, FiLM 2 of 3 datasets "
        "with ΔF1 < −0.50pp). Semantic: 1 warning (A0 Movies −4.14pp). "
        "Every A0-side gain claim would have failed safety.",
        "15. **Next route?** Route C with the inductive-bias caveat (below).",
        "",
        "## Verdict",
        "",
        "### R2-Design-1.6 status: **PARTIAL**",
        "",
        "The dual-parent controlled attribution answered the interaction "
        "question definitively — NO vector realization (CONCAT/PRODDIFF/FiLM "
        "or the semantic residual) reaches frozen GO on EITHER parent; the "
        "best case is B0-only PRODDIFF at the WEAK/GO boundary, and every "
        "A0-side variant is F1-unsafe. The A0 machinery already encodes the "
        "interaction; extra adapters only add redundant correction.",
        "",
        "### Recommended R2-Design-2 route: **C — Factor-Specific Multi-Scale**",
        "",
        "The ONE reproducible cross-parent mechanism is Pt-factor 2-hop "
        "(INDUCTIVE-BIAS SUPPORT ONLY: probe-level, final-residual weak). "
        "Design-2 should therefore enter as: factor-specific multi-scale "
        "propagation (e.g., a zero-init Pt 2-hop correction on the frozen "
        "parent, MixHop/GPR-style but factor-specific), validated through "
        "the D1.6-E matched-init schedule protocol (frozen → warm-start → "
        "gradual unfreeze) that this stage built and left armed. High-pass, "
        "vector interaction and semantic residual routes are closed at "
        "their current realizations.",
        "",
        "Awaiting human/ChatGPT review. No Test has been run.",
    ]
    (SUMMARY_DIR / "R2D16_FINAL_DIAGNOSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[final] saved -> {SUMMARY_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-Design-1.6 summarizer")
    parser.add_argument("--stage", required=True,
                        choices=["interaction", "semantic", "final"])
    args = parser.parse_args()
    if args.stage == "interaction":
        _write_interaction_report()
    elif args.stage == "semantic":
        _write_semantic_report()
    elif args.stage == "final":
        _write_final_report()


if __name__ == "__main__":
    main()
