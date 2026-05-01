"""
One-off analysis to extract numbers needed for a tentative four-model report revision.
Outputs a JSON with all summary stats, plus prints key tables to stdout.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from itertools import combinations

PROC = Path("data/processed")
MODELS = ["gpt4o", "llama", "qwen_1_5b", "qwen_7b"]
PRETTY = {
    "gpt4o": "GPT-4o-mini",
    "llama": "Llama-3-8B-Instruct",
    "qwen_1_5b": "Qwen-2.5-1.5B-Instruct",
    "qwen_7b": "Qwen-2.5-7B-Instruct",
}

questions = {m: pd.read_parquet(PROC / f"questions_{m}.parquet") for m in MODELS}
fdr = {m: pd.read_parquet(PROC / f"fdr_results_{m}.parquet") for m in MODELS}
summaries = {m: pd.read_parquet(PROC / f"subject_summaries_{m}.parquet") for m in MODELS}

# ---------------------------------------------------------------------------
# Per-model headline stats
# ---------------------------------------------------------------------------
print("=" * 70)
print("PER-MODEL HEADLINE STATS")
print("=" * 70)
headline = {}
for m in MODELS:
    q = questions[m]
    f = fdr[m]
    s = summaries[m]
    headline[m] = {
        "name": PRETTY[m],
        "n_questions": len(q),
        "accuracy": float(q["correct"].mean()),
        "mean_confidence": float(q["confidence"].mean()),
        "mean_gap": float(q["calibration_gap"].mean()),
        "n_rejected_alpha05": int(f["reject"].sum()),
        "min_gap_subject": s.sort_values("mean_gap").iloc[0]["subject"],
        "min_gap": float(s["mean_gap"].min()),
        "max_gap_subject": s.sort_values("mean_gap").iloc[-1]["subject"],
        "max_gap": float(s["mean_gap"].max()),
        "gap_range_ratio": float(s["mean_gap"].max() / s["mean_gap"].min()),
        "mean_ece": float(s["ece"].mean()),
        "min_ece_subject": s.sort_values("ece").iloc[0]["subject"],
        "min_ece": float(s["ece"].min()),
        "max_ece_subject": s.sort_values("ece").iloc[-1]["subject"],
        "max_ece": float(s["ece"].max()),
    }
    h = headline[m]
    print(f"\n{h['name']}")
    print(f"  Accuracy: {h['accuracy']:.3f}    Mean confidence: {h['mean_confidence']:.3f}    Mean gap: {h['mean_gap']:.3f}")
    print(f"  FDR rejected: {h['n_rejected_alpha05']}/57")
    print(f"  Mean gap range: {h['min_gap']:.3f} ({h['min_gap_subject']}) -> {h['max_gap']:.3f} ({h['max_gap_subject']})  [ratio {h['gap_range_ratio']:.1f}x]")
    print(f"  ECE range: {h['min_ece']:.3f} ({h['min_ece_subject']}) -> {h['max_ece']:.3f} ({h['max_ece_subject']})")

# ---------------------------------------------------------------------------
# Top-5 most miscalibrated subjects by ECE for each model
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TOP-5 MOST MISCALIBRATED SUBJECTS BY ECE")
print("=" * 70)
top5_by_model = {}
for m in MODELS:
    s = summaries[m].sort_values("ece", ascending=False).head(5)
    print(f"\n{PRETTY[m]}:")
    print(s[["subject", "domain", "n", "ece", "mean_gap", "mean_accuracy"]].to_string(index=False))
    top5_by_model[m] = s[["subject", "domain", "ece", "mean_gap"]].to_dict("records")

# ---------------------------------------------------------------------------
# Top-5 by mean_gap for each model
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TOP-5 BY MEAN GAP (most overconfident subjects)")
print("=" * 70)
top5_gap_by_model = {}
for m in MODELS:
    s = summaries[m].sort_values("mean_gap", ascending=False).head(5)
    print(f"\n{PRETTY[m]}:")
    print(s[["subject", "domain", "n", "mean_gap", "mean_confidence", "mean_accuracy"]].to_string(index=False))
    top5_gap_by_model[m] = s[["subject", "domain", "mean_gap"]].to_dict("records")

# ---------------------------------------------------------------------------
# Subjects that are top-5 in ALL 4 models (universally hardest)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CROSS-MODEL CONSISTENCY: subjects in top-N most-miscalibrated for all 4 models")
print("=" * 70)
for N in [5, 10, 15]:
    sets = [set(summaries[m].sort_values("ece", ascending=False).head(N)["subject"]) for m in MODELS]
    common = set.intersection(*sets)
    print(f"  Top-{N}: {len(common)} subjects in all 4 models -> {sorted(common)}")

# ---------------------------------------------------------------------------
# Pairwise Spearman ECE correlation matrix
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SPEARMAN RANK CORRELATION MATRIX (subject-level ECE)")
print("=" * 70)
common_subjects = sorted(set.intersection(*[set(summaries[m]["subject"]) for m in MODELS]))
ece_matrix = pd.DataFrame({
    m: summaries[m].set_index("subject").loc[common_subjects, "ece"].values
    for m in MODELS
}, index=common_subjects)

corr_ece = ece_matrix.corr(method="spearman")
print("\nECE Spearman correlations:")
print(corr_ece.round(3).to_string())

# Bootstrap CIs for each pair
RNG = np.random.default_rng(42)
def boot_spearman(x, y, n=2000):
    obs, _ = stats.spearmanr(x, y)
    boots = []
    for _ in range(n):
        idx = RNG.integers(0, len(x), len(x))
        rho, _ = stats.spearmanr(x[idx], y[idx])
        boots.append(rho)
    boots = np.array(boots)
    return obs, np.percentile(boots, 2.5), np.percentile(boots, 97.5)

print("\nPairwise ECE Spearman with 95% bootstrap CIs:")
pairwise_ece = {}
for a, b in combinations(MODELS, 2):
    rho, lo, hi = boot_spearman(ece_matrix[a].values, ece_matrix[b].values)
    print(f"  {a} vs {b}: rho = {rho:.3f}  [{lo:.3f}, {hi:.3f}]")
    pairwise_ece[f"{a}__{b}"] = {"rho": rho, "ci_lo": lo, "ci_hi": hi}

# ---------------------------------------------------------------------------
# Mean-gap correlation matrix (signed gap, more interpretable than ECE)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SPEARMAN RANK CORRELATION MATRIX (subject-level mean signed gap)")
print("=" * 70)
gap_matrix = pd.DataFrame({
    m: summaries[m].set_index("subject").loc[common_subjects, "mean_gap"].values
    for m in MODELS
}, index=common_subjects)
corr_gap = gap_matrix.corr(method="spearman")
print(corr_gap.round(3).to_string())

# ---------------------------------------------------------------------------
# Fixed-effects table (intercepts and entropy coefficients)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FIXED EFFECTS SUMMARY (from earlier run; manual entry)")
print("=" * 70)
fixed_effects = {
    "gpt4o":     {"intercept": 0.196, "p_int": "<0.001", "word_count": 0.031, "p_wc": "<0.001",
                  "max_choice_len": 0.001, "p_mcl": 0.848, "has_negation": -0.001, "p_neg": 0.950,
                  "entropy": 0.016, "p_ent": "<0.001"},
    "llama":     {"intercept": 0.197, "p_int": "<0.001", "word_count": -0.005, "p_wc": 0.437,
                  "max_choice_len": -0.001, "p_mcl": 0.838, "has_negation": 0.007, "p_neg": 0.530,
                  "entropy": 0.008, "p_ent": 0.069},
    "qwen_1_5b": {"intercept": 0.157, "p_int": "<0.001", "word_count": 0.004, "p_wc": 0.548,
                  "max_choice_len": -0.002, "p_mcl": 0.700, "has_negation": -0.010, "p_neg": 0.370,
                  "entropy": -0.010, "p_ent": 0.018},
    "qwen_7b":   {"intercept": 0.233, "p_int": "<0.001", "word_count": 0.019, "p_wc": 0.003,
                  "max_choice_len": -0.007, "p_mcl": 0.152, "has_negation": -0.001, "p_neg": 0.917,
                  "entropy": 0.029, "p_ent": "<0.001"},
}
for m, fe in fixed_effects.items():
    print(f"  {PRETTY[m]}: intercept={fe['intercept']:.3f}  entropy_beta={fe['entropy']:+.3f} (p={fe['p_ent']})")

# ---------------------------------------------------------------------------
# ICC summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ICC DECOMPOSITION SUMMARY")
print("=" * 70)
icc = {
    "gpt4o": {"icc_subject": 0.079, "var_subject": 0.013},
    "llama": {"icc_subject": 0.020, "var_subject": 0.004},
    "qwen_1_5b": {"icc_subject": 0.019, "var_subject": 0.004},
    "qwen_7b": {"icc_subject": 0.049, "var_subject": 0.009},
}
for m in MODELS:
    print(f"  {PRETTY[m]}: ICC_subject = {icc[m]['icc_subject']:.3f}, var_subject = {icc[m]['var_subject']:.4f}")

# ---------------------------------------------------------------------------
# Entropy interpretation: relationship between entropy, accuracy, confidence
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ENTROPY-CONFIDENCE-ACCURACY RELATIONSHIP")
print("=" * 70)
entropy_diag = {}
for m in MODELS:
    q = questions[m]
    # Bin by entropy quartiles, look at confidence and accuracy
    q = q.copy()
    q["entropy_q"] = pd.qcut(q["entropy"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    by_q = q.groupby("entropy_q", observed=True).agg(
        n=("question_id", "count"),
        mean_conf=("confidence", "mean"),
        mean_acc=("correct", "mean"),
        mean_gap=("calibration_gap", "mean"),
        mean_ent=("entropy", "mean"),
    )
    # Correlations
    rho_ec, _ = stats.spearmanr(q["entropy"], q["confidence"])
    rho_ea, _ = stats.spearmanr(q["entropy"], q["correct"].astype(float))
    rho_eg, _ = stats.spearmanr(q["entropy"], q["calibration_gap"])
    print(f"\n{PRETTY[m]}:")
    print(by_q.round(3).to_string())
    print(f"  Spearman rho(entropy, confidence) = {rho_ec:.3f}  (mechanically negative)")
    print(f"  Spearman rho(entropy, accuracy)   = {rho_ea:.3f}")
    print(f"  Spearman rho(entropy, gap)        = {rho_eg:.3f}")
    entropy_diag[m] = {
        "rho_ent_conf": float(rho_ec),
        "rho_ent_acc": float(rho_ea),
        "rho_ent_gap": float(rho_eg),
        "by_quartile": by_q.reset_index().assign(
            entropy_q=lambda d: d.entropy_q.astype(str)
        ).to_dict("records"),
    }

# ---------------------------------------------------------------------------
# Save everything
# ---------------------------------------------------------------------------
out = {
    "headline": headline,
    "top5_ece_per_model": top5_by_model,
    "top5_gap_per_model": top5_gap_by_model,
    "ece_corr_matrix": corr_ece.round(3).to_dict(),
    "gap_corr_matrix": corr_gap.round(3).to_dict(),
    "ece_pairwise_with_ci": pairwise_ece,
    "fixed_effects": fixed_effects,
    "icc": icc,
    "entropy_diagnostic": entropy_diag,
    "common_top5_across_all_4": sorted(set.intersection(*[
        set(summaries[m].sort_values("ece", ascending=False).head(5)["subject"]) for m in MODELS
    ])),
    "common_top10_across_all_4": sorted(set.intersection(*[
        set(summaries[m].sort_values("ece", ascending=False).head(10)["subject"]) for m in MODELS
    ])),
}
with open(PROC / "four_model_summary.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nSaved summary to {PROC / 'four_model_summary.json'}")
