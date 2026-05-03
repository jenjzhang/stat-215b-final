"""
Six-model analysis for the expanded report (replaces four_model_analysis.py).

Main story: GPT-4o-mini, Llama-3-8B-Instruct, Qwen-{0.5B, 1.5B fp16, 7B, 14B}-Instruct.
Appendix: Qwen-1.5B 4-bit (precision-effect comparison vs fp16).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from itertools import combinations

PROC = Path("data/processed")
MAIN_MODELS = ["gpt4o", "llama", "qwen_0_5b", "qwen_1_5b_fp16", "qwen_7b", "qwen_14b"]
ALL_MODELS = MAIN_MODELS + ["qwen_1_5b"]   # 4-bit appendix
QWEN_SCALE = ["qwen_0_5b", "qwen_1_5b_fp16", "qwen_7b", "qwen_14b"]

PRETTY = {
    "gpt4o": "GPT-4o-mini",
    "llama": "Llama-3-8B-Instruct",
    "qwen_0_5b": "Qwen-2.5-0.5B-Instruct",
    "qwen_1_5b": "Qwen-2.5-1.5B-Instruct (4-bit)",
    "qwen_1_5b_fp16": "Qwen-2.5-1.5B-Instruct (fp16)",
    "qwen_7b": "Qwen-2.5-7B-Instruct",
    "qwen_14b": "Qwen-2.5-14B-Instruct",
}

questions  = {m: pd.read_parquet(PROC / f"questions_{m}.parquet")           for m in ALL_MODELS}
fdr        = {m: pd.read_parquet(PROC / f"fdr_results_{m}.parquet")          for m in ALL_MODELS}
summaries  = {m: pd.read_parquet(PROC / f"subject_summaries_{m}.parquet")    for m in ALL_MODELS}

# ---------------------------------------------------------------------------
print("=" * 75)
print("PER-MODEL HEADLINE STATS")
print("=" * 75)
headline = {}
for m in ALL_MODELS:
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
        "gap_range_ratio": float(s["mean_gap"].max() / max(s["mean_gap"].min(), 1e-6)),
        "mean_ece": float(s["ece"].mean()),
    }
    h = headline[m]
    print(f"\n{h['name']}")
    print(f"  Acc: {h['accuracy']:.3f}  Conf: {h['mean_confidence']:.3f}  Gap: {h['mean_gap']:.3f}  ECE: {h['mean_ece']:.3f}")
    print(f"  FDR rej: {h['n_rejected_alpha05']}/57   Gap range: {h['min_gap']:.3f} ({h['min_gap_subject']}) -> {h['max_gap']:.3f} ({h['max_gap_subject']})  [{h['gap_range_ratio']:.1f}x]")

# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("TOP-5 MOST MISCALIBRATED SUBJECTS BY ECE")
print("=" * 75)
top5 = {}
for m in MAIN_MODELS:
    s = summaries[m].sort_values("ece", ascending=False).head(5)
    print(f"\n{PRETTY[m]}:")
    print(s[["subject", "domain", "n", "ece", "mean_gap"]].to_string(index=False))
    top5[m] = s[["subject", "ece"]].to_dict("records")

# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("CROSS-MODEL CONSISTENCY (top-N most-miscalibrated across MAIN models)")
print("=" * 75)
for N in [5, 10, 15]:
    sets = [set(summaries[m].sort_values("ece", ascending=False).head(N)["subject"]) for m in MAIN_MODELS]
    common = set.intersection(*sets)
    print(f"  Top-{N} ∩ all 6 main models: {len(common)} subjects -> {sorted(common)}")

print("\nBOTTOM-N (best calibrated) intersection across MAIN models:")
for N in [5, 10, 15]:
    sets = [set(summaries[m].sort_values("ece").head(N)["subject"]) for m in MAIN_MODELS]
    common = set.intersection(*sets)
    print(f"  Bottom-{N} ∩ all 6 main models: {len(common)} subjects -> {sorted(common)}")

# ---------------------------------------------------------------------------
# Cross-model Spearman correlation matrix
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("SPEARMAN RANK CORRELATION MATRIX (subject-level ECE, MAIN models only)")
print("=" * 75)
common_subjects = sorted(set.intersection(*[set(summaries[m]["subject"]) for m in MAIN_MODELS]))
ece_mat = pd.DataFrame({
    m: summaries[m].set_index("subject").loc[common_subjects, "ece"].values
    for m in MAIN_MODELS
}, index=common_subjects)
corr_ece = ece_mat.corr(method="spearman")
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

print("\nPairwise ECE Spearman with 95% bootstrap CIs (MAIN models):")
pairwise_ece = {}
for a, b in combinations(MAIN_MODELS, 2):
    rho, lo, hi = boot_spearman(ece_mat[a].values, ece_mat[b].values)
    print(f"  {PRETTY[a]:<35} vs {PRETTY[b]:<35} rho={rho:.3f}  [{lo:.3f}, {hi:.3f}]")
    pairwise_ece[f"{a}__{b}"] = {"rho": float(rho), "ci_lo": float(lo), "ci_hi": float(hi)}

# Also report Qwen scaling correlations specifically (excluding GPT, Llama)
print("\nWITHIN-QWEN scaling correlations (subset):")
for a, b in combinations(QWEN_SCALE, 2):
    rho, lo, hi = boot_spearman(ece_mat[a].values, ece_mat[b].values)
    print(f"  {PRETTY[a]:<35} vs {PRETTY[b]:<35} rho={rho:.3f}  [{lo:.3f}, {hi:.3f}]")

# ---------------------------------------------------------------------------
# Fixed effects (manual entry from multilevel.py output)
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("FIXED EFFECTS SUMMARY (entropy coefficient is the highlight)")
print("=" * 75)
fixed_effects = {
    "gpt4o":          {"intercept": ( 0.196, "<0.001"), "word_count": ( 0.031, "<0.001"), "max_choice_len": ( 0.001, 0.848), "has_negation": (-0.001, 0.950), "entropy": ( 0.016, "<0.001")},
    "llama":          {"intercept": ( 0.197, "<0.001"), "word_count": (-0.005, 0.437),  "max_choice_len": (-0.001, 0.838), "has_negation": ( 0.007, 0.530), "entropy": ( 0.008, 0.069)},
    "qwen_0_5b":      {"intercept": ( 0.209, "<0.001"), "word_count": (-0.013, 0.048),  "max_choice_len": (-0.028, "<0.001"), "has_negation": ( 0.016, 0.188), "entropy": (-0.063, "<0.001")},
    "qwen_1_5b":      {"intercept": ( 0.157, "<0.001"), "word_count": ( 0.004, 0.548),  "max_choice_len": (-0.002, 0.700), "has_negation": (-0.010, 0.370), "entropy": (-0.010, 0.018)},
    "qwen_1_5b_fp16": {"intercept": ( 0.179, "<0.001"), "word_count": ( 0.003, 0.634),  "max_choice_len": ( 0.005, 0.302), "has_negation": (-0.010, 0.368), "entropy": (-0.004, 0.378)},
    "qwen_7b":        {"intercept": ( 0.233, "<0.001"), "word_count": ( 0.019, 0.003),  "max_choice_len": (-0.007, 0.152), "has_negation": (-0.001, 0.917), "entropy": ( 0.029, "<0.001")},
    "qwen_14b":       {"intercept": ( 0.210, "<0.001"), "word_count": ( 0.023, "<0.001"),"max_choice_len": (-0.007, 0.108), "has_negation": ( 0.005, 0.632), "entropy": ( 0.040, "<0.001")},
}
for m in ALL_MODELS:
    fe = fixed_effects[m]
    print(f"  {PRETTY[m]:<35} intercept={fe['intercept'][0]:+.3f}  entropy_beta={fe['entropy'][0]:+.3f} (p={fe['entropy'][1]})")

# ---------------------------------------------------------------------------
# ICC summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("ICC DECOMPOSITION SUMMARY")
print("=" * 75)
icc = {
    "gpt4o":          {"icc_subject": 0.079, "var_subject": 0.013, "var_residual": 0.150},
    "llama":          {"icc_subject": 0.020, "var_subject": 0.004, "var_residual": 0.190},
    "qwen_0_5b":      {"icc_subject": 0.018, "var_subject": 0.004, "var_residual": 0.224},
    "qwen_1_5b":      {"icc_subject": 0.019, "var_subject": 0.004, "var_residual": 0.203},
    "qwen_1_5b_fp16": {"icc_subject": 0.017, "var_subject": 0.003, "var_residual": 0.197},
    "qwen_7b":        {"icc_subject": 0.049, "var_subject": 0.009, "var_residual": 0.177},
    "qwen_14b":       {"icc_subject": 0.059, "var_subject": 0.010, "var_residual": 0.156},
}
for m in ALL_MODELS:
    print(f"  {PRETTY[m]:<35} ICC_subject = {icc[m]['icc_subject']:.3f}  var_subject = {icc[m]['var_subject']:.4f}")

# ---------------------------------------------------------------------------
# FDR rejection counts at various alpha levels
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("FDR REJECTION COUNTS BY ALPHA")
print("=" * 75)
fdr_counts = {
    "gpt4o":          {0.01: 56, 0.05: 57, 0.10: 57},
    "llama":          {0.01: 55, 0.05: 57, 0.10: 57},
    "qwen_0_5b":      {0.01: 55, 0.05: 55, 0.10: 57},
    "qwen_1_5b":      {0.01: 46, 0.05: 54, 0.10: 54},
    "qwen_1_5b_fp16": {0.01: 55, 0.05: 56, 0.10: 56},
    "qwen_7b":        {0.01: 57, 0.05: 57, 0.10: 57},
    "qwen_14b":       {0.01: 57, 0.05: 57, 0.10: 57},
}
print(f"  {'Model':<35}  α=0.01  α=0.05  α=0.10")
for m in ALL_MODELS:
    c = fdr_counts[m]
    print(f"  {PRETTY[m]:<35}  {c[0.01]:>5}/57  {c[0.05]:>5}/57  {c[0.10]:>5}/57")

# Subjects not rejected by Qwen-1.5B fp16 (the new "lone unrejected" subject)
not_rej = fdr["qwen_1_5b_fp16"][~fdr["qwen_1_5b_fp16"]["reject"]]
print(f"\nQwen-1.5B fp16 subjects NOT rejected at α=0.05:")
print(not_rej[["subject", "domain", "n", "mean_gap", "p_adj"]].to_string(index=False))

# Subjects not rejected by Qwen-0.5B
not_rej_05 = fdr["qwen_0_5b"][~fdr["qwen_0_5b"]["reject"]]
print(f"\nQwen-0.5B subjects NOT rejected at α=0.05:")
print(not_rej_05[["subject", "domain", "n", "mean_gap", "p_adj"]].to_string(index=False))

# ---------------------------------------------------------------------------
# Entropy quartile mechanism (4-panel data)
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("ENTROPY QUARTILE: confidence vs accuracy (the mechanism figure data)")
print("=" * 75)
entropy_diag = {}
for m in ALL_MODELS:
    q = questions[m].copy()
    q["q"] = pd.qcut(q["entropy"], 4, labels=[0, 1, 2, 3])
    by_q = q.groupby("q", observed=True).agg(
        conf=("confidence", "mean"),
        acc=("correct", "mean"),
        gap=("calibration_gap", "mean"),
    ).reindex([0, 1, 2, 3])
    entropy_diag[m] = by_q.to_dict()
    print(f"\n{PRETTY[m]}:")
    print(by_q.round(3).to_string())

# ---------------------------------------------------------------------------
# Save everything
# ---------------------------------------------------------------------------
out = {
    "headline": {m: {**v, "name": PRETTY[m]} for m, v in headline.items()},
    "top5_per_model": top5,
    "ece_corr_matrix": corr_ece.round(3).to_dict(),
    "ece_pairwise_with_ci": pairwise_ece,
    "fixed_effects": {m: {k: list(v) if isinstance(v, tuple) else v for k, v in fe.items()} for m, fe in fixed_effects.items()},
    "icc": icc,
    "fdr_counts": fdr_counts,
    "entropy_diagnostic": entropy_diag,
    "common_top10_main": sorted(set.intersection(*[set(summaries[m].sort_values("ece", ascending=False).head(10)["subject"]) for m in MAIN_MODELS])),
    "common_bottom10_main": sorted(set.intersection(*[set(summaries[m].sort_values("ece").head(10)["subject"]) for m in MAIN_MODELS])),
}
with open(PROC / "six_model_summary.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nSaved summary to {PROC / 'six_model_summary.json'}")
