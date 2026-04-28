"""
Cross-model comparison of subject-level calibration rankings.

Computes Spearman rank correlation of subject ECE between GPT-4o-mini and Llama-3-8B,
with bootstrap confidence intervals. Identifies subjects with concordant vs. discordant
miscalibration (task-driven vs. model-specific).

GPT-4o-mini confidence is computed from OpenAI's top_logprobs=5 API response.
Tokens not in the top-5 are filled with -100.0, which inflates renormalized
confidence toward 1.0. To ensure comparable ECE estimates, GPT-4o-mini is restricted
to questions where all 4 answer tokens appeared in the top-5 response. Llama uses
full-vocabulary softmax logprobs and requires no filtering.

Input:  data/processed/questions_gpt4o.parquet
        data/processed/questions_llama.parquet
Output: data/processed/cross_model_comparison.parquet
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

IN = Path("data/processed")
OUT = Path("data/processed")
N_BOOTSTRAP = 2000
RNG = np.random.default_rng(42)
LOGPROB_COLS = ["logprob_A", "logprob_B", "logprob_C", "logprob_D"]
SENTINEL = -50.0  # logprobs below this are fill-ins for tokens absent from top-5


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, n: int = N_BOOTSTRAP):
    observed, _ = stats.spearmanr(x, y)
    def _one_boot():
        idx = RNG.integers(0, len(x), len(x))
        return stats.spearmanr(x[idx], y[idx])[0]
    boot = np.array([_one_boot() for _ in range(n)])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return observed, ci_lo, ci_hi


def compute_ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 20) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(confidence)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(correct[mask].mean() - confidence[mask].mean())
    return ece


def compute_nlcs(confidence: np.ndarray, correct: np.ndarray) -> float:
    eps = 1e-12
    conf = np.clip(confidence, eps, 1 - eps)
    return float(-np.mean(correct * np.log(conf) + (1 - correct) * np.log(1 - conf)))


def bh_correct(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    reject = p_values <= ranked / n * alpha
    reject_sorted = reject[order]
    for i in range(len(reject_sorted) - 2, -1, -1):
        if reject_sorted[i + 1]:
            reject_sorted[i] = True
    reject[order] = reject_sorted
    return reject


def subject_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-subject ECE and BH-corrected miscalibration test from question-level data."""
    rows = []
    for subj, grp in df.groupby("subject"):
        conf = grp["confidence"].values
        corr = grp["correct"].astype(float).values
        gaps = grp["calibration_gap"].values
        _, p_val = stats.ttest_1samp(gaps, 0.0)
        rows.append({
            "subject": subj,
            "domain": grp["domain"].iloc[0],
            "n": len(grp),
            "ece": compute_ece(conf, corr),
            "nlcs": compute_nlcs(conf, corr),
            "mean_gap": gaps.mean(),
            "p_value": p_val,
        })
    result = pd.DataFrame(rows).set_index("subject")
    result["reject"] = bh_correct(result["p_value"].values)
    return result


def main():
    q_paths = {m: IN / f"questions_{m}.parquet" for m in ["gpt4o", "llama"]}
    for m, p in q_paths.items():
        if not p.exists():
            print(f"Missing {p}, skipping")
            return

    q = {m: pd.read_parquet(p) for m, p in q_paths.items()}

    # Both models now use P(predicted answer) from the full-vocabulary distribution
    # as confidence, so no filtering is needed here. See extract_features.py.
    summaries = {m: subject_stats(q[m]) for m in ["gpt4o", "llama"]}

    common = summaries["gpt4o"].index.intersection(summaries["llama"].index)
    ece_g = summaries["gpt4o"].loc[common, "ece"].values
    ece_l = summaries["llama"].loc[common, "ece"].values

    rho, lo, hi = bootstrap_spearman(ece_g, ece_l)
    print(f"Spearman rank correlation (ECE): rho={rho:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")

    nlcs_g = summaries["gpt4o"].loc[common, "nlcs"].values
    nlcs_l = summaries["llama"].loc[common, "nlcs"].values
    rho_nlcs, lo_nlcs, hi_nlcs = bootstrap_spearman(nlcs_g, nlcs_l)
    print(f"Spearman rank correlation (NLCS): rho={rho_nlcs:.3f}, 95% CI [{lo_nlcs:.3f}, {hi_nlcs:.3f}]")

    df = pd.DataFrame(index=common)
    df["ece_gpt4o"] = summaries["gpt4o"].loc[common, "ece"]
    df["ece_llama"] = summaries["llama"].loc[common, "ece"]
    df["nlcs_gpt4o"] = summaries["gpt4o"].loc[common, "nlcs"]
    df["nlcs_llama"] = summaries["llama"].loc[common, "nlcs"]
    df["reject_gpt4o"] = summaries["gpt4o"].loc[common, "reject"]
    df["reject_llama"] = summaries["llama"].loc[common, "reject"]
    df["domain"] = summaries["gpt4o"].loc[common, "domain"]

    df["agreement"] = "neither"
    df.loc[df["reject_gpt4o"] & df["reject_llama"], "agreement"] = "both (task-driven)"
    df.loc[df["reject_gpt4o"] & ~df["reject_llama"], "agreement"] = "gpt4o only"
    df.loc[~df["reject_gpt4o"] & df["reject_llama"], "agreement"] = "llama only"

    df = df.reset_index().rename(columns={"index": "subject"})
    df.to_parquet(OUT / "cross_model_comparison.parquet", index=False)
    print("\nAgreement breakdown:")
    print(df["agreement"].value_counts().to_string())


if __name__ == "__main__":
    main()
