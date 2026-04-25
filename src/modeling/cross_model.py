"""
Cross-model comparison of subject-level calibration rankings.

Computes Spearman rank correlation of subject ECE between GPT-4o-mini and Llama-3-8B,
with bootstrap confidence intervals. Identifies subjects with concordant vs. discordant
miscalibration (task-driven vs. model-specific).

Input:  data/processed/subject_summaries_gpt4o.parquet
        data/processed/subject_summaries_llama.parquet
        data/processed/fdr_results_gpt4o.parquet
        data/processed/fdr_results_llama.parquet
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


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, n: int = N_BOOTSTRAP):
    observed, _ = stats.spearmanr(x, y)
    def _one_boot():
        idx = RNG.integers(0, len(x), len(x))
        return stats.spearmanr(x[idx], y[idx])[0]
    boot = np.array([_one_boot() for _ in range(n)])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return observed, ci_lo, ci_hi


def main():
    summaries = {}
    fdr = {}
    for model in ["gpt4o", "llama"]:
        s = IN / f"subject_summaries_{model}.parquet"
        f = IN / f"fdr_results_{model}.parquet"
        if not s.exists() or not f.exists():
            print(f"Missing files for {model}, skipping")
            return
        summaries[model] = pd.read_parquet(s).set_index("subject")
        fdr[model] = pd.read_parquet(f).set_index("subject")

    # Align on common subjects
    common = summaries["gpt4o"].index.intersection(summaries["llama"].index)
    ece_g = summaries["gpt4o"].loc[common, "ece"].values
    ece_l = summaries["llama"].loc[common, "ece"].values

    rho, lo, hi = bootstrap_spearman(ece_g, ece_l)
    print(f"Spearman rank correlation (ECE): rho={rho:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")

    # Classify subjects
    df = pd.DataFrame(index=common)
    df["ece_gpt4o"] = summaries["gpt4o"].loc[common, "ece"]
    df["ece_llama"] = summaries["llama"].loc[common, "ece"]
    df["reject_gpt4o"] = fdr["gpt4o"].loc[common, "reject"]
    df["reject_llama"] = fdr["llama"].loc[common, "reject"]
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
