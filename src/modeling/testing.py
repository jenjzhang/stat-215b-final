"""
Multiple testing correction for subject-level miscalibration.

For each subject, tests H0: mean calibration gap == 0 (i.e. perfectly calibrated).
Uses one-sample t-test on question-level calibration gaps, then applies
Benjamini-Hochberg FDR correction at alpha=0.05.

Input:  data/processed/questions_{model}.parquet
Output: data/processed/fdr_results_{model}.parquet
  subject, domain, n, mean_gap, se_gap, t_stat, p_value, p_adj, reject
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

IN = Path("data/processed")
OUT = Path("data/processed")
ALPHA = 0.05


def bh_correction(p_values: np.ndarray, alpha: float = ALPHA):
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    threshold = ranked / n * alpha
    reject = p_values <= threshold
    # Ensure monotonicity: if rank k is rejected, all ranks < k are too
    reject_sorted = reject[order]
    for i in range(len(reject_sorted) - 2, -1, -1):
        if reject_sorted[i + 1]:
            reject_sorted[i] = True
    reject[order] = reject_sorted
    # Adjusted p-values (Benjamini-Hochberg)
    p_adj = np.minimum(1.0, p_values * n / ranked)
    return reject, p_adj


def run(model: str):
    path = IN / f"questions_{model}.parquet"
    if not path.exists():
        print(f"Skipping {model}: {path} not found")
        return

    df = pd.read_parquet(path)
    rows = []
    for subj, grp in df.groupby("subject"):
        gaps = grp["calibration_gap"].values
        t_stat, p_val = stats.ttest_1samp(gaps, 0.0)
        rows.append({
            "subject": subj,
            "domain": grp["domain"].iloc[0],
            "n": len(grp),
            "mean_gap": gaps.mean(),
            "se_gap": gaps.std() / np.sqrt(len(gaps)),
            "t_stat": t_stat,
            "p_value": p_val,
        })

    results = pd.DataFrame(rows)
    p_vals = results["p_value"].values
    reject, p_adj = bh_correction(p_vals)
    results["p_adj"] = p_adj
    results["reject"] = reject
    results = results.sort_values("p_adj")

    out_path = OUT / f"fdr_results_{model}.parquet"
    results.to_parquet(out_path, index=False)

    n_reject = reject.sum()
    print(f"{model}: {n_reject}/{len(results)} subjects rejected at FDR {ALPHA}")
    print(results[results["reject"]].to_string(index=False))
    return results


if __name__ == "__main__":
    for m in ["gpt4o", "llama"]:
        run(m)
