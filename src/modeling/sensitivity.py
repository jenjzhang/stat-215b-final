"""
Sensitivity analysis for the calibration audit.

Three checks:
  1. ECE bin count (10/15/20/25/30) — does subject ranking change?
  2. FDR threshold (alpha = 0.01/0.05/0.10) — how many subjects rejected?
  3. Mixed model structure (2-level vs. 3-level) — does ICC_subject change?

Output: data/processed/sensitivity_{model}.json
"""
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr

IN = Path("data/processed")
OUT = Path("data/processed")

COVARIATES = ["word_count", "max_choice_len", "has_negation", "entropy"]
OUTCOME = "calibration_gap"
BIN_COUNTS = [10, 15, 20, 25, 30]
ALPHA_LEVELS = [0.01, 0.05, 0.10]
BASELINE_BINS = 20


# ---------------------------------------------------------------------------
# ECE helpers
# ---------------------------------------------------------------------------

def compute_ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidence)
    for i in range(n_bins):
        mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidence[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return ece


def subject_ece(df: pd.DataFrame, n_bins: int) -> pd.Series:
    rows = {}
    for subj, grp in df.groupby("subject"):
        rows[subj] = compute_ece(
            grp["confidence"].values,
            grp["correct"].astype(float).values,
            n_bins,
        )
    return pd.Series(rows)


def compute_nlcs(confidence: np.ndarray, correct: np.ndarray) -> float:
    eps = 1e-12
    conf = np.clip(confidence, eps, 1 - eps)
    return float(-np.mean(correct * np.log(conf) + (1 - correct) * np.log(1 - conf)))


def subject_nlcs(df: pd.DataFrame) -> pd.Series:
    rows = {}
    for subj, grp in df.groupby("subject"):
        rows[subj] = compute_nlcs(
            grp["confidence"].values,
            grp["correct"].astype(float).values,
        )
    return pd.Series(rows)


# ---------------------------------------------------------------------------
# BH helper
# ---------------------------------------------------------------------------

def bh_correction(p_values: np.ndarray, alpha: float):
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    threshold = ranked / n * alpha
    reject = p_values <= threshold
    reject_sorted = reject[order]
    for i in range(len(reject_sorted) - 2, -1, -1):
        if reject_sorted[i + 1]:
            reject_sorted[i] = True
    reject[order] = reject_sorted
    return reject


def subject_pvalues(df: pd.DataFrame) -> np.ndarray:
    pvals = []
    for _, grp in df.groupby("subject"):
        _, p = stats.ttest_1samp(grp["calibration_gap"].values, 0.0)
        pvals.append(p)
    return np.array(pvals)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["word_count", "max_choice_len", "entropy"]:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df


def fit_three_level(df: pd.DataFrame):
    df = standardize(df)
    return smf.mixedlm(
        f"{OUTCOME} ~ {' + '.join(COVARIATES)}",
        data=df,
        groups=df["domain"],
        vc_formula={"subject": "0 + C(subject)"},
    ).fit(reml=True)


def fit_two_level(df: pd.DataFrame):
    """Subject-only random effects, no domain level."""
    df = standardize(df)
    return smf.mixedlm(
        f"{OUTCOME} ~ {' + '.join(COVARIATES)}",
        data=df,
        groups=df["subject"],
    ).fit(reml=True)


def icc_three(result) -> dict:
    var_domain = float(result.cov_re.iloc[0, 0]) if result.cov_re.shape[1] > 0 else 0.0
    var_subject = float(result.vcomp[0]) if hasattr(result, "vcomp") and len(result.vcomp) > 0 else 0.0
    var_residual = result.scale
    total = var_domain + var_subject + var_residual
    return {
        "var_domain": var_domain,
        "var_subject": var_subject,
        "var_residual": var_residual,
        "icc_subject": var_subject / total if total > 0 else 0.0,
    }


def icc_two(result) -> dict:
    var_subject = float(result.cov_re.iloc[0, 0]) if result.cov_re.shape[1] > 0 else 0.0
    var_residual = result.scale
    total = var_subject + var_residual
    return {
        "var_subject": var_subject,
        "var_residual": var_residual,
        "icc_subject": var_subject / total if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(model: str):
    path = IN / f"questions_{model}.parquet"
    if not path.exists():
        print(f"Skipping {model}: {path} not found")
        return

    df = pd.read_parquet(path)
    results = {}

    # ------------------------------------------------------------------
    # 1. ECE bin-count sensitivity (and NLCS robustness)
    # ------------------------------------------------------------------
    print(f"\n[{model}] ECE bin-count sensitivity")
    baseline_ece = subject_ece(df, BASELINE_BINS)
    subj_nlcs = subject_nlcs(df)
    bin_results = {}
    for n_bins in BIN_COUNTS:
        ece_n = subject_ece(df, n_bins)
        rho_base, _ = spearmanr(baseline_ece, ece_n)
        rho_nlcs, _ = spearmanr(subj_nlcs, ece_n)
        bin_results[n_bins] = {
            "spearman_vs_baseline": round(float(rho_base), 4),
            "spearman_vs_nlcs": round(float(rho_nlcs), 4),
            "mean_ece": round(float(ece_n.mean()), 4),
        }
        print(f"  bins={n_bins:2d}: mean ECE={ece_n.mean():.4f}, Spearman vs. 20-bin={rho_base:.4f}, vs. NLCS={rho_nlcs:.4f}")
    results["ece_bins"] = bin_results
    results["overall_mean_nlcs"] = round(float(subj_nlcs.mean()), 4)

    # ------------------------------------------------------------------
    # 2. FDR threshold sensitivity
    # ------------------------------------------------------------------
    print(f"\n[{model}] FDR threshold sensitivity")
    p_values = subject_pvalues(df)
    n_subjects = len(p_values)
    fdr_results = {}
    for alpha in ALPHA_LEVELS:
        reject = bh_correction(p_values, alpha)
        n_reject = int(reject.sum())
        fdr_results[alpha] = {"n_rejected": n_reject, "fraction": round(n_reject / n_subjects, 4)}
        print(f"  alpha={alpha}: {n_reject}/{n_subjects} rejected ({n_reject/n_subjects:.1%})")
    results["fdr_thresholds"] = {str(a): v for a, v in fdr_results.items()}

    # ------------------------------------------------------------------
    # 3. Model structure: 2-level vs. 3-level
    # ------------------------------------------------------------------
    print(f"\n[{model}] Model structure sensitivity")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r3 = fit_three_level(df)
        r2 = fit_two_level(df)

    icc3 = icc_three(r3)
    icc2 = icc_two(r2)
    model_results = {
        "three_level": {k: round(v, 4) for k, v in icc3.items()},
        "two_level": {k: round(v, 4) for k, v in icc2.items()},
    }
    print(f"  3-level: icc_subject={icc3['icc_subject']:.4f}, var_subject={icc3['var_subject']:.5f}")
    print(f"  2-level: icc_subject={icc2['icc_subject']:.4f}, var_subject={icc2['var_subject']:.5f}")
    results["model_structure"] = model_results

    out_path = OUT / f"sensitivity_{model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    for m in ["gpt4o", "llama", "qwen_0_5b", "qwen_1_5b", "qwen_1_5b_fp16", "qwen_7b", "qwen_14b"]:
        run(m)
