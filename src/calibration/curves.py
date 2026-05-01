"""
Per-subject calibration curve estimation.

For each subject, fits:
  1. Isotonic regression (monotone reliability curve)
  2. Kernel smoother (unconstrained, for comparison)

Outputs:
  data/processed/calibration_curves_{model}.parquet
    subject, confidence_bins (list), isotonic_acc (list), kernel_acc (list), ece

  data/processed/subject_summaries_{model}.parquet
    subject, domain, n, ece, mean_gap, mean_confidence, mean_accuracy
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from scipy.ndimage import gaussian_filter1d

IN = Path("data/processed")
OUT = Path("data/processed")
MODELS = ["gpt4o", "llama", "qwen_1_5b", "qwen_7b"]
N_BINS = 20


def isotonic_calibration(confidence: np.ndarray, correct: np.ndarray):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(confidence, correct)
    return ir


def kernel_smoother(confidence: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS, sigma: float = 1.5):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_acc = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
        if mask.sum() > 0:
            bin_acc[i] = correct[mask].mean()
    # Fill NaN with neighboring values before smoothing
    valid = ~np.isnan(bin_acc)
    if valid.sum() < 2:
        return bin_centers, bin_acc
    bin_acc_filled = np.interp(bin_centers, bin_centers[valid], bin_acc[valid])
    smoothed = gaussian_filter1d(bin_acc_filled, sigma=sigma)
    return bin_centers, smoothed


def compute_ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
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


def compute_nlcs(confidence: np.ndarray, correct: np.ndarray) -> float:
    eps = 1e-12
    conf = np.clip(confidence, eps, 1 - eps)
    return float(-np.mean(correct * np.log(conf) + (1 - correct) * np.log(1 - conf)))


def process_model(model: str):
    df = pd.read_parquet(IN / f"questions_{model}.parquet")
    subjects = df["subject"].unique()

    curve_rows = []
    summary_rows = []

    for subj in subjects:
        sub = df[df["subject"] == subj]
        conf = sub["confidence"].values
        corr = sub["correct"].astype(float).values
        domain = sub["domain"].iloc[0]

        ir = isotonic_calibration(conf, corr)
        bins = np.linspace(0, 1, N_BINS + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        isotonic_acc = ir.predict(bin_centers)

        _, kernel_acc = kernel_smoother(conf, corr)
        ece = compute_ece(conf, corr)
        nlcs = compute_nlcs(conf, corr)

        curve_rows.append({
            "subject": subj,
            "domain": domain,
            "confidence_bins": bin_centers.tolist(),
            "isotonic_acc": isotonic_acc.tolist(),
            "kernel_acc": kernel_acc.tolist(),
            "ece": ece,
            "nlcs": nlcs,
        })

        summary_rows.append({
            "subject": subj,
            "domain": domain,
            "n": len(sub),
            "ece": ece,
            "nlcs": nlcs,
            "mean_gap": sub["calibration_gap"].mean(),
            "mean_confidence": conf.mean(),
            "mean_accuracy": corr.mean(),
        })

    curves_df = pd.DataFrame(curve_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("ece", ascending=False)

    curves_df.to_parquet(OUT / f"calibration_curves_{model}.parquet", index=False)
    summary_df.to_parquet(OUT / f"subject_summaries_{model}.parquet", index=False)
    print(f"{model}: curves and summaries for {len(subjects)} subjects saved")


def main():
    for model in MODELS:
        path = IN / f"questions_{model}.parquet"
        if not path.exists():
            print(f"Skipping {model}: {path} not found")
            continue
        process_model(model)


if __name__ == "__main__":
    main()
