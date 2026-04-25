"""
Three-level mixed effects model for calibration gap.

Model:
  calibration_gap_ijk = mu + beta @ X_ijk + u_k + v_jk + eps_ijk
  u_k ~ N(0, sigma^2_domain)     (domain random effect)
  v_jk ~ N(0, sigma^2_subject)   (subject-within-domain random effect)

Outputs:
  - Fixed effect estimates (beta) with CIs
  - Variance components (sigma^2_domain, sigma^2_subject, sigma^2_eps)
  - ICC decomposition (fraction of variance at each level)
  - Subject-level BLUPs (shrunken calibration estimates)
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

IN = Path("data/processed")
OUT = Path("data/processed")

COVARIATES = ["word_count", "max_choice_len", "has_negation", "entropy"]
OUTCOME = "calibration_gap"


def fit_model(df: pd.DataFrame):
    # Standardize continuous covariates
    df = df.copy()
    for col in ["word_count", "max_choice_len", "entropy"]:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Three-level nested model: question within subject within domain.
    # groups=domain (outermost), vc_formula=subject (middle level).
    result = smf.mixedlm(
        f"{OUTCOME} ~ {' + '.join(COVARIATES)}",
        data=df,
        groups=df["domain"],
        vc_formula={"subject": "0 + C(subject)"},
    ).fit(reml=True)
    return result


def icc_decomposition(result) -> dict:
    cov_re = result.cov_re
    var_domain = float(cov_re.iloc[0, 0]) if cov_re.shape[1] > 0 else 0.0
    var_subject = float(result.vcomp[0]) if hasattr(result, "vcomp") and len(result.vcomp) > 0 else 0.0
    var_residual = result.scale
    total = var_domain + var_subject + var_residual
    return {
        "icc_domain": var_domain / total,
        "icc_subject": var_subject / total,
        "icc_residual": var_residual / total,
        "var_domain": var_domain,
        "var_subject": var_subject,
        "var_residual": var_residual,
    }


def extract_blups(result, df: pd.DataFrame) -> pd.DataFrame:
    # Empirical Bayes shrinkage: BLUP_j = lambda_j * (y_bar_j - mu_hat_j)
    # where lambda_j = var_subject / (var_subject + var_residual / n_j)
    icc = icc_decomposition(result)
    var_subject = icc["var_subject"]
    var_residual = icc["var_residual"]

    fe = result.fe_params
    covariates_in_model = [c for c in COVARIATES if c in df.columns]
    df = df.copy()
    for col in ["word_count", "max_choice_len", "entropy"]:
        if col in df.columns:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    df["mu_hat"] = fe["Intercept"] + sum(fe.get(c, 0) * df[c] for c in covariates_in_model)

    rows = []
    for subj, grp in df.groupby("subject"):
        n_j = len(grp)
        residual_mean = (grp["calibration_gap"] - grp["mu_hat"]).mean()
        shrinkage = var_subject / (var_subject + var_residual / n_j) if (var_subject + var_residual / n_j) > 0 else 0
        rows.append({
            "subject": subj,
            "blup": shrinkage * residual_mean,
            "domain": grp["domain"].iloc[0],
        })
    return pd.DataFrame(rows)


def run(model: str):
    path = IN / f"questions_{model}.parquet"
    if not path.exists():
        print(f"Skipping {model}: {path} not found")
        return

    df = pd.read_parquet(path)
    print(f"\n=== {model}: n={len(df):,} questions, {df['subject'].nunique()} subjects ===")

    result = fit_model(df)
    print(result.summary())

    icc = icc_decomposition(result)
    print("\nICC Decomposition:")
    for k, v in icc.items():
        print(f"  {k}: {v:.4f}")

    blups = extract_blups(result, df)
    blups.to_parquet(OUT / f"blups_{model}.parquet", index=False)
    print(f"\nBLUPs saved to {OUT / f'blups_{model}.parquet'}")

    return result, icc, blups


if __name__ == "__main__":
    for m in ["gpt4o", "llama"]:
        run(m)
