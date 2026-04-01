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

    formula = f"{OUTCOME} ~ {' + '.join(COVARIATES)} + (1 | domain) + (1 | subject)"
    model = smf.mixedlm(
        formula=f"{OUTCOME} ~ {' + '.join(COVARIATES)}",
        data=df,
        groups=df["subject"],
        exog_re=np.ones(len(df)),
    )
    # Note: statsmodels MixedLM supports two-level nesting natively.
    # For the full three-level (question/subject/domain) model, use the
    # formula interface with variance components:
    vc = {"domain": "0 + C(domain)"}
    result = smf.mixedlm(
        f"{OUTCOME} ~ {' + '.join(COVARIATES)}",
        data=df,
        groups=df["subject"],
        vc_formula=vc,
    ).fit(reml=True)
    return result


def icc_decomposition(result) -> dict:
    var_subject = result.cov_re.iloc[0, 0]
    var_domain = result.vcomp[0] if hasattr(result, "vcomp") else np.nan
    var_residual = result.scale
    total = var_subject + var_domain + var_residual
    return {
        "icc_domain": var_domain / total,
        "icc_subject": var_subject / total,
        "icc_residual": var_residual / total,
        "var_domain": var_domain,
        "var_subject": var_subject,
        "var_residual": var_residual,
    }


def extract_blups(result, df: pd.DataFrame) -> pd.DataFrame:
    re = result.random_effects
    blups = pd.DataFrame([
        {"subject": subj, "blup": vals.iloc[0]}
        for subj, vals in re.items()
    ])
    subject_meta = df.groupby("subject")[["domain", "n"]].first().reset_index() if "n" in df else (
        df.groupby("subject")["domain"].first().reset_index()
    )
    return blups.merge(subject_meta, on="subject")


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
