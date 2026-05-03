"""
Generate all figures for the six-model report revision (replaces four_model_figures.py).
Outputs to report/figures/ with `_6m` suffix.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import combinations
from pathlib import Path
from scipy import stats

PROC = Path("data/processed")
FIG = Path("report/figures")
FIG.mkdir(exist_ok=True)

MAIN_MODELS = ["gpt4o", "llama", "qwen_0_5b", "qwen_1_5b_fp16", "qwen_7b", "qwen_14b"]
QWEN_SCALE_MAIN = ["qwen_1_5b_fp16", "qwen_7b", "qwen_14b"]   # Qwen scaling sweep (cleanest)
QWEN_SCALE_FULL = ["qwen_0_5b", "qwen_1_5b_fp16", "qwen_7b", "qwen_14b"]   # with 0.5B for context
APPENDIX = ["qwen_1_5b", "qwen_1_5b_fp16"]                    # precision comparison

PRETTY = {
    "gpt4o": "GPT-4o-mini",
    "llama": "Llama-3-8B-Instruct",
    "qwen_0_5b": "Qwen-2.5-0.5B-Instruct",
    "qwen_1_5b": "Qwen-2.5-1.5B-Instruct (4-bit)",
    "qwen_1_5b_fp16": "Qwen-2.5-1.5B-Instruct (fp16)",
    "qwen_7b": "Qwen-2.5-7B-Instruct",
    "qwen_14b": "Qwen-2.5-14B-Instruct",
}
SHORT = {
    "gpt4o": "GPT-4o-mini",
    "llama": "Llama-3-8B",
    "qwen_0_5b": "Qwen-0.5B",
    "qwen_1_5b": "Qwen-1.5B (4-bit)",
    "qwen_1_5b_fp16": "Qwen-1.5B",
    "qwen_7b": "Qwen-7B",
    "qwen_14b": "Qwen-14B",
}
COLORS = {
    "gpt4o": "#1f77b4",
    "llama": "#d62728",
    "qwen_0_5b": "#ff7f0e",
    "qwen_1_5b": "#2ca02c",
    "qwen_1_5b_fp16": "#2ca02c",
    "qwen_7b": "#9467bd",
    "qwen_14b": "#8c564b",
}

mpl.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

questions = {m: pd.read_parquet(PROC / f"questions_{m}.parquet") for m in MAIN_MODELS + ["qwen_1_5b"]}
summaries = {m: pd.read_parquet(PROC / f"subject_summaries_{m}.parquet") for m in MAIN_MODELS + ["qwen_1_5b"]}
fdr       = {m: pd.read_parquet(PROC / f"fdr_results_{m}.parquet") for m in MAIN_MODELS + ["qwen_1_5b"]}


# ============================================================
# Figure 1: top-5 / bottom-5 miscalibrated for GPT-4o-mini
# (kept identical to the 4m version — same model, same data)
# ============================================================
def plot_calibration_panel(ax, conf, corr, subject, n_questions, mean_acc, color="#1f77b4"):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(conf, corr)
    p_lo, p_hi = np.percentile(conf, [1, 99])
    grid = np.linspace(p_lo, p_hi, 200)
    iso_pred = ir.predict(grid)

    bins = np.linspace(0, 1, 21)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_acc = np.full(20, np.nan)
    bin_n = np.zeros(20)
    for i in range(20):
        mask = (conf >= bins[i]) & (conf < bins[i+1])
        bin_n[i] = mask.sum()
        if mask.sum() >= 3:
            bin_acc[i] = corr[mask].mean()
    valid = ~np.isnan(bin_acc)

    ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8, alpha=0.7, label="perfect")
    ax.plot(grid, iso_pred, "-", color=color, lw=2.0, label="isotonic")
    if valid.any():
        sizes = np.clip(bin_n[valid] / max(bin_n[valid].max(), 1) * 80, 8, 80)
        ax.scatter(bin_centers[valid], bin_acc[valid], s=sizes, color=color,
                   edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
    rug_y = -0.04
    ax.plot(conf, np.full_like(conf, rug_y), "|", color=color, alpha=0.05, ms=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.08, 1.0)
    ax.set_title(f"{subject.replace('_', ' ')}\n(n={n_questions}, acc={mean_acc:.2f})", fontsize=9)
    ax.grid(alpha=0.3, linewidth=0.5)


def figure_top_or_bottom(model, kind, n_show=5):
    s = summaries[model].sort_values("ece", ascending=(kind == "bottom"))
    subjects = s.head(n_show)
    q = questions[model]
    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3.2), sharey=True)
    color = COLORS[model]
    for ax, (_, row) in zip(axes, subjects.iterrows()):
        sub = q[q["subject"] == row["subject"]]
        plot_calibration_panel(ax, sub["confidence"].values, sub["correct"].astype(float).values,
                               row["subject"], len(sub), row["mean_accuracy"], color=color)
    axes[0].set_ylabel("Empirical accuracy")
    for ax in axes:
        ax.set_xlabel("Confidence")
    label = "most miscalibrated" if kind == "top" else "best calibrated"
    fig.suptitle(f"{PRETTY[model]} — {n_show} {label} subjects (sorted by ECE)", fontsize=11, y=1.02)
    fig.tight_layout()
    out = FIG / f"{kind}5_{label.replace(' ', '_')}_{model}_6m.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


figure_top_or_bottom("gpt4o", "top", 5)
figure_top_or_bottom("gpt4o", "bottom", 5)


# ============================================================
# Figure 2: entropy mechanism (6-panel main, with appendix Qwen-1.5B 4-bit comparison)
# ============================================================
def figure_entropy_mechanism(model_list, suffix, ncols=None):
    n = len(model_list)
    if ncols is None:
        ncols = n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 3.2 * nrows), sharey=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    quartile_labels = ["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"]

    for ax, m in zip(axes, model_list):
        q = questions[m].copy()
        q["q"] = pd.qcut(q["entropy"], 4, labels=[0, 1, 2, 3])
        agg = q.groupby("q", observed=True).agg(
            conf=("confidence", "mean"),
            acc=("correct", "mean"),
        ).reindex([0, 1, 2, 3])
        x = np.arange(4)
        gap_top = agg["conf"].values
        gap_bot = agg["acc"].values

        ax.fill_between(x, gap_bot, gap_top, color=COLORS[m], alpha=0.18, label="gap")
        ax.plot(x, gap_top, "-o", color=COLORS[m], lw=2.0, ms=6, label="confidence")
        ax.plot(x, gap_bot, "--s", color=COLORS[m], lw=1.5, ms=5, mfc="white",
                mew=1.5, label="accuracy")

        for i in [0, 3]:
            gap_size = gap_top[i] - gap_bot[i]
            mid = (gap_top[i] + gap_bot[i]) / 2
            ax.annotate(f"Δ={gap_size:+.2f}", (x[i], mid), fontsize=7,
                        xytext=(0, 0), textcoords="offset points",
                        ha="center", color=COLORS[m])

        ax.set_title(SHORT[m], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(quartile_labels)
        ax.set_xlabel("Entropy quartile")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, linewidth=0.5)
        if ax is axes[0]:
            ax.set_ylabel("Probability")
            ax.legend(loc="lower left", fontsize=7, framealpha=0.9)

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Confidence vs. accuracy by entropy quartile per model "
                 "(shaded = calibration gap)", fontsize=10, y=1.02)
    fig.tight_layout()
    out = FIG / f"entropy_mechanism_{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


figure_entropy_mechanism(MAIN_MODELS, "6m", ncols=6)
figure_entropy_mechanism(APPENDIX, "appendix_quant", ncols=2)


# ============================================================
# Figure 3a: 6x6 Spearman correlation HEATMAP
# ============================================================
def figure_corr_heatmap(model_list, suffix):
    common = sorted(set.intersection(*[set(summaries[m]["subject"]) for m in model_list]))
    ece_mat = pd.DataFrame({
        m: summaries[m].set_index("subject").loc[common, "ece"].values for m in model_list
    }, index=common)
    corr = ece_mat.corr(method="spearman")
    n = len(model_list)

    fig, ax = plt.subplots(figsize=(0.85 * n + 1.5, 0.85 * n + 1.0))
    im = ax.imshow(corr.values, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [SHORT[m] for m in model_list]
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            text_color = "white" if v < 0.6 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color=text_color)
    fig.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.85)
    ax.set_title(f"Cross-model Spearman correlation\n(subject-level ECE, {len(common)} subjects)",
                 fontsize=10)
    fig.tight_layout()
    out = FIG / f"cross_model_heatmap_{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


figure_corr_heatmap(MAIN_MODELS, "6m")


# ============================================================
# Figure 4: coefficient forest plot for the 6 main models
# ============================================================
def figure_coef_plot():
    coef = {
        "Intercept": {
            "gpt4o": (0.196, 0.016), "llama": (0.197, 0.010),
            "qwen_0_5b": (0.209, 0.010),
            "qwen_1_5b_fp16": (0.179, 0.009),
            "qwen_7b": (0.233, 0.014),
            "qwen_14b": (0.210, 0.014),
        },
        "Word count": {
            "gpt4o": (0.031, 0.006), "llama": (-0.005, 0.006),
            "qwen_0_5b": (-0.013, 0.007),
            "qwen_1_5b_fp16": (0.003, 0.006),
            "qwen_7b": (0.019, 0.006),
            "qwen_14b": (0.023, 0.006),
        },
        "Max option length": {
            "gpt4o": (0.001, 0.004), "llama": (-0.001, 0.005),
            "qwen_0_5b": (-0.028, 0.005),
            "qwen_1_5b_fp16": (0.005, 0.005),
            "qwen_7b": (-0.007, 0.005),
            "qwen_14b": (-0.007, 0.004),
        },
        "Negation": {
            "gpt4o": (-0.001, 0.010), "llama": (0.007, 0.011),
            "qwen_0_5b": (0.016, 0.012),
            "qwen_1_5b_fp16": (-0.010, 0.011),
            "qwen_7b": (-0.001, 0.011),
            "qwen_14b": (0.005, 0.010),
        },
        "Entropy": {
            "gpt4o": (0.016, 0.004), "llama": (0.008, 0.004),
            "qwen_0_5b": (-0.063, 0.004),
            "qwen_1_5b_fp16": (-0.004, 0.004),
            "qwen_7b": (0.029, 0.004),
            "qwen_14b": (0.040, 0.003),
        },
    }
    predictors = list(coef.keys())
    n_pred = len(predictors)
    offsets = np.linspace(-0.35, 0.35, len(MAIN_MODELS))

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for i, m in enumerate(MAIN_MODELS):
        xs = [coef[p][m][0] for p in predictors]
        ses = [coef[p][m][1] for p in predictors]
        ys = np.arange(n_pred) + offsets[i]
        ax.errorbar(xs, ys, xerr=[1.96 * s for s in ses], fmt="o",
                    color=COLORS[m], ms=5, capsize=2.5, label=SHORT[m])

    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.set_yticks(np.arange(n_pred))
    ax.set_yticklabels(predictors)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\hat{\beta}$ (with 95% CI; covariates standardized)")
    ax.set_title("Fixed effects from the multilevel model — six models", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = FIG / "coef_plot_6m.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


figure_coef_plot()


# ============================================================
# Figure 5: subject mean gap (6 panels, one per main model)
# ============================================================
def figure_subject_mean_gap():
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), sharey=True)
    axes = axes.flatten()
    for ax, m in zip(axes, MAIN_MODELS):
        f = fdr[m].copy().sort_values("mean_gap")
        x = np.arange(len(f))
        colors_ = ["#d62728" if r else "#7f7f7f" for r in f["reject"].values]
        ax.bar(x, f["mean_gap"].values, color=colors_, edgecolor="black", linewidth=0.3)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"{SHORT[m]}", fontsize=10)
        ax.set_xticks([])
        ax.set_xlabel(f"57 subjects (sorted)")
        if ax is axes[0] or ax is axes[3]:
            ax.set_ylabel("Mean calibration gap")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Subject-level mean calibration gap (red = FDR-rejected at α=0.05)",
                 fontsize=11, y=0.99)
    fig.tight_layout()
    out = FIG / "subject_mean_gap_6m.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


figure_subject_mean_gap()


# ============================================================
# Figure 6: Within-Qwen scaling — intercept, ICC, entropy coef as functions of scale
# ============================================================
def figure_qwen_scaling():
    # Manually pulled from multilevel output
    scale_data = pd.DataFrame({
        "model":      ["Qwen-1.5B", "Qwen-7B", "Qwen-14B"],
        "params_b":   [1.5, 7.0, 14.0],
        "intercept":  [0.179, 0.233, 0.210],
        "intercept_se": [0.009, 0.014, 0.014],
        "icc":        [0.017, 0.049, 0.059],
        "entropy":    [-0.004, 0.029, 0.040],
        "entropy_se": [0.004, 0.004, 0.003],
        "accuracy":   [0.571, 0.685, 0.757],
    })
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    # Intercept
    ax = axes[0]
    ax.errorbar(scale_data["params_b"], scale_data["intercept"],
                yerr=1.96 * scale_data["intercept_se"], fmt="o-", color="#9467bd",
                ms=8, capsize=4, lw=1.5)
    for _, row in scale_data.iterrows():
        ax.annotate(f"  {row['model']}", (row["params_b"], row["intercept"]),
                    fontsize=8, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (B, log scale)")
    ax.set_ylabel(r"Intercept $\hat\mu$")
    ax.set_title("Mean overconfidence", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks([1.5, 7, 14])
    ax.set_xticklabels(["1.5", "7", "14"])

    # ICC
    ax = axes[1]
    ax.plot(scale_data["params_b"], scale_data["icc"], "o-", color="#9467bd",
            ms=8, lw=1.5)
    for _, row in scale_data.iterrows():
        ax.annotate(f"  {row['icc']:.3f}", (row["params_b"], row["icc"]),
                    fontsize=8, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (B, log scale)")
    ax.set_ylabel(r"ICC$_\text{subject}$")
    ax.set_title("Between-subject variance share", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks([1.5, 7, 14])
    ax.set_xticklabels(["1.5", "7", "14"])

    # Entropy coef
    ax = axes[2]
    ax.errorbar(scale_data["params_b"], scale_data["entropy"],
                yerr=1.96 * scale_data["entropy_se"], fmt="o-", color="#9467bd",
                ms=8, capsize=4, lw=1.5)
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    for _, row in scale_data.iterrows():
        ax.annotate(f"  {row['entropy']:+.3f}", (row["params_b"], row["entropy"]),
                    fontsize=8, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (B, log scale)")
    ax.set_ylabel(r"Entropy coefficient $\hat\beta_\text{ent}$")
    ax.set_title("Entropy effect on calibration gap", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks([1.5, 7, 14])
    ax.set_xticklabels(["1.5", "7", "14"])

    fig.suptitle("Within-Qwen scaling: ICC and entropy coefficient grow monotonically with scale",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    out = FIG / "qwen_scaling_6m.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


figure_qwen_scaling()


print("\nAll figures saved with `_6m` suffix to report/figures/")
