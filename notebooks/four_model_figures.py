"""
Generate all figures needed for the four-model report revision.

Outputs to report/figures/ with `_4m` suffix to avoid clobbering existing figures
until the user confirms which to keep.
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

MODELS = ["gpt4o", "llama", "qwen_1_5b", "qwen_7b"]
PRETTY = {
    "gpt4o": "GPT-4o-mini",
    "llama": "Llama-3-8B-Instruct",
    "qwen_1_5b": "Qwen-2.5-1.5B-Instruct",
    "qwen_7b": "Qwen-2.5-7B-Instruct",
}
SHORT = {
    "gpt4o": "GPT-4o-mini",
    "llama": "Llama-3-8B",
    "qwen_1_5b": "Qwen-1.5B",
    "qwen_7b": "Qwen-7B",
}
COLORS = {
    "gpt4o": "#1f77b4",
    "llama": "#d62728",
    "qwen_1_5b": "#2ca02c",
    "qwen_7b": "#9467bd",
}

mpl.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

questions = {m: pd.read_parquet(PROC / f"questions_{m}.parquet") for m in MODELS}
summaries = {m: pd.read_parquet(PROC / f"subject_summaries_{m}.parquet") for m in MODELS}
fdr      = {m: pd.read_parquet(PROC / f"fdr_results_{m}.parquet") for m in MODELS}


# ============================================================
# Figure 1: top-5 miscalibrated GPT-4o-mini (truncated to data support)
# ============================================================
def plot_calibration_panel(ax, conf, corr, subject, n_questions, mean_acc, color="#1f77b4"):
    """Plot isotonic curve + scatter, masking extrapolated region."""
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(conf, corr)

    # Only plot curve over observed support
    p_lo, p_hi = np.percentile(conf, [1, 99])
    grid = np.linspace(p_lo, p_hi, 200)
    iso_pred = ir.predict(grid)

    # Empirical bin scatter (20 bins) — only show bins with data
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

    # Rug plot of confidence values
    rug_y = -0.04
    ax.plot(conf, np.full_like(conf, rug_y), "|", color=color, alpha=0.05, ms=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.08, 1.0)
    ax.set_title(f"{subject.replace('_', ' ')}\n(n={n_questions}, acc={mean_acc:.2f})",
                 fontsize=9)
    ax.grid(alpha=0.3, linewidth=0.5)


def figure_top_or_bottom(model, kind, n_show=5):
    """kind: 'top' (most miscalibrated) or 'bottom' (best calibrated)."""
    s = summaries[model].sort_values("ece", ascending=(kind == "bottom"))
    subjects = s.head(n_show)
    q = questions[model]

    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3.2), sharey=True)
    color = COLORS[model]
    for ax, (_, row) in zip(axes, subjects.iterrows()):
        sub = q[q["subject"] == row["subject"]]
        plot_calibration_panel(
            ax,
            sub["confidence"].values,
            sub["correct"].astype(float).values,
            row["subject"],
            len(sub),
            row["mean_accuracy"],
            color=color,
        )
    axes[0].set_ylabel("Empirical accuracy")
    for ax in axes:
        ax.set_xlabel("Confidence")
    label = "most miscalibrated" if kind == "top" else "best calibrated"
    fig.suptitle(f"{PRETTY[model]} — {n_show} {label} subjects (sorted by ECE)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = FIG / f"{kind}5_{label.replace(' ', '_')}_{model}_4m.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


figure_top_or_bottom("gpt4o", "top", 5)
figure_top_or_bottom("gpt4o", "bottom", 5)


# ============================================================
# Figure 2: entropy mechanism (4-panel)
# ============================================================
def figure_entropy_mechanism():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    quartile_labels = ["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"]

    for ax, m in zip(axes, MODELS):
        q = questions[m].copy()
        q["q"] = pd.qcut(q["entropy"], 4, labels=[0, 1, 2, 3])
        agg = q.groupby("q", observed=True).agg(
            conf=("confidence", "mean"),
            acc=("correct", "mean"),
        ).reindex([0, 1, 2, 3])
        x = np.arange(4)
        gap_top = agg["conf"].values
        gap_bot = agg["acc"].values

        # Shaded gap
        ax.fill_between(x, gap_bot, gap_top, color=COLORS[m], alpha=0.18, label="gap")
        ax.plot(x, gap_top, "-o", color=COLORS[m], lw=2.0, ms=6, label="confidence")
        ax.plot(x, gap_bot, "--s", color=COLORS[m], lw=1.5, ms=5, mfc="white",
                mew=1.5, label="accuracy")

        # Annotate Q1 and Q4 gap sizes
        for i in [0, 3]:
            gap_size = gap_top[i] - gap_bot[i]
            mid = (gap_top[i] + gap_bot[i]) / 2
            ax.annotate(f"Δ={gap_size:.2f}", (x[i], mid), fontsize=8,
                        xytext=(0, 0), textcoords="offset points",
                        ha="center", color=COLORS[m])

        ax.set_title(SHORT[m], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(quartile_labels)
        ax.set_xlabel("Entropy quartile")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, linewidth=0.5)
        if m == MODELS[0]:
            ax.set_ylabel("Probability")
            ax.legend(loc="lower left", fontsize=7, framealpha=0.9)

    fig.suptitle("Confidence vs. accuracy by entropy quartile — the gap between solid (confidence) "
                 "and dashed (accuracy) is the calibration error",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    out = FIG / "entropy_mechanism_4m.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


figure_entropy_mechanism()


# ============================================================
# Figure 3a: 4x4 Spearman correlation HEATMAP
# ============================================================
def figure_corr_heatmap():
    common = sorted(set.intersection(*[set(summaries[m]["subject"]) for m in MODELS]))
    ece_mat = pd.DataFrame({
        m: summaries[m].set_index("subject").loc[common, "ece"].values for m in MODELS
    }, index=common)
    corr = ece_mat.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    im = ax.imshow(corr.values, vmin=0.7, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    labels = [SHORT[m] for m in MODELS]
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    for i in range(4):
        for j in range(4):
            v = corr.values[i, j]
            text_color = "white" if v < 0.85 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=10, color=text_color)
    fig.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
    ax.set_title("Cross-model Spearman correlation\n(subject-level ECE, 57 subjects)",
                 fontsize=10)
    fig.tight_layout()
    out = FIG / "cross_model_heatmap_4m.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ============================================================
# Figure 3b: 4x4 pairs plot (scatter on lower triangle, corr on upper)
# ============================================================
def figure_corr_pairsplot():
    common = sorted(set.intersection(*[set(summaries[m]["subject"]) for m in MODELS]))
    ece_mat = pd.DataFrame({
        m: summaries[m].set_index("subject").loc[common, "ece"].values for m in MODELS
    }, index=common)
    corr = ece_mat.corr(method="spearman")

    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for i, mi in enumerate(MODELS):
        for j, mj in enumerate(MODELS):
            ax = axes[i, j]
            if i == j:
                # Diagonal: model name
                ax.text(0.5, 0.5, SHORT[mi], ha="center", va="center",
                        fontsize=11, fontweight="bold", color=COLORS[mi],
                        transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
            elif i > j:
                # Lower triangle: scatter
                ax.scatter(ece_mat[mj], ece_mat[mi], s=14,
                           color=COLORS[mi], alpha=0.5, edgecolor="white", lw=0.4)
                ax.set_xlim(0, ece_mat.values.max() * 1.05)
                ax.set_ylim(0, ece_mat.values.max() * 1.05)
                ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.6, alpha=0.6)
                ax.tick_params(labelsize=7)
                if i < 3:
                    ax.set_xticklabels([])
                if j > 0:
                    ax.set_yticklabels([])
            else:
                # Upper triangle: correlation value
                rho = corr.values[i, j]
                ax.text(0.5, 0.5, f"ρ = {rho:.2f}", ha="center", va="center",
                        fontsize=12, transform=ax.transAxes,
                        color="black" if rho < 0.9 else "darkred")
                ax.set_xticks([])
                ax.set_yticks([])

    # Outer labels
    for j, mj in enumerate(MODELS):
        if j != 3:
            axes[3, j].set_xlabel(f"{SHORT[mj]} ECE")
    for i, mi in enumerate(MODELS):
        if i != 0:
            axes[i, 0].set_ylabel(f"{SHORT[mi]} ECE")

    fig.suptitle("Pairwise subject-level ECE across four models",
                 fontsize=11, y=0.995)
    fig.tight_layout()
    out = FIG / "cross_model_pairs_4m.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


figure_corr_heatmap()
figure_corr_pairsplot()


# ============================================================
# Figure 4: coefficient forest plot for all 4 models
# ============================================================
def figure_coef_plot():
    coef = {
        "Intercept":           {"gpt4o": ( 0.196, 0.016), "llama": ( 0.197, 0.010),
                                "qwen_1_5b": ( 0.157, 0.010), "qwen_7b": ( 0.233, 0.014)},
        "Word count":          {"gpt4o": ( 0.031, 0.006), "llama": (-0.005, 0.006),
                                "qwen_1_5b": ( 0.004, 0.006), "qwen_7b": ( 0.019, 0.006)},
        "Max option length":   {"gpt4o": ( 0.001, 0.004), "llama": (-0.001, 0.005),
                                "qwen_1_5b": (-0.002, 0.005), "qwen_7b": (-0.007, 0.005)},
        "Negation":            {"gpt4o": (-0.001, 0.010), "llama": ( 0.007, 0.011),
                                "qwen_1_5b": (-0.010, 0.011), "qwen_7b": (-0.001, 0.011)},
        "Entropy":             {"gpt4o": ( 0.016, 0.004), "llama": ( 0.008, 0.004),
                                "qwen_1_5b": (-0.010, 0.004), "qwen_7b": ( 0.029, 0.004)},
    }
    predictors = list(coef.keys())
    n_pred = len(predictors)
    offsets = np.linspace(-0.30, 0.30, len(MODELS))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, m in enumerate(MODELS):
        xs = [coef[p][m][0] for p in predictors]
        ses = [coef[p][m][1] for p in predictors]
        ys = np.arange(n_pred) + offsets[i]
        ax.errorbar(xs, ys, xerr=[1.96 * s for s in ses], fmt="o",
                    color=COLORS[m], ms=5, capsize=3, label=SHORT[m])

    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.set_yticks(np.arange(n_pred))
    ax.set_yticklabels(predictors)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\hat{\beta}$ (with 95% CI; covariates standardized)")
    ax.set_title("Fixed effects from the multilevel model — four models",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = FIG / "coef_plot_4m.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


figure_coef_plot()


# ============================================================
# Figure 5: subject mean gap (4 panels, one per model)
# ============================================================
def figure_subject_mean_gap():
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)

    for ax, m in zip(axes, MODELS):
        f = fdr[m].copy().sort_values("mean_gap")
        x = np.arange(len(f))
        colors_ = ["#d62728" if r else "#7f7f7f" for r in f["reject"].values]
        ax.bar(x, f["mean_gap"].values, color=colors_, edgecolor="black", linewidth=0.3)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(SHORT[m], fontsize=10)
        ax.set_xticks([])
        ax.set_xlabel(f"57 subjects (sorted)")
        if m == MODELS[0]:
            ax.set_ylabel("Mean calibration gap")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Subject-level mean calibration gap (red = FDR-rejected at α=0.05; "
                 "gray = not rejected)", fontsize=10, y=1.02)
    fig.tight_layout()
    out = FIG / "subject_mean_gap_4m.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


figure_subject_mean_gap()

print("\nAll figures saved with `_4m` suffix to report/figures/")
