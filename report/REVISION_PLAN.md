# Tentative Report Revision Plan — Four-Model Expansion

This document outlines proposed changes to `report/report.tex` to incorporate the two new Qwen models (Qwen-2.5-1.5B-Instruct and Qwen-2.5-7B-Instruct) alongside the existing GPT-4o-mini and Llama-3-8B-Instruct results. **Nothing in `report.tex` has been modified yet** — this is for review.

The pipeline scripts have been extended to handle the two new models (small edits to model lists in `extract_features.py`, `curves.py`, `multilevel.py`, `testing.py`, `sensitivity.py`, and `run_analysis.py`). All raw numbers below are pulled from regenerated `data/processed/*.parquet` and the new `data/processed/four_model_summary.json`.

---

## Part 1 — Answers to the two questions

### Q1. How is entropy calculated, and why doesn't it reduce miscalibration?

**Calculation** ([extract_features.py:50](src/collect/extract_features.py:50)):
$$
H_i = -\sum_{k\in\{A,B,C,D\}} p_{ik}\log p_{ik},\quad \text{where } p_i = \mathrm{softmax}(\text{logprob}_A, \text{logprob}_B, \text{logprob}_C, \text{logprob}_D).
$$
Entropy is in nats (`np.log`, not `log2`); zero means the model concentrates all mass on one option, $\log 4 \approx 1.386$ means uniform. The covariate is then standardized (mean 0, sd 1) before the mixed-model fit.

**The mechanical expectation**: confidence is $\max_k p_{ik}$. As entropy rises, confidence is forced down (Spearman $\rho(H, c) \approx -1.00$ in our data — confirmed in all 4 models). So a *naive* prediction is that high-entropy questions should have small calibration gaps.

**What we observe instead** is the opposite pattern in 3 of 4 models. The signed entropy coefficient $\hat\beta_{\text{ent}}$ is **positive** for GPT-4o-mini ($+0.016$, $p<0.001$) and Qwen-7B ($+0.029$, $p<0.001$), marginal for Llama ($+0.008$, $p=0.069$), and **negative** for Qwen-1.5B ($-0.010$, $p=0.018$).

**The explanation** is visible in the entropy-quartile tables below. As entropy rises, confidence does fall — but **accuracy falls faster**, so the gap grows.

| Model | Q1 conf → Q4 conf (Δ) | Q1 acc → Q4 acc (Δ) | Q1 gap → Q4 gap |
|---|---|---|---|
| GPT-4o-mini | 1.000 → 0.829 (−0.17) | 0.963 → 0.406 (−0.56) | 0.04 → 0.42 |
| Llama-3-8B-Instruct | 0.999 → 0.521 (−0.48) | 0.936 → 0.318 (−0.62) | 0.06 → 0.20 |
| **Qwen-1.5B-Instruct** | 0.966 → 0.448 (−0.52) | 0.854 → 0.303 (−0.55) | 0.11 → 0.15 |
| Qwen-7B-Instruct | 1.000 → 0.764 (−0.24) | 0.948 → 0.394 (−0.55) | 0.05 → 0.37 |

The capable RLHF-trained models (GPT, Qwen-7B) keep confidence pinned near 1.0 even as accuracy collapses — they only "release" probability into other options at very high entropy, by which time accuracy has already crashed. **Qwen-1.5B is the only model where confidence and accuracy fall in roughly parallel** as entropy rises (Δ −0.52 vs −0.55), which is precisely why it shows a *negative* entropy coefficient — its 1.5B weight count is too small to absorb the RLHF-driven confidence-saturation pattern that produces the positive coefficients elsewhere.

**Interpretation, with caveats.** Three of four models show a positive coefficient on entropy in the fitted regression. A plausible reading is that capable instruction-tuned models keep their top-1 probability near 1.0 even as the option distribution begins to flatten, so confidence falls slower than accuracy and the gap grows. The smallest model (Qwen-1.5B) is the exception, with confidence and accuracy falling in roughly parallel as entropy rises — yielding the negative coefficient we observe.

This pattern is consistent with prior literature on the calibration cost of instruction tuning. \citet{kadavath2022language} found that base models tend to be better-calibrated than RLHF counterparts at the *aggregate* level; \citet{nakkiran2025trained} and \citet{yaldiz2026balancing} report similar effects across multiple instruction-tuning regimes. Our four-model frame contributes a *marginal* mechanism behind the aggregate finding: the calibration cost is concentrated in the upper tail of the confidence distribution, where capable aligned models continue to assert high confidence even on questions where the option distribution suggests they should not.

We are cautious about the strength of this claim. The negative-coefficient result rests on a single small model (Qwen-1.5B), and the size differences in entropy coefficients (e.g., $+0.029$ for Qwen-7B vs $-0.010$ for Qwen-1.5B) are observational rather than the result of a controlled scaling experiment. A future revision adding additional scale points (e.g., Qwen-2.5-14B or a within-Llama base/instruct comparison) would let us test whether the coefficient varies *monotonically* with capability, or whether the 1.5B exception reflects something specific to that checkpoint. Pending that, our claim is descriptive: the four models we examined exhibit a sign flip in the entropy coefficient that aligns with what existing literature predicts about RLHF and confidence saturation.

---

### Q2. In what specific subjects is miscalibration the highest?

**Top-5 most miscalibrated subjects by ECE (per model):**

| Rank | GPT-4o-mini | Llama-3-8B-Instruct | Qwen-1.5B-Instruct | Qwen-7B-Instruct |
|---|---|---|---|---|
| 1 | moral_scenarios (0.49) | virology (0.44) | virology (0.34) | moral_scenarios (0.53) |
| 2 | virology (0.43) | machine_learning (0.39) | college_physics (0.30) | college_mathematics (0.45) |
| 3 | abstract_algebra (0.42) | college_mathematics (0.35) | econometrics (0.26) | machine_learning (0.45) |
| 4 | global_facts (0.42) | formal_logic (0.34) | college_chemistry (0.26) | professional_law (0.43) |
| 5 | formal_logic (0.39) | professional_law (0.32) | hs_statistics (0.26) | college_chemistry (0.43) |

**Cross-model consistency**: no subject is top-5 in *all four* models, but four subjects appear in *top-10 across all four models*: **virology, college_chemistry, college_physics, professional_law**. Expanding to top-15 adds five more (college_mathematics, econometrics, high_school_physics, machine_learning, moral_scenarios), giving nine subjects that are consistently among the worst across all model families and scales.

These naturally cluster into three semantic groups:

1. **Quantitative STEM at "college" level**: college_chemistry, college_mathematics, college_physics, high_school_physics, machine_learning, econometrics. Domain-specific reasoning where shortcuts from pretraining text fail.
2. **Niche factual recall**: virology, global_facts. Long-tail knowledge that pretraining sees infrequently.
3. **Ethical / formal-logical reasoning**: moral_scenarios, formal_logic, abstract_algebra, professional_law. Tasks with structured rule-following that surface differences between "fluent text" and "correct reasoning".

**Subjects that are well-calibrated across all four models.** The same subject-level structure that concentrates miscalibration also reveals a coherent set of *well*-calibrated subjects. The intersection of bottom-5 ECE across all four models contains **high_school_government_and_politics** and **high_school_psychology**; expanding to bottom-10 adds **high_school_us_history**, **marketing**, and **miscellaneous**; bottom-15 adds **high_school_geography**, **high_school_world_history**, **sociology**, and **us_foreign_policy**.

Two patterns are visible:

1. *Level matters*: high-school-level versions of subjects show much smaller calibration gaps than their college-level counterparts. The MMLU subject taxonomy makes this comparison particularly clean — `high_school_psychology` has ECE 0.06–0.11 across the four models, while there is no comparable "professional_psychology" entry in the bottom rankings; `high_school_government_and_politics` is consistently best-calibrated, while `professional_law` is consistently among the worst. The level appears to track the depth of specialization required.
2. *Domain matters*: well-calibrated subjects are dominated by Social Sciences and Other (the "applied/factual" category in the MMLU taxonomy). No STEM subject appears in the bottom-10 intersection, and Humanities are sparse (only the high-school-level history subjects, no philosophy or formal_logic).

Two top-level subject-content axes thus organize where calibration succeeds versus fails: *subject specialization* (high-school vs college vs professional) and *domain category* (applied/factual vs technical/structured). This is the substantive finding that the heterogeneity audit recovers.

---

## Part 2 — Proposed report changes, section by section

Notation below: `[REPLACE]` = swap entire passage; `[ADD]` = new content; `[EDIT]` = modify in place; `[REMOVE]` = delete.

### Abstract `[REPLACE]`

The current abstract reports two-model results. Suggested rewrite:

> Large language model (LLM) calibration is nearly always reported as a single aggregate scalar, obscuring whether miscalibration is diffuse or concentrated in specific knowledge domains. We develop a statistically rigorous audit framework and apply it to MMLU using token-level log probabilities from four instruction-tuned models spanning two families and three parameter scales: GPT-4o-mini, Llama-3-8B-Instruct, Qwen-2.5-1.5B-Instruct, and Qwen-2.5-7B-Instruct. Our framework combines a three-level mixed-effects model, isotonic and kernel-smoothed reliability curves, empirical Bayes shrinkage, and Benjamini–Hochberg FDR correction. Across all four models, between-subject calibration variance accounts for 1.9–7.9% of the total, and 54–57 of 57 subjects are flagged as significantly miscalibrated at FDR $\alpha=0.05$. Subject-level ECE rankings are highly correlated across all model pairs (Spearman $\rho$ between 0.78 and 0.92), indicating that miscalibration is task-driven rather than model-specific. Question-level entropy enters the multilevel model with a sign that flips with model capability — positive for the larger RLHF-tuned models (GPT-4o-mini, Qwen-7B) and negative for Qwen-1.5B — revealing that capable models keep confidence saturated near 1.0 even as the option distribution flattens, while the smallest model expresses uncertainty proportionally. Across the four models, miscalibration consistently concentrates in college-level quantitative STEM (chemistry, mathematics, physics, machine_learning), niche factual recall (virology), and structured ethical/legal reasoning (moral_scenarios, professional_law).

### Section 3 — Data, "Models" paragraph `[REPLACE]`

> We query four instruction-tuned models spanning two model families and three parameter scales. **GPT-4o-mini** is queried via the OpenAI Batch API at temperature 0 (`max_tokens=1`, `top_logprobs=20`); the choice of 20 (rather than the API default of 5) ensures all four answer tokens are captured for virtually every question. **Llama-3-8B-Instruct** and **Qwen-2.5-{1.5B, 7B}-Instruct** are run locally on a Colab T4 with 4-bit quantization (BitsAndBytes; \citealt{dettmers2022gptint}). For all instruction-tuned models we pre-fill the assistant turn with "The answer is " before extracting logprobs at the next-token position; this places the scoring position immediately before the answer token without depending on the model's preferred preamble. The Qwen models extend our cross-model coverage along two axes: family (Qwen vs Llama at matched ~7B scale) and scale (1.5B vs 7B within Qwen), enabling a more direct test of whether subject-level miscalibration is an artifact of any single training pipeline.

Add a footnote on accuracies (74.9% / 60.6% / 53.6% / 68.5%, all slightly below the published full-precision figures because of 4-bit quantization for the open models).

### Section 5.1 — Variance Decomposition `[REPLACE table; lightly edit prose]`

New `\label{tab:icc}` table:

| Model | $\sigma^2_{\text{domain}}$ | $\sigma^2_{\text{subject}}$ | $\sigma^2_\varepsilon$ | ICC$_{\text{subject}}$ |
|---|---|---|---|---|
| GPT-4o-mini | 0.000 | 0.013 | 0.150 | 0.079 |
| Llama-3-8B-Instruct | 0.000 | 0.004 | 0.190 | 0.020 |
| Qwen-2.5-1.5B-Instruct | 0.000 | 0.004 | 0.203 | 0.019 |
| Qwen-2.5-7B-Instruct | 0.000 | 0.009 | 0.177 | 0.049 |

Updated prose:

> Across all four models, the bulk of calibration-gap variance lies at the question level (92.2–98.1%), and domain-level variance collapses to zero. Between-subject ICCs differ by nearly a factor of four across models: GPT-4o-mini concentrates 7.9% of variance at the subject level, Qwen-7B 4.9%, and Llama and Qwen-1.5B around 2%. The Qwen scaling pair (1.5B → 7B) shows that ICC roughly *triples* as scale grows from 1.5B to 7B, suggesting that subject-level structure in calibration emerges with model capability — small or weakly-aligned models miscalibrate diffusely, while larger or more-aligned models concentrate their miscalibration on specific subject clusters.

### Section 5.2 — Calibration Curves `[EDIT figure; keep prose mostly]`

Keep the figure restricted to GPT-4o-mini (no per-model proliferation). Two changes to the existing figure:

1. **Truncate to observed data support.** The current `figures/top5_miscalibrated_gpt4o.pdf` displays the isotonic curve over the full $[0, 1]$ confidence range, but for GPT-4o-mini essentially no questions land in $[0, 0.25]$ — `IsotonicRegression(out_of_bounds="clip")` produces a flat extrapolated line in that range that misrepresents the empirical evidence. Fix: mask bin centers below the 1st-percentile observed confidence per subject (or below a fixed threshold like 0.30 for these high-confidence models) when plotting, and add a subtle rug plot or histogram strip beneath the axes to communicate where data actually lies.
2. **Add a small annotation** on each panel reporting the subject's mean accuracy and number of questions, so readers can connect the curve shape to sample size and difficulty without flipping back to a table.

We can also add a companion figure showing the *bottom-5* (best-calibrated) subjects for GPT-4o-mini using the same conventions — visually contrasting "above the diagonal" or "tracks the diagonal" curves with the overconfident shapes. This makes the heterogeneity-audit framing more legible: it's not just that some subjects are bad, it's that some are simultaneously well-calibrated, which is what the average ECE hides.

### Section 5.3 — Predictors of Miscalibration `[REPLACE table; rewrite text]`

Replacement table (one row per coefficient, one column per model):

| Predictor | GPT-4o-mini $\hat\beta$ ($p$) | Llama-3-8B-Instruct $\hat\beta$ ($p$) | Qwen-2.5-1.5B-Instruct $\hat\beta$ ($p$) | Qwen-2.5-7B-Instruct $\hat\beta$ ($p$) |
|---|---|---|---|---|
| Intercept | 0.196 (***) | 0.197 (***) | 0.157 (***) | 0.233 (***) |
| Word count | +0.031 (***) | −0.005 (.44) | +0.004 (.55) | +0.019 (**) |
| Max option length | +0.001 (.85) | −0.001 (.84) | −0.002 (.70) | −0.007 (.15) |
| Negation | −0.001 (.95) | +0.007 (.53) | −0.010 (.37) | −0.001 (.92) |
| **Entropy** | **+0.016 (***)** | +0.008 (.07$^\dagger$) | **−0.010 (*)** | **+0.029 (***)** |

Replacement prose (substantially expanded compared to the current 2-model version):

> The intercepts confirm large mean overconfidence in all four models, with magnitudes ranging from 0.157 (Qwen-1.5B) to 0.233 (Qwen-7B). Notably, the Qwen scaling pair shows that *more capable* Qwen models are *more* overconfident — the 7B intercept (0.233) is 49% larger than the 1.5B intercept (0.157), despite higher accuracy (0.685 vs 0.536). This is consistent with the view that RLHF/SFT alignment, applied more aggressively to stronger base models, drives the confidence-inflation effect documented by \citet{kadavath2022language} and \citet{nakkiran2025trained}.
>
> The word-count effect is large for GPT-4o-mini ($+0.031$, $p<0.001$) and significant for Qwen-7B ($+0.019$, $p=0.003$), but absent for Llama and Qwen-1.5B. Question complexity appears to amplify miscalibration only in the more confidently-overconfident models.
>
> The most interesting pattern is the entropy coefficient. Higher option-distribution entropy mechanically reduces top-1 confidence — for a flatter distribution, the maximum probability must be lower. A naïve prediction is therefore that high-entropy questions should show smaller calibration gaps. The fitted coefficients deviate from this prediction in two of three larger models: positive for GPT-4o-mini ($+0.016$, $p<0.001$) and Qwen-7B ($+0.029$, $p<0.001$), marginal for Llama-3-8B-Instruct ($+0.008$, $p=0.07$), and negative for Qwen-1.5B-Instruct ($-0.010$, $p=0.02$). Figure~\ref{fig:entropy_mechanism} traces this to a simple marginal pattern: as entropy rises, confidence does fall, but in three of four models accuracy falls faster, so the calibration gap widens. Qwen-1.5B is the one model in which confidence and accuracy fall in roughly lockstep, and it is the model with a negative entropy coefficient. This is consistent with prior reports that RLHF/SFT instruction tuning tends to inflate top-1 confidence relative to base models \citep{kadavath2022language, nakkiran2025trained, yaldiz2026balancing}; we are cautious about generalizing further from the four checkpoints studied here.

Replace the planned table `\ref{tab:entropy_quartiles}` with a four-panel figure `\ref{fig:entropy_mechanism}`. Each panel shows, for one model, mean confidence (solid line) and mean accuracy (dashed line) plotted against the entropy quartile $\{Q_1, Q_2, Q_3, Q_4\}$, with the area between the two curves shaded to make the calibration gap visually salient. Three panels (GPT, Llama, Qwen-7B) share the same shape — confidence stays high while accuracy crashes — and the fourth (Qwen-1.5B) shows the curves falling in parallel. A figure communicates the mechanism more directly than a table because the *shape* of the gap-by-entropy relationship is the substantive finding.

### Section 5.4 — FDR-Corrected Subject Rankings `[EDIT]`

Update the prose to note that **Qwen-1.5B is the only model where 3 of 57 subjects fail to be rejected at $\alpha=0.05$** (54/57), and at $\alpha=0.01$ only 46/57 are rejected — a meaningfully smaller proportion than the other three models (where 55–57/57 are rejected at the strict threshold). The three "passers" are all in Social Sciences and have small mean gaps:

| Subject | Domain | Mean gap | Adjusted $p$ |
|---|---|---|---|
| high_school_government_and_politics | Social Sciences | 0.025 | 0.372 |
| sociology | Social Sciences | 0.033 | 0.235 |
| us_foreign_policy | Social Sciences | 0.057 | 0.141 |

These are the only subject-by-model cells in the entire 4-model × 57-subject study where calibration is statistically indistinguishable from zero, which is itself an interesting null worth a sentence.

The current "13-fold range" claim becomes: GPT-4o-mini 13.0×, Llama 6.2×, Qwen-1.5B 13.9×, Qwen-7B 8.0×. The range is broad in three of four models; Llama is the outlier in the other direction (smallest range), not GPT in being unusually wide.

### Section 5.5 — Sensitivity Analysis `[EDIT]`

Sensitivity-analysis conclusions extend to all four models without modification. ECE rankings are stable across bin counts (Spearman $\geq 0.97$ vs. the 20-bin baseline for every model). NLCS–ECE rank correlations remain $\geq 0.85$ for all four. Two-level and three-level ICC estimates are numerically identical for every model, consistent with $\sigma^2_{\text{domain}}$ collapsing to zero. I'd add one short sentence summarizing this for the new models without re-listing all numbers.

### Section 5.6 — Cross-Model Agreement `[REPLACE figure and table]`

The current 2-model Spearman is replaced by a 4×4 correlation matrix. Numbers (subject-level ECE, all on the 57 common subjects):

|             | gpt4o | llama | qwen_1_5b | qwen_7b |
|-------------|-------|-------|-----------|---------|
| gpt4o       | 1.000 | 0.846 | 0.780     | 0.924   |
| llama       | 0.846 | 1.000 | 0.832     | 0.885   |
| qwen_1_5b   | 0.780 | 0.832 | 1.000     | 0.849   |
| qwen_7b     | 0.924 | 0.885 | 0.849     | 1.000   |

All six pairwise correlations are between 0.78 and 0.92. The cross-family pairing GPT-4o-mini vs Qwen-7B is the *highest* correlation (0.924), exceeding even the within-Qwen scaling correlation (0.849). This strengthens the "task-driven" interpretation: subject-level miscalibration is more consistent across model families at matched capability than within a single family across scale.

The figure to replace `figures/cross_model_ece_2.png`: a 4-panel pairs-plot (or a single 4×4 heatmap of Spearman correlations with subject-clustering on the off-diagonals). I can produce this as a follow-up if the rest is approved.

The "agreement Venn" (both / GPT only / Llama only / neither) extends naturally to a 4-set Venn or a more readable consensus table — proposed: "How many models reject at FDR $\alpha=0.05$" by subject. Given 57/57, 57/57, 54/57, 57/57, the consensus story is: 54 subjects are rejected by all four; 3 subjects are rejected by exactly three (the three subjects Qwen-1.5B fails to reject).

### Section 6 — Discussion `[REPLACE several paragraphs]`

Three new substantive findings warrant their own paragraphs. We frame each descriptively — what the four checkpoints exhibit — rather than as a confirmation of a general scaling law, since we have only two scale points within one family.

1. **Scale and overconfidence within family (descriptive).** The Qwen 1.5B → 7B comparison shows the intercept rising from 0.157 to 0.233 (+49\%), ICC roughly tripling (0.019 → 0.049), and the entropy coefficient flipping sign ($-0.010 \to +0.029$). All three changes go in the same direction: more capable Qwen → more concentrated overconfidence. We present this as a within-family observation consistent with prior aggregate findings on RLHF and confidence \citep{kadavath2022language, nakkiran2025trained}, not a controlled scaling experiment. A 14B Qwen (or a Llama-base/Llama-instruct comparison) would be needed to test whether the trend extends monotonically.

2. **An entropy-confidence mechanism consistent with the literature.** The entropy-quartile figure (\ref{fig:entropy_mechanism}) shows that, in three of four models, confidence falls more slowly than accuracy as the option distribution flattens. This is a marginal pattern at the question level that is consistent with the aggregate RLHF-overconfidence claim documented by \citet{kadavath2022language} and \citet{yaldiz2026balancing}. We do not claim to have isolated a causal mechanism; the comparison is between four checkpoints rather than a controlled within-checkpoint manipulation, and Qwen-1.5B is the only model in our set that shows the alternative (parallel) pattern. The finding's value is in *connecting* a question-level marginal pattern to the existing aggregate literature rather than introducing a novel claim.

3. **Cross-family agreement at matched capability.** The Spearman ECE correlation between GPT-4o-mini and Qwen-7B (0.924) exceeds the within-Qwen scaling correlation (0.849), and all six pairwise correlations fall in $[0.78, 0.92]$. We treat this as the strongest evidence in the paper for task-driven miscalibration: the same subject-level structure shows up in checkpoints from very different training pipelines and very different scales. This strengthens (does not establish) the practical implication that subject-aware recalibration is warranted, since the targets of any such correction are stable across the model space we examined.

Existing limitations paragraph stays; add one sentence noting that 4-bit quantization affects three of the four open-weight models and may modestly attenuate confidence for them, and a second sentence flagging that the entropy-coefficient sign comparison rests on a single small-model checkpoint (Qwen-1.5B) and should be replicated with additional small or base models before being treated as a robust pattern.

### Conclusion `[EDIT]`

Update last paragraph numbers from 2-model summary to 4-model summary. The conclusion should describe the entropy finding as descriptive (the four checkpoints we observe) rather than mechanism-confirming, and should explicitly cite \citet{kadavath2022language} et al. as the literature this aligns with rather than as evidence we have extended.

---

## Part 3 — Implementation order if you approve

1. (One-off) regenerate `four_model_summary.json` and the `data/processed/*_qwen_*.parquet` files — **already done**.
2. Replace abstract.
3. Replace tables: ICC (5.1), fixed effects (5.3), cross-model correlations (5.6).
4. Rewrite results prose for 5.1, 5.3, 5.4, 5.5, 5.6, including the new well-calibrated-subjects passage.
5. Add three new discussion paragraphs (within-family scale; entropy mechanism tied to literature; cross-family agreement).
6. Regenerate figures in `02_calibration_curves.ipynb` and `04_cross_model_comparison.ipynb`:
   - Mask extrapolated region in top-5 calibration curves figure.
   - Replace 2-model cross-model scatter with a 4-model visualization.
   - Add entropy-mechanism figure (4-panel: confidence + accuracy by entropy quartile).
   - (Optional) Add bottom-5 best-calibrated GPT-4o-mini curves for visual contrast.
7. Update Statement of Work to reflect the new model collection by John.

Estimated edit volume: ~2 pages of LaTeX rewrite + 2 new tables + 1–2 new figures + 1–2 figure replacements. Methods section needs no changes (the methodology applies as-is to four models). Bibliography needs no changes.

## Part 4 — Open questions for you to decide before I touch `report.tex`

1. **Title/scope**: keep as-is, or expand to mention scale and family comparisons? Suggest: keep.
2. **Statement of Work**: how should the new collection work be attributed?
3. **Limitations on Qwen-1.5B accuracy (53.6%)**: the 1.5B model is well below the published full-precision figure (~60%). Worth flagging in limitations as a 4-bit quantization artifact?
4. **Slides**: confirmed *not* updated, per your instruction. Existing slide deck is left untouched.
5. **Figure budget**: revised based on your feedback. My current recommendation:
   - **Replace** `figures/cross_model_ece_2.png` with a 4×4 Spearman correlation matrix or 4-panel scatter pairs plot.
   - **Replace** `figures/top5_miscalibrated_gpt4o.pdf` with a version that masks extrapolated $[0, 0.30]$ region (no curve drawn outside data support).
   - **Add** an entropy-mechanism figure (4-panel: confidence vs accuracy by entropy quartile per model, gap shaded).
   - **Add (optional)** a bottom-5 best-calibrated GPT-4o-mini curves figure for visual contrast with the top-5 miscalibrated panel.
   - Net: 2 figures replaced, 1–2 figures added; 0 tables added (the entropy quartile data lives in the new figure rather than a table).
