# STAT 215B Final Project — Plan & Progress Tracker

**Title**: A Statistical Audit of LLM Calibration Heterogeneity Across Knowledge Domains

---

## Overview

LLM calibration is nearly always reported as a single aggregate scalar. This project asks whether that aggregate masks systematic miscalibration in specific knowledge domains, developing a statistically rigorous framework to detect and characterize that heterogeneity. The framing is analogous to bias auditing in the fairness literature: just as reporting a single accuracy number across demographic groups is insufficient for a fairness audit, reporting aggregate ECE is insufficient for a calibration audit.

**Methods**: multilevel regression, calibration curve clustering, empirical Bayes shrinkage, multiple testing correction  
**Benchmark**: MMLU (~14,000 questions, 57 subjects, 4 domains)  
**Models**: GPT-4o-mini (OpenAI API, top_logprobs=20) and Llama-3-8B-Instruct (HuggingFace, 4-bit quantized on Colab T4)

---

## Research Questions

1. How much of the variance in LLM calibration error is attributable to domain vs. subject vs. individual question?
2. Which question- and subject-level features predict the direction and magnitude of miscalibration?
3. Are calibration patterns consistent across models, or model-specific?

---

## Pipeline Steps

> **Legend**: ✅ code written | ⏳ needs to be run | ❌ not started

### Step 1: Confidence Elicitation — ✅ COMPLETE (GPT), ⏳ IN PROGRESS (Llama)

- [x] `src/collect/download_mmlu.py` — downloads `cais/mmlu`, outputs `data/raw/mmlu.parquet` ✅
- [x] `src/collect/query_gpt4o.py` — GPT-4o-mini via OpenAI Batch API, `top_logprobs=20`, output: `data/raw/gpt4o_responses_top20.parquet` ✅ **DONE**
- [x] `notebooks/colab_llama.ipynb` — Llama-3-8B-Instruct on Colab T4, 4-bit quantization, chat template prompt, output: `data/raw/llama_instruct_responses.parquet` ⏳ **RUNNING ON COLAB**
- **Model change (2026-04-26)**: Switched from Llama-3-8B base → Llama-3-8B-Instruct so both models use instruction-following format and comparable confidence distributions. Base model spreads probability over non-answer continuations ("The answer is A", etc.), making renormalized P(A|{A,B,C,D}) incomparable to GPT.
- **Methodology fix (2026-04-26)**: Changed GPT querying from `top_logprobs=5` to `top_logprobs=20`. With top_logprobs=5, 55% of questions had at least one answer token missing (assigned -100 sentinel), inflating renormalized confidence artificially. top_logprobs=20 captures all four answer tokens reliably.
- **Confidence measure**: Both models use softmax-renormalized P(answer | {A,B,C,D}) — this is the right quantity for ECE (only the probability on the chosen answer matters) and is now comparable across models.

### Step 2: Outcome and Features — ✅ COMPLETE (GPT), ⏳ pending Llama

- [x] `src/collect/extract_features.py` ✅ **RUN**
  - [x] Signed calibration gap: $\text{gap}_i = c_i - y_i$
  - [x] Word count
  - [x] Max option length *(proposal said "average and max"; only max implemented)*
  - [x] Negation indicator (`not`, `except`, `least`, `never`, `no`)
  - [x] Entropy of the A/B/C/D probability distribution
  - [x] Confidence ($c_i$ = max renormalized softmax prob over {A,B,C,D})
- **Decision (2026-04-23)**: Use signed calibration gap as primary outcome. Reasons: sign directly answers RQ2 (over vs. underconfidence), random effects are interpretable as subject-level miscalibration, and BH test of H₀: mean gap = 0 is exactly the calibration audit question.
- **Limitation to note**: variance of $c_i - y_i$ is not constant across subjects (harder subjects have noisier outcomes since $\text{var}(y_i) = p_j(1-p_j)$); mixed model assumes homoskedastic residuals — check residual plots after fitting.

### Step 3: Nonparametric Calibration Curves — ✅ COMPLETE (GPT), ⏳ pending Llama

- [x] `src/calibration/curves.py` ✅ **RUN**
  - [x] Isotonic regression per subject (monotone fit of accuracy on confidence, sklearn)
  - [x] Kernel smoother per subject (Gaussian σ=1.5, unconstrained comparison)
  - [x] ECE per subject (integrated absolute deviation, 20-bin bucketing)
  - [x] Per-subject summary: n, ece, mean_gap, mean_confidence, mean_accuracy
- **Gap**: Divergence flagging between isotonic and kernel fits is not automated — check visually in `notebooks/02_calibration_curves.ipynb`

### Step 4: Multilevel Regression, Shrinkage, and Multiple Testing — ✅ COMPLETE (GPT), ⏳ pending Llama

- [x] `src/modeling/multilevel.py` — three-level mixed model via `statsmodels.mixedlm`, REML ✅ **RUN**

$$\text{outcome}_{ijk} = \mu + \mathbf{x}_{ijk}^\top \boldsymbol{\beta} + u_k + v_{jk} + \varepsilon_{ijk}$$

  where $u_k \sim \mathcal{N}(0, \sigma^2_\text{domain})$, $v_{jk} \sim \mathcal{N}(0, \sigma^2_\text{subject})$

  - [x] Fixed effects: `word_count`, `max_choice_len`, `has_negation`, `entropy` (all standardized)
  - [x] ICC variance decomposition ($\sigma^2_\text{domain}$, $\sigma^2_\text{subject}$, $\sigma^2_\varepsilon$) → answers RQ1
  - [x] Subject-level BLUPs ($\hat{v}_{jk}$) as shrunken calibration estimates, output to `data/processed/blups_{model}.parquet`
- [x] `src/modeling/testing.py` — one-sample t-test per subject + BH FDR correction ✅ **RUN**
  - [x] H₀: mean calibration gap = 0 per subject; BH at α=0.05 with monotonicity enforcement
  - [x] Output: `data/processed/fdr_results_{model}.parquet` sorted by adjusted p-value
- **GPT-4o-mini results (2026-04-26)**:
  - Intercept = 0.196 (p<0.001) — strong uniform overconfidence driven by RLHF
  - Significant predictors: `word_count` (β=0.031, p<0.001), `entropy` (β=0.016, p<0.001)
  - Non-significant: `max_choice_len`, `has_negation`
  - ICC: icc_domain=0.000, icc_subject=0.079, icc_residual=0.922
  - FDR: 57/57 subjects rejected — expected given intercept of 0.196 with ~250 questions/subject; substantive finding is heterogeneity in gap magnitude (range: 0.037 to 0.487)
- **Note on icc_domain=0**: With only 4 domain groups, statsmodels cannot reliably identify domain-level variance. Report total between-subject variance (7.9%) without clean domain/subject partition; note as limitation.

### Step 5: Cross-Model Replication — ⏳ pending Llama-Instruct data

- [x] `src/modeling/cross_model.py` ✅ — rewritten to recompute ECE from question-level data directly
  - [x] Spearman rank correlation of ECE across subjects, bootstrap CIs (2000 replicates, seed=42)
  - [x] Four-way agreement: both (task-driven) / GPT-4o only / Llama only / neither
  - [x] Output: `data/processed/cross_model_comparison.parquet`
- **Methodology fix (2026-04-26)**: cross_model.py now recomputes subject-level ECE and BH rejection from question-level parquets rather than relying on pre-aggregated summaries, ensuring the comparison uses the same filtered/consistent data for both models.

### Step 6: Visualization & Notebooks — ⏳ pending Llama-Instruct data

- [x] `notebooks/01_eda.ipynb` ✅
- [x] `notebooks/02_calibration_curves.ipynb` — isotonic vs. kernel curves, top-5 miscalibrated subjects ✅
- [x] `notebooks/03_multilevel_model.ipynb` — ICC decomposition, FDR-rejected subject list ✅
- [x] `notebooks/04_cross_model_comparison.ipynb` — ECE scatter, domain ECE bar chart, coefficient forest plot, subject mean_gap bar chart ✅
- [ ] **Re-run all notebooks** after Llama-Instruct pipeline completes ⏳
- [ ] **Write final report** (`report/report.tex`) ❌

---

## Deliverables

| # | Deliverable | Status |
|---|---|---|
| 1 | **Variance decomposition**: ICC from multilevel model (domain / subject / question) | GPT done ✅, Llama pending ⏳ |
| 2 | **FDR-corrected subject rankings**: robustly miscalibrated subjects after shrinkage + BH | GPT done ✅, Llama pending ⏳ |
| 3 | **Predictors of miscalibration**: question-level features → over- vs. underconfidence | GPT done ✅, Llama pending ⏳ |
| 4 | **Cross-model agreement**: task-driven vs. model-specific miscalibration | Pending both models ⏳ |
| 5 | **Final report** | Not started ❌ |

---

## Feasibility

| Component | Time | Cost |
|---|---|---|
| Download MMLU + feature extraction | 1–2 hrs | Free |
| API calls (14k questions, two models) | ~2 hrs runtime | ~$0.50 |
| Isotonic + kernel calibration curves | 2–3 hrs | — |
| Multilevel regression | 2–3 hrs | — |
| EB shrinkage + multiple testing | 2 hrs | — |
| Cross-model comparison + writing | — | — |

Full pipeline runs on a laptop, no GPU required. Data collection can run overnight.

---

## Data

- **MMLU**: `cais/mmlu` on HuggingFace — ~14,000 four-option multiple-choice questions, 57 subjects, 4 broad domains (STEM, Social Sciences, Humanities, Other)
- Three-level hierarchy: questions → subjects → domains
- Raw data: `data/` (gitignored)
- Model outputs: stored in `data/` after querying

---

## Key References

- Guo et al. [2017] — ECE, temperature scaling
- Kadavath et al. [2022] — token-level log probs as calibration signal; base models better calibrated than RLHF models
- Xiong et al. [2024] — all confidence elicitation methods struggle on professional knowledge tasks
- Luo et al. [2025] — subject-specific temperature scaling (implicit acknowledgment of heterogeneity)
- Tan et al. [2026] — BaseCal: base model signals for recalibration
- Nakkiran et al. [2025], Yaldiz et al. [2026] — instruction tuning effects on calibration
- Benjamini & Hochberg [1995] — FDR correction
- Barocas, Hardt, Narayanan [2023] — subgroup auditing in fairness (motivating analogy)
