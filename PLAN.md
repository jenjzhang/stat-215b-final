# STAT 215B Final Project — Plan & Progress Tracker

**Title**: A Statistical Audit of LLM Calibration Heterogeneity Across Knowledge Domains  
**Authors**: Jennifer Zhang, John Wright

---

## Overview

LLM calibration is nearly always reported as a single aggregate scalar. This project asks whether that aggregate masks systematic miscalibration in specific knowledge domains, developing a statistically rigorous framework to detect and characterize that heterogeneity. The framing is analogous to bias auditing in the fairness literature: just as reporting a single accuracy number across demographic groups is insufficient for a fairness audit, reporting aggregate ECE is insufficient for a calibration audit.

**Methods**: multilevel regression, calibration curve clustering, empirical Bayes shrinkage, multiple testing correction  
**Benchmark**: MMLU (~14,000 questions, 57 subjects, 4 domains)  
**Models**: GPT-4o-mini (OpenAI API, ~$0.50) and Llama-3-8B base (HuggingFace, free)

---

## Research Questions

1. How much of the variance in LLM calibration error is attributable to domain vs. subject vs. individual question?
2. Which question- and subject-level features predict the direction and magnitude of miscalibration?
3. Are calibration patterns consistent across models, or model-specific?

---

## Pipeline Steps

> **Legend**: ✅ code written | ⏳ needs to be run | ❌ not started

### Step 1: Confidence Elicitation — code complete, not yet run

- [x] `src/collect/download_mmlu.py` — downloads `cais/mmlu`, outputs `data/raw/mmlu.parquet` ✅
- [x] `src/collect/query_gpt4o.py` — GPT-4o-mini via OpenAI API, token-level logprobs (`logprobs=True, top_logprobs=5`), checkpoint-resumable ✅
- [x] `src/collect/query_llama.py` — Llama-3-8B base via HuggingFace, batch size 8, completion-style prompt, checkpoint-resumable ✅
- [ ] **RUN**: `make collect` (~2 hrs, ~$0.50) ⏳
- **Notes**: Base Llama uses completion-style format, not instruction prompt — note as limitation in cross-model comparison

### Step 2: Outcome and Features — code complete, not yet run

- [x] `src/collect/extract_features.py` ✅
  - [x] Signed calibration gap: $\text{gap}_i = c_i - y_i$
  - [x] Word count
  - [x] Max option length *(proposal said "average and max"; only max implemented)*
  - [x] Negation indicator (`not`, `except`, `least`, `never`, `no`)
  - [x] Entropy of the A/B/C/D probability distribution
  - [x] Confidence ($c_i$ = max softmax prob from logprobs)
- [ ] **RUN**: `make features` ⏳
- **Decision (2026-04-23)**: Use signed calibration gap as primary outcome. Reasons: sign directly answers RQ2 (over vs. underconfidence), random effects are interpretable as subject-level miscalibration, and BH test of H₀: mean gap = 0 is exactly the calibration audit question. Log-likelihood kept as a robustness check if needed — it is a proper scoring rule and would tell us "what predicts worse probabilistic accuracy" rather than directional miscalibration.
- **Limitation to note**: variance of $c_i - y_i$ is not constant across subjects (harder subjects have noisier outcomes since $\text{var}(y_i) = p_j(1-p_j)$); mixed model assumes homoskedastic residuals — check residual plots after fitting.
- **Note**: Subject-level accuracy (difficulty proxy) is computed at the modeling stage, not here — fine.

### Step 3: Nonparametric Calibration Curves — code complete, not yet run

- [x] `src/calibration/curves.py` ✅
  - [x] Isotonic regression per subject (monotone fit of accuracy on confidence, sklearn)
  - [x] Kernel smoother per subject (Gaussian σ=1.5, unconstrained comparison)
  - [x] ECE per subject (integrated absolute deviation, 20-bin bucketing)
  - [x] Per-subject summary: n, ece, mean_gap, mean_confidence, mean_accuracy
- [ ] **RUN**: `make curves` ⏳
- **Gap**: Divergence flagging between isotonic and kernel fits is not automated — check visually in `notebooks/02_calibration_curves.ipynb`

### Step 4: Multilevel Regression, Shrinkage, and Multiple Testing — code complete, not yet run

- [x] `src/modeling/multilevel.py` — three-level mixed model via `statsmodels.mixedlm`, REML ✅

$$\text{outcome}_{ijk} = \mu + \mathbf{x}_{ijk}^\top \boldsymbol{\beta} + u_k + v_{jk} + \varepsilon_{ijk}$$

  where $u_k \sim \mathcal{N}(0, \sigma^2_\text{domain})$, $v_{jk} \sim \mathcal{N}(0, \sigma^2_\text{subject})$

  - [x] Fixed effects: `word_count`, `max_choice_len`, `has_negation`, `entropy` (all standardized)
  - [x] ICC variance decomposition ($\sigma^2_\text{domain}$, $\sigma^2_\text{subject}$, $\sigma^2_\varepsilon$) → answers RQ1
  - [x] Subject-level BLUPs ($\hat{v}_{jk}$) as shrunken calibration estimates, output to `data/processed/blups_{model}.parquet`
- [x] `src/modeling/testing.py` — one-sample t-test per subject + BH FDR correction ✅
  - [x] H₀: mean calibration gap = 0 per subject; BH at α=0.05 with monotonicity enforcement
  - [x] Output: `data/processed/fdr_results_{model}.parquet` sorted by adjusted p-value
- [ ] **RUN**: `make model` ⏳
- **Lead**: Jennifer

### Step 5: Cross-Model Replication — code complete, not yet run

- [x] `src/modeling/cross_model.py` ✅
  - [x] Spearman rank correlation of ECE across subjects, bootstrap CIs (2000 replicates, seed=42)
  - [x] Four-way agreement: both (task-driven) / GPT-4o only / Llama only / neither
  - [x] Output: `data/processed/cross_model_comparison.parquet`
- [ ] **RUN**: `make compare` ⏳
- **Lead**: Both

### Step 6: Visualization & Notebooks — scaffolded, not yet run

- [x] `notebooks/01_eda.ipynb` — question counts, accuracy range per subject ✅
- [x] `notebooks/02_calibration_curves.ipynb` — isotonic vs. kernel curves, top-5 miscalibrated subjects, outputs `report/figures/top5_miscalibrated_{model}.pdf` ✅
- [x] `notebooks/03_multilevel_model.ipynb` — ICC decomposition, FDR-rejected subject list ✅
- [x] `notebooks/04_cross_model_comparison.ipynb` — ECE scatter colored by agreement, outputs `report/figures/cross_model_ece.pdf` ✅
- [ ] **RUN** all four notebooks after pipeline executes ⏳
- [ ] **Write final report** ❌

---

## Deliverables

| # | Deliverable | Status |
|---|---|---|
| 1 | **Variance decomposition**: ICC from multilevel model (domain / subject / question) | Code ready ⏳ |
| 2 | **FDR-corrected subject rankings**: robustly miscalibrated subjects after shrinkage + BH | Code ready ⏳ |
| 3 | **Predictors of miscalibration**: question-level features → over- vs. underconfidence | Code ready ⏳ |
| 4 | **Cross-model agreement**: task-driven vs. model-specific miscalibration | Code ready ⏳ |
| 5 | **Final report** | Not started ❌ |

---

## Division of Labor

| Component | Lead |
|---|---|
| MMLU download + feature extraction | John |
| GPT-4o-mini + Llama API querying | John |
| Isotonic + kernel calibration curves | John |
| Three-level mixed effects model | Jennifer |
| Empirical Bayes shrinkage (BLUPs) | Jennifer |
| BH multiple testing correction | Jennifer |
| Cross-model comparison | Both |
| Writing | Both |

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
