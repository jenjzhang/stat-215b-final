# A Statistical Audit of LLM Calibration Heterogeneity Across Knowledge Domains

STAT 215B Final Project — Jennifer Zhang & John Wright

## Overview

We develop a statistically rigorous framework to detect and characterize domain-level miscalibration in LLMs, applying it to the MMLU benchmark across GPT-4o-mini and Llama-3-8B. Methods include multilevel regression, isotonic calibration curves, empirical Bayes shrinkage, and Benjamini-Hochberg FDR correction.

## Repo Structure

```
final/
├── src/
│   ├── collect/        # John: MMLU download, API querying, feature extraction
│   ├── calibration/    # John: isotonic + kernel calibration curves
│   └── modeling/       # Jennifer: multilevel model, EB shrinkage, BH testing
├── notebooks/          # Exploratory and final analysis
├── data/               # gitignored — see Setup
└── report/figures/     # Generated plots (gitignored)
```

## Setup

```bash
conda env create -f environment.yml
conda activate 215b-final
```

Create a `.env` file (gitignored) with your OpenAI key:
```
OPENAI_API_KEY=sk-...
```

## Pipeline

Run the full pipeline step by step:

```bash
# 1. Download MMLU
python src/collect/download_mmlu.py

# 2. Query models (can run overnight)
python src/collect/query_gpt4o.py
python src/collect/query_llama.py   # requires GPU or patience on CPU

# 3. Extract features
python src/collect/extract_features.py

# 4. Calibration curves per subject
python src/calibration/curves.py

# 5. Multilevel model + FDR testing
python src/modeling/run_analysis.py

# 6. Cross-model comparison
python src/modeling/cross_model.py
```

Or run all at once: `make all`

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

## Research Questions

1. **Variance decomposition**: How much calibration heterogeneity is attributable to domain vs. subject vs. question?
2. **Predictors**: Which question/subject features predict miscalibration direction and magnitude?
3. **Cross-model**: Are calibration patterns task-driven or model-specific?
