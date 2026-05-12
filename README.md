# A Statistical Audit of LLM Calibration Heterogeneity Across Knowledge Domains

STAT 215B Final Project — Jennifer Zhang & John Wright

## Overview

We develop a statistically rigorous framework to detect and characterize subject-level miscalibration in LLMs, applying it to the MMLU benchmark across seven instruction-tuned models from three families: GPT-4o-mini, Llama-3-8B-Instruct, and Qwen-2.5-{0.5B, 1.5B, 3B, 7B, 14B}-Instruct. Methods include a three-level mixed-effects model, isotonic calibration curves, and Benjamini-Hochberg FDR correction.

## Repo Structure

```
final/
├── src/
│   ├── collect/        # MMLU download, API querying, feature extraction
│   ├── calibration/    # Isotonic + kernel calibration curves
│   └── modeling/       # Multilevel model, BH testing, sensitivity analysis
├── notebooks/
│   ├── seven_model_analysis.py   # Final analysis (all 7 models)
│   ├── seven_model_figures.py    # Final figures
│   ├── colab_mmlu_inference.ipynb  # Open-weight model inference (Colab GPU)
│   ├── colab_llama.ipynb           # Llama inference (Colab GPU)
│   └── 01–05_*.ipynb               # Exploratory notebooks
├── data/               # gitignored — see Setup
├── report/
│   ├── report.pdf / report.tex         # Full report
│   └── report_workshop.pdf / .tex      # ICML workshop submission
└── environment.yml
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

The core pipeline (GPT-4o-mini) runs via `make`:

```bash
make all
```

Or step by step:

```bash
# 1. Download MMLU
python src/collect/download_mmlu.py

# 2. Query GPT-4o-mini via OpenAI Batch API
python src/collect/query_gpt4o.py

# 3. Extract question-level features
python src/collect/extract_features.py

# 4. Fit calibration curves per subject
python src/calibration/curves.py

# 5. Multilevel model + FDR testing
PYTHONPATH=. python src/modeling/run_analysis.py

# 6. Cross-model comparison
python src/modeling/cross_model.py
```

Open-weight models (Llama, Qwen) require a GPU and were run on Google Colab — see `notebooks/colab_mmlu_inference.ipynb` and `notebooks/colab_llama.ipynb`.

Once all model outputs are collected, the final analysis and figures are produced by:

```bash
python notebooks/seven_model_analysis.py
python notebooks/seven_model_figures.py
```

## Research Questions

1. **Where** is calibration good and bad within MMLU? (Variance decomposition, FDR-corrected subject rankings)
2. **How consistent** are these patterns across models? (Cross-model Spearman correlations)
3. **What mechanisms** produce the heterogeneity? (Question-level predictors, within-family scaling)
