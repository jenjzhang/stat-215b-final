.PHONY: all collect features curves model compare clean

# Full pipeline
all: collect features curves model compare

# Step 1: Download MMLU and query both models
collect:
	python src/collect/download_mmlu.py
	python src/collect/query_gpt4o.py
	python src/collect/query_llama.py

# Step 2: Extract question-level features from raw responses
features:
	python src/collect/extract_features.py

# Step 3: Fit calibration curves per subject
curves:
	python src/calibration/curves.py

# Step 4: Multilevel model, EB shrinkage, multiple testing
model:
	PYTHONPATH=. python src/modeling/run_analysis.py

# Step 5: Cross-model comparison
compare:
	python src/modeling/cross_model.py

clean:
	rm -f data/processed/*.csv data/processed/*.parquet
