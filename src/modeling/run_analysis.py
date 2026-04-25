"""
Entry point: run multilevel model + multiple testing for both models.
"""
from pathlib import Path
from src.modeling import multilevel, testing

MODELS = [m for m in ["gpt4o", "llama"]
          if Path(f"data/processed/questions_{m}.parquet").exists()]

if __name__ == "__main__":
    for model in MODELS:
        multilevel.run(model)
        testing.run(model)
