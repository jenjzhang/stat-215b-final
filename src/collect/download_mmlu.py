"""
Download MMLU from HuggingFace and save as parquet.

Output: data/raw/mmlu.parquet
Columns: question_id, subject, domain, question, choices (list), answer (int 0-3)
"""
from datasets import load_dataset
import pandas as pd
from pathlib import Path

DOMAIN_MAP = {
    # STEM
    "abstract_algebra": "STEM", "anatomy": "STEM", "astronomy": "STEM",
    "college_biology": "STEM", "college_chemistry": "STEM", "college_computer_science": "STEM",
    "college_mathematics": "STEM", "college_physics": "STEM", "computer_security": "STEM",
    "conceptual_physics": "STEM", "electrical_engineering": "STEM", "elementary_mathematics": "STEM",
    "formal_logic": "STEM", "global_facts": "STEM", "high_school_biology": "STEM",
    "high_school_chemistry": "STEM", "high_school_computer_science": "STEM",
    "high_school_mathematics": "STEM", "high_school_physics": "STEM",
    "high_school_statistics": "STEM", "machine_learning": "STEM",
    # Social Sciences
    "econometrics": "Social Sciences", "high_school_government_and_politics": "Social Sciences",
    "high_school_macroeconomics": "Social Sciences", "high_school_microeconomics": "Social Sciences",
    "high_school_psychology": "Social Sciences", "human_sexuality": "Social Sciences",
    "professional_psychology": "Social Sciences", "public_relations": "Social Sciences",
    "security_studies": "Social Sciences", "sociology": "Social Sciences",
    "us_foreign_policy": "Social Sciences",
    # Humanities
    "formal_logic": "Humanities", "high_school_european_history": "Humanities",
    "high_school_us_history": "Humanities", "high_school_world_history": "Humanities",
    "international_law": "Humanities", "jurisprudence": "Humanities",
    "logical_fallacies": "Humanities", "moral_disputes": "Humanities",
    "moral_scenarios": "Humanities", "philosophy": "Humanities",
    "prehistory": "Humanities", "professional_law": "Humanities",
    "world_religions": "Humanities",
    # Other
    "business_ethics": "Other", "clinical_knowledge": "Other",
    "college_medicine": "Other", "global_facts": "Other",
    "human_aging": "Other", "management": "Other", "marketing": "Other",
    "medical_genetics": "Other", "miscellaneous": "Other", "nutrition": "Other",
    "professional_accounting": "Other", "professional_medicine": "Other",
    "virology": "Other",
}

OUT = Path("data/raw/mmlu.parquet")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("cais/mmlu", "all", split="test")
    df = ds.to_pandas()
    df = df.rename(columns={"subject": "subject", "answer": "answer"})
    df["domain"] = df["subject"].map(DOMAIN_MAP).fillna("Other")
    df = df.reset_index(drop=True)
    df.index.name = "question_id"
    df = df.reset_index()
    df.to_parquet(OUT, index=False)
    print(f"Saved {len(df):,} questions to {OUT}")


if __name__ == "__main__":
    main()
