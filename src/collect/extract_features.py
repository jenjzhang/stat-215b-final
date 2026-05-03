"""
Join MMLU metadata with model responses and compute question-level features.

Inputs:  data/raw/mmlu.parquet
         data/raw/gpt4o_responses.parquet
         data/raw/llama_responses.parquet
Outputs: data/processed/questions_gpt4o.parquet
         data/processed/questions_llama.parquet

Question-level features added:
  - word_count: number of words in the question
  - max_choice_len: max word count across the four choices
  - has_negation: 1 if question contains not/never/except/least
  - confidence: max softmax prob (= model's stated confidence)
  - entropy: entropy of the softmax distribution over A/B/C/D
  - calibration_gap: confidence - correct (signed, positive = overconfident)
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import softmax

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

NEGATION_RE = re.compile(r"\b(not|never|except|least|no)\b", re.IGNORECASE)
LOGPROB_COLS = ["logprob_A", "logprob_B", "logprob_C", "logprob_D"]


def add_features(mmlu: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
    df = mmlu.merge(responses, on="question_id", how="inner")

    # Question-level text features
    df["word_count"] = df["question"].str.split().str.len()
    df["max_choice_len"] = df["choices"].apply(
        lambda cs: max(len(c.split()) for c in cs)
    )
    df["has_negation"] = df["question"].apply(
        lambda q: int(bool(NEGATION_RE.search(q)))
    )

    # Confidence = P(predicted answer | {A,B,C,D}), renormalized over the four choices.
    # Both models use top_logprobs=20 (GPT) or full-vocab logits (Llama-Instruct), so
    # all four answer tokens are reliably captured and the renormalization is accurate.
    logprobs = df[LOGPROB_COLS].values
    probs = softmax(logprobs, axis=1)
    df["confidence"] = probs.max(axis=1)
    df["entropy"] = -(probs * np.log(probs + 1e-12)).sum(axis=1)

    # Signed calibration gap: positive = overconfident
    df["calibration_gap"] = df["confidence"] - df["correct"].astype(float)

    return df


def main():
    mmlu = pd.read_parquet(RAW / "mmlu.parquet")

    for model, fname in [
        ("gpt4o", "gpt4o_responses_top20.parquet"),
        ("llama", "llama_instruct_responses_final.parquet"),
        ("qwen_0_5b", "qwen25_0_5b_instruct_responses.parquet"),
        ("qwen_1_5b", "qwen25_1_5b_instruct_responses.parquet"),
        ("qwen_1_5b_fp16", "qwen25_1_5b_instruct_fp16_responses.parquet"),
        ("qwen_7b", "qwen25_7b_instruct_responses.parquet"),
        ("qwen_14b", "qwen25_14b_instruct_responses.parquet"),
    ]:
        resp_path = RAW / fname
        if not resp_path.exists():
            print(f"Skipping {model}: {resp_path} not found")
            continue
        responses = pd.read_parquet(resp_path)
        df = add_features(mmlu, responses)
        out_path = OUT / f"questions_{model}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"{model}: {len(df):,} questions → {out_path}")


if __name__ == "__main__":
    main()
