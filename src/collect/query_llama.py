"""
Query Llama-3-8B (base) for each MMLU question using completion-style prompting.
Extracts token-level log probabilities over answer tokens.

Input:  data/raw/mmlu.parquet
Output: data/raw/llama_responses.parquet
Columns: question_id, logprob_A, logprob_B, logprob_C, logprob_D, predicted (str), correct (bool)
"""
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

IN = Path("data/raw/mmlu.parquet")
OUT = Path("data/raw/llama_responses.parquet")
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
ANSWER_TOKENS = ["A", "B", "C", "D"]

# Completion-style prompt for base model (no system/instruction formatting)
PROMPT_TEMPLATE = """\
Question: {question}

Choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer:"""


def build_prompt(row: pd.Series) -> str:
    choices = row["choices"]
    return PROMPT_TEMPLATE.format(
        question=row["question"],
        choice_a=choices[0], choice_b=choices[1],
        choice_c=choices[2], choice_d=choices[3],
    )


def get_answer_token_ids(tokenizer) -> dict[str, int]:
    return {tok: tokenizer.encode(f" {tok}", add_special_tokens=False)[0] for tok in ANSWER_TOKENS}


def query_batch(model, tokenizer, prompts: list[str], token_ids: dict) -> list[dict]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    # logits at the last token position for each prompt
    last_logits = outputs.logits[:, -1, :]
    log_probs = torch.log_softmax(last_logits, dim=-1).cpu()

    results = []
    for i in range(len(prompts)):
        row = {f"logprob_{tok}": log_probs[i, tid].item() for tok, tid in token_ids.items()}
        predicted = max(row, key=row.get).replace("logprob_", "")
        results.append({**row, "predicted": predicted})
    return results


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    df = pd.read_parquet(IN)
    if OUT.exists():
        done = pd.read_parquet(OUT)["question_id"].tolist()
        df = df[~df["question_id"].isin(done)]
        print(f"Resuming: {len(df):,} questions remaining")

    token_ids = get_answer_token_ids(tokenizer)
    BATCH_SIZE = 8
    SAVE_EVERY = 100  # save checkpoint every 100 questions
    results = []

    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[i : i + BATCH_SIZE]
        prompts = [build_prompt(row) for _, row in batch.iterrows()]
        batch_results = query_batch(model, tokenizer, prompts, token_ids)
        for j, (_, row) in enumerate(batch.iterrows()):
            res = batch_results[j]
            res["question_id"] = row["question_id"]
            res["correct"] = res["predicted"] == ANSWER_TOKENS[row["answer"]]
            results.append(res)

        if len(results) >= SAVE_EVERY:
            out_df = pd.DataFrame(results)
            if OUT.exists():
                out_df = pd.concat([pd.read_parquet(OUT), out_df], ignore_index=True)
            out_df.to_parquet(OUT, index=False)
            results = []

    if results:
        out_df = pd.DataFrame(results)
        if OUT.exists():
            out_df = pd.concat([pd.read_parquet(OUT), out_df], ignore_index=True)
        out_df.to_parquet(OUT, index=False)

    print(f"Saved {len(pd.read_parquet(OUT)):,} responses to {OUT}")


if __name__ == "__main__":
    main()
