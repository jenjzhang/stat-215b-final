"""
Query GPT-4o-mini for each MMLU question and collect token-level log probabilities.

Input:  data/raw/mmlu.parquet
Output: data/raw/gpt4o_responses.parquet
Columns: question_id, logprob_A, logprob_B, logprob_C, logprob_D, predicted (str), correct (bool)
"""
import os
import pandas as pd
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

IN = Path("data/raw/mmlu.parquet")
OUT = Path("data/raw/gpt4o_responses.parquet")
ANSWER_TOKENS = ["A", "B", "C", "D"]

PROMPT_TEMPLATE = """\
The following is a multiple choice question. Answer with a single letter: A, B, C, or D.

{question}

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


def query_one(client: OpenAI, prompt: str) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        logprobs=True,
        top_logprobs=5,
        temperature=0,
    )
    top = {t.token: t.logprob for t in resp.choices[0].logprobs.content[0].top_logprobs}
    import math
    logprobs = {f"logprob_{tok}": top.get(tok, -100.0) for tok in ANSWER_TOKENS}
    predicted = max(logprobs, key=logprobs.get).replace("logprob_", "")
    return {**logprobs, "predicted": predicted}


def main():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    df = pd.read_parquet(IN)

    # Resume from checkpoint if partial run exists
    if OUT.exists():
        done = pd.read_parquet(OUT)["question_id"].tolist()
        df = df[~df["question_id"].isin(done)]
        print(f"Resuming: {len(df):,} questions remaining")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = build_prompt(row)
        res = query_one(client, prompt)
        res["question_id"] = row["question_id"]
        answer_letter = ANSWER_TOKENS[row["answer"]]
        res["correct"] = res["predicted"] == answer_letter
        results.append(res)

    out_df = pd.DataFrame(results)
    if OUT.exists():
        out_df = pd.concat([pd.read_parquet(OUT), out_df], ignore_index=True)
    out_df.to_parquet(OUT, index=False)
    print(f"Saved {len(out_df):,} responses to {OUT}")


if __name__ == "__main__":
    main()
