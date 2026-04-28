"""
Query GPT-4o-mini for each MMLU question and collect token-level log probabilities.

Input:  data/raw/mmlu.parquet
Output: data/raw/gpt4o_responses.parquet
Columns: question_id, logprob_A, logprob_B, logprob_C, logprob_D, predicted (str), correct (bool)
"""
import json
import os
import re
import time
import pandas as pd
from pathlib import Path
from openai import OpenAI, RateLimitError, APIConnectionError
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

IN = Path("data/raw/mmlu.parquet")
OUT = Path("data/raw/gpt4o_responses_top20.parquet")
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
    while True:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                logprobs=True,
                top_logprobs=20,
                temperature=0,
            )
            top = {t.token: t.logprob for t in resp.choices[0].logprobs.content[0].top_logprobs}
            logprobs = {f"logprob_{tok}": top.get(tok, -100.0) for tok in ANSWER_TOKENS}
            predicted = max(logprobs, key=logprobs.get).replace("logprob_", "")
            return {**logprobs, "predicted": predicted}
        except RateLimitError as e:
            match = re.search(r"try again in ([0-9.]+)s", str(e))
            wait = float(match.group(1)) if match else 60.0
            tqdm.write(f"Rate limit hit — sleeping {wait:.0f}s")
            time.sleep(wait + 1)
        except APIConnectionError:
            tqdm.write("Connection error — retrying in 30s")
            time.sleep(30)


def main():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    df = pd.read_parquet(IN)

    # Resume from checkpoint if partial run exists
    if OUT.exists():
        done = pd.read_parquet(OUT)["question_id"].tolist()
        df = df[~df["question_id"].isin(done)]
        print(f"Resuming: {len(df):,} questions remaining")

    SAVE_EVERY = 100
    results = []
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        prompt = build_prompt(row)
        res = query_one(client, prompt)
        res["question_id"] = row["question_id"]
        answer_letter = ANSWER_TOKENS[row["answer"]]
        res["correct"] = res["predicted"] == answer_letter
        results.append(res)
        time.sleep(0.5)

        if (i + 1) % SAVE_EVERY == 0:
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


def batch_mode():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    df = pd.read_parquet(IN)

    if OUT.exists():
        done = pd.read_parquet(OUT)["question_id"].tolist()
        df = df[~df["question_id"].isin(done)]
        print(f"Submitting {len(df):,} questions as batch")

    # preserve original question_id type for merging later
    qid_map = {str(row["question_id"]): row["question_id"] for _, row in df.iterrows()}
    answer_map = {str(row["question_id"]): ANSWER_TOKENS[row["answer"]] for _, row in df.iterrows()}

    lines = [
        json.dumps({
            "custom_id": str(row["question_id"]),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": build_prompt(row)}],
                "max_tokens": 1,
                "logprobs": True,
                "top_logprobs": 5,
                "temperature": 0,
            },
        })
        for _, row in df.iterrows()
    ]

    batch_file = client.files.create(file=("batch.jsonl", "\n".join(lines).encode()), purpose="batch")
    print(f"Uploaded file: {batch_file.id}")

    BATCH_ID_FILE = Path("data/raw/batch_id.txt")

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    BATCH_ID_FILE.write_text(batch.id)
    print(f"Batch submitted: {batch.id}")
    print(f"Saved to {BATCH_ID_FILE} — close your laptop and run --collect later to download results.")


def collect_mode():
    BATCH_ID_FILE = Path("data/raw/batch_id.txt")
    if not BATCH_ID_FILE.exists():
        print("No batch_id.txt found — run --batch first.")
        return

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    batch_id = BATCH_ID_FILE.read_text().strip()
    batch = client.batches.retrieve(batch_id)
    print(f"Status: {batch.status}  ({batch.request_counts.completed}/{batch.request_counts.total} done)")

    if batch.status != "completed":
        print("Batch not finished yet — try again later.")
        return

    df_all = pd.read_parquet(IN)
    qid_map = {str(r["question_id"]): r["question_id"] for _, r in df_all.iterrows()}
    answer_map = {str(r["question_id"]): ANSWER_TOKENS[r["answer"]] for _, r in df_all.iterrows()}

    content = client.files.content(batch.output_file_id).text
    results = []
    for line in content.strip().split("\n"):
        item = json.loads(line)
        qid = item["custom_id"]
        top = {t["token"]: t["logprob"] for t in item["response"]["body"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
        logprobs = {f"logprob_{tok}": top.get(tok, -100.0) for tok in ANSWER_TOKENS}
        predicted = max(logprobs, key=logprobs.get).replace("logprob_", "")
        results.append({"question_id": qid_map[qid], **logprobs, "predicted": predicted, "correct": predicted == answer_map[qid]})

    out_df = pd.DataFrame(results)
    if OUT.exists():
        out_df = pd.concat([pd.read_parquet(OUT), out_df], ignore_index=True)
    out_df.to_parquet(OUT, index=False)
    BATCH_ID_FILE.unlink()
    print(f"Saved {len(out_df):,} total responses to {OUT}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true", help="Submit batch job and exit")
    parser.add_argument("--collect", action="store_true", help="Download completed batch results")
    args = parser.parse_args()
    if args.batch:
        batch_mode()
    elif args.collect:
        collect_mode()
    else:
        main()
