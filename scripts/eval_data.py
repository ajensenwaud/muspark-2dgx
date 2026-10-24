#!/usr/bin/env python3
"""Export the evaluation items once (stdlib only, no `datasets`), `--n` items per task sampled with
random.Random(seed) over the test split, sorted by index:
  gsm8k     GSM8K test from its original publisher (github.com/openai/grade-school-math, MIT)
  mmlu_pro  TIGER-Lab/MMLU-Pro test (12,032 rows, MIT), rows fetched one by one from the Hugging Face datasets
            server in the split's own order (the order `datasets.load_dataset` gives)
Same sampler and seed as the MiMo recipe's tools/mimo_eval_data.py (n=500, seed 1234 there), so an n=500 export here
should give the same items (same row order as the HF copy); the default n=200 is a different subset.

  python3 scripts/eval_data.py results/evals [--n 200 --seed 1234] [--tasks gsm8k,mmlu_pro]
"""

import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
ROWS_URL = "https://datasets-server.huggingface.co/rows?"
MMLU_PRO_ROWS = 12032


def get_json(url: str) -> dict:
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception:                                  # noqa: BLE001  (rate limits: back off and retry)
            if attempt == 7:
                raise
            time.sleep(5 * 2 ** min(attempt, 4))
    raise RuntimeError(url)


def gsm8k(n: int, seed: int) -> list:
    with urllib.request.urlopen(GSM8K_URL, timeout=120) as r:
        rows = [json.loads(line) for line in r.read().decode().splitlines() if line.strip()]
    assert len(rows) == 1319, len(rows)
    pick = sorted(random.Random(seed).sample(range(len(rows)), n))
    return [{"id": int(i), "question": rows[i]["question"],
             "gold": rows[i]["answer"].split("####")[-1].strip().replace(",", "")} for i in pick]


def mmlu_pro(n: int, seed: int) -> list:
    q = {"dataset": "TIGER-Lab/MMLU-Pro", "config": "default", "split": "test", "offset": 0, "length": 1}
    total = get_json(ROWS_URL + urllib.parse.urlencode(q))["num_rows_total"]
    assert total == MMLU_PRO_ROWS, total
    pick = sorted(random.Random(seed).sample(range(total), n))
    rows = {}
    for page in sorted({i // 100 for i in pick}):          # pages of 100 rows (the server's limit), paced
        q["offset"], q["length"] = page * 100, 100
        for row in get_json(ROWS_URL + urllib.parse.urlencode(q))["rows"]:
            rows[row["row_idx"]] = row["row"]
        time.sleep(1.0)
    items = []
    for i in pick:
        r = rows[i]
        items.append({"id": int(i), "question": r["question"], "options": list(r["options"]), "gold": r["answer"],
                      "category": r["category"]})
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tasks", default="gsm8k")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for task in a.tasks.split(","):
        items = {"gsm8k": gsm8k, "mmlu_pro": mmlu_pro}[task](a.n, a.seed)
        (out / f"items-{task}.json").write_text(json.dumps(items, indent=0))
        print(f"wrote {len(items)} {task} items (seed {a.seed}) to {out}/items-{task}.json")


if __name__ == "__main__":
    main()
