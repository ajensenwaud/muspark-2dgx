#!/usr/bin/env python3
"""Task accuracy through the served endpoint (stdlib only): GSM8K (and MMLU-Pro when its items exist), greedy, the
items exported by scripts/eval_data.py (sampled with --temperature > 0). Adapted from the MiMo recipe's scripts/eval_tasks.py. Muse Spark always
reasons; its chat template takes `reasoning_effort` (minimal/low/medium/high) through chat_template_kwargs, and the
reasoning tokens count against max_tokens, so the budget is larger than MiMo's thinking-off 1024.

  python3 scripts/eval_tasks.py http://127.0.0.1:8892 Muse-Spark-Open-UD-IQ2_XXS results/evals --tag A-iq2xxs \
                                [--tasks gsm8k] [--concurrency 4] [--limit N] [--reasoning-effort low] [--max-tokens N]
                                [--temperature 1.0 --top-p 1.0 --top-k 0 --min-p 0 --seed 1234] [--ids 123,149]

Writes results/evals/<tag>-<task>.jsonl (every reply) and <tag>-<task>.score.json (accuracy, a 95% interval,
unparsed and truncated counts, by category). Resumes: items already in the .jsonl are not asked again.
"""

import argparse
import concurrent.futures as cf
import json
import math
import re
import time
import urllib.request
from pathlib import Path

LETTERS = "ABCDEFGHIJ"
MAX_TOKENS = {"gsm8k": 4096, "mmlu_pro": 6144}
EFFORT = "low"
SAMPLING = {"temperature": 0, "top_k": 1, "seed": 1234}   # greedy default (the request body of every earlier tag)


def prompt(task: str, it: dict) -> str:
    if task == "gsm8k":
        return (f"{it['question']}\n\nSolve the problem step by step. On the last line, write the final answer as "
                f"'Answer: <number>'.")
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(it["options"]))
    return (f"{it['question']}\n\nOptions:\n{opts}\n\nThink step by step, then on the last line write the final "
            f"answer as 'Answer: <letter>'.")


def parse(task: str, text: str) -> str | None:
    if task == "gsm8k":
        m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text)
        nums = m or re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums:
            return None
        v = nums[-1].replace(",", "").rstrip(".")
        try:
            f = float(v)
        except ValueError:
            return None
        return str(int(f)) if f == int(f) else str(f)
    m = re.findall(r"Answer:\s*\(?([A-J])\)?", text)
    if not m:
        m = re.findall(r"\b(?:answer is|answer:)\s*\(?([A-J])\)?", text, re.I)
    if not m:                                              # e.g. "This corresponds to option **F**."
        m = re.findall(r"\boption\s*\**\(?([A-J])\)?\**", text, re.I)
    return m[-1].upper() if m else None


def ask(url: str, model: str, task: str, it: dict) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt(task, it)}],
            "max_tokens": MAX_TOKENS[task], **SAMPLING,
            "chat_template_kwargs": {"reasoning_effort": EFFORT}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                j = json.load(r)
            break
        except Exception as exc:                          # noqa: BLE001  (retried, then recorded)
            if attempt == 2:
                return {"id": it["id"], "error": str(exc)}
            time.sleep(5)
    msg = j["choices"][0]["message"]
    text = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    got = parse(task, text)
    gold = it["gold"]
    ok = got is not None and (float(got) == float(gold) if task == "gsm8k" else got == gold)
    return {"id": it["id"], "got": got, "gold": gold, "correct": bool(ok), "finish": j["choices"][0].get("finish_reason"),
            "tokens": j["usage"]["completion_tokens"], "category": it.get("category"), "text": text,
            "reasoning_chars": len(reasoning), "prompt_tokens": j["usage"].get("prompt_tokens"),
            **({"reasoning": reasoning} if SAVE_REASONING else {})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tasks", default="gsm8k")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--max-tokens", type=int, default=0, help="override the per-task budget")
    ap.add_argument("--timeout", type=int, default=7200, help="per-request seconds (a 30K-token reply at 4 streams ~45 min)")
    ap.add_argument("--save-reasoning", action="store_true", help="keep each reply's reasoning text in the jsonl")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 (default): greedy, sent as temperature 0, top_k 1")
    ap.add_argument("--top-p", type=float, default=1.0, help="sampled runs only")
    ap.add_argument("--top-k", type=int, default=0, help="sampled runs only; 0 = disabled (Zig: 0 or -1; llama.cpp: <= 0)")
    ap.add_argument("--min-p", type=float, default=0.0, help="sampled runs only; sent explicitly (llama.cpp defaults to 0.05)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ids", default="", help="comma-separated item ids to ask (default: all)")
    a = ap.parse_args()
    global EFFORT, TIMEOUT, SAVE_REASONING, SAMPLING
    if a.temperature > 0:
        SAMPLING = {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "min_p": a.min_p, "seed": a.seed}
    else:
        SAMPLING = {"temperature": 0, "top_k": 1, "seed": a.seed}
    EFFORT = a.reasoning_effort
    TIMEOUT, SAVE_REASONING = a.timeout, a.save_reasoning
    if a.max_tokens:
        for k in MAX_TOKENS:
            MAX_TOKENS[k] = a.max_tokens
    d = Path(a.dir)
    for task in a.tasks.split(","):
        items = json.loads((d / f"items-{task}.json").read_text())
        if a.ids:
            want = {int(x) for x in a.ids.split(",")}
            items = [it for it in items if it["id"] in want]
        if a.limit:
            items = items[:a.limit]
        raw = d / f"{a.tag}-{task}.jsonl"
        done = {}
        if raw.exists():
            for line in raw.read_text().splitlines():
                r = json.loads(line)
                if "error" not in r:
                    r["got"] = parse(task, r["text"])         # scores always use the current parser
                    r["correct"] = r["got"] is not None and (
                        float(r["got"]) == float(r["gold"]) if task == "gsm8k" else r["got"] == r["gold"])
                    done[r["id"]] = r
        todo = [it for it in items if it["id"] not in done]
        t0 = time.time()
        with raw.open("a") as f, cf.ThreadPoolExecutor(a.concurrency) as ex:
            for r in ex.map(lambda it: ask(a.url.rstrip("/"), a.model, task, it), todo):
                f.write(json.dumps(r) + "\n")
                f.flush()
                if "error" not in r:
                    done[r["id"]] = r
        rs = [done[it["id"]] for it in items if it["id"] in done]
        n, c = len(rs), sum(r["correct"] for r in rs)
        acc = c / n if n else 0.0
        half = 1.96 * math.sqrt(acc * (1 - acc) / n) if n else 0.0
        cats: dict = {}
        for r in rs:
            if r.get("category"):
                cats.setdefault(r["category"], [0, 0])
                cats[r["category"]][0] += r["correct"]
                cats[r["category"]][1] += 1
        score = {"tag": a.tag, "task": task, "n": n, "correct": c, "acc": round(acc, 4), "ci95": round(half, 4),
                 "unparsed": sum(r["got"] is None for r in rs), "truncated": sum(r["finish"] == "length" for r in rs),
                 "gen_tokens": sum(r["tokens"] for r in rs), "wall_s": round(time.time() - t0, 1),
                 "by_category": cats or None, "settings": {"reasoning_effort": EFFORT, **SAMPLING,
                                                            "max_tokens": MAX_TOKENS[task], "concurrency": a.concurrency}}
        (d / f"{a.tag}-{task}.score.json").write_text(json.dumps(score, indent=1))
        print(json.dumps({k: v for k, v in score.items() if k != "by_category"}), flush=True)


if __name__ == "__main__":
    main()
