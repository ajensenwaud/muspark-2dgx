#!/usr/bin/env python3
"""Prompt-cache contracts against a served Muse Spark endpoint (stdlib only; nothing to install on the host).

The engine's prompt cache (TF_MUSE_PROMPT_CACHE=1) must give byte-identical replies whether a request resumes from a
kept prompt state or runs fresh on a cold server (resumed == fresh), with or without the drafter (drafted == plain),
alone or beside other streams (concurrent == solo). Two modes:

  record  on a serve WITHOUT the cache (the cold reference): runs every scenario, builds the multi-turn transcripts
          from this serve's own replies, writes each request body and its reply's token sha to REF (JSON lines).
  replay  on any other arm (cache on, drafter on, kv8 ...): sends the recorded bodies again in four phases (solo in
          order; four conversations at once; a burst of four requests sharing one prefix, which resumes the same
          kept state in four slots; more conversations than slots, round robin) plus the window-edge sweep, and
          compares every reply's token_sha with REF. Exit 1 on any difference.

Scenarios: multi-turn chats on the model's own replies (the warm-pass path), a scripted agent session whose tool
outputs grow the history (the history-mark path), a long document conversation (--long tokens), one sampled
conversation (temperature 0.8, seed 7: the keyed sampler), and a sweep of resume points across the sliding window's
edge (2047 / 2048 / 2049), the ring (2176), prompt chunks (1024, 4096) and decode windows (128 rows), each hit
exactly through /tokenize. The chat template prints the date: compare arms recorded on the same day only.

  python3 tools/pc/pc_contracts.py record --base http://127.0.0.1:8888 --ref pc/ref.jsonl
  python3 tools/pc/pc_contracts.py replay --base http://127.0.0.1:8888 --ref pc/ref.jsonl \\
      --out pc/cache-plain.jsonl
"""

import argparse
import json
import sys
import threading
import time
import urllib.request

MODEL = "Muse-Spark-Open"
KW = {"reasoning_effort": "low"}
WORDS = ("amber basil cedar delta ember fable garnet harbor island jasper kettle lantern meadow nectar orbit pebble "
         "quartz ripple saddle timber umber velvet willow xenon yarrow zephyr").split()
EDGES = [127, 128, 129, 1023, 1024, 1025, 2046, 2047, 2048, 2049, 2050, 2175, 2176, 2177, 4095, 4096, 4097]


def post(base, path, body, timeout=3600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def find(obj, key):
    """The first value under `key` anywhere in a JSON reply (the engine's runtime extras sit in a nested object)."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            got = find(v, key)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = find(v, key)
            if got is not None:
                return got
    return None


def chat(base, body):
    t0 = time.time()
    j = post(base, "/v1/chat/completions", body)
    msg = j["choices"][0]["message"]
    usage = j.get("usage") or {}
    return {"sha": find(j, "token_sha"), "content": msg.get("content") or "", "tool_calls": msg.get("tool_calls"),
            "finish": j["choices"][0].get("finish_reason"), "prompt": usage.get("prompt_tokens"),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "ttft": find(j, "time_to_first_token"), "prefill_s": find(j, "prefill_seconds"), "wall": round(time.time() - t0, 2)}


def history_len(base, messages, tools=None):
    body = {"model": MODEL, "messages": messages, "add_generation_prompt": False, "chat_template_kwargs": KW}
    if tools:
        body["tools"] = tools
    j = post(base, "/tokenize", body, timeout=120)
    toks = j.get("tokens") or j.get("ids") or j.get("input_ids")
    return len(toks) if toks is not None else int(j["count"])


def filler(n, seed=0):
    return " ".join(WORDS[(i * 7 + seed) % len(WORDS)] for i in range(n))


def body(messages, max_tokens, tools=None, sampled=False):
    b = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "chat_template_kwargs": KW,
         "temperature": 0.8 if sampled else 0}
    if sampled:
        b["seed"] = 7
    if tools:
        b["tools"] = tools
    return b


TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file of the repository.",
                                           "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                          "required": ["path"]}}}]


def record(a):
    """Every scenario on the cold serve; each turn's body uses this serve's reply to the turn before."""
    out = open(a.ref, "w")
    n = 0

    def emit(scenario, conv, turn, b, mark=None):
        nonlocal n
        r = chat(a.base, b)
        row = {"scenario": scenario, "conv": conv, "turn": turn, "body": b, **r}
        if mark is not None:
            row["mark"] = mark  # where the cache keeps the history's state (the edge sweep's resume point)
        out.write(json.dumps(row) + "\n")
        out.flush()
        n += 1
        print(f"{scenario:8s} {conv:10s} t{turn} prompt={r['prompt']} wall={r['wall']}s sha={str(r['sha'])[:12]} "
              f"finish={r['finish']}", flush=True)
        return r

    questions = ["Name three uses of basalt.", "Which of those is the oldest practice?", "Explain it in two sentences.",
                 "Now give one risk of it."]
    # multi-turn chats on the model's own replies: 6 conversations (more than the 4 slots for the churn phase)
    for c in range(6):
        sys_msg = {"role": "system", "content": f"You are a concise assistant. Notes: {filler(150 + 400 * c, c)}"}
        msgs = [sys_msg]
        for t, q in enumerate(questions[:3 if c >= 2 else 4]):
            msgs.append({"role": "user", "content": f"{q} (conversation {c})"})
            r = emit("chat", f"chat{c}", t, body(list(msgs), a.max_tokens, sampled=(c == 5)))
            msgs.append({"role": "assistant", "content": r["content"]})
    # the burst: four last turns sharing conversation 0's history (recorded one at a time here)
    hist = []
    for line in open(a.ref):
        e = json.loads(line)
        if e["conv"] == "chat0":
            hist = e["body"]["messages"] + [{"role": "assistant", "content": e["content"]}]
    for k in range(4):
        emit("burst", "chat0", 10 + k, body(hist + [{"role": "user", "content": f"One more fact, number {k + 1}."}], a.max_tokens))
    # a scripted agent session: tool outputs grow the history by ~1.5K tokens a step (crossing the window and chunks)
    msgs = [{"role": "system", "content": "You are a coding agent. Use the tools, then answer briefly."},
            {"role": "user", "content": "Find where the cache is sized and tell me the default."}]
    for t in range(5):
        r = emit("agent", "agent", t, body(list(msgs), a.max_tokens, tools=TOOLS))
        call_id = f"call_{t}"
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {
            "name": "read_file", "arguments": json.dumps({"path": f"src/part{t}.zig"})}}]})
        msgs.append({"role": "tool", "tool_call_id": call_id, "content": f"// part {t}\n" + filler(1100, t)})
    if a.long > 0:  # a long document: every turn re-reads it on the cold serve
        doc = {"role": "user", "content": "Document:\n" + filler(int(a.long * 0.75), 3) + "\n\nSay 'ready'."}
        msgs = [{"role": "system", "content": "Answer questions about the document in one sentence."}, doc]
        for t, q in enumerate(["ready?", "Which word appears first?", "Which word appears last?"]):
            if t > 0:
                msgs.append({"role": "user", "content": q})
            r = emit("long", "long", t, body(list(msgs), a.max_tokens))
            msgs.append({"role": "assistant", "content": r["content"]})
    # the edge sweep: the first request keeps a mark at exactly T tokens (its history), the second resumes there
    for T in EDGES:
        base_msgs = [{"role": "system", "content": "Reply with one word."}]
        lo, hi = 0, T
        while lo < hi:  # the most filler words whose history stays at or under T
            mid = (lo + hi + 1) // 2
            if history_len(a.base, base_msgs + [{"role": "user", "content": filler(mid, T)}]) <= T:
                lo = mid
            else:
                hi = mid - 1
        msgs = base_msgs + [{"role": "user", "content": filler(lo, T)}]
        for pad in ["", " ok", " ok.", " ok ok", ".", " a", " a a"]:
            got = history_len(a.base, base_msgs + [{"role": "user", "content": filler(lo, T) + pad}])
            if got == T:
                msgs = base_msgs + [{"role": "user", "content": filler(lo, T) + pad}]
                break
        at = history_len(a.base, msgs)
        emit("edge", f"edge{T}", 0, body(msgs, 1), mark=at)
        follow = msgs + [{"role": "assistant", "content": "Noted."}, {"role": "user", "content": "Now one more word."}]
        emit("edge", f"edge{T}", 1, body(follow, a.max_tokens), mark=at)
    print(f"recorded {n} requests into {a.ref}")


def replay(a):
    ref = [json.loads(line) for line in open(a.ref)]
    results, lock = [], threading.Lock()

    def send(e, phase):
        b = dict(e["body"])
        if a.draft_false:
            b["draft"] = False
        try:
            r = chat(a.base, b)
        except Exception as x:  # a failed request is a failed contract
            r = {"sha": None, "error": str(x), "cached": None, "prompt": None, "wall": None, "ttft": None}
        ok = r["sha"] == e["sha"] and r["sha"] is not None
        row = {"phase": phase, "scenario": e["scenario"], "conv": e["conv"], "turn": e["turn"], "ok": ok,
               "ref_sha": e["sha"], **{k: r.get(k) for k in ("sha", "prompt", "cached", "ttft", "prefill_s", "wall", "error")}}
        if "mark" in e:
            row["mark"] = e["mark"]
        with lock:
            results.append(row)
            print(f"{phase:10s} {e['conv']:10s} t{e['turn']:<2d} {'ok ' if ok else 'DIFF'} prompt={r.get('prompt')} "
                  f"cached={r.get('cached')} ttft={r.get('ttft')} wall={r.get('wall')}", flush=True)

    def convs(rows):
        by = {}
        for e in rows:
            by.setdefault(e["conv"], []).append(e)
        return [sorted(v, key=lambda e: e["turn"]) for v in by.values()]

    def parallel(groups, phase, width):
        """`width` workers, each taking whole groups in order (a group's requests one after another)."""
        it = iter(groups)
        it_lock = threading.Lock()

        def worker():
            while True:
                with it_lock:
                    g = next(it, None)
                if g is None:
                    return
                for e in g:
                    send(e, phase)
        ts = [threading.Thread(target=worker) for _ in range(width)]
        [t.start() for t in ts]
        [t.join() for t in ts]

    phases = a.phases.split(",")
    main = [e for e in ref if e["scenario"] in ("chat", "agent", "long")]
    if "solo" in phases:
        for e in main:
            send(e, "solo")
    if "concurrent" in phases:  # four conversations at once, each in turn order
        parallel(convs([e for e in main if e["scenario"] != "long"]), "concurrent", 4)
    if "burst" in phases:  # the chat0 history first (alone), then four requests resuming its state at once
        burst = [e for e in ref if e["scenario"] == "burst"]
        warm = [e for e in ref if e["conv"] == "chat0" and e["scenario"] == "chat"]
        for e in warm:
            send(e, "burst-warm")
        parallel([[e] for e in burst], "burst", 4)
    if "churn" in phases:  # six conversations round robin on four slots: kept states end and requests run fresh
        cs = convs([e for e in ref if e["scenario"] == "chat"])
        turns = max(len(c) for c in cs)
        for t in range(turns):
            parallel([[c[t]] for c in cs if t < len(c)], "churn", 2)
    if "edge" in phases:
        for e in [e for e in ref if e["scenario"] == "edge"]:
            send(e, "edge")
    with open(a.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    bad = [r for r in results if not r["ok"]]
    resumed = [r for r in results if (r.get("cached") or 0) > 0]
    edge_hits = [r for r in results if r["phase"] == "edge" and r["turn"] == 1 and r.get("cached") == r.get("mark")]
    edges = [r for r in results if r["phase"] == "edge" and r["turn"] == 1]
    summary = {"requests": len(results), "identical": len(results) - len(bad), "different": len(bad),
               "resumed": len(resumed), "cached_tokens": sum(r.get("cached") or 0 for r in results),
               "prompt_tokens": sum(r.get("prompt") or 0 for r in results),
               "edge_resumed_at_mark": f"{len(edge_hits)}/{len(edges)}"}
    print("SUMMARY " + json.dumps(summary))
    with open(a.out.replace(".jsonl", "") + "-summary.json", "w") as f:
        json.dump({"summary": summary, "different": bad}, f, indent=1)
    sys.exit(1 if bad else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["record", "replay"])
    p.add_argument("--base", default="http://127.0.0.1:8888")
    p.add_argument("--ref", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--long", type=int, default=20000, help="tokens of the long document scenario (0: none)")
    p.add_argument("--phases", default="solo,concurrent,burst,churn,edge")
    p.add_argument("--draft-false", action="store_true", help="send \"draft\": false with every request")
    a = p.parse_args()
    a.base = a.base.rstrip("/")
    if a.mode == "record":
        record(a)
    else:
        if not a.out:
            p.error("replay needs --out")
        replay(a)


if __name__ == "__main__":
    main()
