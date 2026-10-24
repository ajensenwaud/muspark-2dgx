#!/usr/bin/env python3
"""Drafted == plain on spark-bench's fixed texts (stdlib only): each of the 12 decode prompts with a fixed tag, greedy,
thinking off as spark-bench sends it, ignore_eos, N tokens; once with drafting (serial), once with `draft: false`
(serial), and the drafted set again 4 at a time. PASS when every drafted reply (content, reasoning and token sha) is
byte-identical to its draft:false reply. Prints per request tok/s and the server's speculative counters.

  python3 tools/g6/drafted_exact.py http://127.0.0.1:8888 Muse-Spark-Open OUT.json [--tokens 256] [--sb DIR]
"""
import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--tokens", type=int, default=256)
ap.add_argument("--sb", default=str(Path.home() / "muse/spark-bench"))
a = ap.parse_args()
sys.path.insert(0, a.sb + "/scripts")
from bench import WORKLOADS  # noqa: E402  (spark-bench's own prompts)

texts = [(f"{wl}{i}", f"[run {i:08x}] " + p) for wl, ps in WORKLOADS.items() for i, p in enumerate(ps)]


def call(text, draft):
    body = {"model": a.model, "messages": [{"role": "user", "content": text}], "max_tokens": a.tokens,
            "temperature": 0, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    if not draft:
        body["draft"] = False
    req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.load(r)
    m = j["choices"][0]["message"]
    rt = j.get("tensorfold") or {}  # the server's runtime extras: token_sha, tokens_per_second
    sp = j.get("speculative") or {}
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
            "sha": rt.get("token_sha"), "tps": rt.get("tokens_per_second"), "wall": time.time() - t0,
            "tokens": j["usage"]["completion_tokens"], "spec": sp}


res = {"plain": {}, "drafted": {}, "drafted4": {}}
for name, t in texts:
    res["plain"][name] = call(t, False)
for name, t in texts:
    res["drafted"][name] = call(t, True)
with ThreadPoolExecutor(4) as ex:
    for (name, _), r in zip(texts, ex.map(lambda nt: call(nt[1], True), texts)):
        res["drafted4"][name] = r
bad = []
key = lambda r: (r["content"], r["reasoning"], r["sha"])  # noqa: E731
for name, _ in texts:
    p, d, d4 = res["plain"][name], res["drafted"][name], res["drafted4"][name]
    same, same4 = key(p) == key(d), key(p) == key(d4)
    if not (same and same4):
        bad.append(name)
    sp = d["spec"]
    print(f"{name:8s} plain {p['tps'] or 0:6.2f} tok/s  drafted {d['tps'] or 0:6.2f} tok/s  rounds {sp.get('rounds')} "
          f"accepted {sp.get('accepted')}/{sp.get('drafted')}  sha {str(d['sha'])[:12]}  serial {'==' if same else '!='}  "
          f"x4 {'==' if same4 else '!='}", flush=True)
res["pass"] = not bad
res["differ"] = bad
json.dump(res, open(a.out, "w"), indent=1)
print("PASS" if not bad else f"FAIL {bad}")
sys.exit(0 if not bad else 1)
