#!/usr/bin/env python3
"""Smoke test of a served Muse Spark endpoint (stdlib only): one plain chat and one tool call, reasoning effort low,
greedy. Writes every raw response plus timings to OUT (JSON).

  python3 scripts/smoke.py http://127.0.0.1:8892 Muse-Spark-Open-UD-IQ2_XXS results/A-llamacpp-UD-IQ2_XXS/smoke.json
"""

import json
import sys
import time
import urllib.request

url, model, out = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]

CASES = {
    "chat": {"messages": [{"role": "user", "content":
             "In three sentences, explain why the sky is blue, then give the boiling point of water at sea level "
             "in Celsius and Fahrenheit."}], "max_tokens": 4096},
    "tool_call": {"messages": [{"role": "user", "content": "What is the weather like in Sydney right now?"}],
                  "tools": [{"type": "function", "function": {
                      "name": "get_weather", "description": "Current weather for a city.",
                      "parameters": {"type": "object", "properties": {
                          "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                          "required": ["city"]}}}],
                  "max_tokens": 1024},
}

res = {"url": url, "model": model, "cases": {}}
for name, body in CASES.items():
    body = {"model": model, "temperature": 0, "chat_template_kwargs": {"reasoning_effort": "low"}, **body}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.load(r)
    wall = time.time() - t0
    msg = j["choices"][0]["message"]
    res["cases"][name] = {"request": body, "wall_s": round(wall, 2), "response": j}
    print(f"== {name}: finish={j['choices'][0].get('finish_reason')} wall={wall:.1f}s usage={j.get('usage')} "
          f"timings={ {k: round(v, 2) for k, v in (j.get('timings') or {}).items() if isinstance(v, float)} }")
    print("reasoning:", (msg.get("reasoning_content") or "")[:400].replace("\n", " "))
    print("content:", (msg.get("content") or "")[:800])
    if msg.get("tool_calls"):
        print("tool_calls:", json.dumps(msg["tool_calls"]))
with open(out, "w") as f:
    json.dump(res, f, indent=1)
