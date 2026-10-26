#!/usr/bin/env python3
"""A Muse Spark serve's memory per rank before it loads (stdlib only), and the verdict start.sh acts on.

Per rank: weights ((pack bytes - embedding) / world; the engine keeps the embedding in host memory, read from the
pack's file mapping), the drafter pack / world, the fp16 KV caches for CONTEXT x SLOTS (full layers at the context,
sliding layers' rings at their window) and FIXED_GIB: everything else the serve holds on one rank, measured (CUDA
context, kernels, graphs, scratch, NCCL, the prompt cache budget, the process's host memory). Against AVAIL (each node's MemAvailable now, no serve running) it prints the free memory left per rank and
exits 3 when that is under MIN_FREE_GIB; it also prints the largest context (a multiple of 4096) that keeps it.

  estimate.py PACK [--drafter DIR] --context N --slots N --avail-mib A [A2 ...] [--min-free-gib 3] [--fixed-gib F]
"""
import argparse
import glob
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("pack")
ap.add_argument("--drafter", default="")
ap.add_argument("--context", type=int, required=True)
ap.add_argument("--slots", type=int, default=4)
ap.add_argument("--world", type=int, default=2)
ap.add_argument("--fixed-gib", type=float, default=float(os.environ.get("MEM_FIXED_GIB", "4.0")))
ap.add_argument("--avail-mib", type=int, nargs="+", required=True, help="MemAvailable per node now, MiB")
ap.add_argument("--min-free-gib", type=float, default=3.0)
ap.add_argument("--rows", type=int, default=128)  # forward.DECODE_ROWS
a = ap.parse_args()
G = 1 << 30

cfg = json.load(open(os.path.join(a.pack, "config.json")))
tc = cfg.get("text_config", cfg)
hidden, vocab = tc["hidden_size"], tc["vocab_size"]
hd, hk = tc["head_dim"], tc["num_key_value_heads"] // a.world
types = tc["layer_types"]
full = sum(t == "full_attention" for t in types)
slide = len(types) - full
ring = (tc["sliding_window"] + 1 + a.rows + 63) // 64 * 64  # forward.ringSize

pack = sum(os.path.getsize(f) for f in glob.glob(os.path.join(a.pack, "*.safetensors")))
embed = vocab * hidden * 2
weights = (pack - embed) / a.world
drafter = 0.0
if a.drafter:
    drafter = sum(os.path.getsize(f) for f in glob.glob(os.path.join(a.drafter, "*.safetensors"))) / a.world


def kv(ctx):
    per = 2 * a.slots * ctx * hk * hd * 2
    return full * per + slide * 2 * a.slots * ring * hk * hd * 2


def need(ctx):
    return weights + drafter + kv(ctx) + a.fixed_gib * G


avail = min(a.avail_mib) * (1 << 20)
free = avail - need(a.context)
best = 0
for c in range(4096, 1048576 + 1, 4096):
    if avail - need(c) >= a.min_free_gib * G:
        best = c
ok = free >= a.min_free_gib * G
print(f"memory estimate per rank: weights {weights / G:.1f} + drafter {drafter / G:.2f} + KV {kv(a.context) / G:.2f} "
      f"({a.context} x {a.slots}, {full} full + {slide} sliding layers, fp16) + fixed {a.fixed_gib:.1f}"
      f" = {need(a.context) / G:.1f} GiB; MemAvailable now {min(a.avail_mib) / 1024:.1f} GiB (lowest node) -> "
      f"{free / G:.1f} GiB left (need >= {a.min_free_gib:.1f}); largest context that keeps it: {best}")
print(f"VERDICT {'OK' if ok else 'REFUSE'} {best}")
sys.exit(0 if ok else 3)
