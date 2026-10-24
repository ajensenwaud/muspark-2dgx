#!/usr/bin/env python3
"""Integrity of an EXL3 pack (stdlib only, header reads, no tensor loads): every shard's size equals its header's
data end (catches truncation, e.g. a disk that filled during write-out), every index entry exists in its shard,
and the EXL3 matrix count (`.trellis` keys) equals the expected value when given.

  python3 verify_exl3_pack.py <pack dir> [expected matrices]

Runs on the host (no torch). Compare the count against a known-good pack of the same architecture (for example the
public baseline pack): MiMo-V2.6-Flash text packs hold 36,316.
"""

import json
import os
import struct
import sys


def main() -> None:
    d = sys.argv[1]
    expect = int(sys.argv[2]) if len(sys.argv) > 2 else None
    idx_path = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        idx = json.load(open(idx_path))["weight_map"]
    else:   # small packs: one model.safetensors, no index
        with open(os.path.join(d, "model.safetensors"), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            idx = {k: "model.safetensors" for k in json.loads(f.read(n)) if k != "__metadata__"}
    keys_in = {}
    bad = []
    for shard in sorted(set(idx.values())):
        p = os.path.join(d, shard)
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        end = max((v["data_offsets"][1] for k, v in h.items() if k != "__metadata__"), default=0)
        size = os.path.getsize(p)
        if size != 8 + n + end:
            bad.append(f"{shard}: size {size} != header end {8 + n + end}")
        keys_in[shard] = set(h) - {"__metadata__"}
    missing = [k for k, s in idx.items() if k not in keys_in[s]]
    mats = sum(k.endswith(".trellis") for k in idx)
    print(f"{os.path.basename(d.rstrip('/'))}: {len(keys_in)} shards, {len(idx)} tensors, {mats} EXL3 matrices, "
          f"{len(bad)} truncated, {len(missing)} missing")
    for b in bad + missing[:5]:
        print("  ", b)
    ok = not bad and not missing and (expect is None or mats == expect)
    print("OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
