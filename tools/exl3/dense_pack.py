#!/usr/bin/env python3
"""Assemble and check a dense-requantized EXL3 pack (standard library only, runs on the host).

A dense requantization (exllamav3 `python3 -m exllamav3.conversion.requant_dense`, patches-exl3 0005) writes only the
new dense linears of each decoder block to <work>/qtensors/<block key>.safetensors. This tool builds the output pack
from the base pack plus one or more such qtensors dirs (one per node of a split run):

- same shard files and the same tensor-to-shard map as the base pack; inside each shard the tensors keep the base
  order, a requantized linear's new tensors (trellis, suh, svh, mul1, ...) taking the place of its old ones;
- every other tensor (routed experts, embedding, norms, router, residual gates, expert score bias, head) is a byte
  copy of the base pack's;
- model.safetensors.index.json (new total size), config.json (quantization_config: bits recomputed over all EXL3
  matrices except the head, as the converter reports it, plus a `dense_requant` record) and quantization_config.json
  (the base pack's, with the requantized modules' stored tensors and bits_per_weight updated, as exllamav3's
  create_quantization_config_json would write them); every other file of the base pack is copied.

  dense_pack.py assemble --base <pack> --qtensors <work>/qtensors [<work2>/qtensors ...] --out <new pack> \
      [--expect-bits 3] [--note "..."]
  dense_pack.py check --base <pack> --pack <new pack> --qtensors <dir> [<dir> ...] [--sample N]

`check` re-reads both packs: every kept tensor byte-identical to the base pack's (all of them, or every Nth with
--sample N), every requantized tensor byte-identical to its qtensors source, the same key set apart from the
requantized linears, shard sizes equal to their header ends, and prints the bits per tensor kind.
quantization_config.json is written last: its presence marks a complete pack.
"""

import argparse
import collections
import json
import os
import re
import shutil
import struct
import sys
import time

EXL3_SUBKEYS = ("su", "sv", "suh", "svh", "trellis", "bias", "mcg", "mul1")
TORCH_DTYPES = {"F16": "torch.float16", "BF16": "torch.bfloat16", "F32": "torch.float32", "I16": "torch.int16",
                "I32": "torch.int32", "I64": "torch.int64", "U8": "torch.uint8", "I8": "torch.int8",
                "U32": "torch.uint32", "F64": "torch.float64"}
CHUNK = 64 << 20


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h, 8 + n


def split_key(key):
    """'<linear>.<sub>' -> (linear, sub) for EXL3 subkeys, else (None, None)"""
    base, _, sub = key.rpartition(".")
    return (base, sub) if sub in EXL3_SUBKEYS else (None, None)


def weight_map(pack):
    p = os.path.join(pack, "model.safetensors.index.json")
    if os.path.exists(p):
        return json.load(open(p))["weight_map"], True
    h, _ = read_header(os.path.join(pack, "model.safetensors"))
    return {k: "model.safetensors" for k in h if k != "__metadata__"}, False


def copy_range(src_fd, dst_fd, src_off, dst_off, n):
    while n > 0:
        try:
            got = os.copy_file_range(src_fd, dst_fd, min(n, 1 << 30), src_off, dst_off)
        except OSError:
            got = 0
        if got <= 0:
            data = os.pread(src_fd, min(n, CHUNK), src_off)
            if not data:
                raise IOError("short read")
            got = os.pwrite(dst_fd, data, dst_off)
        src_off += got
        dst_off += got
        n -= got


def collect_new(qdirs):
    """key -> (file, abs data offset, n_bytes, dtype, shape), and linear -> set(keys), from the qtensors files"""
    new, linears = {}, collections.defaultdict(set)
    for d in qdirs:
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".safetensors"):
                continue
            path = os.path.join(d, fn)
            h, base = read_header(path)
            for k, v in h.items():
                if k == "__metadata__":
                    continue
                lin, sub = split_key(k)
                assert lin, f"{path}: {k} is not an EXL3 linear tensor"
                assert k not in new, f"{k} in two qtensors dirs ({new.get(k, ('?',))[0]} and {path})"
                a, b = v["data_offsets"]
                new[k] = (path, base + a, b - a, v["dtype"], v["shape"])
                linears[lin].add(k)
    return new, linears


def bits_of(shape):
    b = shape[-1] / 16
    return int(b) if float(b).is_integer() else b


def final_bits(shapes):
    """converter's 'Final bitrate (excluding head)': weighted over every EXL3 matrix but lm_head"""
    num = den = 0
    for k, s in shapes.items():
        if k.startswith("lm_head"):
            continue
        n = s[0] * 16 * s[1] * 16
        num += n * s[2] / 16
        den += n
    return num / den


def kind_of(lin):
    if lin.startswith("lm_head"):
        return "head"
    if re.search(r"\.experts\.\d+\.", lin + "."):
        return "experts." + lin.rsplit(".", 1)[-1]
    return lin.rsplit(".", 1)[-1]


def assemble(a):
    base, out = a.base, a.out
    wm, indexed = weight_map(base)
    new, linears = collect_new(a.qtensors)
    for lin, ks in linears.items():
        assert f"{lin}.trellis" in wm, f"{lin}: no such EXL3 linear in the base pack"
        assert f"{lin}.trellis" in ks and ({f"{lin}.suh", f"{lin}.su"} & ks) and ({f"{lin}.svh", f"{lin}.sv"} & ks), \
            f"{lin}: incomplete new tensors {sorted(ks)}"
    replaced_old = {k for k in wm if split_key(k)[0] in linears}
    print(f" -- base {base}: {len(wm)} tensors in {len(set(wm.values()))} shards; replacing {len(linears)} linears "
          f"({len(replaced_old)} tensors -> {len(new)})", flush = True)
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f" ## {out} is not empty")
    os.makedirs(out, exist_ok = True)

    new_wm = {}
    total = 0
    shards = sorted(set(wm.values()))
    t0 = time.time()
    for si, shard in enumerate(shards):
        src = os.path.join(base, shard)
        h, data0 = read_header(src)
        meta = h.pop("__metadata__", None)
        order = sorted(h, key = lambda k: h[k]["data_offsets"][0])
        entries, plan, off = {}, [], 0
        for k in order:
            lin, sub = split_key(k)
            if lin in linears:
                # A replaced linear's new tensors go where its base trellis was (its other base tensors may sit in
                # another shard); every base tensor of it is dropped
                if sub != "trellis":
                    continue
                for nk in sorted(linears[lin]):
                    f, a0, n, dt, shape = new[nk]
                    entries[nk] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + n]}
                    plan.append((f, a0, off, n))
                    off += n
            else:
                v = h[k]
                a0, b0 = v["data_offsets"]
                entries[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + b0 - a0]}
                plan.append((src, data0 + a0, off, b0 - a0))
                off += b0 - a0
        if meta is not None:
            entries = {"__metadata__": meta, **entries}
        hdr = json.dumps(entries, separators = (",", ":")).encode()
        hdr += b" " * ((8 - len(hdr) % 8) % 8)
        dpos = 8 + len(hdr)
        tmp = os.path.join(out, shard + ".part")
        fds = {}
        fd = os.open(tmp, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.ftruncate(fd, dpos + off)
            os.pwrite(fd, struct.pack("<Q", len(hdr)) + hdr, 0)
            for f, a0, o, n in plan:
                if f not in fds:
                    fds[f] = os.open(f, os.O_RDONLY)
                copy_range(fds[f], fd, a0, dpos + o, n)
            os.fsync(fd)
        finally:
            os.close(fd)
            for x in fds.values():
                os.close(x)
        os.replace(tmp, os.path.join(out, shard))
        for k in entries:
            if k != "__metadata__":
                new_wm[k] = shard
        total += off
        print(f" -- {shard}: {len(entries)} tensors, {off / 1e9:.2f} GB  [{si + 1}/{len(shards)}, "
              f"{time.time() - t0:.0f} s]", flush = True)

    missing = set(linears) - {split_key(k)[0] for k in new_wm}
    assert not missing, f"linears not placed: {sorted(missing)[:3]}"
    assert set(new_wm) == (set(wm) - replaced_old) | set(new), "key set mismatch"

    if indexed:
        idx = json.load(open(os.path.join(base, "model.safetensors.index.json")))
        idx.setdefault("metadata", {})["total_size"] = total
        idx["weight_map"] = new_wm
        with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
            f.write(json.dumps(idx, indent = 4))

    # Shapes of every EXL3 matrix in the new pack (bits) from the headers just written
    shapes = {}
    for shard in shards:
        h, _ = read_header(os.path.join(out, shard))
        for k, v in h.items():
            if k.endswith(".trellis"):
                shapes[k[: -len(".trellis")]] = v["shape"]
    fb = round(final_bits(shapes), 2)
    by_kind = collections.defaultdict(collections.Counter)
    for lin in linears:
        by_kind[kind_of(lin)][bits_of(shapes[lin])] += 1
    dense_bits = {k: dict(sorted(c.items())) for k, c in sorted(by_kind.items())}
    if a.expect_bits is not None:
        bad = [lin for lin in linears if bits_of(shapes[lin]) not in a.expect_bits]
        assert not bad, f"unexpected bits: {[(l, bits_of(shapes[l])) for l in bad[:3]]}"

    # Other files: config.json, quantization_config.json rewritten; everything else copied
    for fn in sorted(os.listdir(base)):
        p = os.path.join(base, fn)
        if fn.endswith(".safetensors") or fn in ("model.safetensors.index.json", "config.json",
                                                  "quantization_config.json"):
            continue
        if os.path.isdir(p):
            shutil.copytree(p, os.path.join(out, fn))
        else:
            shutil.copy2(p, os.path.join(out, fn))
    record = {
        "base": os.path.basename(os.path.normpath(base)),
        "base_bits": None,
        "requantized_linears": len(linears),
        "bits_by_kind": dense_bits,
        "tool": "exllamav3.conversion.requant_dense + dense_pack.py",
    }
    if a.note:
        record["note"] = a.note
    cfg = json.load(open(os.path.join(base, "config.json")))
    q = cfg["quantization_config"]
    record["base_bits"] = q.get("bits")
    q["bits"] = fb
    q["dense_requant"] = record
    with open(os.path.join(out, "config.json"), "w") as f:
        f.write(json.dumps(cfg, indent = 4))

    qpath = os.path.join(base, "quantization_config.json")
    if os.path.exists(qpath):
        qc = json.load(open(qpath))
        qc["bits"] = fb
        qc["dense_requant"] = record
        ts = qc.get("tensor_storage", {})
        for lin, ks in linears.items():
            ent = ts.get(lin)
            assert ent is not None, f"{lin}: not in the base pack's quantization_config.json"
            old = ent.get("stored_tensors", {})
            st = {}
            for k in [k for k in old if k in ks] + sorted(ks - set(old)):     # the base entry's (canonical) order
                _, _, n, dt, shape = new[k]
                st[k] = {"shape": shape, "n_bytes": n, "dtype": TORCH_DTYPES.get(dt, old.get(k, {}).get("dtype"))}
            ent["stored_tensors"] = st
            ent["quant_format"] = "exl3"
            ent["bits_per_weight"] = bits_of(new[f"{lin}.trellis"][4])
            mk = f"{lin}.mul1"
            if mk in new:
                f, a0, n, dt, _ = new[mk]
                with open(f, "rb") as fh:
                    fh.seek(a0)
                    ent["mul1_multiplier"] = int.from_bytes(fh.read(n), "little")
        with open(os.path.join(out, "quantization_config.json"), "w") as f:
            f.write(json.dumps(qc, indent = 4))
    print(f" -- {out}: {len(new_wm)} tensors, {total / 1e9:.2f} GB, bits {fb} (base {record['base_bits']}); "
          f"requantized: {json.dumps(dense_bits)}", flush = True)


def same(fd1, off1, fd2, off2, n):
    """byte equality of two file ranges (chunked compare, no hashing)"""
    while n > 0:
        m = min(n, CHUNK)
        a, b = os.pread(fd1, m, off1), os.pread(fd2, m, off2)
        if len(a) != m or a != b:
            return False
        off1 += m
        off2 += m
        n -= m
    return True


def check(a):
    bwm, _ = weight_map(a.base)
    pwm, _ = weight_map(a.pack)
    new, linears = collect_new(a.qtensors) if a.qtensors else ({}, {})
    replaced_old = {k for k in bwm if split_key(k)[0] in linears}
    errors = []
    if set(pwm) != (set(bwm) - replaced_old) | set(new):
        errors.append(f"key sets differ: {len(set(pwm) ^ ((set(bwm) - replaced_old) | set(new)))} keys")
    if set(bwm.values()) != set(pwm.values()):
        errors.append("shard file sets differ")
    hdrs = {}
    for pack, wm in ((a.base, bwm), (a.pack, pwm)):
        for shard in sorted(set(wm.values())):
            p = os.path.join(pack, shard)
            h, d0 = read_header(p)
            end = max((v["data_offsets"][1] for k, v in h.items() if k != "__metadata__"), default = 0)
            if os.path.getsize(p) != d0 + end:
                errors.append(f"{p}: size != header end")
            hdrs[p] = (h, d0)
    fds = {}
    def fd(p):
        if p not in fds:
            fds[p] = os.open(p, os.O_RDONLY)
        return fds[p]
    kept = sorted(k for k in pwm if k not in new)
    sample = kept if not a.sample else kept[:: a.sample]
    t0 = time.time()
    n_bytes = 0
    for i, k in enumerate(sample):
        bp, pp = os.path.join(a.base, bwm[k]), os.path.join(a.pack, pwm[k])
        (bh, bd0), (ph, pd0) = hdrs[bp], hdrs[pp]
        bv, pv = bh[k], ph[k]
        if bv["dtype"] != pv["dtype"] or bv["shape"] != pv["shape"]:
            errors.append(f"{k}: dtype/shape differ")
            continue
        n = bv["data_offsets"][1] - bv["data_offsets"][0]
        if not same(fd(bp), bd0 + bv["data_offsets"][0], fd(pp), pd0 + pv["data_offsets"][0], n):
            errors.append(f"{k}: bytes differ")
        n_bytes += n
        if i and i % 5000 == 0:
            print(f"    {i}/{len(sample)} kept tensors compared ({n_bytes / 1e9:.0f} GB, {time.time() - t0:.0f} s)",
                  flush = True)
    for k, (f, a0, n, dt, shape) in new.items():
        pp = os.path.join(a.pack, pwm[k])
        ph, pd0 = hdrs[pp]
        pv = ph[k]
        if pv["dtype"] != dt or pv["shape"] != shape or not same(fd(f), a0, fd(pp), pd0 + pv["data_offsets"][0], n):
            errors.append(f"{k}: differs from its qtensors source")
    for x in fds.values():
        os.close(x)
    kinds = collections.defaultdict(collections.Counter)
    for k in pwm:
        if k.endswith(".trellis"):
            p = os.path.join(a.pack, pwm[k])
            kinds[kind_of(k[: -len(".trellis")])][bits_of(hdrs[p][0][k]["shape"])] += 1
    kinds = {k: dict(c) for k, c in sorted(kinds.items())}
    print(f" -- kept tensors compared: {len(sample)} of {len(kept)} ({n_bytes / 1e9:.1f} GB) in {time.time() - t0:.0f} s; "
          f"requantized tensors compared: {len(new)}")
    print(f" -- EXL3 matrices by kind and bits: {json.dumps(kinds)}")
    for e in errors[:20]:
        print("   ", e)
    print("OK" if not errors else f"FAIL ({len(errors)} errors)")
    return 0 if not errors else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest = "cmd", required = True)
    p = sub.add_parser("assemble")
    p.add_argument("--base", required = True)
    p.add_argument("--qtensors", nargs = "+", required = True)
    p.add_argument("--out", required = True)
    p.add_argument("--expect-bits", type = float, nargs = "+", default = None)
    p.add_argument("--note", default = None)
    p = sub.add_parser("check")
    p.add_argument("--base", required = True)
    p.add_argument("--pack", required = True)
    p.add_argument("--qtensors", nargs = "*", default = [])
    p.add_argument("--sample", type = int, default = 0, help = "compare every Nth kept tensor (0: all)")
    a = ap.parse_args()
    if a.cmd == "assemble":
        assemble(a)
    else:
        sys.exit(check(a))


if __name__ == "__main__":
    main()
