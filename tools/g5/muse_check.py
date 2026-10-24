#!/usr/bin/env python3
"""G5 parity: score tf-muse m1 dumps against exllamav3's answer key (tools/g5/muse_ref.py), against the noise floor.

  python3 tools/g5/muse_check.py REF_DIR ZIG_DIR [--json out.json] [--teacher TEACHER_ZIG_DIR]

Per text: KL(ref || zig) mean / max and top-1 agreement over every position, the same against exllamav3's own
prefix-vs-full noise floor (meta.json), true-token NLL under both engines, greedy prefix match. With captures
(MUSE_CAPTURE=1): per layer, free-running error of each block's output (and of the post-attention residual)
relative to the layer's own update |ref_out - ref_in|, beside exllamav3's per-layer prefix noise; and per token
(tokens_*: median, p99, outlier count, aggregate without outliers) to set routing near-tie flips apart. With a
teacher-forced run (MUSE_TEACHER=1, MUSE_CAPTURE=1): the same per layer with every layer fed the reference's input,
which localises a wiring bug to its layer and sublayer. CPU only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load(path: Path, dtype, shape=None):
    a = np.fromfile(path, dtype=dtype)
    t = torch.from_numpy(a)
    return t.view(*shape) if shape is not None else t


def per_layer(ref_states, ref_attn, got_res, got_attn):
    """[(block rel, attn rel)] per layer; rel = |got - ref| / |ref_out - ref_in| over all rows."""
    rows = []
    for i in range(got_res.shape[0]):
        r_in, r_out = ref_states[i], ref_states[i + 1]
        delta = (r_out - r_in).norm()
        blk = float((got_res[i] - r_out).norm() / delta)
        att = float((got_attn[i] - ref_attn[i]).norm() / (ref_attn[i] - r_in).norm()) if got_attn is not None else None
        rows.append((blk, att))
    return rows


def token_spread(ref_states, got_res):
    """Per layer, the block error per token |got_t - ref_t| / |ref_out_t - ref_in_t|: (median, p99, tokens over 10x the
    median, aggregate rel without them). Routing near-ties (top-8 margins ~1e-5, exllamav3 patch 0004) put a few
    tokens on another expert; those show as isolated outliers, a wiring bug as a raised median."""
    out = []
    for i in range(got_res.shape[0]):
        r_in, r_out = ref_states[i], ref_states[i + 1]
        num = (got_res[i] - r_out).norm(dim=-1)
        den = (r_out - r_in).norm(dim=-1).clamp_min(1e-30)
        t = num / den
        med = float(t.median())
        bad = t > 10 * med
        keep = ~bad
        agg = float(num[keep].pow(2).sum().sqrt() / den[keep].pow(2).sum().sqrt()) if keep.any() else None
        out.append([round(med, 5), round(float(torch.quantile(t.double(), 0.99)), 5), int(bad.sum()),
                    round(agg, 5) if agg is not None else None])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("zig")
    ap.add_argument("--teacher", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    ref, zig = Path(a.ref), Path(a.zig)
    meta = json.loads((ref / "meta.json").read_text())
    V = meta["vocab"]
    report = {}
    for name in (ref / "names.txt").read_text().split():
        ids = load(ref / f"{name}.ids", np.int32).long()
        T = ids.numel()
        rl = load(ref / f"{name}.ref_logits.f32", np.float32, (T, V))
        zf = zig / f"{name}.logits.f32"
        if not zf.exists():
            continue
        zl = load(zf, np.float32, (T, V))
        kls, nll_r, nll_z = [], 0.0, 0.0
        for i in range(0, T, 256):  # chunked: long texts' [T, 202048] copies add up
            lp = torch.log_softmax(rl[i:i + 256].double(), -1)
            lq = torch.log_softmax(zl[i:i + 256].double(), -1)
            kls.append((lp.exp() * (lp - lq)).sum(-1))
            j = min(i + 256, T - 1) - i
            if j > 0:
                nxt = ids[i + 1:i + 1 + j, None]
                nll_r += float(-lp[:j].gather(1, nxt).sum())
                nll_z += float(-lq[:j].gather(1, nxt).sum())
        kl = torch.cat(kls)
        nll_r, nll_z = nll_r / (T - 1), nll_z / (T - 1)
        top1 = float((rl.argmax(-1) == zl.argmax(-1)).double().mean())
        m = meta["texts"][name]
        floor = {k: v["kl"] for k, v in m["noise"].items()}
        floor_kl = max(floor.values())
        g_ref = m["greedy"]
        g_zig = load(zig / f"{name}.greedy.i32", np.int32).tolist() if (zig / f"{name}.greedy.i32").exists() else []
        match = next((i for i, (x, y) in enumerate(zip(g_zig, g_ref)) if x != y), min(len(g_zig), len(g_ref)))
        worst = torch.topk(kl, min(5, T))
        r = {
            "tokens": T, "kl": float(kl.mean()), "kl_max": float(kl.max()), "top1": top1,
            "noise_kl": floor, "kl_over_floor": float(kl.mean()) / floor_kl if floor_kl > 0 else None,
            "nll_ref": nll_r, "nll_zig": nll_z, "nll_rel": (nll_z - nll_r) / nll_r,
            "greedy_match": f"{match}/{len(g_ref)}",
            "worst": [(int(i), round(float(v), 4)) for v, i in zip(worst.values, worst.indices)],
        }
        L = meta["layers"]
        if (zig / f"{name}.res.f32").exists():
            D = rl.new_empty(0)  # placeholder for shape inference below
            st = load(ref / f"{name}.states.f32", np.float32)
            D = st.numel() // ((L + 1) * T)
            st = st.view(L + 1, T, D)
            at = load(ref / f"{name}.attn_ref.f32", np.float32, (L, T, D))
            gr = load(zig / f"{name}.res.f32", np.float32)
            Lz = gr.numel() // (T * D)
            gr = gr.view(Lz, T, D)
            ga = load(zig / f"{name}.attn.f32", np.float32, (Lz, T, D))
            r["layers_free"] = [[round(x, 5) if x is not None else None for x in row] for row in per_layer(st, at, gr, ga)]
            r["tokens_free"] = token_spread(st, gr)
            nk = list(m["noise"].values())[-1].get("layers_rel")
            r["layers_noise"] = [round(x, 5) for x in nk] if nk else None
            if a.teacher and (Path(a.teacher) / f"{name}.res.f32").exists():
                tr = load(Path(a.teacher) / f"{name}.res.f32", np.float32, (Lz, T, D))
                ta = load(Path(a.teacher) / f"{name}.attn.f32", np.float32, (Lz, T, D))
                r["layers_teacher"] = [[round(x, 5) for x in row] for row in per_layer(st, at, tr, ta)]
                r["tokens_teacher"] = token_spread(st, tr)
        report[name] = r
        print(f"[check] {name}: T {T}  KL {r['kl']:.5f} (max {r['kl_max']:.4f})  floor {floor_kl:.5f}  "
              f"ratio {r['kl_over_floor']:.2f}  top1 {top1:.4f}  NLL ref {nll_r:.4f} zig {nll_z:.4f} "
              f"({100 * r['nll_rel']:+.2f}%)  greedy {r['greedy_match']}", flush=True)
        for key in ("layers_free", "layers_noise", "layers_teacher", "tokens_free", "tokens_teacher"):
            if key in r and r[key]:
                print(f"    {key}: {r[key]}", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
