#!/usr/bin/env python3
"""G6 drafter gate (a): reference DFlash2 proposals under the fixed protocol, compared with the Zig drafter's.

The Zig side (`tf-muse dflash` with MUSE_DUMP_TAPS=1, both TP ranks over the full pack) writes, per text:
  <name>.seq.i32     prompt + the target's greedy continuation
  <name>.drafts.i32  [G, block - 1]: the Zig drafter's proposals from greedy token i (anchor at position P + i)
  <name>.taps.f16    [P + G, taps * hidden]: the drafter's inputs, the target's residuals after layers 1/16/30/44/59
This tool feeds the same anchors and taps to the reference trainer's DFlash2DraftModel (tools/g6/dflash_ref_model.py,
z-lab/dflash verbatim), with the target's raw embedding rows and its EXL3 head from the pack (exllamav3), and writes
<name>.ref_drafts.i32 plus a JSON summary: per-depth token agreement and each side's mean accepted length against the
greedy text (exactly what a greedy verify would accept at that position).

Weights: --weights bf16 (the released drafter: the quantisation's cost shows in the gap) or exl3 (the drafter pack the
Zig engine runs, dequantised by exllamav3: isolates the port). Note on exllamav3: its muse_spark head already applies
output_multiplier and the softcap, and stock DFlash2Model.sample_from_state applies them again; the trainer applies
them once (compute_logits on the raw lm_head), so this tool sets the drafter's own scale to 1 / no softcap.

Run in the recipe image on one GPU (~15 GB): tools/g6/dflash_gate.sh.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_target_parts(pack, dev):
    from exllamav3 import Config, Model

    cfg = Config.from_directory(pack)
    model = Model.from_config(cfg)
    emb = cfg.stc.get_tensor("model.language_model.embed_tokens.weight", dev, allow_bf16=True)
    head = model.modules[model.logit_layer_idx]
    head.load(dev)
    vocab = cfg.vocab_size

    def output_head(h):
        shape = h.shape
        y = head.forward(h.reshape(-1, shape[-1]).half().contiguous(), {})
        return y[..., :vocab].float().reshape(*shape[:-1], -1)

    return emb, output_head, vocab


def walk(m):
    yield m
    for c in getattr(m, "modules", []) or []:
        yield from walk(c)


def build_drafter(drafter_dir, weights, exl3_dir, dev):
    from transformers import Qwen3Config
    from dflash_ref_model import DFlash2DraftModel

    if weights == "bf16":
        model = DFlash2DraftModel.from_pretrained(drafter_dir, dtype=torch.bfloat16).to(dev)
    else:
        from exllamav3 import Config, Model

        qcfg = Qwen3Config.from_pretrained(drafter_dir)
        with torch.device("meta"):
            model = DFlash2DraftModel(qcfg)
        model = model.to_empty(device=dev).to(torch.bfloat16)
        ecfg = Config.from_directory(exl3_dir)
        emodel = Model.from_config(ecfg)
        linears = {}
        for top in emodel.modules:
            for m in walk(top):
                if type(m).__name__ == "Linear":
                    linears[m.key] = m
        done = set()
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                lin = linears[name]
                lin.load(dev)
                w = lin.inner.get_weight_tensor()   # [in, out] for EXL3 and fp16 inners alike
                assert tuple(w.shape) == tuple(mod.weight.shape[::-1]), (name, tuple(w.shape))
                mod.weight.data.copy_(w.T.to(torch.bfloat16))
                lin.unload()
                done.add(name + ".weight")
        for name, p in model.named_parameters():
            if name in done:
                continue
            key = name
            if name.startswith("candidate_selector.") and name.endswith("_codebook.weight"):
                key = name[: -len(".weight")]
            t = ecfg.stc.get_tensor(key, dev, allow_bf16=True)
            p.data.copy_(t.reshape(p.shape).to(torch.bfloat16))
        model.rotary_emb = type(model.rotary_emb)(qcfg, device=dev)
    model.eval()
    # the target's EXL3 head already scales and softcaps: apply them once (see the docstring)
    model.config.dflash_config = dict(getattr(model.config, "dflash_config", {}) or {})
    model.config.dflash_config["output_multiplier"] = 1.0
    model.config.dflash_config["final_logit_softcapping"] = 0
    return model


@torch.inference_mode()
def ref_proposals(model, emb, head, seq, taps, P, G, block, window, mask_id, dev, max_anchors):
    out = np.full((G, block - 1), -1, dtype=np.int32)
    n = G if max_anchors <= 0 else min(G, max_anchors)
    for i in range(n):
        a = P + i
        lo = max(0, a - (window - 1))
        th = torch.from_numpy(taps[lo:a].astype(np.float16)).to(dev).to(torch.bfloat16)[None]
        ids = torch.tensor([[int(seq[a])] + [mask_id] * (block - 1)], device=dev)
        noise = F.embedding(ids, emb).to(torch.bfloat16)
        pos = torch.arange(lo, a + block, device=dev)[None]
        h = model(target_hidden=th, noise_embedding=noise, position_ids=pos, past_key_values=None, use_cache=False)
        h = h[:, 1:, :]
        toks, _, _ = model.propose(h, ids[:, 0], head, 0.0)
        out[i] = toks[0].cpu().numpy().astype(np.int32)
    return out, n


def accepted(drafts, seq, P):
    G, D = drafts.shape
    acc = []
    for i in range(G):
        a = 0
        while a < D and P + i + 1 + a < len(seq) and drafts[i, a] == seq[P + i + 1 + a]:
            a += 1
        acc.append(a)
    return np.array(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="the target EXL3 pack (embedding, head)")
    ap.add_argument("--drafter", required=True, help="the bf16 DFlash2 release (config.json + safetensors)")
    ap.add_argument("--drafter-exl3", default=None, help="the drafter's EXL3 pack (with --weights exl3)")
    ap.add_argument("--weights", choices=["bf16", "exl3"], default="exl3")
    ap.add_argument("--dump", required=True, help="tf-muse dflash OUT_DIR (seq, drafts, taps)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-anchors", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    cfg = json.load(open(os.path.join(args.drafter, "config.json")))
    dfc = cfg.get("dflash_config", {})
    block, mask_id, window = dfc["block_size"], dfc["mask_token_id"], cfg.get("sliding_window", 2048)
    ntaps, hidden = len(dfc["target_layer_ids"]), cfg["hidden_size"]
    emb, head, vocab = load_target_parts(args.pack, dev)
    model = build_drafter(args.drafter, args.weights, args.drafter_exl3, dev)
    os.makedirs(args.out, exist_ok=True)
    names = sorted(f[: -len(".drafts.i32")] for f in os.listdir(args.dump) if f.endswith(".drafts.i32"))
    summary = {"weights": args.weights, "block": block, "texts": {}}
    for name in names:
        seq = np.fromfile(os.path.join(args.dump, f"{name}.seq.i32"), dtype=np.int32)
        zig = np.fromfile(os.path.join(args.dump, f"{name}.drafts.i32"), dtype=np.int32).reshape(-1, block - 1)
        taps = np.fromfile(os.path.join(args.dump, f"{name}.taps.f16"), dtype=np.float16).reshape(-1, ntaps * hidden)
        G = zig.shape[0]
        P = len(seq) - G
        ref, n = ref_proposals(model, emb, head, seq, taps, P, G, block, window, mask_id, dev, args.max_anchors)
        ref.tofile(os.path.join(args.out, f"{name}.ref_drafts.i32"))
        z, r = zig[:n], ref[:n]
        agree = [(z[:, k] == r[:, k]).mean().item() for k in range(block - 1)]
        prefix_eq = (z == r).all(axis=1).mean().item()
        acc_z, acc_r = accepted(z, seq, P), accepted(r, seq, P)
        row = {"prompt": int(P), "anchors": int(n), "agree_by_depth": agree, "whole_block_equal": prefix_eq,
               "mean_accepted_zig": float(acc_z.mean()), "mean_accepted_ref": float(acc_r.mean())}
        summary["texts"][name] = row
        print(f"[dflash_ref] {name}: anchors {n}, agree d1 {agree[0]:.3f} d7 {agree[-1]:.3f}, whole block "
              f"{prefix_eq:.3f}; mean accepted zig {acc_z.mean():.3f} ref {acc_r.mean():.3f}", flush=True)
    za = np.mean([t["mean_accepted_zig"] for t in summary["texts"].values()])
    ra = np.mean([t["mean_accepted_ref"] for t in summary["texts"].values()])
    a1 = np.mean([t["agree_by_depth"][0] for t in summary["texts"].values()])
    # pass: the Zig drafter accepts within 10 % of the reference and agrees on >= 90 % of first drafts (same weights)
    ok = za >= 0.9 * ra and (args.weights == "bf16" or a1 >= 0.9)
    summary.update({"mean_accepted_zig": za, "mean_accepted_ref": ra, "agree_d1": a1, "pass": bool(ok)})
    json.dump(summary, open(os.path.join(args.out, f"dflash_ref_{args.weights}.json"), "w"), indent=1)
    print(f"[dflash_ref] {args.weights}: mean accepted zig {za:.3f} ref {ra:.3f}, first-draft agreement {a1:.3f}: "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
