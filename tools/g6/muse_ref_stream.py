#!/usr/bin/env python3
"""G6 answer key on the FULL pack (213 GB: no single GPU holds it), streamed module by module through exllamav3.

  python3 tools/g6/muse_ref_stream.py MODEL OUT_DIR [--texts tools/g5/muse_texts.json] [--max-len 512] [--states]

Writes what tools/g5/muse_ref.py writes, for tf-muse m1 at TP=2 and tools/g5/muse_check.py: names.txt, <name>.ids,
<name>.ref_logits.f32 [T, vocab], meta.json with the noise floor (forwards over the first T/2 and T-17 tokens against
the full forward, logits KL and per-layer errors), and with --states <name>.states.f32 [L + 1, T, D]. Each module
(embedding, embedding norm, 62 blocks, final norm, head) is loaded once, applied to every sequence (the texts and
their prefixes), and unloaded: one pass over the pack (~3.4 GB a layer) instead of a whole-model load. No greedy
continuation (a full pass a token); muse_check.py then reports greedy 0/0. Same calls as tools/g2/dryrun_forward.py
(flash_attn_nc, no cache). One GPU, ~10 GB; run in the recipe image (muse-tf:bbf461065a, router on fp32 input).
"""

import argparse
import json
import time
from pathlib import Path

import torch


def kl_stats(ref, got):
    kls = []
    for i in range(0, ref.shape[0], 256):
        lp = torch.log_softmax(ref[i:i + 256].float(), -1)
        lq = torch.log_softmax(got[i:i + 256].float(), -1)
        kls.append((lp.exp() * (lp - lq)).sum(-1))
    kl = torch.cat(kls)
    return {"kl": float(kl.mean()), "kl_max": float(kl.max()), "top1": float((ref.argmax(-1) == got.argmax(-1)).float().mean())}


def true_nll(logits, ids):
    total, T = 0.0, ids.numel()
    for i in range(0, T - 1, 256):
        j = min(i + 256, T - 1)
        lp = torch.log_softmax(logits[i:j].float(), -1)
        total += float(-lp.gather(1, ids[i + 1:j + 1, None].long()).sum())
    return total / (T - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--texts", default=str(Path(__file__).resolve().parents[1] / "g5" / "muse_texts.json"))
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--states", action="store_true")
    ap.add_argument("--names", default="")
    a = ap.parse_args()

    from exllamav3 import Config, Model, Tokenizer

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    config = Config.from_directory(a.model)
    model = Model.from_config(config)
    tok = Tokenizer.from_config(config)
    V = config.vocab_size
    texts = json.loads(Path(a.texts).read_text())
    if a.names:
        texts = {k: v for k, v in texts.items() if k in a.names.split(",")}
    seqs = {}   # key -> ids [1, T]; keys: name, name@k (prefixes for the noise floor)
    for name, text in texts.items():
        if not text.startswith("<|begin_of_text|>"):
            text = "<|begin_of_text|>" + text
        ids = tok.encode(text, add_bos=False, encode_special_tokens=True)[:, : a.max_len]
        T = ids.shape[1]
        seqs[name] = ids
        for k in (T // 2, T - 17):
            seqs[f"{name}@{k}"] = ids[:, :k]
    dev = torch.device("cuda:0")
    params = {"attn_mode": "flash_attn_nc"}
    x = {k: v for k, v in seqs.items()}
    states = {k: [] for k in seqs}
    first, last = model.first_block_idx, model.last_kv_module_idx
    for i, m in enumerate(model.modules):
        t1 = time.perf_counter()
        m.load(dev)
        with torch.inference_mode():
            for k in seqs:
                y = m.forward(m.prepare_for_device(x[k], params), dict(params))
                x[k] = y
                if first - 1 <= i <= last:   # the embedding norm's output, then every block's output
                    states[k].append(y.detach().float()[0].cpu())
        m.unload()
        torch.cuda.empty_cache()
        print(f"[ref-stream] module {i + 1}/{len(model.modules)} {type(m).__name__} {time.perf_counter() - t1:.1f}s", flush=True)
    logits = {k: v.float()[0, :, :V].cpu() for k, v in x.items()}
    meta = {"model": a.model, "layers": config.num_hidden_layers, "vocab": V, "streamed": True, "texts": {}}
    for name in texts:
        ids = seqs[name]
        T = ids.shape[1]
        ids[0].to(torch.int32).numpy().tofile(out / f"{name}.ids")
        logits[name].contiguous().numpy().tofile(out / f"{name}.ref_logits.f32")
        if a.states:
            torch.stack(states[name]).numpy().tofile(out / f"{name}.states.f32")
        noise = {}
        for k in (T // 2, T - 17):
            row = kl_stats(logits[name][:k], logits[f"{name}@{k}"])
            sa, sb = states[f"{name}@{k}"], states[name]
            row["layers_rel"] = [float((sa[j][:k] - sb[j][:k]).norm() / (sb[j][:k] - sb[j - 1][:k]).norm())
                                 for j in range(1, len(sb))]
            noise[f"prefix_{k}"] = row
        nll = true_nll(logits[name], ids[0])
        meta["texts"][name] = {"tokens": T, "nll": nll, "noise": noise, "greedy": [], "greedy_text": ""}
        print(f"[ref-stream] {name}: {T} tokens, nll {nll:.4f}, floor KL "
              f"{max(v['kl'] for v in noise.values()):.5f}", flush=True)
    (out / "names.txt").write_text("\n".join(texts.keys()) + "\n")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[ref-stream] wrote {out} in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
