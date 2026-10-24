#!/usr/bin/env python3
"""
Dense requantization, split runs: fetch only the DENSE linears' unquantized weights of some decoder layers (attention
q/k/v/o/gate_proj and the latent mlp.pre_expert_proj / post_expert_proj `.weight`, bf16 in the source) from
tools/g3/range_server.py into one local file per layer, exact bytes (raw copy). ~0.57 GB a Muse Spark layer instead of
the ~12 GB a whole layer holds; the second node of a split dense requantization needs nothing else from the source
(everything else comes from the base EXL3 pack). Standard library only.

  fetch_dense.py --url http://<worker CX7 address>:8899 --index <copy of the source's model.safetensors.index.json> \
      --out <dir> --layers 32-61

Output: <dir>/dense-NN.safetensors (written as .part, renamed when complete; complete files are skipped on rerun).
"""
import argparse, os, re, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "g3"))
import fetch_layers as fl          # noqa: E402  (Remote, fetch_group: the G3 range fetcher)

DENSE = re.compile(r"\.layers\.\d+\.(self_attn\.(q|k|v|o|gate)_proj|mlp\.(pre|post)_expert_proj)\.weight$")
_group_keys = fl.group_keys


def dense_keys(weight_map, group):
    keys = [k for k in _group_keys(weight_map, group) if DENSE.search(k)]
    assert len(keys) == 7, f"layer {group}: {len(keys)} dense tensors in the index, expected 7"
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required = True)
    ap.add_argument("--index", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--layers", required = True)
    ap.add_argument("--threads", type = int, default = 8)
    a = ap.parse_args()
    fl.group_keys = dense_keys
    os.makedirs(a.out, exist_ok = True)
    remote = fl.Remote(a.url, a.index)
    t_all, n_all = time.time(), 0
    for layer in fl.parse_layers(a.layers):
        path = os.path.join(a.out, f"dense-{layer:02}.safetensors")
        if os.path.exists(path):
            print(f" -- {os.path.basename(path)} present, skipped", flush = True)
            continue
        t0 = time.time()
        n = fl.fetch_group(remote, layer, path, a.threads)
        n_all += n
        print(f" -- {os.path.basename(path)}: {n / 1e9:.2f} GB in {time.time() - t0:.1f} s", flush = True)
    print(f" -- done: {n_all / 1e9:.1f} GB in {time.time() - t_all:.0f} s", flush = True)


if __name__ == "__main__":
    main()
