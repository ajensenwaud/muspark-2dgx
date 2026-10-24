#!/usr/bin/env python3
"""
G3 split, step 1 on the second node: produce the converter's calibration state at the input of decoder layer L by
running the UNQUANTISED source layers start..L-1 over the converter's own calibration rows, and write it as a
resumable exllamav3 conversion job (work dir with args.json + ckpt/) whose next module is layer L. The second
node's converter then resumes that job with -r and quantises layers L..61, the final norm and the head.

Seed: the first node's checkpoint taken when its job stood at layer 0 (ckpt/job.json next_module_idx 2 = after
embed_tokens + embed_norm): ckpt/state.safetensors (250 x [1, 2048, 8192] fp32, the normed embeddings of the
calibration rows), ckpt/original_input_ids.safetensors, args.json. Each layer runs exactly the converter's
state-advance loop (convert_model.py main(), "Forward pass") with the unquantised module (LinearFP16, the same
modules and forward the converter's capture pass uses): params {attn_mode: flash_attn_nc, input_ids, quant_preserve}.

Layers whose per-layer file (tools/g3/fetch_layers.py) is not in --src are fetched from the range server into
--stream-dir just before use (the next one is prefetched while the current one runs) and that stream file is
removed after the layer is done.

Also, cheaply: per-layer and per-expert maxima of the fp16 hand-off points (latent input z, expert gate/up outputs,
the fp32 intermediate silu(g)*u before its fp16 cast, the down projection output, the routed sum), for the expert
precision scan (G3 task 2). Optionally carries the first --probe-rows rows on through layers L..probe_to-1
(unquantised as well) and saves them as real hidden states at the input of layer probe_to.

  state_gen.py --src /src --url http://<worker CX7 address>:8899 --index /meta/model.safetensors.index.json \
      --seed /seed --work /work --upto 33 --stream-dir /stream --stats /out/stats.json \
      [--probe-rows 4 --probe-to 61 --probe-out /out/probe-in-61.safetensors]
"""
import argparse, json, os, shutil, sys, threading, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_layers as fl

FP16_MIN_NORMAL = 6.103515625e-05
FP16_MAX = 65504.0


def load_list(path):
    from safetensors import safe_open
    out = []
    with safe_open(path, framework = "pt", device = "cpu") as f:
        keys = set(f.keys())
        if "tensor" in keys:
            return f.get_tensor("tensor")
        i = 0
        while f"tensor.{i}" in keys:
            out.append(f.get_tensor(f"tensor.{i}"))
            i += 1
    return out


def save_list(tensors, path):
    from exllamav3.loader.safetensors_alt import save_file    # the converter's own writer (save_tensor)
    save_file({f"tensor.{i}": t for i, t in enumerate(tensors)}, path)


class Stats:
    """running maxima per layer (and per expert for the expert hand-offs), kept on the GPU until the layer ends"""
    def __init__(self, num_experts, device):
        self.E = num_experts
        self.dev = device
        self.reset()

    def reset(self):
        z = lambda *s: torch.zeros(*s, dtype = torch.float, device = self.dev)
        self.g = z(self.E); self.u = z(self.E); self.a = z(self.E); self.d = z(self.E)
        self.a_small = z(self.E)          # count of nonzero |silu(g)*u| below fp16's smallest normal
        self.a_n = z(self.E)              # elements seen
        self.d_rms_min = torch.full((self.E,), float("inf"), device = self.dev)
        self.misc = {k: z(()) for k in ("mlp_in", "z", "acc", "moe_out", "attn_in", "block_out")}
        self.rows = 0

    def mx(self, name, t):
        self.misc[name] = torch.maximum(self.misc[name], t.detach().abs().amax().float())

    def report(self):
        r = {k: float(v) for k, v in self.misc.items()}
        def top(t, n = 3):
            v, i = t.topk(n)
            return [[int(a), float(b)] for a, b in zip(i.tolist(), v.tolist())]
        r.update({
            "g_max": float(self.g.max()), "u_max": float(self.u.max()), "a_max": float(self.a.max()),
            "d_max": float(self.d.max()),
            "a_subnormal_frac": float(self.a_small.sum() / self.a_n.sum().clamp(min = 1)),
            "d_rms_min": float(self.d_rms_min.min()),
            "top_experts_g": top(self.g), "top_experts_u": top(self.u), "top_experts_a": top(self.a),
            "rows": self.rows,
        })
        return r


def instrument(block, stats):
    """same arithmetic as MuseSparkMoE.expert_forward / forward (sens_w None), plus maxima"""
    mlp = block.mlp

    def expert_forward(idx, z, params, sens_w = None):
        assert sens_w is None
        g = mlp.gates[idx].forward(z, params)
        u = mlp.ups[idx].forward(z, params)
        a32 = F.silu(g.float()) * u.float()
        a = a32.half()
        d = mlp.downs[idx].forward(a, params)
        stats.g[idx] = torch.maximum(stats.g[idx], g.abs().amax().float())
        stats.u[idx] = torch.maximum(stats.u[idx], u.abs().amax().float())
        aa = a32.abs()
        stats.a[idx] = torch.maximum(stats.a[idx], aa.amax())
        stats.a_small[idx] += ((aa < FP16_MIN_NORMAL) & (aa > 0)).sum()
        stats.a_n[idx] += aa.numel()
        stats.d[idx] = torch.maximum(stats.d[idx], d.abs().amax().float())
        stats.d_rms_min[idx] = torch.minimum(stats.d_rms_min[idx], d.float().square().mean(-1).sqrt().amin())
        return mlp.post_expert_norm.forward(d, params)

    mlp.expert_forward = expert_forward
    lin_fwd, lout_fwd, mlp_fwd = mlp.latent_in.forward, mlp.latent_out.forward, mlp.forward

    def latent_in(x, params, out_dtype = None):
        y = lin_fwd(x, params, out_dtype)
        stats.mx("z", y)
        return y

    def latent_out(x, params, out_dtype = None):
        stats.mx("acc", x)
        y = lout_fwd(x, params, out_dtype)
        stats.mx("moe_out", y)
        return y

    def fwd(x, params, out_dtype = None):
        stats.mx("mlp_in", x)
        return mlp_fwd(x, params, out_dtype)

    mlp.latent_in.forward = latent_in
    mlp.latent_out.forward = latent_out
    mlp.forward = fwd


class LayerFiles:
    """local per-layer file, else fetched into stream_dir (one layer prefetched ahead)"""
    def __init__(self, src, stream_dir, url, index):
        self.src, self.stream_dir = src, stream_dir
        self.remote = fl.Remote(url, index) if url else None
        self.pending = {}
        if stream_dir:
            os.makedirs(stream_dir, exist_ok = True)

    def local(self, layer):
        p = fl.group_file(self.src, layer)
        return p if os.path.exists(p) else None

    def _fetch(self, layer):
        p = fl.group_file(self.stream_dir, layer)
        if not os.path.exists(p):
            t0 = time.time()
            n = fl.fetch_group(self.remote, layer, p)
            print(f"    fetched layer {layer}: {n / 1e9:.1f} GB in {time.time() - t0:.1f} s", flush = True)
        return p

    def prefetch(self, layer):
        if layer is None or self.local(layer) or layer in self.pending:
            return
        res = {}
        th = threading.Thread(target = lambda: res.update(p = self._fetch(layer)), daemon = True)
        th.start()
        self.pending[layer] = (th, res)

    def get(self, layer):
        p = self.local(layer)
        if p:
            return p, False
        self.prefetch(layer)
        th, res = self.pending.pop(layer)
        th.join()
        assert "p" in res, f"fetch of layer {layer} failed"
        return res["p"], True


def run_layer(model, config, layer, files, rows, state, ids, dev, stats, next_layer = None, batch = 1,
              verify_batch = False):
    from exllamav3.util.memory import free_mem
    path, streamed = files.get(layer)
    files.prefetch(next_layer)
    config.stc.add_tensor_files(path, warn_if_override = False)
    module = model.modules[model.first_block_idx + layer]
    assert module.layer_idx == layer
    t0 = time.time()
    defer = module.can_defer_load()
    if defer:
        config.stc.begin_deferred_load()
    try:
        module.load(dev)
    finally:
        if defer:
            config.stc.end_deferred_load()
    config.stc.close()
    t_load = time.time() - t0
    saved = (module.mlp.expert_forward, module.mlp.forward, module.mlp.latent_in.forward, module.mlp.latent_out.forward)
    instrument(module, stats)
    stats.reset()
    t0 = time.time()
    bad = []
    verify = None
    for c in range(0, len(rows), batch):
        chunk = rows[c : c + batch]
        if verify_batch and c == 0 and len(chunk) > 1:
            # the same rows one at a time, as the converter's loop runs them (batching only changes GEMM shapes)
            single = []
            for i in chunk:
                p1 = {"attn_mode": "flash_attn_nc", "input_ids": ids[i], "quant_preserve": {}}
                single.append(module.forward(module.prepare_for_device(state[i], p1), p1).float().cpu())
        params = {"attn_mode": "flash_attn_nc", "input_ids": torch.cat([ids[i] for i in chunk], dim = 0),
                  "quant_preserve": {}}
        x = module.prepare_for_device(torch.cat([state[i] for i in chunk], dim = 0), params)
        stats.mx("attn_in", x)
        rs = module.forward(x, params)
        stats.mx("block_out", rs)
        rs = rs.cpu()
        for j, i in enumerate(chunk):
            r = rs[j : j + 1].clone()
            if not torch.isfinite(r).all().item():
                bad.append(i)
            state[i] = r
            stats.rows += 1
        if verify_batch and c == 0 and len(chunk) > 1:
            verify = max(float((s - state[i].float()).norm() / s.norm()) for s, i in zip(single, chunk))
            print(f"    batch {len(chunk)} vs one row at a time: max rel l2 {verify:.3e}", flush = True)
    torch.cuda.synchronize()
    t_fwd = time.time() - t0
    rep = stats.report()
    rep.update({"seconds_load": round(t_load, 1), "seconds_forward": round(t_fwd, 1), "nonfinite_rows": bad,
                "batch": batch, "batch_vs_single_rel_l2": verify})
    (module.mlp.expert_forward, module.mlp.forward, module.mlp.latent_in.forward, module.mlp.latent_out.forward) = saved
    module.unload()
    free_mem()
    if streamed:
        os.remove(path)     # this tool's own stream copy of the layer (re-fetchable); never a source file
    return rep


def write_job(work, seed, next_idx, state, ids, bad_rows):
    ck_new = os.path.join(work, "ckpt_new")
    os.makedirs(ck_new, exist_ok = True)
    save_list(state, os.path.join(ck_new, "state.safetensors"))
    shutil.copyfile(os.path.join(seed, "original_input_ids.safetensors"), os.path.join(ck_new, "original_input_ids.safetensors"))
    job = {"next_module_idx": next_idx, "q_strategy": None, "bad_rows": sorted(bad_rows)}
    with open(os.path.join(ck_new, "job.json"), "w") as f:
        json.dump(job, f, indent = 4)
    ck = os.path.join(work, "ckpt")
    if os.path.exists(ck):
        old = os.path.join(work, "ckpt_old")
        assert not os.path.exists(old), f"{old} exists (left over); move it away first"
        os.rename(ck, old)
        os.rename(ck_new, ck)
        shutil.rmtree(old)    # this tool's own previous checkpoint, superseded
    else:
        os.rename(ck_new, ck)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required = True, help = "model dir: source config/tokenizer files (+ any per-layer files)")
    ap.add_argument("--url", default = None)
    ap.add_argument("--index", default = None)
    ap.add_argument("--seed", required = True)
    ap.add_argument("--work", required = True)
    ap.add_argument("--upto", type = int, required = True, help = "L: the state is produced at the input of layer L")
    ap.add_argument("--stream-dir", default = None)
    ap.add_argument("--stats", required = True)
    ap.add_argument("--ckpt-every", type = int, default = 8)
    ap.add_argument("--batch", type = int, default = 10, help = "calibration rows per forward (the converter runs 1)")
    ap.add_argument("--verify-batch", action = "store_true", help = "first layer: also run the first batch row by row")
    ap.add_argument("--probe-rows", type = int, default = 0)
    ap.add_argument("--probe-to", type = int, default = 61)
    ap.add_argument("--probe-out", default = None)
    args = ap.parse_args()

    from exllamav3 import Config, Model
    torch.set_grad_enabled(False)
    dev = torch.device("cuda:0")
    config = Config.from_directory(args.src)
    model = Model.from_config(config)
    first = model.first_block_idx
    files = LayerFiles(args.src, args.stream_dir, args.url, args.index)

    # Start: our own checkpoint if one exists (resume), else the seed
    stats_all = json.load(open(args.stats)) if os.path.exists(args.stats) else {"layers": {}, "probe_layers": {}}
    ck = os.path.join(args.work, "ckpt")
    src_ck = ck if os.path.exists(os.path.join(ck, "job.json")) else args.seed
    job = json.load(open(os.path.join(src_ck, "job.json")))
    start = job["next_module_idx"] - first
    assert 0 <= start <= args.upto, f"checkpoint at layer {start}, target {args.upto}"
    print(f" -- state at layer {start} from {src_ck}; target layer {args.upto}", flush = True)
    state = load_list(os.path.join(src_ck, "state.safetensors"))
    ids = load_list(os.path.join(args.seed, "original_input_ids.safetensors"))
    bad_rows = set(job.get("bad_rows") or [])
    if not os.path.exists(os.path.join(args.work, "args.json")):
        os.makedirs(args.work, exist_ok = True)
        shutil.copyfile(os.path.join(args.seed, "args.json"), os.path.join(args.work, "args.json"))
    stats = Stats(config.num_experts, dev)

    t_all = time.time()
    for layer in range(start, args.upto):
        rows = [i for i in range(len(state)) if i not in bad_rows]
        nxt = layer + 1 if layer + 1 < args.upto else (args.upto if args.probe_rows else None)
        t0 = time.time()
        rep = run_layer(model, config, layer, files, rows, state, ids, dev, stats, nxt, args.batch,
                        args.verify_batch and layer == start)
        bad_rows |= set(rep["nonfinite_rows"])
        rep["seconds_total"] = round(time.time() - t0, 1)
        stats_all["layers"][str(layer)] = rep
        print(f" -- layer {layer}: {rep['seconds_total']} s (load {rep['seconds_load']}, fwd {rep['seconds_forward']}); "
              f"g {rep['g_max']:.1f} u {rep['u_max']:.1f} a {rep['a_max']:.1f} d {rep['d_max']:.2f} "
              f"a<fp16min {rep['a_subnormal_frac']:.2e} d_rms_min {rep['d_rms_min']:.2e} acc {rep['acc']:.1f}"
              + (f" NONFINITE {rep['nonfinite_rows']}" if rep["nonfinite_rows"] else ""), flush = True)
        json.dump(stats_all, open(args.stats, "w"), indent = 1)
        done = layer + 1
        if done == args.upto or (done - start) % args.ckpt_every == 0:
            t0 = time.time()
            write_job(args.work, args.seed, first + done, state, ids, bad_rows)
            print(f" -- checkpoint: next module {first + done} (layer {done}) in {time.time() - t0:.0f} s", flush = True)
    print(f" -- state generation: {args.upto - start} layers in {time.time() - t_all:.0f} s", flush = True)
    stats_all["state_gen_seconds"] = stats_all.get("state_gen_seconds", 0) + round(time.time() - t_all)
    json.dump(stats_all, open(args.stats, "w"), indent = 1)

    if args.probe_rows:
        rows = [i for i in range(len(state)) if i not in bad_rows][: args.probe_rows]
        pstate = {i: state[i] for i in rows}
        del state
        t_all = time.time()
        for layer in range(args.upto, args.probe_to):
            nxt = layer + 1 if layer + 1 < args.probe_to else None
            rep = run_layer(model, config, layer, files, rows, pstate, ids, dev, stats, nxt, args.batch)
            stats_all["probe_layers"][str(layer)] = rep
            print(f" -- probe layer {layer}: g {rep['g_max']:.1f} u {rep['u_max']:.1f} a {rep['a_max']:.1f} "
                  f"d {rep['d_max']:.2f} a<fp16min {rep['a_subnormal_frac']:.2e} d_rms_min {rep['d_rms_min']:.2e} "
                  f"acc {rep['acc']:.1f}", flush = True)
            json.dump(stats_all, open(args.stats, "w"), indent = 1)
        from safetensors.torch import save_file
        save_file({f"row.{i}": pstate[i].contiguous() for i in rows} |
                  {f"ids.{i}": ids[i].contiguous() for i in rows}, args.probe_out)
        print(f" -- probe rows {rows} at the input of layer {args.probe_to}: {args.probe_out} "
              f"({time.time() - t_all:.0f} s)", flush = True)


if __name__ == "__main__":
    main()
