#!/usr/bin/env python3
"""
G3 split: fetch whole decoder layers (or the head) of the source checkpoint from tools/g3/range_server.py into
local safetensors files, one file per layer, with the source's exact bytes, dtypes, shapes and tensor names (raw
copy, nothing decoded). exllamav3 globs every *.safetensors in a model directory, so a directory holding the source's
config/tokenizer files plus some of these per-layer files is a valid source for the layers it holds.
Standard library only (runs on the host or in any image).

  fetch_layers.py --url http://<worker CX7 address>:8899 --index <local copy of model.safetensors.index.json> \
      --out <dir> --layers 33-61 --head

Groups: layer N = every tensor named model.language_model.layers.N.*; head = lm_head.weight +
model.language_model.norm.weight; embed = model.language_model.embed_tokens.weight (layer checks only). Output files: layer-NN.safetensors, head.safetensors (written as .part, renamed
when complete; an existing complete file is skipped, so reruns resume).
"""
import argparse, http.client, json, os, re, struct, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

P = "model.language_model"
PIECE = 64 << 20


class Remote:
    def __init__(self, url, index_path):
        u = urlparse(url)
        self.host, self.port = u.hostname, u.port or 80
        self.weight_map = json.load(open(index_path))["weight_map"]
        self.headers = {}
        self.local = threading.local()
        self.lock = threading.Lock()

    def conn(self):
        c = getattr(self.local, "c", None)
        if c is None:
            c = self.local.c = http.client.HTTPConnection(self.host, self.port, timeout = 600)
        return c

    def get_range(self, f, a, b, sink = None):
        """bytes [a, b) of file f; with sink(offset, chunk) the data is streamed instead of returned"""
        for attempt in range(6):
            try:
                c = self.conn()
                c.request("GET", "/" + f, headers = {"Range": f"bytes={a}-{b - 1}"})
                r = c.getresponse()
                if r.status != 206:
                    raise IOError(f"{f} [{a},{b}): HTTP {r.status}")
                if sink is None:
                    data = r.read()
                    if len(data) != b - a:
                        raise IOError(f"{f} [{a},{b}): short read {len(data)}")
                    return data
                got = 0
                while got < b - a:
                    chunk = r.read(min(8 << 20, b - a - got))
                    if not chunk:
                        raise IOError(f"{f} [{a},{b}): connection closed at {got}")
                    sink(got, chunk)
                    got += len(chunk)
                return None
            except Exception as e:
                self.local.c = None
                if attempt == 5:
                    raise
                print(f"   retry {attempt + 1} {f} [{a},{b}): {e}", flush = True)
                time.sleep(3)

    def header(self, f):
        with self.lock:
            if f in self.headers:
                return self.headers[f]
        n = struct.unpack("<Q", self.get_range(f, 0, 8))[0]
        h = json.loads(self.get_range(f, 8, 8 + n))
        with self.lock:
            self.headers[f] = (h, 8 + n)
        return self.headers[f]


def group_keys(weight_map, group):
    if group == "head":
        keys = ["lm_head.weight", f"{P}.norm.weight"]
    elif group == "embed":
        keys = [f"{P}.embed_tokens.weight"]
    else:
        pre = f"{P}.layers.{int(group)}."
        keys = [k for k in weight_map if k.startswith(pre)]
    assert keys and all(k in weight_map for k in keys), f"group {group}: tensors missing from the index"
    return sorted(keys)


def fetch_group(remote, group, out_path, threads = 8):
    """write every tensor of group into out_path (atomic rename); returns bytes written"""
    keys = group_keys(remote.weight_map, group)
    shards = sorted(set(remote.weight_map[k] for k in keys))
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(remote.header, shards))
    entries, src, off = {}, {}, 0
    for k in keys:
        f = remote.weight_map[k]
        h, base = remote.headers[f]
        e = h[k]
        a, b = e["data_offsets"]
        entries[k] = {"dtype": e["dtype"], "shape": e["shape"], "data_offsets": [off, off + b - a]}
        src[k] = (f, base + a, base + b)
        off += b - a
    entries["__metadata__"] = {"format": "pt"}
    hdr = json.dumps(entries, separators = (",", ":")).encode()
    hdr += b" " * ((8 - len(hdr) % 8) % 8)
    data0 = 8 + len(hdr)
    total = data0 + off

    tmp = out_path + ".part"
    fd = os.open(tmp, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, total)
        os.pwrite(fd, struct.pack("<Q", len(hdr)) + hdr, 0)
        tasks = []
        for k in keys:
            f, a, b = src[k]
            d = data0 + entries[k]["data_offsets"][0]
            for p in range(a, b, PIECE):
                q = min(b, p + PIECE)
                tasks.append((f, p, q, d + (p - a)))

        def run(t):
            f, a, b, d = t
            remote.get_range(f, a, b, sink = lambda o, chunk: os.pwrite(fd, chunk, d + o))
            return b - a

        with ThreadPoolExecutor(threads) as ex:
            got = sum(ex.map(run, tasks))
        assert got == off, f"group {group}: fetched {got} of {off} bytes"
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, out_path)
    return total


def group_file(out_dir, group):
    if group in ("head", "embed"):
        return os.path.join(out_dir, f"{group}.safetensors")
    return os.path.join(out_dir, f"layer-{int(group):02}.safetensors")


def parse_layers(s):
    out = []
    for part in s.split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required = True)
    ap.add_argument("--index", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--layers", default = "")
    ap.add_argument("--head", action = "store_true")
    ap.add_argument("--embed", action = "store_true", help = "also the token embedding (only the layer checks need it)")
    ap.add_argument("--threads", type = int, default = 8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok = True)
    remote = Remote(args.url, args.index)
    groups = parse_layers(args.layers) + (["head"] if args.head else []) + (["embed"] if args.embed else [])
    t_all, n_all = time.time(), 0
    for g in groups:
        path = group_file(args.out, g)
        if os.path.exists(path):
            print(f" -- {os.path.basename(path)} present, skipped", flush = True)
            continue
        t0 = time.time()
        n = fetch_group(remote, g, path, args.threads)
        n_all += n
        dt = time.time() - t0
        print(f" -- {os.path.basename(path)}: {n / 1e9:.2f} GB in {dt:.1f} s ({n / 1e9 / dt:.2f} GB/s)", flush = True)
    dt = time.time() - t_all
    print(f" -- done: {n_all / 1e9:.1f} GB in {dt:.0f} s", flush = True)


if __name__ == "__main__":
    main()
