#!/usr/bin/env python3
"""
Split conversion: read-only HTTP server with byte ranges for the source checkpoint on the first node, so the second node can fetch single tensors
(one layer at a time) over the CX7 link without a full copy of the 509 GB source. Standard library only.

  GET /<file>            whole file
  GET /<file> + Range: bytes=a-b   206 with that slice (one range per request)
  HEAD /<file>           size only

Serves regular files directly inside --root (no subdirectories, no listing). Bind it to the CX7 address only.
Run in the recipe image on the first node, CPU only, source mounted read-only:
  docker run -d --name muse-range --network host --user $(id -u):$(id -g) -v $SRC_DIR:/src:ro \
    -v $PWD/tools:/rtools:ro --entrypoint python3 muse-tf:latest /rtools/g3/range_server.py --root /src \
    --bind <its CX7 address> --port 8899
"""
import argparse, os, re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RANGE = re.compile(r"bytes=(\d+)-(\d*)$")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"          # keep-alive: one connection per fetch thread
    root = "/src"

    def log_message(self, fmt, *args):
        pass

    def _path(self):
        name = self.path.lstrip("/")
        if not name or "/" in name or name.startswith("."):
            return None
        p = os.path.join(self.root, name)
        return p if os.path.isfile(p) else None

    def _head(self, send_body):
        p = self._path()
        if p is None:
            self.send_error(404)
            return
        size = os.path.getsize(p)
        a, b, code = 0, size, 200
        r = self.headers.get("Range")
        if r:
            m = RANGE.match(r.strip())
            if not m:
                self.send_error(416)
                return
            a = int(m.group(1))
            b = int(m.group(2)) + 1 if m.group(2) else size
            if a >= size or b > size or a >= b:
                self.send_error(416)
                return
            code = 206
        self.send_response(code)
        self.send_header("Content-Length", str(b - a))
        self.send_header("Accept-Ranges", "bytes")
        if code == 206:
            self.send_header("Content-Range", f"bytes {a}-{b - 1}/{size}")
        self.end_headers()
        if not send_body:
            return
        with open(p, "rb") as f:
            off, left = a, b - a
            while left > 0:
                n = os.sendfile(self.connection.fileno(), f.fileno(), off, min(left, 1 << 26))
                if n <= 0:
                    break
                off += n
                left -= n

    def do_HEAD(self):
        self._head(False)

    def do_GET(self):
        self._head(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default = "/src")
    ap.add_argument("--bind", required = True)
    ap.add_argument("--port", type = int, default = 8899)
    args = ap.parse_args()
    Handler.root = args.root
    srv = ThreadingHTTPServer((args.bind, args.port), Handler)
    srv.daemon_threads = True
    print(f"serving {args.root} on {args.bind}:{args.port}", flush = True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
