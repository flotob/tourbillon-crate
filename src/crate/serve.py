"""Serves the crate to Strudel: /strudel.json maps <record>_<stem> to its bar files.

In Strudel: samples('http://localhost:5455/strudel.json'), then s("<record>_bass").n(<bar>).
"""

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def manifest(crate: Path, base: str) -> dict:
    sounds = {"_base": base}
    for meta_file in sorted(crate.glob("*/record.json")):
        meta = json.loads(meta_file.read_text())
        name = meta["name"]
        for stem in meta["stems"]:
            sounds[f"{name}_{stem}"] = [f"{name}/bars/{stem}/{bar:03d}.flac" for bar in range(meta["bars"])]
    return sounds


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, crate: Path, **kwargs):
        self.crate = crate
        super().__init__(*args, directory=str(crate), **kwargs)

    def end_headers(self):
        # the app loads samples from another origin
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_GET(self):
        if self.path.split("?")[0] != "/strudel.json":
            return super().do_GET()
        host = self.headers.get("Host", f"localhost:{self.server.server_port}")
        body = json.dumps(manifest(self.crate, f"http://{host}/")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def serve(crate: Path, port: int):
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, crate=crate))
    print(f"serving {crate} at http://localhost:{port}/strudel.json")
    server.serve_forever()
