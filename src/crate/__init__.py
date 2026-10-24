import argparse
import json
import os
from pathlib import Path

# next to tourbillon's own data on macOS
DEFAULT_CRATE = Path.home() / "Library/Application Support/Tourbillon/crate"
DEFAULT_PORT = 5455


def snippet(meta: dict, port: int) -> str:
    name, bpm = meta["name"], meta["bpm"]
    start = min(32, max(0, meta["bars"] - 4))
    bars = " ".join(str(bar) for bar in range(start, start + 4))
    return f"""samples('http://localhost:{port}/strudel.json')
setcpm({bpm:g}/4) // one cycle is one bar of {name}
$: s("{name}_drums").n("<{bars}>")
$: s("{name}_bass").n("<{bars}>")"""


def main() -> None:
    parser = argparse.ArgumentParser(prog="crate", description="Turns tracks into bar-sliced stems for Strudel.")
    parser.add_argument("--crate", type=Path, default=Path(os.environ.get("CRATE_DIR", DEFAULT_CRATE)))
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="separate a track and slice it into bars")
    add.add_argument("file", type=Path)
    add.add_argument("--name", help="sound name prefix in strudel (letters and digits)")
    add.add_argument("--bpm", type=float, help="force the tempo when the detected one is half or double")
    commands.add_parser("list", help="list the records in the crate")
    serve_cmd = commands.add_parser("serve", help="serve the crate to strudel")
    serve_cmd.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    if args.command == "add":
        from crate.record import add_record

        folder = add_record(args.file, args.crate, name=args.name, bpm=args.bpm)
        print("\nin tourbillon (with `crate serve` running):\n")
        print(snippet(json.loads((folder / "record.json").read_text()), DEFAULT_PORT))
    elif args.command == "list":
        for meta_file in sorted(args.crate.glob("*/record.json")):
            meta = json.loads(meta_file.read_text())
            print(f"{meta['name']:<24} {meta['bpm']:7.2f} bpm  {meta['bars']:4d} bars  {meta['title']}")
    elif args.command == "serve":
        from crate.serve import serve

        serve(args.crate, args.port)
