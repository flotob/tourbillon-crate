import argparse
import json
import os
import sys
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


def add_json(args) -> int:
    """add, reporting to the app as one json object per line on stdout; everything else goes to stderr."""
    out, sys.stdout = sys.stdout, sys.stderr

    def emit(event, **fields):
        out.write(json.dumps({"event": event, **fields}) + "\n")
        out.flush()

    try:
        from crate.record import add_record

        folder, existing = add_record(
            args.file,
            args.crate,
            name=args.name,
            bpm=args.bpm,
            log=lambda message: emit("log", message=message),
            progress=lambda step, fraction: emit("progress", step=step, fraction=round(fraction, 3)),
            started=lambda name: emit("started", name=name),
        )
    except Exception as err:  # noqa: BLE001 - any failure goes to the app as an error event
        emit("error", message=f"{type(err).__name__}: {err}")
        return 1
    emit("done", name=folder.name, existing=existing)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="crate", description="Turns tracks into bar-sliced stems for Strudel.")
    parser.add_argument("--crate", type=Path, default=Path(os.environ.get("CRATE_DIR", DEFAULT_CRATE)))
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="separate a track and slice it into bars")
    add.add_argument("file", type=Path)
    add.add_argument("--name", help="sound name prefix in strudel (letters and digits)")
    add.add_argument("--bpm", type=float, help="force the tempo when the detected one is half or double")
    add.add_argument("--json", action="store_true", help="report progress as json lines, for the app")
    analyze = commands.add_parser("analyze", help="recompute what plays where, e.g. after an update")
    analyze.add_argument("names", nargs="*", help="records to analyze, default all")
    commands.add_parser("list", help="list the records in the crate")
    serve_cmd = commands.add_parser("serve", help="serve the crate to strudel")
    serve_cmd.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    if args.command == "add" and args.json:
        sys.exit(add_json(args))
    elif args.command == "add":
        from crate.record import add_record

        folder, _ = add_record(args.file, args.crate, name=args.name, bpm=args.bpm)
        print("\nin tourbillon (with `crate serve` running):\n")
        print(snippet(json.loads((folder / "record.json").read_text()), DEFAULT_PORT))
    elif args.command == "analyze":
        from crate.record import reanalyze, records

        for folder in records(args.crate):
            if not args.names or folder.name in args.names:
                reanalyze(folder)
                print(f"analyzed {folder.name}")
    elif args.command == "list":
        from crate.record import records

        for folder in records(args.crate):
            meta = json.loads((folder / "record.json").read_text())
            print(f"{meta['name']:<16} {meta['bpm']:7.2f} bpm  {meta['bars']:4d} bars  {meta['title']}")
    elif args.command == "serve":
        from crate.serve import serve

        serve(args.crate, args.port)
