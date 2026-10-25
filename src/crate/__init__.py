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


def run_json(task) -> int:
    """Runs task(log, progress, emit) for the app, which reads one json object per line from stdout.

    Everything else goes to stderr, including what native libraries print.
    """
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def emit(event, **fields):
        out.write(json.dumps({"event": event, **fields}) + "\n")
        out.flush()

    try:
        task(
            lambda message: emit("log", message=message),
            lambda step, fraction: emit("progress", step=step, fraction=round(fraction, 3)),
            emit,
        )
    except Exception as err:  # noqa: BLE001 - any failure goes to the app as an error event
        emit("error", message=f"{type(err).__name__}: {err}")
        return 1
    return 0


def find_record(crate: Path, name: str) -> Path:
    folder = crate / name
    if not (folder / "record.json").exists():
        raise SystemExit(f"no record called {name} in {crate}")
    return folder


def add(args, log=print, progress=lambda step, fraction: None, emit=None):
    from crate.record import add_record

    folder, existing = add_record(
        args.file,
        args.crate,
        name=args.name,
        bpm=args.bpm,
        log=log,
        progress=progress,
        started=lambda name: emit and emit("started", name=name),
    )
    if emit:
        emit("done", name=folder.name, existing=existing)
    else:
        print("\nin tourbillon (with `crate serve` running):\n")
        print(snippet(json.loads((folder / "record.json").read_text()), DEFAULT_PORT))


def tempo(args, log=print, progress=lambda step, fraction: None, emit=None):
    from crate.tempo import make_version

    folder, existing = make_version(find_record(args.crate, args.name), args.bpm, log=log, progress=progress)
    if emit:
        emit("done", name=args.name, bpm=float(folder.name), existing=existing)


def analyze(args, log=print, progress=lambda step, fraction: None, emit=None):
    from crate.record import reanalyze, records

    folders = [folder for folder in records(args.crate) if not args.names or folder.name in args.names]
    for i, folder in enumerate(folders):
        progress("analyzing", i / len(folders))
        reanalyze(folder)
        log(f"analyzed {folder.name}")
    if emit:
        emit("done", names=[folder.name for folder in folders])


def main() -> None:
    parser = argparse.ArgumentParser(prog="crate", description="Turns tracks into bar-sliced stems for Strudel.")
    parser.add_argument("--crate", type=Path, default=Path(os.environ.get("CRATE_DIR", DEFAULT_CRATE)))
    commands = parser.add_subparsers(dest="command", required=True)
    add_cmd = commands.add_parser("add", help="separate a track and slice it into bars")
    add_cmd.add_argument("file", type=Path)
    add_cmd.add_argument("--name", help="sound name prefix in strudel (letters and digits)")
    add_cmd.add_argument("--bpm", type=float, help="force the tempo when the detected one is half or double")
    tempo_cmd = commands.add_parser("tempo", help="make a version of a record at another tempo, keeping its pitch")
    tempo_cmd.add_argument("name")
    tempo_cmd.add_argument("bpm", type=float)
    analyze_cmd = commands.add_parser("analyze", help="recompute key and what plays where, e.g. after an update")
    analyze_cmd.add_argument("names", nargs="*", help="records to analyze, default all")
    for command in (add_cmd, tempo_cmd, analyze_cmd):
        command.add_argument("--json", action="store_true", help="report progress as json lines, for the app")
    commands.add_parser("list", help="list the records in the crate")
    serve_cmd = commands.add_parser("serve", help="serve the crate to strudel")
    serve_cmd.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    tasks = {"add": add, "tempo": tempo, "analyze": analyze}
    if args.command in tasks and args.json:
        sys.exit(run_json(lambda log, progress, emit: tasks[args.command](args, log, progress, emit)))
    elif args.command in tasks:
        tasks[args.command](args)
    elif args.command == "list":
        from crate.record import records
        from crate.tempo import tempo_label, versions

        for folder in records(args.crate):
            meta = json.loads((folder / "record.json").read_text())
            key = meta.get("key", "")
            also = ", ".join(tempo_label(bpm) for bpm in versions(folder))
            print(
                f"{meta['name']:<16} {meta['bpm']:7.2f} bpm  {key:<9} {meta['bars']:4d} bars  {meta['title']}"
                + (f"  (also at {also} bpm)" if also else "")
            )
    elif args.command == "serve":
        from crate.serve import serve

        serve(args.crate, args.port)
