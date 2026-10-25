"""Versions of a record at other tempos, stretched with Rubber Band so the pitch stays the same.

Layout, next to the record's own bars:
    <name>/tempo/<bpm>/<stem>/NNN.flac  bar NNN of a stem at <bpm>, the same music as bars/<stem>/NNN.flac
    <name>/tempo/<bpm>/version.json     how it was made

Rubber Band's faster R2 engine keeps kicks sharp, where its finer R3 engine softens them and lets
them start early. A time map pins every beat of the grid to its new time, so bars stay on the beat
over a whole record; what is left is a constant delay of a few milliseconds, which is measured on
the drums and cut away like on the original.
"""

import json
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from crate.grid import Grid, align_to_transients
from crate.record import write

TEMPO = "tempo"
# stretching further than this sounds bad, and nobody mixes records that far apart
MAX_RATIO = 1.5


def tempo_label(bpm: float) -> str:
    """Folder name of a version: 124.0 -> "124", 123.456 -> "123.46"."""
    return f"{round(bpm, 2):g}"


def versions(folder: Path) -> list[float]:
    """Tempos of a record's complete versions."""
    base = folder / TEMPO
    if not base.exists():
        return []
    return sorted(float(d.name) for d in base.iterdir() if d.is_dir() and not d.name.startswith("."))


def time_map(grid: Grid, duration: float, ratio: float, samplerate: int) -> str:
    """Source and target frame of every beat, in Rubber Band's --timemap format."""
    first = grid.first_downbeat % grid.beat_period
    beats = np.arange(first, duration, grid.beat_period)
    return "".join(f"{round(t * samplerate)} {round(t * ratio * samplerate)}\n" for t in beats)


def stretch(source: Path, target: Path, ratio: float, timemap: Path, progress=lambda fraction: None):
    """Stretches one file to ratio times its length, reporting Rubber Band's two passes as 0..1."""
    try:
        process = subprocess.Popen(
            ["rubberband", "--fast", "--timemap", str(timemap), "--time", f"{ratio:.12f}", str(source), str(target)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        raise RuntimeError("rubberband not found, install it with `brew install rubberband`") from None
    # progress comes as "\r 42%", once per pass ("Pass 1: Studying...", "Pass 2: Processing...")
    second_pass, text, lines = False, b"", []
    while chunk := process.stderr.read1(4096):
        text += chunk
        *done, text = re.split(rb"[\r\n]", text)
        for line in (part.decode(errors="replace").strip() for part in done):
            if percent := re.fullmatch(r"(\d+)%", line):
                progress((int(percent[1]) / 100 + second_pass) / 2)
            elif line.startswith("Pass 2"):
                second_pass = True
            elif line:
                lines.append(line)
    if process.wait() != 0:
        raise RuntimeError(f"rubberband failed on {source.name}: {' / '.join(lines[-3:])}")


def cut(audio: np.ndarray, start: int, end: int) -> np.ndarray:
    """audio[start:end], padded with silence where it reaches past either end."""
    before = max(0, -start)
    part = audio[max(0, start) : max(0, end)]
    after = max(0, end - start - before - len(part))
    return np.pad(part, ((before, after), (0, 0)))


def make_version(folder: Path, bpm: float, log=print, progress=lambda step, fraction: None) -> tuple[Path, bool]:
    """Makes the record in folder playable at bpm. Returns the version's folder and whether it was there already."""
    meta = json.loads((folder / "record.json").read_text())
    label = tempo_label(bpm)
    bpm = float(label)
    if abs(bpm - meta["bpm"]) < 0.005:
        raise ValueError(f"{meta['name']} is {label} bpm already")
    ratio = meta["bpm"] / bpm
    if not 1 / MAX_RATIO <= ratio <= MAX_RATIO:
        raise ValueError(f"{label} bpm is too far from {meta['name']}'s {meta['bpm']:g} bpm")
    target = folder / TEMPO / label
    if target.exists():
        log(f"{meta['name']} at {label} bpm is there already")
        return target, True

    partial = folder / TEMPO / f".{label}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    begin = time.monotonic()
    samplerate, grid, stems = meta["samplerate"], Grid(**meta["grid"]), meta["stems"]
    timemap = partial / "timemap.txt"
    timemap.write_text(time_map(grid, meta["duration"], ratio, samplerate))

    log(f"stretching {meta['name']} from {meta['bpm']:g} to {label} bpm…")
    progress("stretching", 0.0)
    done, lock, reported = dict.fromkeys(stems, 0.0), threading.Lock(), [0]

    def report(stem, fraction):
        with lock:
            done[stem] = fraction
            percent = int(100 * sum(done.values()) / len(done))
            if percent > reported[0]:
                reported[0] = percent
                progress("stretching", percent / 100)

    # one rubberband per stem; each uses a single core
    with ThreadPoolExecutor(max_workers=len(stems)) as pool:
        jobs = [
            pool.submit(
                stretch,
                folder / "stems" / f"{stem}.flac",
                partial / f"{stem}.wav",
                ratio,
                timemap,
                lambda fraction, stem=stem: report(stem, fraction),
            )
            for stem in stems
        ]
        for job in jobs:
            job.result()
    stretched = time.monotonic()

    progress("writing", 0.0)
    audio = {stem: sf.read(partial / f"{stem}.wav", always_2d=True)[0] for stem in stems}
    # the same bars as the original on the scaled grid, moved onto the hits; bar n stays bar n even if that moves
    # bar 0 before the start
    first = grid.first_downbeat * ratio
    shift = 0.0
    if "drums" in audio:
        scaled = replace(grid, bpm=bpm, first_downbeat=first)
        shift = align_to_transients(scaled, audio["drums"], samplerate, search_ms=15).shift_ms / 1000
    bar_length = 240 / bpm
    starts = [round((first + shift + bar * bar_length) * samplerate) for bar in range(meta["bars"] + 1)]
    for i, stem in enumerate(stems):
        progress("writing", i / len(stems))
        for bar in range(meta["bars"]):
            write(partial / stem / f"{bar:03d}.flac", cut(audio[stem], starts[bar], starts[bar + 1]), samplerate)
        (partial / f"{stem}.wav").unlink()
    timemap.unlink()
    version = {
        "bpm": bpm,
        "from_bpm": meta["bpm"],
        "engine": "rubberband R2 (--fast) with a time map on every beat",
        "shift_ms": round(shift * 1000, 2),
        "created_at": time.time(),
        "seconds": {"stretching": stretched - begin, "total": time.monotonic() - begin},
    }
    (partial / "version.json").write_text(json.dumps(version, indent=1))
    partial.rename(target)
    log(f"done in {time.monotonic() - begin:.1f}s (onsets moved {shift * 1000:+.1f}ms): {target}")
    return target, False
