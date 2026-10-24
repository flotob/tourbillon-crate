"""Turns an audio file into a record: stems, a bar grid, and one audio file per bar and stem.

Layout of a record in the crate:
    <name>/record.json        metadata: tempo, grid, stems, bars
    <name>/stems/<stem>.flac  whole stems, for later processing
    <name>/bars/<stem>/NNN.flac  bar NNN of a stem, starting at the first downbeat
"""

import hashlib
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from crate.grid import align_to_transients, fit_grid

MODEL = "htdemucs"
# the original audio, sliced like the stems
MIX = "mix"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_name(text: str) -> str:
    # strudel sound names: letters and digits only, since _ separates bank and sound
    name = re.sub(r"[^a-z0-9]", "", text.lower())[:24]
    return name or "record"


def find_by_hash(crate: Path, digest: str):
    for meta in crate.glob("*/record.json"):
        if json.loads(meta.read_text()).get("hash") == digest:
            return meta.parent.name


def unique_name(crate: Path, name: str) -> str:
    candidate, n = name, 2
    while (crate / candidate).exists():
        candidate, n = f"{name}{n}", n + 1
    return candidate


def best_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def separate(path: Path, device: str, log):
    from demucs.api import Separator

    try:
        separator = Separator(model=MODEL, device=device)
        mix, stems = separator.separate_audio_file(path)
    except Exception as err:
        if device == "cpu":
            raise
        log(f"separation on {device} failed ({err}), using the cpu")
        separator = Separator(model=MODEL, device="cpu")
        mix, stems = separator.separate_audio_file(path)
    as_array = lambda tensor: tensor.cpu().numpy().T  # (frames, channels)
    return separator.samplerate, as_array(mix), {name: as_array(audio) for name, audio in stems.items()}


def detect_beats(mix: np.ndarray, samplerate: int, device: str):
    from beat_this.inference import Audio2Beats

    tracker = Audio2Beats(checkpoint_path="final0", device=device if device != "mps" else "cpu")
    beats, downbeats = tracker(mix.mean(axis=1), samplerate)
    return beats, downbeats


def write(path: Path, audio: np.ndarray, samplerate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1, 1), samplerate, subtype="PCM_16")


def add_record(source: Path, crate: Path, name: str | None = None, bpm: float | None = None, log=print) -> Path:
    crate.mkdir(parents=True, exist_ok=True)
    digest = file_hash(source)
    existing = find_by_hash(crate, digest)
    if existing:
        log(f"already in the crate as {existing}")
        return crate / existing
    name = unique_name(crate, record_name(name or source.stem))
    device = best_device()

    started = time.monotonic()
    log(f"separating stems with {MODEL} on {device}…")
    samplerate, mix, stems = separate(source, device, log)
    separated = time.monotonic()
    log(f"  {separated - started:.1f}s")

    log("detecting beats…")
    beats, downbeats = detect_beats(mix, samplerate, device)
    grid = align_to_transients(fit_grid(beats, downbeats, bpm=bpm), stems["drums"], samplerate)
    duration = len(mix) / samplerate
    bars = grid.bar_count(duration)
    log(
        f"  {grid.bpm:.2f} bpm, first downbeat at {grid.first_downbeat:.4f}s (moved {grid.shift_ms:+.1f}ms onto the hits), {bars} bars, "
        f"beats off the grid by {grid.median_error_ms:.1f}ms (median) / {grid.max_error_ms:.1f}ms (max)"
    )

    folder = crate / name
    tracks = {MIX: mix, **stems}
    for stem, audio in tracks.items():
        write(folder / "stems" / f"{stem}.flac", audio, samplerate)
        # bar boundaries rounded to samples; consecutive bars join without a gap
        starts = [round(grid.bar_start(bar) * samplerate) for bar in range(bars + 1)]
        for bar in range(bars):
            write(folder / "bars" / stem / f"{bar:03d}.flac", audio[starts[bar] : starts[bar + 1]], samplerate)

    meta = {
        "name": name,
        "title": source.stem,
        "source": str(source.resolve()),
        "hash": digest,
        "duration": duration,
        "samplerate": samplerate,
        "bpm": grid.bpm,
        "bars": bars,
        "grid": asdict(grid),
        "stems": list(tracks),
        "model": MODEL,
        "created_at": time.time(),
        "seconds": {"separation": separated - started, "total": time.monotonic() - started},
    }
    (folder / "record.json").write_text(json.dumps(meta, indent=2))
    log(f"done in {time.monotonic() - started:.1f}s: {folder}")
    return folder
