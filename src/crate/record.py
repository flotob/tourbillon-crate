"""Turns an audio file into a record: stems, a bar grid, and one audio file per bar and stem.

Layout of a record in the crate:
    <name>/record.json           metadata: tempo, key, grid, stems, bars, what plays where
    <name>/stems/<stem>.flac     whole stems, for later processing
    <name>/bars/<stem>/NNN.flac  bar NNN of a stem, starting at the first downbeat
    <name>/tempo/<bpm>/...       the bars at other tempos (see tempo.py)

A record is written to .<name>.partial first and renamed when complete.
"""

import hashlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from crate.analysis import bar_levels, playing_ranges
from crate.grid import Grid, align_to_transients, fit_grid
from crate.key import detect_key

MODEL = "htdemucs"
# the original audio, sliced like the stems
MIX = "mix"
# version of the record.json layout; 2 added the key
FORMAT = 2


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tags(path: Path) -> dict:
    """Title and artist from the file's tags, if ffprobe finds any."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags=title,artist", "-of", "json", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        tags = json.loads(result.stdout).get("format", {}).get("tags", {})
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}
    tags = {key.lower(): value.strip() for key, value in tags.items() if isinstance(value, str)}
    return {key: tags[key] for key in ("title", "artist") if tags.get(key)}


def describe(path: Path) -> tuple[str, str | None]:
    """Title and artist of a track, from its tags or else its file name ("01 Artist - Title.mp3")."""
    tags = read_tags(path)
    if "title" in tags:
        return tags["title"], tags.get("artist")
    stem = re.sub(r"^\d{1,3}[\s._-]+", "", path.stem).strip() or path.stem
    artist, _, title = stem.rpartition(" - ")
    return (title or stem), (artist or None)


def record_name(title: str) -> str:
    """A short strudel sound prefix: "Nata Alma (Club Edit)" -> "nataalma".

    Letters and digits only, since _ separates the record from the stem, starting with a letter.
    """
    words = re.sub(r"[(\[].*?[)\]]", "", title)
    name = re.sub(r"[^a-z0-9]", "", words.lower())[:16] or re.sub(r"[^a-z0-9]", "", title.lower())[:16]
    if not name:
        return "record"
    return name if name[0].isalpha() else f"r{name}"[:16]


def records(crate: Path):
    """Folders of complete records."""
    return sorted(meta.parent for meta in crate.glob("*/record.json") if not meta.parent.name.startswith("."))


def find_by_hash(crate: Path, digest: str):
    for folder in records(crate):
        if json.loads((folder / "record.json").read_text()).get("hash") == digest:
            return folder.name


def unique_name(crate: Path, name: str) -> str:
    candidate, n = name, 2
    while (crate / candidate).exists():
        candidate, n = f"{name}{n}", n + 1
    return candidate


def best_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def separate(path: Path, device: str, log, progress):
    from demucs.api import Separator

    def callback(info):
        # called when a chunk starts and ends; models in a bag run one after another
        if info.get("state") == "end" and info.get("audio_length"):
            models = info.get("models", 1) or 1
            done = info.get("model_idx_in_bag", 0) + info["segment_offset"] / info["audio_length"]
            progress("separating", min(1.0, done / models))

    try:
        separator = Separator(model=MODEL, device=device, callback=callback)
        mix, stems = separator.separate_audio_file(path)
    except Exception as err:
        if device == "cpu":
            raise
        log(f"separation on {device} failed ({err}), using the cpu")
        separator = Separator(model=MODEL, device="cpu", callback=callback)
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


def bar_starts(grid: Grid, bars: int, samplerate: int) -> list[int]:
    # bar boundaries rounded to samples; consecutive bars join without a gap
    return [round(grid.bar_start(bar) * samplerate) for bar in range(bars + 1)]


def analyze(tracks: dict[str, np.ndarray], starts: list[int], samplerate: int) -> dict:
    """Key and what plays where, for record.json."""
    levels = {stem: bar_levels(audio, starts) for stem, audio in tracks.items()}
    playing = {stem: playing_ranges(stem_levels) for stem, stem_levels in levels.items() if stem != MIX}
    stems = {stem: audio for stem, audio in tracks.items() if stem != MIX}
    return {**detect_key(stems or tracks, samplerate), "levels": levels, "playing": playing}


def add_record(
    source: Path,
    crate: Path,
    name: str | None = None,
    bpm: float | None = None,
    log=print,
    progress=lambda step, fraction: None,
    started=lambda name: None,
) -> tuple[Path, bool]:
    """Adds a track to the crate. Returns the record's folder and whether it was there already."""
    crate.mkdir(parents=True, exist_ok=True)
    digest = file_hash(source)
    existing = find_by_hash(crate, digest)
    if existing:
        log(f"already in the crate as {existing}")
        return crate / existing, True
    title, artist = describe(source)
    name = unique_name(crate, record_name(name or title))
    started(name)
    partial = crate / f".{name}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    device = best_device()

    begin = time.monotonic()
    log(f"separating stems with {MODEL} on {device}…")
    progress("separating", 0.0)
    samplerate, mix, stems = separate(source, device, log, progress)
    separated = time.monotonic()
    log(f"  {separated - begin:.1f}s")

    log("detecting beats…")
    progress("beats", 0.0)
    beats, downbeats = detect_beats(mix, samplerate, device)
    grid = align_to_transients(fit_grid(beats, downbeats, bpm=bpm), stems["drums"], samplerate)
    duration = len(mix) / samplerate
    bars = grid.bar_count(duration)
    log(
        f"  {grid.bpm:.2f} bpm, first downbeat at {grid.first_downbeat:.4f}s (moved {grid.shift_ms:+.1f}ms onto the "
        f"hits), {bars} bars, beats off the grid by {grid.median_error_ms:.1f}ms (median) / "
        f"{grid.max_error_ms:.1f}ms (max)"
    )

    tracks = {MIX: mix, **stems}
    starts = bar_starts(grid, bars, samplerate)
    for i, (stem, audio) in enumerate(tracks.items()):
        progress("writing", i / len(tracks))
        write(partial / "stems" / f"{stem}.flac", audio, samplerate)
        for bar in range(bars):
            write(partial / "bars" / stem / f"{bar:03d}.flac", audio[starts[bar] : starts[bar + 1]], samplerate)

    meta = {
        "format": FORMAT,
        "name": name,
        "title": title,
        "artist": artist,
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
        "seconds": {"separation": separated - begin, "total": time.monotonic() - begin},
        **analyze(tracks, starts, samplerate),
        "beats": [round(float(t), 3) for t in beats],
        "downbeats": [round(float(t), 3) for t in downbeats],
    }
    log(f"  key {meta['key']} ({meta['camelot']}, strength {meta['key_strength']})")
    (partial / "record.json").write_text(json.dumps(meta, indent=1))
    folder = crate / name
    partial.rename(folder)
    log(f"done in {time.monotonic() - begin:.1f}s: {folder}")
    return folder, False


def reanalyze(folder: Path):
    """Recomputes the key and what plays where from the stems on disk, e.g. for records made by an older version."""
    meta = json.loads((folder / "record.json").read_text())
    grid = Grid(**meta["grid"])
    tracks = {stem: sf.read(folder / "stems" / f"{stem}.flac")[0] for stem in meta["stems"]}
    meta.update(analyze(tracks, bar_starts(grid, meta["bars"], meta["samplerate"]), meta["samplerate"]))
    if "format" not in meta and Path(meta["source"]).exists():
        meta["title"], meta["artist"] = describe(Path(meta["source"]))
    meta["format"] = FORMAT
    # replaced in one step, since the app may read it at any time
    partial = folder / ".record.json.partial"
    partial.write_text(json.dumps(meta, indent=1))
    partial.replace(folder / "record.json")
