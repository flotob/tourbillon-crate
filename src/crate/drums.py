"""Drum patterns: which drum hits on which 16th of every bar, how loud, and a kit cut from the record itself.

ADTOF (Zehren et al., 2023) finds the hits in the drums stem. It is free for non-commercial use only
(CC BY-NC-SA 4.0), so it is not part of crate: it is installed on first use into its own environment
(<crate>/.tools/adtof) and runs there as a separate program (adtof/run.py).

Layout, next to the record's bars:
    <name>/drums/drums.json      every hit as [bar, 16th, velocity, ms off the 16th], per drum
    <name>/drums/<drum>.flac     one hit of each drum, where it plays alone: bd, sd, hh, perc, cr

In strudel the kit is a bank, so a pattern plays on the record's drums or any drum machine:
    s("bd*4, ~ sd ~ sd, [~ hh]*4").bank("<name>")
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import soundfile as sf

from crate.grid import Grid, onset_strength
from crate.record import write
from crate.tempo import cut

ADTOF_URL = "https://github.com/MZehren/ADTOF"
ADTOF_COMMIT = "b3968fb332f69b65ee07c089fc62f436503755db"
ADTOF_PYTHON = "3.11"
# raise to reinstall ADTOF's environment, e.g. when requirements.txt changes
ADTOF_SETUP = "1"
TOOLS = ".tools"
FOLDER = "drums"
STEPS = 16

# ADTOF's classes (General MIDI) and their names in strudel; ADTOF's toms are mostly other percussion in
# electronic music, like shakers and congas
DRUMS = {35: "bd", 38: "sd", 42: "hh", 47: "perc", 49: "cr"}
# fewer hits than this are taken for mistakes
MIN_HITS = 4
# where in the spectrum each drum's loudness is measured, in Hz, so a kick under a hi-hat doesn't count
BANDS = {"bd": (30, 150), "sd": (150, 5000), "hh": (6000, 16000), "perc": (2000, 12000), "cr": (4000, 16000)}
# shortest and longest hit of the kit, in seconds
LENGTHS = {"bd": (0.15, 0.6), "sd": (0.12, 0.5), "hh": (0.05, 0.4), "perc": (0.05, 0.4), "cr": (0.3, 2.0)}
# the kick's tail is filtered out of the bright drums, in Hz
HIGHPASS = {"hh": 300, "perc": 300, "cr": 300}
# the kick rings on under hi-hats and percussion: after its attack, it is followed and cut below this, in Hz
KICK_TAIL = 1500
# hits closer than this play together
TOGETHER = 0.03
# velocities in steps of 3 dB, down to -18 dB
VELOCITY_STEP_DB, VELOCITY_FLOOR_DB = 3.0, -18.0


def uv() -> str:
    found = os.environ.get("UV") or shutil.which("uv")
    if not found:
        raise RuntimeError("uv not found, install it from https://docs.astral.sh/uv/")
    return found


def ensure_adtof(tools: Path, log=print, progress=lambda step, fraction: None) -> Path:
    """Python of ADTOF's environment, installed on first use (about 1.5 GB). A partial install is redone."""
    env = tools / "adtof"
    python = env / "bin" / "python"
    ready = env / ".ready"
    if python.exists() and ready.exists() and ready.read_text().strip() == ADTOF_SETUP:
        return python
    log("installing ADTOF into its own environment, once (about 1.5 GB)…")
    log(f"ADTOF is free for non-commercial use only (CC BY-NC-SA 4.0): {ADTOF_URL}")
    shutil.rmtree(env, ignore_errors=True)
    requirements = Path(__file__).parent / "adtof" / "requirements.txt"
    install = [uv(), "pip", "install", "--python", str(python)]
    steps = [
        ("python", [uv(), "venv", "--python", ADTOF_PYTHON, str(env)]),
        # madmom builds against the numpy and cython in the environment
        ("build tools", [*install, "cython==3.3.0", "numpy==1.26.4", "setuptools"]),
        ("dependencies", [*install, "--no-build-isolation-package", "madmom", "-r", str(requirements)]),
        ("adtof", [*install, "--no-deps", f"adtof @ git+{ADTOF_URL}@{ADTOF_COMMIT}"]),
    ]
    for i, (what, command) in enumerate(steps):
        progress("installing_adtof", i / len(steps))
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            lines = [line for line in result.stderr.splitlines() if line.strip()]
            raise RuntimeError(f"installing ADTOF failed at {what}: {' / '.join(lines[-3:])}")
    ready.write_text(ADTOF_SETUP)
    return python


def run_adtof(python: Path, audio: Path) -> dict[str, list[float]]:
    """Hit times in seconds per drum name, from ADTOF in its own environment."""
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "hits.json"
        result = subprocess.run(
            [str(python), "-I", str(Path(__file__).parent / "adtof" / "run.py"), str(audio), str(output)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not output.exists():
            lines = [line for line in result.stderr.splitlines() if line.strip()]
            raise RuntimeError(f"ADTOF failed: {' / '.join(lines[-3:])}")
        hits = json.loads(output.read_text())
    return {DRUMS[int(label)]: sorted(t for t, _ in found) for label, found in hits.items() if int(label) in DRUMS}


def attack(mono: np.ndarray, samplerate: int, t: float, search: float = 0.015) -> float:
    """Where the hit near t starts: ADTOF reports hits on a 10 ms grid, a cut needs the millisecond."""
    lo = max(0, round((t - search) * samplerate))
    strength, step = onset_strength(mono[lo : round((t + search) * samplerate)], samplerate)
    return lo / samplerate + int(np.argmax(strength)) * step if len(strength) else t


def spectrum(mono: np.ndarray, samplerate: int, t: float, length: float) -> tuple[np.ndarray, np.ndarray]:
    """Power spectrum of the audio from t on, and its frequencies."""
    start = max(0, round(t * samplerate))
    part = mono[start : start + round(length * samplerate)]
    size = 4096
    if len(part) < 64:
        return np.zeros(size // 2 + 1), np.fft.rfftfreq(size, 1 / samplerate)
    power = np.abs(np.fft.rfft(part * np.hanning(len(part)), n=size)) ** 2
    return power, np.fft.rfftfreq(size, 1 / samplerate)


def band_level(mono: np.ndarray, samplerate: int, t: float, band: tuple[float, float]) -> float:
    """Loudness of a hit in dB, in the band that belongs to its drum."""
    power, freqs = spectrum(mono, samplerate, t - 0.002, 0.03)
    return float(10 * np.log10(power[(freqs >= band[0]) & (freqs < band[1])].sum() + 1e-12))


def profile(mono: np.ndarray, samplerate: int, t: float, low: float = 40, high: float = 16000) -> np.ndarray:
    """How a hit sounds, roughly: its loudness in eight bands from low to high Hz, in dB."""
    power, freqs = spectrum(mono, samplerate, t, 0.05)
    edges = np.geomspace(low, high, 9)
    return np.array([10 * np.log10(power[(freqs >= lo) & (freqs < hi)].sum() + 1e-12) for lo, hi in pairwise(edges)])


def velocities(levels_db) -> np.ndarray:
    """0..1 per hit, relative to the drum's loud hits, in steps of 3 dB; within 3 dB of them is 1, so a
    drum that always plays at the same level doesn't flicker between two steps."""
    levels = np.asarray(levels_db, dtype=float)
    below = np.clip(np.percentile(levels, 90) - levels, 0, -VELOCITY_FLOOR_DB)
    return np.round(10 ** (-np.floor(below / VELOCITY_STEP_DB) * VELOCITY_STEP_DB / 20), 2)


def place(times, levels_db, grid: Grid, bars: int) -> list[list]:
    """[bar, 16th, velocity, ms off the 16th] per hit, the louder one where two fall on the same 16th."""
    step = grid.bar_length / STEPS
    found = {}
    for t, velocity in zip(times, velocities(levels_db)):
        position = (t - grid.first_downbeat) / step
        index = round(position)
        if 0 <= index < bars * STEPS and velocity > found.get(index, [0, 0, 0])[2]:
            found[index] = [index // STEPS, index % STEPS, float(velocity), round((position - index) * step * 1000, 1)]
    return [found[index] for index in sorted(found)]


def filtered(audio: np.ndarray, samplerate: int, above: float | None = None, below: float | None = None):
    """audio without what is under `above` and over `below` Hz, each edge an octave wide."""
    size = 1 << int(np.ceil(np.log2(2 * max(1, len(audio)))))
    freqs = np.fft.rfftfreq(size, 1 / samplerate)
    gain = np.ones_like(freqs)
    if above:
        gain *= 0.5 - 0.5 * np.cos(np.pi * np.clip((freqs - above / 2) / (above / 2), 0, 1))
    if below:
        gain *= 0.5 + 0.5 * np.cos(np.pi * np.clip((freqs - below) / below, 0, 1))
    shape = (-1,) + (1,) * (audio.ndim - 1)
    return np.fft.irfft(np.fft.rfft(audio, n=size, axis=0) * gain.reshape(shape), n=size, axis=0)[: len(audio)]


def envelope(audio: np.ndarray, samplerate: int, hop: float = 0.005) -> np.ndarray:
    """Loudness in dB per 5 ms."""
    size = round(hop * samplerate)
    frames = len(audio) // size
    return 10 * np.log10(np.mean(audio[: frames * size].reshape(frames, size) ** 2, axis=1) + 1e-12)


def ring(
    mono: np.ndarray, samplerate: int, t: float, longest: float, below: float | None = None
) -> tuple[float, float, float]:
    """How the hit at t rings out. ADTOF misses quiet hits, so this listens for them; with below, only to what
    is under that frequency.

    Returns how long it rings before it fades out, something else starts or longest is reached (seconds),
    how loud the audio just before it is (dB) and how loud it still is at that point (dB, -50 when it
    faded out), both relative to the hit.
    """
    hop = 0.005
    part = cut(mono[:, None], round((t - 0.02) * samplerate), round((t + longest) * samplerate))[:, 0]
    if below:
        part = filtered(part, samplerate, below=below)
    lead = round(0.02 * samplerate)
    before, after = (
        envelope(part[: lead - round(0.002 * samplerate)], samplerate, hop),
        envelope(part[lead:], samplerate, hop),
    )
    if len(after) < 2:
        return 0.0, 0.0, 0.0
    peak = int(np.argmax(after[:6]))
    top = after[peak]
    quiet_before = float(np.max(before) - top) if len(before) else -60.0
    lowest = top
    for k in range(peak + 1, len(after)):
        if after[k] < top - 50:
            return k * hop, quiet_before, -50.0
        lowest = min(lowest, after[k])
        # rising again by 6 dB: another hit starts in this frame
        if after[k] > lowest + 6 and after[k] > top - 45:
            return (k - 1) * hop, quiet_before, float(lowest - top)
    return len(after) * hop, quiet_before, float(after[-1] - top)


def choose_hit(drum: str, times: np.ndarray, others: dict[str, np.ndarray], mono: np.ndarray, samplerate: int) -> dict:
    """The hit to cut for the kit: one that sounds like most of the drum's hits and rings out alone.

    others are the hit times of the other drums. Returns the hit's time, how long it can be and whether
    it is clean, i.e. nothing else plays over it.
    """
    kick = drum == "bd"
    below = KICK_TAIL if kick else None
    # what cuts its ring short: everything, or for the kick what sounds under KICK_TAIL
    rivals = np.sort(np.concatenate([times] + [t for other, t in others.items() if not kick or other in ("bd", "sd")]))
    other_times = np.sort(np.concatenate(list(others.values()) or [np.zeros(0)]))
    shortest, longest = LENGTHS[drum]
    # a kick is compared by its own range, since hi-hats often play over it
    profiles = np.array([profile(mono, samplerate, t, *((30, 2000) if kick else ())) for t in times])
    typical = np.median(profiles, axis=0)
    levels = np.array([band_level(mono, samplerate, t, BANDS[drum]) for t in times])
    loud = np.percentile(levels, 75)
    best = None
    for i, t in enumerate(times):
        together = len(other_times) and np.min(np.abs(other_times - t)) < TOGETHER
        later = rivals[rivals > t + TOGETHER]
        until = min(longest, later[0] - t) if len(later) else longest
        room, before_db, left_db = ring(mono, samplerate, t, until, below)
        if until == longest and room >= until - 0.005:
            # long enough: cut at its longest, not cut off by another hit
            left_db = -50.0
        clean = not together and room >= shortest
        # dB that the hit is off: how it sounds, how loud it is, how much of it is cut off, what plays before it
        score = (
            np.mean(np.abs(profiles[i] - typical))
            + 0.5 * abs(levels[i] - loud)
            + 12 * max(0.0, 50 + left_db) / 50
            + 0.5 * max(0.0, before_db + 30)
            + (0 if clean else 100)
        )
        if best is None or score < best[0]:
            best = (
                score,
                {"time": round(float(t), 4), "length": round(float(max(room, shortest)), 4), "clean": bool(clean)},
            )
    return best[1]


def one_shot(audio: np.ndarray, samplerate: int, drum: str, start: float, length: float) -> np.ndarray:
    """A hit cut from start for length seconds, from a millisecond before it, faded out at the end."""
    shot = cut(audio, round((start - 0.001) * samplerate), round((start + length) * samplerate)).astype(np.float64)
    if drum in HIGHPASS:
        shot = filtered(shot, samplerate, above=HIGHPASS[drum])
    if drum == "bd":
        # the attack whole, then only the low end, without the hi-hats over the tail
        blend = np.clip((np.arange(len(shot)) / samplerate - 0.02) / 0.02, 0, 1)[:, None]
        shot = shot * (1 - blend) + filtered(shot, samplerate, below=KICK_TAIL) * blend
    fade_in = min(len(shot), round(0.001 * samplerate))
    shot[:fade_in] *= np.linspace(0, 1, fade_in)[:, None]
    fade_out = min(len(shot), round(min(0.03, 0.25 * length) * samplerate))
    shot[len(shot) - fade_out :] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, fade_out)))[:, None]
    return shot


def drum_patterns(folder: Path, hits: dict[str, list[float]], log=print) -> dict:
    """Writes the record's drums folder (see the module docstring) from hit times per drum. Returns drums.json."""
    meta = json.loads((folder / "record.json").read_text())
    grid = Grid(**meta["grid"])
    audio, samplerate = sf.read(folder / "stems" / "drums.flac", always_2d=True)
    mono = audio.mean(axis=1)
    found = {drum: np.array(times) for drum, times in hits.items() if len(times) >= MIN_HITS}
    partial = folder / f".{FOLDER}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir()
    drums, placed = {}, {}
    for drum, times in found.items():
        times = np.array([attack(mono, samplerate, t) for t in times])
        levels = [band_level(mono, samplerate, t, BANDS[drum]) for t in times]
        placed[drum] = place(times, levels, grid, meta["bars"])
        others = {other: np.array(t) for other, t in hits.items() if other != drum}
        kit = choose_hit(drum, times, others, mono, samplerate)
        write(partial / f"{drum}.flac", one_shot(audio, samplerate, drum, kit["time"], kit["length"]), samplerate)
        drums[drum] = {"hits": len(placed[drum]), "kit": kit}
        log(
            f"  {drum}: {len(placed[drum])} hits, kit from {kit['time']:.2f}s ({kit['length'] * 1000:.0f} ms"
            + ("" if kit["clean"] else ", with other drums over it")
            + ")"
        )
    result = {
        "format": 1,
        "model": f"ADTOF Frame_RNN, trained on all of ADTOF's data, at {ADTOF_COMMIT[:7]}",
        "steps": STEPS,
        "created_at": time.time(),
        "drums": drums,
        "hits": placed,
    }
    (partial / "drums.json").write_text(json.dumps(result))
    shutil.rmtree(folder / FOLDER, ignore_errors=True)
    partial.rename(folder / FOLDER)
    return result


def find_drums(folder: Path, crate: Path, log=print, progress=lambda step, fraction: None) -> dict:
    """Finds the drum patterns of a record and cuts its kit, installing ADTOF first if needed."""
    begin = time.monotonic()
    python = ensure_adtof(crate / TOOLS, log, progress)
    log(f"finding the drums of {folder.name} with ADTOF…")
    progress("transcribing", 0.0)
    hits = run_adtof(python, folder / "stems" / "drums.flac")
    progress("cutting", 0.0)
    result = drum_patterns(folder, hits, log)
    log(f"done in {time.monotonic() - begin:.1f}s: {folder / FOLDER}")
    return result
