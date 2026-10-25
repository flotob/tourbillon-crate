import importlib
import json
import stat
from dataclasses import asdict

import numpy as np
import pytest
import soundfile as sf

from crate.grid import Grid

# crate's __init__ has a `drums` command that shadows the module as an attribute
drums = importlib.import_module("crate.drums")

SR = 44100


def kick(length=0.4):
    """A 55 Hz thump that decays to silence over length seconds."""
    t = np.arange(int(length * SR)) / SR
    return 0.8 * np.sin(2 * np.pi * 55 * t) * np.exp(-t / (length / 8))


def hat(length=0.06, seed=0):
    """A short burst of high noise."""
    noise = np.random.default_rng(seed).normal(0, 0.3, int(length * SR))
    noise = np.diff(noise, prepend=0)  # tilted towards the highs
    return noise * np.exp(-np.arange(len(noise)) / (length * SR / 6))


def add(audio, sound, at):
    start = round(at * SR)
    audio[start : start + len(sound)] += sound[: len(audio) - start]


def synthetic_record(folder, bpm=120.0, first_downbeat=0.1, bars=4):
    """Kicks on every beat (quieter in the last bar), hats on the offbeats except in bar 1. Returns the hit times."""
    beat = 60 / bpm
    duration = first_downbeat + bars * 4 * beat + 0.5
    audio = np.zeros(int(duration * SR))
    hits = {"bd": [], "hh": []}
    for i in range(bars * 4):
        t = first_downbeat + i * beat
        add(audio, kick() * (0.4 if i >= (bars - 1) * 4 else 1), t)
        hits["bd"].append(t)
        if i // 4 != 1:
            add(audio, hat(seed=i), t + beat / 2)
            hits["hh"].append(t + beat / 2)
    (folder / "stems").mkdir(parents=True)
    sf.write(folder / "stems" / "drums.flac", np.stack([audio, audio], axis=1), SR, subtype="PCM_16")
    grid = Grid(
        bpm=bpm, beats_per_bar=4, first_downbeat=first_downbeat, median_error_ms=0, max_error_ms=0, beats_used=0
    )
    meta = {"name": "synth", "bpm": bpm, "samplerate": SR, "grid": asdict(grid), "bars": bars, "duration": duration}
    (folder / "record.json").write_text(json.dumps(meta))
    return hits


def test_velocities_in_steps_of_3_db():
    levels = [0.0] * 10 + [-1, -2.9, -3.1, -7, -40]
    assert drums.velocities(levels).tolist() == [1.0] * 10 + [1.0, 1.0, 0.71, 0.5, 0.13]


def test_place_on_16ths():
    grid = Grid(bpm=120, beats_per_bar=4, first_downbeat=0.1, median_error_ms=0, max_error_ms=0, beats_used=0)
    # a 16th is 125 ms; the second hit is 10 ms late, the third falls on the same 16th but is louder,
    # the last is before the first bar
    times = [0.1, 0.235, 0.24, 0.1 + 2.0 + 0.375, 0.05]
    placed = drums.place(times, [0, -10, 0, -4, 0], grid, bars=2)
    assert placed == [[0, 0, 1.0, 0.0], [0, 1, 1.0, 15.0], [1, 3, 0.71, 0.0]]


def test_ring_stops_where_another_hit_starts():
    audio = np.zeros(SR)
    add(audio, kick(0.5), 0.1)
    add(audio, kick(0.5), 0.3)
    length, before_db, left_db = drums.ring(audio, SR, 0.1, 0.6)
    assert length == pytest.approx(0.2, abs=0.011)
    assert before_db < -50
    # still loud where the next one starts
    assert -30 < left_db < -5

    alone = np.zeros(SR)
    add(alone, kick(0.3), 0.1)
    length, _, left_db = drums.ring(alone, SR, 0.1, 0.6)
    assert length < 0.3
    assert left_db == -50


def test_kick_rings_under_hats():
    audio = np.zeros(SR)
    add(audio, kick(0.5), 0.1)
    add(audio, hat(), 0.3)
    # over the whole spectrum the hat cuts it short; below the hat it rings on
    assert drums.ring(audio, SR, 0.1, 0.6)[0] < 0.21
    assert drums.ring(audio, SR, 0.1, 0.6, below=drums.KICK_TAIL)[0] > 0.25


def test_drum_patterns(tmp_path):
    hits = synthetic_record(tmp_path)
    result = drums.drum_patterns(tmp_path, hits, log=lambda message: None)
    assert result == json.loads((tmp_path / "drums" / "drums.json").read_text())
    assert set(result["drums"]) == {"bd", "hh"}
    bd, hh = result["hits"]["bd"], result["hits"]["hh"]
    assert [(bar, step) for bar, step, *_ in bd] == [(bar, step) for bar in range(4) for step in (0, 4, 8, 12)]
    assert [(bar, step) for bar, step, *_ in hh] == [(bar, step) for bar in (0, 2, 3) for step in (2, 6, 10, 14)]
    # the last bar is 8 dB quieter, which rounds up to -6 dB
    assert {velocity for bar, _, velocity, _ in bd if bar == 3} == {0.5}
    assert {velocity for bar, _, velocity, _ in bd if bar < 3} == {1.0}
    assert max(abs(offset) for *_, offset in bd + hh) < 3

    # the kit: the kick from a bar without hats is not needed, since it rings on under them; the hat is
    # cut before the next kick
    kit_bd, kit_hh = result["drums"]["bd"]["kit"], result["drums"]["hh"]["kit"]
    assert kit_bd["clean"] and kit_hh["clean"]
    assert kit_bd["length"] > 0.25
    assert kit_hh["length"] < 0.25
    shot, samplerate = sf.read(tmp_path / "drums" / "bd.flac")
    assert samplerate == SR
    assert len(shot) == pytest.approx(kit_bd["length"] * SR, abs=0.002 * SR)
    # it starts right at the hit
    assert np.argmax(np.abs(shot[:, 0]) > 0.05) < 0.003 * SR
    assert (tmp_path / "drums" / "hh.flac").exists()
    assert not (tmp_path / ".drums.partial").exists()


def test_rare_drums_are_left_out(tmp_path):
    hits = synthetic_record(tmp_path)
    hits["cr"] = [1.0, 2.0]
    assert "cr" not in drums.drum_patterns(tmp_path, hits, log=lambda message: None)["drums"]


def fake_python(path, output):
    """A stand-in for ADTOF's python that writes output wherever run.py would."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\ncat > \"$4\" <<'EOF'\n{json.dumps(output)}\nEOF\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_run_adtof_names_the_drums(tmp_path):
    python = tmp_path / "bin" / "python"
    fake_python(python, {"35": [[0.5, 0.9], [0.1, 0.8]], "42": [[0.3, 0.5]], "47": [], "60": [[1.0, 0.5]]})
    assert drums.run_adtof(python, tmp_path / "drums.flac") == {"bd": [0.1, 0.5], "hh": [0.3], "perc": []}


def test_adtof_is_installed_once(tmp_path, monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[1] == "venv":
            fake_python(tmp_path / "adtof" / "bin" / "python", {})
        return type("Result", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(drums.subprocess, "run", run)
    monkeypatch.setenv("UV", "uv")
    python = drums.ensure_adtof(tmp_path, log=lambda message: None)
    assert python == tmp_path / "adtof" / "bin" / "python"
    assert commands[0][:4] == ["uv", "venv", "--python", "3.11"]
    assert commands[-1][-2:] == ["--no-deps", f"adtof @ git+{drums.ADTOF_URL}@{drums.ADTOF_COMMIT}"]
    # ready: nothing to do
    commands.clear()
    assert drums.ensure_adtof(tmp_path, log=lambda message: None) == python
    assert commands == []


def test_a_failed_install_says_where(tmp_path, monkeypatch):
    def run(command, **kwargs):
        return type("Result", (), {"returncode": 1, "stderr": "error: no network\n"})()

    monkeypatch.setattr(drums.subprocess, "run", run)
    monkeypatch.setenv("UV", "uv")
    with pytest.raises(RuntimeError, match="at python: error: no network"):
        drums.ensure_adtof(tmp_path, log=lambda message: None)
