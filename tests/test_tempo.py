import json
import shutil
from dataclasses import asdict

import numpy as np
import pytest
import soundfile as sf

from crate.grid import Grid
from crate.tempo import cut, make_version, tempo_label, time_map, versions

SR = 44100


def test_tempo_label():
    assert tempo_label(124.0) == "124"
    assert tempo_label(124.00000000000001) == "124"
    assert tempo_label(123.5) == "123.5"
    assert tempo_label(123.456) == "123.46"


def test_time_map_pins_every_beat():
    grid = Grid(bpm=120, beats_per_bar=4, first_downbeat=1.1, median_error_ms=0, max_error_ms=0, beats_used=0)
    lines = time_map(grid, duration=4, ratio=0.5, samplerate=1000).splitlines()
    # beats at 0.1, 0.6, 1.1, ... 3.6 s, including the pickup before the first downbeat
    assert lines[0] == "100 50"
    assert lines[2] == "1100 550"
    assert len(lines) == 8


def test_cut_pads_with_silence():
    audio = np.arange(10, dtype=float).reshape(-1, 1)
    assert cut(audio, 2, 5)[:, 0].tolist() == [2, 3, 4]
    assert cut(audio, -2, 2)[:, 0].tolist() == [0, 0, 0, 1]
    assert cut(audio, 8, 12)[:, 0].tolist() == [8, 9, 0, 0]


def synthetic_record(folder, bpm=120.0, first_downbeat=0.1, bars=8):
    """A click on every beat (louder on the downbeat) in the drums, a held low A in the bass."""
    beat = 60 / bpm
    duration = first_downbeat + bars * 4 * beat + 0.5
    frames = int(duration * SR)
    drums = np.zeros(frames)
    click = np.random.default_rng(0).normal(0, 0.5, int(0.03 * SR)) * np.exp(-np.arange(int(0.03 * SR)) / 200)
    for i in range(bars * 4):
        start = round((first_downbeat + i * beat) * SR)
        drums[start : start + len(click)] += click * (1 if i % 4 == 0 else 0.5)
    bass = 0.3 * np.sin(2 * np.pi * 55 * np.arange(frames) / SR)
    stems = {"drums": drums, "bass": bass}
    for stem, audio in stems.items():
        sf.write(folder / "stems" / f"{stem}.flac", np.stack([audio, audio], axis=1), SR, subtype="PCM_16")
    grid = Grid(
        bpm=bpm, beats_per_bar=4, first_downbeat=first_downbeat, median_error_ms=0, max_error_ms=0, beats_used=0
    )
    meta = {"name": "synth", "bpm": bpm, "samplerate": SR, "grid": asdict(grid), "stems": list(stems), "bars": bars}
    (folder / "record.json").write_text(json.dumps({**meta, "duration": duration}))


@pytest.mark.skipif(not shutil.which("rubberband"), reason="needs the rubberband command")
def test_version_keeps_bars_on_the_beat_and_the_pitch(tmp_path):
    (tmp_path / "stems").mkdir()
    synthetic_record(tmp_path)
    folder, existing = make_version(tmp_path, 126, log=lambda message: None)
    assert not existing
    assert folder == tmp_path / "tempo" / "126"
    assert versions(tmp_path) == [126.0]
    assert json.loads((folder / "version.json").read_text())["from_bpm"] == 120

    bar_frames = 240 / 126 * SR
    for bar in range(8):
        drums, _ = sf.read(folder / "drums" / f"{bar:03d}.flac")
        assert abs(len(drums) - bar_frames) <= 1
        # the downbeat click starts the bar, within 2 ms
        assert np.argmax(np.abs(drums[:, 0]) > 0.05) < 0.002 * SR
    bass, _ = sf.read(folder / "bass" / "004.flac")
    spectrum = np.abs(np.fft.rfft(bass[:, 0] * np.hanning(len(bass))))
    assert np.fft.rfftfreq(len(bass), 1 / SR)[spectrum.argmax()] == pytest.approx(55, abs=1)

    # asking again finds it
    assert make_version(tmp_path, 126.001, log=lambda message: None) == (folder, True)


def test_version_needs_another_tempo(tmp_path):
    (tmp_path / "stems").mkdir()
    synthetic_record(tmp_path, bars=2)
    with pytest.raises(ValueError, match="already"):
        make_version(tmp_path, 120)
    with pytest.raises(ValueError, match="too far"):
        make_version(tmp_path, 240)
