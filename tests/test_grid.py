import numpy as np
import pytest

from crate.grid import Grid, align_to_transients, fit_grid


def synthetic_beats(bpm=128.0, first_downbeat=0.5, bars=64, jitter_ms=8.0, seed=0, pickup_beats=1):
    rng = np.random.default_rng(seed)
    period = 60 / bpm
    # a pickup beat before the first downbeat, as many tracks have
    indices = np.arange(-pickup_beats, bars * 4)
    beats = first_downbeat + indices * period + rng.normal(0, jitter_ms / 1000, len(indices))
    beats = beats[beats >= 0]
    downbeats = first_downbeat + np.arange(bars) * 4 * period
    return beats, downbeats


def test_recovers_tempo_and_first_downbeat():
    beats, downbeats = synthetic_beats(bpm=128.0, first_downbeat=0.512)
    grid = fit_grid(beats, downbeats)
    assert grid.bpm == pytest.approx(128.0, abs=1e-9)  # snapped to the whole number
    assert grid.first_downbeat == pytest.approx(0.512, abs=0.004)
    assert grid.median_error_ms < 10


def test_keeps_fractional_tempo():
    beats, downbeats = synthetic_beats(bpm=122.5, first_downbeat=0.2, jitter_ms=2)
    grid = fit_grid(beats, downbeats)
    assert grid.bpm == pytest.approx(122.5, abs=0.01)


def test_ignores_misdetected_and_missing_beats():
    beats, downbeats = synthetic_beats(bpm=124.0, first_downbeat=1.0)
    beats = np.delete(beats, np.arange(60, 90))  # a breakdown without detected beats
    beats = np.sort(np.append(beats, [30.11, 31.37]))  # two beats off the grid
    grid = fit_grid(beats, downbeats)
    assert grid.bpm == pytest.approx(124.0, abs=1e-9)
    assert grid.first_downbeat == pytest.approx(1.0, abs=0.004)


def test_downbeat_phase_from_downbeats_not_first_beat():
    # the first detected beat is the third beat of a bar
    beats, downbeats = synthetic_beats(bpm=130.0, first_downbeat=0.9, pickup_beats=2, jitter_ms=1)
    grid = fit_grid(beats, downbeats)
    assert grid.first_downbeat == pytest.approx(0.9, abs=0.003)


def test_forced_bpm_for_half_time_detections():
    # the tracker only found every second beat
    beats, downbeats = synthetic_beats(bpm=128.0, first_downbeat=0.5, jitter_ms=1)
    grid = fit_grid(beats[1::2], downbeats, bpm=128)
    assert grid.bpm == 128
    assert grid.first_downbeat == pytest.approx(0.5, abs=0.003)


def test_bar_count_and_starts():
    beats, downbeats = synthetic_beats(bpm=120.0, first_downbeat=0.0, jitter_ms=0)
    grid = fit_grid(beats, downbeats)
    assert grid.bar_length == pytest.approx(2.0)
    assert grid.bar_count(duration=129.0) == 64
    assert grid.bar_start(10) == pytest.approx(20.0)


def test_beats_quantized_to_tracker_frames():
    # beat this reports times on a 50 fps frame grid; a 6 minute track has ~770 beats
    beats, downbeats = synthetic_beats(bpm=128.0, first_downbeat=0.37, bars=192, jitter_ms=3)
    beats, downbeats = np.round(beats * 50) / 50, np.round(downbeats * 50) / 50
    grid = fit_grid(beats, downbeats)
    assert grid.bpm == pytest.approx(128.0, abs=1e-9)
    assert grid.first_downbeat == pytest.approx(0.37, abs=0.006)
    # frames alone put beats up to 10 ms off, jitter adds to that
    assert grid.median_error_ms < 8
    assert grid.max_error_ms < 25


def test_spurious_detections_do_not_shift_the_count():
    beats, downbeats = synthetic_beats(bpm=126.0, first_downbeat=0.6, bars=128, jitter_ms=2)
    beats = np.round(beats * 50) / 50
    # extra detections between beats, e.g. on a syncopated clap
    beats = np.sort(np.append(beats, beats[100:110] + 0.11))
    grid = fit_grid(beats, downbeats)
    assert grid.bpm == pytest.approx(126.0, abs=1e-9)
    assert grid.first_downbeat == pytest.approx(0.6, abs=0.006)


def test_aligns_grid_to_transients():
    sr, bpm, first = 44100, 128.0, 0.395
    period = 60 / bpm
    audio = np.zeros(int(sr * 20))
    rng = np.random.default_rng(2)
    hit = rng.normal(0, 0.5, 2000) * np.exp(-np.arange(2000) / 300)
    for beat in range(int((20 - first - 0.1) / period)):
        start = round((first + beat * period) * sr)
        audio[start : start + len(hit)] += hit
    late = Grid(
        bpm=bpm, beats_per_bar=4, first_downbeat=first + 0.012, median_error_ms=5, max_error_ms=10, beats_used=40
    )
    aligned = align_to_transients(late, audio, sr)
    assert aligned.first_downbeat == pytest.approx(first, abs=0.0015)
    assert aligned.shift_ms == pytest.approx(-12, abs=1.5)
