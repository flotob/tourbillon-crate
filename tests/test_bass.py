import numpy as np

from crate.bass import bass_notes, merge_glides, quiet_level

SR = 44100
BPM = 120
STEP = 60 / BPM / 4
FIRST = 0.1


def tone(midi, length, glide_from=None, decay=None):
    """A bass tone: four harmonics, optionally starting at another pitch and sliding over 80 ms, optionally decaying."""
    t = np.arange(int(length * SR)) / SR
    freq = np.full_like(t, 440 * 2 ** ((midi - 69) / 12))
    if glide_from is not None:
        start = 440 * 2 ** ((glide_from - 69) / 12)
        slide = t < 0.08
        freq[slide] = start + (freq[slide] - start) * t[slide] / 0.08
    phase = 2 * np.pi * np.cumsum(freq) / SR
    wave = sum(np.sin(k * phase) / k for k in range(1, 5))
    envelope = np.exp(-t / decay) if decay else np.ones_like(t)
    envelope[: int(0.003 * SR)] *= np.linspace(0, 1, int(0.003 * SR))
    return 0.3 * wave * envelope


def synthetic_bass():
    """Four bars at 120 bpm: a held A1; D2 for half a bar, a rest, F1; A1 struck on every beat; F1 dropping in from G1."""
    audio = np.zeros(int((FIRST + 4 * 16 * STEP + 0.5) * SR))

    def put(bar, step, steps, sound):
        start = round((FIRST + (bar * 16 + step) * STEP) * SR)
        audio[start : start + len(sound)] += sound[: int(steps * STEP * SR)]

    put(0, 0, 16, tone(33, 16 * STEP))
    put(1, 0, 8, tone(38, 8 * STEP))
    put(1, 12, 4, tone(29, 4 * STEP))
    for beat in range(4):
        put(2, beat * 4, 4, tone(33, 4 * STEP, decay=0.15))
    put(3, 0, 16, tone(29, 16 * STEP, glide_from=31))
    return audio


def test_bass_notes():
    audio = synthetic_bass()
    starts = [FIRST + bar * 16 * STEP for bar in range(5)]
    notes = bass_notes(audio, SR, starts, quiet_db=-50)
    assert notes == [
        [0, 0, 16, 33],
        [1, 0, 8, 38],
        [1, 12, 4, 29],
        [2, 0, 4, 33],
        [2, 4, 4, 33],
        [2, 8, 4, 33],
        [2, 12, 4, 33],
        [3, 0, 16, 29],
    ]


def test_merge_glides():
    # [bar, 16th, length, midi, struck, loudest]
    drop = [[0, 0, 1, 31, 1, -10], [0, 1, 15, 29, 0, -10]]
    assert merge_glides(drop) == [[0, 0, 16, 29, 1, -10]]
    # struck again, or too far: two notes
    struck = [[0, 0, 1, 31, 1, -10], [0, 1, 15, 29, 1, -10]]
    assert len(merge_glides(struck)) == 2
    leap = [[0, 0, 1, 36, 1, -10], [0, 1, 15, 29, 0, -10]]
    assert len(merge_glides(leap)) == 2


def test_quiet_level():
    assert quiet_level([-60] * 5 + [-15] * 20) == -45
    assert quiet_level([-60] * 25) == -60
