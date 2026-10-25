import numpy as np

from crate.key import camelot, detect_key

SR = 44100


def tone(freqs, seconds):
    t = np.arange(int(seconds * SR)) / SR
    # a few harmonics, like a synth
    wave = sum(np.sin(2 * np.pi * f * h * t) / h for f in freqs for h in (1, 2, 3))
    return wave * np.exp(-t) / len(freqs)


def note(name, octave=4):
    index = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"].index(name)
    return 440 * 2 ** ((index - 9) / 12 + octave - 4)


def test_camelot():
    assert camelot(9, "minor") == "8A"  # A minor
    assert camelot(0, "major") == "8B"  # C major
    assert camelot(4, "minor") == "9A"  # E minor
    assert camelot(8, "minor") == "1A"  # G# minor
    assert camelot(11, "major") == "1B"  # B major
    assert camelot(1, "major") == "3B"  # Db major


def test_detects_a_minor_without_the_drums():
    chords = [["A", "C", "E"], ["D", "F", "A"], ["E", "G", "B"], ["A", "C", "E"]]
    other = np.concatenate([tone([note(n) for n in chord], 2) for chord in chords * 4])
    bass = np.concatenate([tone([note(chord[0], 2)], 2) for chord in chords * 4])
    # loud noise in the drums would pull a key estimate anywhere; it is left out
    drums = np.random.default_rng(0).normal(0, 1, len(other))
    key = detect_key({"drums": drums, "bass": bass, "other": other}, SR)
    assert key["key"] == "A minor"
    assert key["camelot"] == "8A"
    assert 0 < key["key_strength"] <= 1
