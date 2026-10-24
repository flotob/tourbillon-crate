"""Fits a constant-tempo bar grid to detected beats, which suits electronic music.

A tracker reports beats with small timing errors and occasional misses. One straight
line through all of them (beat time = offset + index * period) gives a grid that does
not drift over a whole track, and bar-aligned slices cut from it loop cleanly.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

# tempos of electronic music are almost always whole numbers
SNAP_BPM_WITHIN = 0.03


@dataclass
class Grid:
    bpm: float
    beats_per_bar: int
    # time of the first downbeat at or after the start of the track
    first_downbeat: float
    # how far detected beats are from the grid, in milliseconds
    median_error_ms: float
    max_error_ms: float
    beats_used: int
    # how far aligning to the transients moved the grid
    shift_ms: float = 0.0

    @property
    def beat_period(self) -> float:
        return 60 / self.bpm

    @property
    def bar_length(self) -> float:
        return self.beats_per_bar * self.beat_period

    def bar_count(self, duration: float) -> int:
        return max(0, math.floor((duration - self.first_downbeat) / self.bar_length + 1e-6))

    def bar_start(self, bar: int) -> float:
        return self.first_downbeat + bar * self.bar_length


def _fit_line(times, indices):
    design = np.vstack([np.ones_like(indices), indices]).T
    (offset, period), *_ = np.linalg.lstsq(design, times, rcond=None)
    return offset, period


def _count_beats(beats, period):
    """Beat number of each detection, from the gaps between neighbours.

    Rounding each gap on its own keeps a slightly wrong period from adding up over a whole
    track. Detections less than half a beat apart share a number and end up as outliers.
    """
    return np.concatenate([[0.0], np.cumsum(np.round(np.diff(beats) / period))])


def fit_grid(beats, downbeats, beats_per_bar=4, bpm=None) -> Grid:
    """beats and downbeats in seconds; bpm forces the tempo, e.g. when the tracker found half or double time."""
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 8:
        raise ValueError(f"only {len(beats)} beats detected, too few for a grid")

    if bpm:
        period = 60 / bpm
        indices = _count_beats(beats, period)
        offset = float(np.median(beats - indices * period))
    else:
        # trackers report times on a frame grid (beat this: 20 ms), so single gaps are off by up to a frame;
        # the mean over all regular gaps is not
        gaps = np.diff(beats)
        regular = np.abs(gaps - np.median(gaps)) < 0.25 * np.median(gaps)
        indices = _count_beats(beats, float(np.mean(gaps[regular])))
        offset, period = _fit_line(beats, indices)

    # with a good period the line itself numbers the beats; refit without beats far off it
    for _ in range(3):
        indices = np.round((beats - offset) / period)
        residuals = beats - (offset + indices * period)
        keep = np.abs(residuals) < 0.15 * period
        if keep.sum() < 8:
            raise ValueError("detected beats do not follow a constant tempo")
        if bpm:
            offset = float(np.median(beats[keep] - indices[keep] * period))
        else:
            offset, period = _fit_line(beats[keep], indices[keep])

    fitted_bpm = 60 / period
    if not bpm and abs(fitted_bpm - round(fitted_bpm)) < SNAP_BPM_WITHIN:
        period = 60 / round(fitted_bpm)
        indices = np.round((beats - offset) / period)
        offset = float(np.median(beats[keep] - indices[keep] * period))

    indices = np.round((beats - offset) / period).astype(int)
    errors_ms = np.abs(beats - (offset + indices * period))[keep] * 1000

    # which beat of the bar is the downbeat: majority vote over the detected downbeats
    downbeats = np.asarray(downbeats, dtype=float)
    if len(downbeats):
        downbeat_indices = np.round((downbeats - offset) / period).astype(int)
        phase = int(np.bincount(downbeat_indices % beats_per_bar, minlength=beats_per_bar).argmax())
    else:
        phase = int(indices[keep][0]) % beats_per_bar

    # first grid downbeat at or after the start, tolerating a few milliseconds before it
    first = math.ceil((-0.005 - offset) / period)
    first += (phase - first) % beats_per_bar
    return Grid(
        bpm=60 / period,
        beats_per_bar=beats_per_bar,
        first_downbeat=float(offset + first * period),
        median_error_ms=float(np.median(errors_ms)),
        max_error_ms=float(np.max(errors_ms)),
        beats_used=int(keep.sum()),
    )


def onset_strength(audio, samplerate, hop_ms=1.0):
    """Rise of loudness per millisecond: peaks where a hit starts."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    hop = max(1, round(samplerate * hop_ms / 1000))
    frames = len(mono) // hop
    energy = np.sqrt(np.mean(mono[: frames * hop].reshape(frames, hop) ** 2, axis=1))
    loudness = np.log1p(1000 * energy)
    return np.maximum(0, np.diff(loudness, prepend=loudness[0])), hop / samplerate


def align_to_transients(grid: Grid, audio, samplerate, search_ms=50.0) -> Grid:
    """Shifts the grid so its beats land on the hits in the audio (the drums stem works best).

    Beat trackers report beats on a coarse frame grid and slightly late. Slices have to start
    right at the transient, or the attack of the kick is cut off and layering flams.
    """
    strength, step = onset_strength(audio, samplerate)
    beats = grid.first_downbeat + np.arange(int(len(strength) * step / grid.beat_period)) * grid.beat_period

    def score(shift):
        frames = np.round((beats + shift) / step).astype(int)
        return strength[frames[(frames >= 0) & (frames < len(strength))]].sum()

    shifts = np.arange(-search_ms, search_ms + 0.25, 0.25) / 1000
    best = float(max(shifts, key=score))
    first = grid.first_downbeat + best
    # moving earlier can uncover a whole bar before the old first downbeat
    first -= math.floor(first / grid.bar_length) * grid.bar_length
    return replace(grid, first_downbeat=first, shift_ms=best * 1000)
