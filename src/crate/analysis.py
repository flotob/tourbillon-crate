"""What plays where: the loudness of each stem per bar, and the bar ranges in which each stem plays."""

from itertools import pairwise

import numpy as np

SILENCE_DB = -60.0


def bar_levels(audio: np.ndarray, starts: list[int]) -> list[float]:
    """Loudness of each bar in dBFS (RMS), between consecutive sample positions in starts."""
    levels = []
    for start, end in pairwise(starts):
        rms = np.sqrt(np.mean(np.square(audio[start:end], dtype=np.float64))) if end > start else 0.0
        levels.append(round(max(SILENCE_DB, 20 * np.log10(rms + 1e-12)), 1))
    return levels


def active_bars(levels, below_peak_db: float = 18.0, floor_db: float = -45.0) -> np.ndarray:
    """Bars in which a stem plays: close to its own loud bars. Separation leaves quiet bleed in every stem."""
    levels = np.asarray(levels, dtype=float)
    active = levels > max(floor_db, np.percentile(levels, 95) - below_peak_db)
    # majority of three: a single-bar blip or dropout doesn't count
    padded = np.pad(active, 1, mode="edge").astype(int)
    return (padded[:-2] + padded[1:-1] + padded[2:]) >= 2


def playing_ranges(levels, max_gap: int = 2, min_bars: int = 2) -> list[list[int]]:
    """Bar ranges [start, end) in which a stem plays, joining ranges split by short pauses (e.g. between
    vocal phrases)."""
    ranges = []
    for bar, active in enumerate(active_bars(levels)):
        if not active:
            continue
        if ranges and bar - ranges[-1][1] <= max_gap:
            ranges[-1][1] = bar + 1
        else:
            ranges.append([bar, bar + 1])
    return [r for r in ranges if r[1] - r[0] >= min_bars]
