"""The bass line as notes on the bar grid: which note sounds on each 16th, where notes start and how long they last.

A bass line plays one note at a time, so a pitch tracker for one voice follows it: YIN (de Cheveigné and Kawahara,
2002, in essentia). Transcribers made for chords, like Basic Pitch, need analysis windows of over a second for the
lowest octave and smear sub-bass notes by as much. Each 16th gets the median pitch of its second half, after the
pitch drop that many synth basses start their notes with; a note starts where the pitch changes, where the bass is
struck again (gets louder at once) and at every bar line.
"""

from itertools import pairwise

import numpy as np

from crate.analysis import SILENCE_DB

STEPS = 16
# YIN on the bass stem resampled to RATE, in frames of FRAME samples (93 ms, two periods of the lowest note) every
# HOP samples (11.6 ms)
RATE, FRAME, HOP = 5512, 512, 64
LOWEST_HZ, HIGHEST_HZ = 25.0, 400.0
CONFIDENCE = 0.5
# this far below the bass's loud bars counts as silence
QUIET_DB = 30.0
# louder by this much from just before a 16th to just after: the note is struck again
STRIKE_DB = 6.0
# a note a single 16th long must be this much louder than silence, else it is bleed from other instruments
BLIP_DB = 12.0
# how far a pitch drop or glide into a note may start from it, in semitones
GLIDE = 3


def track(mono: np.ndarray, samplerate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pitch in Hz (0 where there is none), YIN's confidence and frame centres in seconds, every HOP / RATE s."""
    import essentia

    essentia.log.infoActive = False
    essentia.log.warningActive = False
    import essentia.standard as es

    low = es.Resample(inputSampleRate=samplerate, outputSampleRate=RATE, quality=1)(mono.astype(np.float32))
    yin = es.PitchYin(frameSize=FRAME, sampleRate=RATE, minFrequency=LOWEST_HZ, maxFrequency=HIGHEST_HZ)
    pitches, confidences = [], []
    for frame in es.FrameGenerator(low, frameSize=FRAME, hopSize=HOP, startFromZero=True):
        pitch, confidence = yin(frame)
        pitches.append(pitch)
        confidences.append(confidence)
    times = (np.arange(len(pitches)) * HOP + FRAME / 2) / RATE
    return np.array(pitches), np.array(confidences), times


def loudness(mono: np.ndarray, samplerate: int, hop: float = 0.005, window: float = 0.04) -> np.ndarray:
    """dB every 5 ms (at 0, 5, 10 ms...), each over 40 ms around it: longer than a period of the lowest notes, whose
    loudness swings with their waveform over shorter windows."""
    energy = np.concatenate([[0.0], np.cumsum(mono.astype(np.float64) ** 2)])
    centres = np.arange(0, len(mono), round(hop * samplerate))
    half = round(window / 2 * samplerate)
    lo, hi = np.clip(centres - half, 0, len(mono)), np.clip(centres + half, 0, len(mono))
    return 10 * np.log10((energy[hi] - energy[lo]) / np.maximum(1, hi - lo) + 1e-12)


def merge_glides(notes: list[list]) -> list[list]:
    """A 16th a few semitones off that leads into a longer note without a new strike is a pitch drop or glide
    into that note. Notes are [bar, 16th, length, midi, struck, loudest dB]."""
    merged = []
    for note in notes:
        last = merged[-1] if merged else None
        if (
            last
            and last[2] == 1
            and note[4] == 0
            and note[0] == last[0]
            and note[1] == last[1] + 1
            and note[2] >= 2
            and abs(note[3] - last[3]) <= GLIDE
        ):
            last[2] += note[2]
            last[3] = note[3]
            last[5] = max(last[5], note[5])
        else:
            merged.append(list(note))
    return merged


def bass_notes(mono: np.ndarray, samplerate: int, starts: list[float], quiet_db: float) -> list[list[int]]:
    """[bar, 16th, length in 16ths, midi note] for the bass line in mono; starts are the bar start times (one more
    than there are bars) and quiet_db the loudness below which the bass is silent."""
    pitches, confidences, times = track(mono, samplerate)
    voiced = (pitches > 0) & (confidences > CONFIDENCE)
    level = loudness(mono, samplerate)
    frame_level = np.interp(times, np.arange(len(level)) * 0.005, level)
    voiced &= frame_level > quiet_db

    def between(t0, t1):
        return slice(*np.searchsorted(times, [t0, t1]))

    def level_between(t0, t1):
        part = level[max(0, round(t0 / 0.005)) : max(0, round(t1 / 0.005) + 1)]
        return part if len(part) else np.array([quiet_db])

    notes = []  # [bar, 16th, length, midi, struck, loudest dB]
    for bar, (start, end) in enumerate(pairwise(starts)):
        step = (end - start) / STEPS
        playing = None
        for s in range(STEPS):
            t = start + s * step
            frames = between(t + 0.4 * step, t + 0.9 * step)
            if voiced[frames].sum() < 0.5 * max(1, frames.stop - frames.start):
                playing = None
                continue
            midi = int(np.rint(np.median(69 + 12 * np.log2(pitches[frames][voiced[frames]] / 440))))
            struck = level_between(t + 0.02, t + 0.06).max() - level_between(t - 0.06, t).min() > STRIKE_DB
            loudest = float(level_between(t, t + step).max())
            if playing is not None and midi == playing[3] and not struck:
                playing[2] += 1
                playing[5] = max(playing[5], loudest)
            else:
                playing = [bar, s, 1, midi, int(struck), loudest]
                notes.append(playing)
    return [note[:4] for note in merge_glides(notes) if note[2] > 1 or note[5] > quiet_db + BLIP_DB]


def quiet_level(levels_db) -> float:
    """Loudness below which the bass is silent, from its loudness per bar."""
    return max(SILENCE_DB, float(np.percentile(levels_db, 95)) - QUIET_DB)
