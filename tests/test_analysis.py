import numpy as np

from crate.analysis import SILENCE_DB, active_bars, bar_levels, outline, playing_ranges


def test_bar_levels():
    samplerate = 1000
    loud = np.full((samplerate, 2), 0.5)
    audio = np.concatenate([loud, np.zeros((samplerate, 2)), loud * 0.1])
    levels = bar_levels(audio, [0, 1000, 2000, 3000])
    assert levels[0] == round(20 * np.log10(0.5), 1)
    assert levels[1] == SILENCE_DB
    assert levels[2] == round(20 * np.log10(0.05), 1)


def test_outline():
    # two bars of 1000 and 1003 samples; the first loud in its first quarter only, the second quiet throughout
    audio = np.concatenate([np.full((250, 2), 0.5), np.zeros((750, 2)), np.full((1003, 2), 0.05)])
    levels = outline(audio, [0, 1000, 2003], 4)
    assert levels == [-6, -60, -60, -60, -26, -26, -26, -26]
    # as loud overall as the bar
    power = np.mean([10 ** (db / 10) for db in levels[4:]])
    assert round(10 * np.log10(power)) == round(bar_levels(audio, [1000, 2003])[0])


def test_bleed_is_not_playing():
    # a vocal stem: sung bars around -15 dB, separation bleed around -40 dB
    levels = [-40.0] * 8 + [-15.0] * 8 + [-40.0] * 8
    assert list(np.flatnonzero(active_bars(levels))) == list(range(8, 16))


def test_single_bar_blips_and_dropouts():
    levels = [-50.0] * 4 + [-12.0] + [-50.0] * 4 + [-12.0] * 3 + [-50.0] + [-12.0] * 3
    active = active_bars(levels)
    assert not active[4]
    assert active[9:16].all()


def test_ranges_join_short_pauses():
    # two vocal phrases with a two-bar pause, then a long pause
    levels = [-50.0] * 4 + [-15.0] * 6 + [-50.0] * 2 + [-15.0] * 6 + [-50.0] * 8 + [-15.0] * 4
    assert playing_ranges(levels) == [[4, 18], [26, 30]]


def test_silent_stem_plays_nowhere():
    assert playing_ranges([SILENCE_DB] * 16) == []
