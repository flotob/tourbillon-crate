# crate

Turns tracks into bar-sliced stems for [Tourbillon](https://github.com/flotob/tourbillon), so parts of other
producers' records can be mixed into a live-coded set.

For each audio file, `crate add`:

1. separates stems with [Demucs](https://github.com/adefossez/demucs) (`htdemucs`: drums, bass, vocals, other),
   on the Apple Silicon GPU when available
2. detects beats and downbeats with [Beat This!](https://github.com/CPJKU/beat_this)
3. fits a constant-tempo bar grid (`src/crate/grid.py`) and aligns it to the hits in the drums stem, so each
   slice starts right at the transient
4. writes one file per bar and stem, plus the original mix sliced the same way
5. measures the loudness of each stem per bar and finds the bar ranges where each stem plays
   (`src/crate/analysis.py`)
6. finds the key with [Essentia](https://essentia.upf.edu)'s profile for electronic dance music, listening to
   everything but the drums (`src/crate/key.py`)

`crate tempo` makes a version of a record at another tempo without changing its pitch (`src/crate/tempo.py`):
[Rubber Band](https://breakfastquay.com/rubberband/) stretches each stem, with every beat pinned to its new time,
and the result is cut into the same bars. Its faster R2 engine is used on purpose: it keeps kicks sharp, where the
finer R3 engine softens them and lets them start early. An 8½ minute record takes about 25 seconds.

Bar `n` of a stem is sample `n` of the sound `<record>_<stem>`:

```js
setcpm(118/4) // one cycle is one bar at the record's tempo
$: s("nataalma_drums").n("<64 65 66 67>")
$: s("nataalma_vocals").n("<64 65 66 67>")
```

## In Tourbillon

Tourbillon runs crate itself: drop an audio file on the app (or use the crate tab) to import it. Its sounds are
loaded automatically, and the agent can look up records, their keys and where their parts play. When a record has a
version at the tempo a track plays at, Tourbillon plays that version. Tourbillon looks for this
repository next to its own (`../crate` or `../tourbillon-crate`), or at `TOURBILLON_CRATE_PROJECT`.

## On the command line

Requires [uv](https://docs.astral.sh/uv/), ffmpeg and rubberband (`brew install ffmpeg rubberband`; Homebrew's
ffmpeg brings rubberband along).

```bash
uv run crate add ~/Music/track.mp3 --name acid   # --bpm 128 forces the tempo if half or double was detected
uv run crate tempo acid 124                      # the record at 124 bpm, same pitch
uv run crate list
uv run crate analyze                            # recomputes key and what plays where for existing records
uv run crate serve                              # for Strudel without Tourbillon
```

With `crate serve` running, load the crate in Strudel with `samples('http://localhost:5455/strudel.json')`.

Records go to `~/Library/Application Support/Tourbillon/crate` (or `--crate` / `CRATE_DIR`). Audio is only
processed locally and never leaves your machine.

## A record

```
<name>/record.json                 tempo, key, grid, stems, bars, levels and playing ranges per stem, beats
<name>/stems/<stem>.flac           whole stems
<name>/bars/<stem>/NNN.flac        bar NNN of a stem, starting at the first downbeat
<name>/tempo/<bpm>/<stem>/NNN.flac the same bar at another tempo
<name>/tempo/<bpm>/version.json    how that version was made
```

`key` is like `A minor`, `camelot` its place on the Camelot wheel (`8A`), and `key_strength` (0 to 1) how clearly
the music follows that key.

In `record.json`, `levels` holds each stem's loudness per bar in dBFS and `playing` the bar ranges
`[start, end)` in which each stem plays.

## Development

```bash
uv run pytest
uv run ruff check && uv run ruff format
```

## License

[AGPL-3.0-or-later](LICENSE), like Tourbillon. Demucs and Beat This! (code and model weights) are MIT licensed;
the models are downloaded on first use. Essentia is AGPL-3.0 licensed. Rubber Band (GPL-2.0-or-later) runs as a
separate program. Only import music you have the rights to use: the separated stems are
derived from the original recording.
