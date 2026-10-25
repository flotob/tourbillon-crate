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

Bar `n` of a stem is sample `n` of the sound `<record>_<stem>`:

```js
setcpm(118/4) // one cycle is one bar at the record's tempo
$: s("nataalma_drums").n("<64 65 66 67>")
$: s("nataalma_vocals").n("<64 65 66 67>")
```

## In Tourbillon

Tourbillon runs crate itself: drop an audio file on the app (or use the crate tab) to import it. Its sounds are
loaded automatically, and the agent can look up records and where their parts play. Tourbillon looks for this
repository next to its own (`../crate` or `../tourbillon-crate`), or at `TOURBILLON_CRATE_PROJECT`.

## On the command line

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
uv run crate add ~/Music/track.mp3 --name acid   # --bpm 128 forces the tempo if half or double was detected
uv run crate list
uv run crate analyze                            # recomputes what plays where for existing records
uv run crate serve                              # for Strudel without Tourbillon
```

With `crate serve` running, load the crate in Strudel with `samples('http://localhost:5455/strudel.json')`.

Records go to `~/Library/Application Support/Tourbillon/crate` (or `--crate` / `CRATE_DIR`). Audio is only
processed locally and never leaves your machine.

## A record

```
<name>/record.json           tempo, grid, stems, bars, levels and playing ranges per stem, beats
<name>/stems/<stem>.flac     whole stems
<name>/bars/<stem>/NNN.flac  bar NNN of a stem, starting at the first downbeat
```

In `record.json`, `levels` holds each stem's loudness per bar in dBFS and `playing` the bar ranges
`[start, end)` in which each stem plays.

## Development

```bash
uv run pytest
uv run ruff check && uv run ruff format
```

## License

[AGPL-3.0-or-later](LICENSE), like Tourbillon. Demucs and Beat This! (code and model weights) are MIT licensed;
the models are downloaded on first use. Only import music you have the rights to use: the separated stems are
derived from the original recording.
