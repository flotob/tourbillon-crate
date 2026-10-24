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

`crate serve` serves the crate to Strudel. Bar `n` of a stem is sample `n` of the sound `<record>_<stem>`:

```js
samples('http://localhost:5455/strudel.json')
setcpm(128/4) // one cycle is one bar at the record's tempo
$: s("acid_drums").n("<32 33 34 35>")
$: s("acid_bass").n("<32 33 34 35>")
```

Status: first test. Tourbillon integration (crate tab, agent tools), key detection, sections, tempo-matched
renders and transcription to patterns come next.

## Usage

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
uv run crate add ~/Music/track.mp3 --name acid   # --bpm 128 forces the tempo if half or double was detected
uv run crate list
uv run crate serve
```

Records go to `~/Library/Application Support/Tourbillon/crate` (or `--crate` / `CRATE_DIR`). Audio is only
processed locally.

## Development

```bash
uv run pytest
uv run ruff check && uv run ruff format
```
