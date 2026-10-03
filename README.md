# Phase — Autonomous DJ MVP

Phase is a runnable first-sprint implementation of the attached autonomous AI DJ specification. It deliberately separates creative decisions from deterministic audio execution: the operator console proposes direction, while a reproducible DSP pipeline analyzes PCM WAV files and renders phrase-length transitions.

## What works now

- Browser-based venue/operator console with energy direction, skip, emergency fade, requests, intelligent queue, set trajectory, and engine telemetry.
- Dependency-light WAV ingestion, mono conversion, and resampling to 44.1 kHz.
- BPM and beat timestamp estimation from an onset envelope.
- Manual transition points expressed through a typed `TransitionPlan`.
- Tempo normalization, downbeat-aligned duration, equal-power crossfade, low-frequency handoff, peak management, and WAV export.
- Three deterministic techniques: bass swap, EQ blend, and quick cut.
- Synthetic, rights-safe demo tracks and an end-to-end render button.
- Automated checks for output duration, clipping, sample rate, channel count, and BPM sanity.

The included implementation is a production-shaped research harness, not a claim of venue-ready source separation or pitch-preserving time stretch. Its extension points are intentionally clear: replace interpolation with Rubber Band, add Essentia/librosa analysis, run Demucs offline, and place learned rankers above the same transition contract.

## Run it

Python 3.10+ and NumPy are the only runtime requirements.

```bash
cd /Users/keshavvarma/Downloads/autonomous-dj
PYTHONPATH=src python3 -m autonomous_dj serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), then select a transition technique and click **Render transition**. The app generates two demo tracks, renders a real WAV, and starts playback in the browser.

To use a virtual environment instead:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
autonomous-dj serve
```

## Analyze and render your own WAV files

The built-in decoder accepts uncompressed PCM WAV (8/16/24/32-bit). Convert MP3/AAC/FLAC to WAV before ingestion in this dependency-light MVP.

```bash
PYTHONPATH=src python3 -m autonomous_dj analyze path/to/track.wav

PYTHONPATH=src python3 -m autonomous_dj render \
  path/to/source.wav path/to/destination.wav data/renders/my-transition.wav \
  --source-start 96 --destination-start 0 \
  --source-bpm 122 --destination-bpm 124 \
  --target-bpm 123 --bars 16 --technique bass_swap
```

## Test it

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Architecture

```text
Operator console → decision/control API → TransitionPlan
                                           ↓
PCM WAV → analysis → tempo normalization → bass/EQ envelopes → headroom → WAV
                              ↓
                        render metrics + telemetry
```

Key locations:

- `apps/web/` — zero-build responsive operator console
- `src/autonomous_dj/audio.py` — ingestion, analysis, time scaling, EQ split, rendering
- `src/autonomous_dj/models.py` — track, transition, metrics, and operator contracts
- `src/autonomous_dj/server.py` — API, static app, demo catalog, telemetry
- `tests/` — measurable first-sprint exit criteria
- `data/catalog/` and `data/renders/` — generated demo inputs and outputs

## Next build phases

1. Replace the lightweight BPM estimator with validated beat/downbeat, key, phrase, section, vocal, and loudness analysis.
2. Add a transition-candidate labeling workflow and train an explainable XGBoost pairwise ranker.
3. Add offline Demucs stems and higher-quality Rubber Band tempo adjustment.
4. Persist tracks, set runs, requests, transitions, and feedback in PostgreSQL/pgvector.
5. Promote the heuristic queue to a contextual set planner with explicit event constraints and a target energy curve.

