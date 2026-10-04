# Phase — Autonomous DJ

Phase is an ML-ready autonomous DJ system designed to learn how to choose and execute musically coherent transitions. It is built around a simple idea: a learned model makes the creative decisions—what should play next, where the transition should happen, and which technique should be used—while a deterministic DSP engine performs the mix safely and reproducibly.

The current MVP provides the complete audio-analysis, transition-planning, rendering, and operator-control foundation. Its decision engine is presently a heuristic baseline; the included licensed house-music dataset and stable `TransitionPlan` contract are the foundation for training and evaluating the learned ranking model described below.

## What it does

Phase takes tracks from analysis to a playable DJ transition:

1. **Analyzes audio** to estimate BPM, beat timestamps, energy, duration, and peak level.
2. **Represents a DJ decision** as a typed transition plan containing the source and destination tracks, cue points, target BPM, phrase length, and mixing technique.
3. **Normalizes tempo and aligns phrases** so both tracks meet on a shared beat grid.
4. **Renders the transition** using equal-power crossfades, low-frequency handoffs, EQ blending, or quick cuts.
5. **Measures the result** for duration, peak level, clipping, and beat-alignment error.
6. **Keeps a human in control** through a browser console with energy direction, requests, queue visibility, skip, and emergency fade controls.

```text
Tracks → audio features → transition candidates → decision model → TransitionPlan
                                                                      ↓
Operator intent ───────────────────────────────────────────────→ DSP renderer
                                                                      ↓
                                                        WAV + quality metrics
```

## Machine learning focus

The ML problem is framed as **learning to rank transition candidates**, not generating raw audio. For every possible next track and transition point, the model can score musical compatibility and choose the candidate that best follows the current track, the desired energy trajectory, and operator constraints.

### Current baseline

- Onset-envelope autocorrelation estimates tempo in the 70–180 BPM range.
- RMS-derived energy and peak measurements describe track intensity and headroom.
- A heuristic planner represents the initial decision baseline (`heuristic-v0.1`).
- Every decision is passed to the renderer through the same typed `TransitionPlan` interface that a trained model will use.

### Training path

- **Training data:** `data/training/cc_by_house/` contains a manifest and CC BY 4.0 house tracks that may be used for waveform analysis and model development with attribution.
- **Candidate features:** tempo distance, beat/downbeat alignment, energy change, key compatibility, phrase/section position, vocal overlap, loudness, recency, and operator intent.
- **Labels:** accepted/rejected transition pairs, pairwise preferences, render-quality checks, and eventual operator or listener feedback.
- **Model:** an explainable pairwise ranker (planned: XGBoost) scores candidate transitions and exposes the factors behind each choice.
- **Evaluation:** ranking quality is combined with deterministic audio checks such as clipping, transition duration, and beat alignment.

This separation makes experimentation safer: the learned ranker can be retrained or replaced without changing the audio engine, and model output must satisfy a validated transition contract before any audio is rendered.

> **Model status:** this repository does not yet contain trained model weights or a training command. The working MVP uses signal processing and a heuristic decision baseline; supervised ranker training is the next ML phase.

## What works now

- Browser-based venue/operator console with energy direction, skip, emergency fade, requests, intelligent queue, set trajectory, and engine telemetry.
- Dependency-light WAV ingestion, mono conversion, and resampling to 44.1 kHz.
- BPM and beat timestamp estimation from an onset envelope.
- Manual transition points expressed through a typed `TransitionPlan`.
- Tempo normalization, downbeat-aligned duration, equal-power crossfade, low-frequency handoff, peak management, and WAV export.
- Three deterministic techniques: bass swap, EQ blend, and quick cut.
- Synthetic, rights-safe demo tracks and an end-to-end render button.
- Automated checks for output duration, clipping, sample rate, channel count, and BPM sanity.

The included implementation is a production-shaped research harness, not a claim of venue-ready source separation or pitch-preserving time stretch. Its extension points are intentionally clear: replace interpolation with Rubber Band, add Essentia/librosa analysis, run Demucs offline, and place the learned ranker above the existing transition contract.

## Tech stack

| Layer | Technology | Role |
| --- | --- | --- |
| Audio + ML foundation | Python 3.10+, NumPy | Waveform processing, feature extraction, tempo analysis, mixing, and model-ready feature pipelines |
| Audio I/O | Python `wave`, PCM WAV at 44.1 kHz | Dependency-light decoding and reproducible output |
| Decision contracts | Python dataclasses | Typed track analysis, transition plans, render metrics, and operator state |
| API + runtime | Python `http.server` | Local control API, telemetry, static app hosting, and render endpoints |
| Operator UI | HTML, CSS, vanilla JavaScript | Real-time queue, energy controls, transition selection, and playback |
| Training data | CC BY audio + CSV manifests | Licensed waveform corpus and traceable metadata for ML experiments |
| Testing | Python `unittest` | BPM sanity, timing, format, headroom, and clipping checks |
| Planned ML/audio upgrades | XGBoost, Essentia/librosa, Demucs, Rubber Band, PostgreSQL/pgvector | Learned ranking, richer features, source separation, high-quality time stretch, and feedback storage |

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
Licensed audio → feature extraction → candidate/label dataset → ranker training
                                                               ↓
Operator console → decision/control API → heuristic or trained ranker → TransitionPlan
                                                                        ↓
PCM WAV → analysis → tempo normalization → bass/EQ envelopes → headroom → WAV
                              ↓                                         ↓
                    model features                         metrics + telemetry
```

Key locations:

- `apps/web/` — zero-build responsive operator console
- `src/autonomous_dj/audio.py` — ingestion, analysis, time scaling, EQ split, rendering
- `src/autonomous_dj/models.py` — track, transition, metrics, and operator contracts
- `src/autonomous_dj/server.py` — API, static app, demo catalog, telemetry
- `scripts/build_cc_house_dataset.py` — licensed training-corpus builder and manifest generator
- `data/training/cc_by_house/` — attributed CC BY house-audio training material
- `data/playlists/` — evaluation metadata kept separate from licensed waveform data
- `tests/` — measurable first-sprint exit criteria
- `data/catalog/` and `data/renders/` — generated demo inputs and outputs

## Next build phases

1. Replace the lightweight BPM estimator with validated beat/downbeat, key, phrase, section, vocal, and loudness analysis.
2. Add a transition-candidate labeling workflow and train an explainable XGBoost pairwise ranker.
3. Add offline Demucs stems and higher-quality Rubber Band tempo adjustment.
4. Persist tracks, set runs, requests, transitions, and feedback in PostgreSQL/pgvector.
5. Promote the heuristic queue to a contextual set planner with explicit event constraints and a target energy curve.
