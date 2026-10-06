# Phase — Autonomous DJ

## What the project does

Phase is an ML-powered autonomous DJ system that analyzes songs, selects the best track to play next, and creates smooth, beat-aligned transitions. It extracts musical features—including BPM, beat timestamps, musical key, phrase structure, energy, loudness, and vocal activity—and uses them to understand how tracks fit together.

Its trained ranking model scores possible tracks and transition points using musical compatibility, energy progression, crowd feedback, and operator intent. The highest-ranked decision is converted into a transition plan and executed by a deterministic audio engine using bass swaps, EQ blends, or quick cuts.

Phase continuously improves from accepted and rejected transitions, pairwise DJ preferences, and performance feedback. A browser-based operator console keeps a human in control of set energy, requests, the upcoming queue, skips, emergency fades, and rendered-transition playback.

## Tech stack

| Technology | Use |
| --- | --- |
| Python 3.10+ | Audio analysis, transition planning, API, and application logic |
| NumPy | Waveform processing, feature extraction, tempo normalization, and audio mixing |
| XGBoost | Trained pairwise ranking model for track and transition selection |
| librosa | BPM, beat, key, energy, loudness, and musical-structure analysis |
| scikit-learn | Feature preprocessing, dataset splitting, training, and model evaluation |
| Python `wave` | Reading and writing 44.1 kHz PCM WAV files |
| Python `http.server` | Local API and web application server |
| Python dataclasses | Typed track, transition, metrics, and operator-state models |
| HTML, CSS, and JavaScript | Browser-based DJ operator console and audio playback |
| CSV and CC BY audio data | Licensed training audio, metadata, transition candidates, and labels |
| Python `unittest` | Automated audio-engine and transition validation |

## Run the app

The deterministic demo only needs NumPy:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
autonomous-dj serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Choose any two demo songs,
assign them to Deck A and Deck B, select a transition style and length, and click
**Generate my mix**. The studio renders a real WAV and displays its waveform,
playhead, crossfader, deck gains, tempo, duration, and peak level during playback.
The recommendation engine uses a documented musical-compatibility baseline until
a trained model is present, then loads the learned ranker automatically.

## ML phase

The ML pipeline is implemented as reproducible CLI stages rather than a
notebook. It includes:

- librosa extraction for tempo, key/Camelot code, RMS energy, loudness,
  vocal-band activity, spectral centroid, and phrase confidence;
- transition features for tempo/key compatibility, energy intent, loudness,
  vocal overlap, timbral distance, and phrase quality;
- a CSV labeling workflow supporting rejected (`0`), acceptable (`1`), and
  preferred (`2`) candidates;
- an XGBoost `rank:pairwise` model with source-context group splitting, so the
  same decision context cannot leak into train and test sets;
- NDCG@5 and pairwise-accuracy evaluation, persisted metadata, feature
  importance, and per-decision feature contributions;
- live accepted/rejected and winner/loser feedback at `POST /api/feedback`;
- automatic model serving in the operator console, with a deterministic
  heuristic fallback when an artifact has not been trained yet.

Install the ML dependencies:

```bash
pip install -e '.[ml]'
```

On macOS, XGBoost also needs the OpenMP runtime: `brew install libomp`.

Extract features from the included CC BY house dataset:

```bash
autonomous-dj ml-extract \
  data/training/cc_by_house/manifest.csv \
  data/training/track_features.csv
```

Generate the candidate sheet:

```bash
autonomous-dj ml-candidates \
  data/training/track_features.csv \
  data/training/transition_labels.csv \
  --target-energy 7 --direction maintain
```

Fill the `relevance` column with `0`, `1`, or `2`. Label at least three source
contexts and use multiple relevance levels in each; more independent contexts
produce a more meaningful holdout score. Then train:

```bash
autonomous-dj ml-train \
  data/training/transition_labels.csv \
  data/models/transition_ranker
```

Inspect ranking output and its top feature contributions:

```bash
autonomous-dj ml-rank \
  data/training/track_features.csv 01-in-search-of-sunrise \
  --model-dir data/models/transition_ranker \
  --target-energy 7 --direction energy_up
```

Restart `autonomous-dj serve` after training. `/api/status` reports the loaded
model name and returns the model-ranked queue.

## Train on 1,000 songs in GitHub Actions

The repository includes a manual cloud workflow at
`.github/workflows/train-fma-1000.yml`. It downloads the official Free Music
Archive metadata and precomputed features inside a temporary GitHub runner,
selects exactly 1,000 Electronic tracks, builds 24 candidate transitions per
source track, and trains the pairwise ranker. It does **not** download, commit,
or redistribute the audio files.

After pushing this repository to GitHub:

1. Open the repository's **Actions** tab.
2. Select **Train ranker on 1,000 FMA tracks**.
3. Choose **Run workflow** and keep the default values.
4. When it finishes, download the `phase-fma-1000-*` artifact.

The artifact contains:

- `fma_catalog.csv` — 1,000 track records and Phase-compatible features;
- `weak_transition_labels.csv` — 24,000 reproducible transition examples;
- `model/ranker.json` — the trained XGBoost model;
- `model/metadata.json` — evaluation metrics and training provenance;
- `dataset_report.json` — catalog size, genre, seed, and label source.

To use the downloaded model locally, copy its two model files into
`data/models/transition_ranker/` and restart the server. The initial labels are
explicitly marked as heuristic weak supervision. Live DJ feedback collected by
`POST /api/feedback` should be used for later human-supervised retraining.

The workflow uses the research-oriented, Creative Commons-licensed
[Free Music Archive dataset](https://github.com/mdeff/fma). Its metadata is CC
BY 4.0; individual audio licenses remain those selected by each artist.

## Feedback contract

Record an accepted or rejected candidate:

```json
{"source_id":"midnight-circuit","candidate_id":"afterimage","accepted":true}
```

Or record a stronger pairwise preference:

```json
{"source_id":"midnight-circuit","winner_id":"afterimage","loser_id":"soft-focus"}
```

Send either payload to `POST /api/feedback`. Rows are appended to
`data/training/feedback.csv` using the exact schema consumed by `ml-train`.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
