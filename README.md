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
