# Phase — Autonomous DJ

## What the project does

Phase is an autonomous DJ system that analyzes songs and creates smooth, beat-aligned transitions between them. It estimates each track's BPM, beat timestamps, energy, and peak level, then uses that information to normalize tempo and render phrase-length mixes.

The project supports bass swaps, EQ blends, and quick cuts. It also includes a browser-based operator console for controlling set energy, managing requests, viewing the upcoming queue, skipping tracks, triggering an emergency fade, and playing rendered transitions.

The current version uses a heuristic decision model and deterministic digital signal processing. It provides the foundation for training a machine-learning model to rank tracks and select transitions based on musical compatibility, energy, and operator intent.

## Tech stack

| Technology | Use |
| --- | --- |
| Python 3.10+ | Audio analysis, transition planning, API, and application logic |
| NumPy | Waveform processing, BPM estimation, tempo normalization, and audio mixing |
| Python `wave` | Reading and writing 44.1 kHz PCM WAV files |
| Python `http.server` | Local API and web application server |
| Python dataclasses | Typed track, transition, metrics, and operator-state models |
| HTML, CSS, and JavaScript | Browser-based DJ operator console and audio playback |
| CSV and CC BY audio data | Licensed training data and metadata for future ML development |
| Python `unittest` | Automated audio-engine and transition validation |
