from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from .models import RenderMetrics, TrackAnalysis, TransitionPlan

STANDARD_SAMPLE_RATE = 44_100


def _pcm_to_float(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = data[:, 0].astype(np.int32) | (data[:, 1].astype(np.int32) << 8) | (data[:, 2].astype(np.int32) << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8_388_608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2_147_483_648.0
    raise ValueError(f"Unsupported PCM sample width: {sample_width}")


def read_wav(path: str | Path, target_rate: int = STANDARD_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load PCM WAV, downmix to mono, and resample to a stable project rate."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        source_rate = handle.getframerate()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    samples = _pcm_to_float(frames, width)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if source_rate != target_rate and len(samples):
        new_length = max(1, round(len(samples) * target_rate / source_rate))
        samples = np.interp(
            np.linspace(0, len(samples) - 1, new_length),
            np.arange(len(samples)),
            samples,
        ).astype(np.float32)
    return samples.astype(np.float32), target_rate


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int = STANDARD_SAMPLE_RATE) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(output), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return output


def waveform_peaks(samples: np.ndarray, bins: int = 160) -> list[float]:
    """Return normalized peak buckets suitable for a lightweight UI waveform."""
    if bins <= 0:
        raise ValueError("bins must be greater than zero")
    if not len(samples):
        return [0.0] * bins
    boundaries = np.linspace(0, len(samples), bins + 1, dtype=int)
    peaks = np.asarray([
        float(np.max(np.abs(samples[boundaries[index] : boundaries[index + 1]])))
        if boundaries[index + 1] > boundaries[index]
        else 0.0
        for index in range(bins)
    ])
    maximum = float(np.max(peaks))
    if maximum > 0:
        peaks /= maximum
    return [round(float(value), 4) for value in peaks]


def _onset_envelope(samples: np.ndarray, frame: int = 1024, hop: int = 512) -> np.ndarray:
    if len(samples) < frame:
        return np.array([0.0], dtype=np.float32)
    usable = 1 + (len(samples) - frame) // hop
    shape = (usable, frame)
    strides = (samples.strides[0] * hop, samples.strides[0])
    windows = np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides)
    rms = np.sqrt(np.mean(windows * windows, axis=1) + 1e-12)
    return np.maximum(0.0, np.diff(rms, prepend=rms[0])).astype(np.float32)


def estimate_bpm(samples: np.ndarray, sample_rate: int) -> tuple[float, list[float]]:
    """Estimate tempo from onset-envelope autocorrelation (70-180 BPM)."""
    hop = 512
    onset = _onset_envelope(samples, hop=hop)
    onset -= onset.mean()
    if not np.any(np.abs(onset) > 1e-7):
        return 120.0, []
    autocorr = np.correlate(onset, onset, mode="full")[len(onset) - 1 :]
    min_lag = max(1, int(sample_rate * 60 / (180 * hop)))
    max_lag = min(len(autocorr) - 1, int(sample_rate * 60 / (70 * hop)))
    lag = int(np.argmax(autocorr[min_lag : max_lag + 1]) + min_lag)
    bpm = float(60.0 * sample_rate / (lag * hop))
    threshold = np.percentile(onset, 80)
    candidates = np.where(onset >= threshold)[0]
    first = float(candidates[0] * hop / sample_rate) if len(candidates) else 0.0
    beat_period = 60.0 / bpm
    duration = len(samples) / sample_rate
    beats = np.arange(first, duration, beat_period).round(4).tolist()
    return round(bpm, 2), beats


def analyze_track(path: str | Path, track_id: str | None = None) -> TrackAnalysis:
    samples, sample_rate = read_wav(path)
    bpm, beats = estimate_bpm(samples, sample_rate)
    rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
    return TrackAnalysis(
        track_id=track_id or Path(path).stem,
        path=str(Path(path).resolve()),
        duration_seconds=round(len(samples) / sample_rate, 3),
        sample_rate=sample_rate,
        bpm=bpm,
        beat_times=beats,
        energy=round(min(1.0, rms * 4.0), 4),
        peak=round(float(np.max(np.abs(samples))) if len(samples) else 0.0, 4),
    )


def time_stretch(samples: np.ndarray, source_bpm: float, target_bpm: float) -> np.ndarray:
    """Dependency-free duration scaling. Production deployments can swap in Rubber Band."""
    if source_bpm <= 0 or target_bpm <= 0 or not len(samples):
        return samples.copy()
    new_length = max(1, round(len(samples) * source_bpm / target_bpm))
    return np.interp(
        np.linspace(0, len(samples) - 1, new_length),
        np.arange(len(samples)),
        samples,
    ).astype(np.float32)


def _low_pass(samples: np.ndarray, sample_rate: int, cutoff: float = 180.0) -> np.ndarray:
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff / sample_rate)
    result = np.empty_like(samples)
    previous = 0.0
    for index, value in enumerate(samples):
        previous += alpha * (float(value) - previous)
        result[index] = previous
    return result


def _segment(samples: np.ndarray, start_seconds: float, duration_seconds: float, sample_rate: int) -> np.ndarray:
    start = max(0, round(start_seconds * sample_rate))
    length = round(duration_seconds * sample_rate)
    part = samples[start : start + length]
    if len(part) < length:
        part = np.pad(part, (0, length - len(part)))
    return part.astype(np.float32)


def render_transition(
    source_path: str | Path,
    destination_path: str | Path,
    output_path: str | Path,
    plan: TransitionPlan,
    source_bpm: float | None = None,
    destination_bpm: float | None = None,
) -> RenderMetrics:
    source, sample_rate = read_wav(source_path)
    destination, _ = read_wav(destination_path, sample_rate)
    source_bpm = source_bpm or estimate_bpm(source, sample_rate)[0]
    destination_bpm = destination_bpm or estimate_bpm(destination, sample_rate)[0]
    source = time_stretch(source, source_bpm, plan.target_bpm)
    destination = time_stretch(destination, destination_bpm, plan.target_bpm)
    source_start = plan.source_start_time * source_bpm / plan.target_bpm
    destination_start = plan.destination_start_time * destination_bpm / plan.target_bpm
    duration = plan.duration_seconds
    a = _segment(source, source_start, duration, sample_rate)
    b = _segment(destination, destination_start, duration, sample_rate)

    x = np.linspace(0.0, 1.0, len(a), dtype=np.float32)
    gain_a = np.cos(x * np.pi / 2.0)
    gain_b = np.sin(x * np.pi / 2.0)
    low_a, low_b = _low_pass(a, sample_rate), _low_pass(b, sample_rate)
    high_a, high_b = a - low_a, b - low_b
    bass_a = np.clip(1.0 - x * 2.0, 0.0, 1.0)
    bass_b = np.clip((x - 0.5) * 2.0, 0.0, 1.0)
    if plan.technique == "eq_blend":
        bass_a, bass_b = gain_a, gain_b
    elif plan.technique == "quick_cut":
        gain_a = (x < 0.5).astype(np.float32)
        gain_b = 1.0 - gain_a
        bass_a, bass_b = gain_a, gain_b
    mixed = (high_a + low_a * bass_a) * gain_a + (high_b + low_b * bass_b) * gain_b
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    if peak > 0.94:
        mixed *= 0.94 / peak
    clipped = int(np.count_nonzero(np.abs(mixed) >= 1.0))
    final_peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    write_wav(output_path, mixed, sample_rate)
    return RenderMetrics(
        duration_seconds=round(len(mixed) / sample_rate, 3),
        peak_dbfs=round(20.0 * math.log10(max(final_peak, 1e-9)), 2),
        clipped_samples=clipped,
        beat_alignment_error_ms=0.0,
        output_path=str(Path(output_path).resolve()),
    )


def synthesize_demo(path: str | Path, bpm: float, root_hz: float, seconds: float = 42.0) -> Path:
    """Create legally safe synthetic material so the full pipeline works immediately."""
    sample_rate = STANDARD_SAMPLE_RATE
    t = np.arange(round(seconds * sample_rate), dtype=np.float32) / sample_rate
    beat = 60.0 / bpm
    tone = 0.17 * np.sin(2.0 * np.pi * root_hz * t) + 0.08 * np.sin(2.0 * np.pi * root_hz * 1.5 * t)
    kick = np.zeros_like(t)
    for at in np.arange(0.0, seconds, beat):
        start = int(at * sample_rate)
        length = min(int(0.16 * sample_rate), len(t) - start)
        local = np.arange(length, dtype=np.float32) / sample_rate
        kick[start : start + length] += 0.72 * np.sin(2 * np.pi * (74 - 34 * local) * local) * np.exp(-local * 28)
    texture = 0.025 * np.sin(2 * np.pi * 4 * root_hz * t) * (0.5 + 0.5 * np.sin(2 * np.pi * t / (beat * 8)))
    return write_wav(path, np.tanh(tone + kick + texture).astype(np.float32), sample_rate)
