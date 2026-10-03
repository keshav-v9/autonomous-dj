from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class TrackAnalysis:
    track_id: str
    path: str
    duration_seconds: float
    sample_rate: int
    bpm: float
    beat_times: list[float]
    energy: float
    peak: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class TransitionPlan:
    source_track_id: str
    destination_track_id: str
    source_start_time: float
    destination_start_time: float
    transition_bars: int = 16
    target_bpm: float = 124.0
    technique: str = "bass_swap"
    beats_per_bar: int = 4

    @property
    def duration_seconds(self) -> float:
        return self.transition_bars * self.beats_per_bar * 60.0 / self.target_bpm

    def to_dict(self) -> dict:
        result = asdict(self)
        result["duration_seconds"] = self.duration_seconds
        return result


@dataclass(slots=True)
class RenderMetrics:
    duration_seconds: float
    peak_dbfs: float
    clipped_samples: int
    beat_alignment_error_ms: float
    output_path: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class OperatorState:
    energy_target: float = 7.2
    mode: str = "maintain"
    current_track: str = "Midnight Circuit"
    next_track: str = "Afterimage"
    requests: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    telemetry: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

