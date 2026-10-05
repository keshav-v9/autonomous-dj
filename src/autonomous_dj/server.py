from __future__ import annotations

import json
import mimetypes
import sys
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .audio import analyze_track, render_transition, synthesize_demo
from .feedback import append_feedback
from .ml import MusicFeatures, TransitionRanker
from .models import OperatorState, TransitionPlan

ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "apps" / "web"
DATA_ROOT = ROOT / "data"
state = OperatorState()
ranker = TransitionRanker.load(DATA_ROOT / "models" / "transition_ranker")

TRACKS = [
    {"id": "midnight-circuit", "title": "Midnight Circuit", "artist": "Demo System", "bpm": 122.0, "key": "8A", "energy": 6.8, "duration": "04:18", "color": "violet"},
    {"id": "afterimage", "title": "Afterimage", "artist": "Demo System", "bpm": 124.0, "key": "8A", "energy": 7.6, "duration": "03:52", "color": "lime"},
    {"id": "glass-horizon", "title": "Glass Horizon", "artist": "Demo System", "bpm": 125.0, "key": "9A", "energy": 8.1, "duration": "04:04", "color": "cyan"},
    {"id": "soft-focus", "title": "Soft Focus", "artist": "Demo System", "bpm": 120.0, "key": "7A", "energy": 5.9, "duration": "03:46", "color": "amber"},
]


def _runtime_features(track: dict) -> MusicFeatures:
    """Adapt demo metadata to the same feature contract used by trained models."""
    return MusicFeatures(
        track_id=track["id"],
        path="",
        duration_seconds=0.0,
        bpm=float(track["bpm"]),
        key=str(track["key"]),
        camelot=str(track["key"]),
        energy=float(track["energy"]) / 10.0,
        loudness_db=-18.0 + float(track["energy"]) * 0.8,
        vocal_activity=float(track.get("vocal_activity", 0.35)),
        spectral_centroid=float(track.get("spectral_centroid", 2100.0)),
        phrase_seconds=16.0 * 60.0 / float(track["bpm"]),
        phrase_confidence=float(track.get("phrase_confidence", 0.75)),
    )


def ranked_queue() -> list[dict]:
    by_title = {track["title"]: track for track in TRACKS}
    source_track = by_title.get(state.current_track, TRACKS[0])
    source = _runtime_features(source_track)
    candidates = [track for track in TRACKS if track["id"] != source.track_id and track["title"] not in state.blocked]
    ranked = ranker.rank(source, [_runtime_features(track) for track in candidates], state.energy_target, state.mode)
    metadata = {track["id"]: track for track in candidates}
    return [{**metadata[item.track_id], **item.to_dict()} for item in ranked]


def _track_by_id(track_id: str) -> dict:
    try:
        return next(track for track in TRACKS if track["id"] == track_id)
    except StopIteration as exc:
        raise ValueError(f"Unknown track id: {track_id}") from exc


def ensure_demo_audio() -> tuple[Path, Path]:
    catalog = DATA_ROOT / "catalog"
    source = catalog / "midnight-circuit.wav"
    destination = catalog / "afterimage.wav"
    if not source.exists():
        synthesize_demo(source, 122.0, 110.0)
    if not destination.exists():
        synthesize_demo(destination, 124.0, 130.81)
    return source, destination


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stdout.write(f"[operator] {fmt % args}\n")

    def _json(self, payload: dict | list, status: int = 200) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            queue = ranked_queue()
            if queue:
                state.next_track = queue[0]["title"]
            self._json({
                "state": state.to_dict(),
                "tracks": TRACKS,
                "ranked_queue": queue,
                "engine": {"status": "ready", "model": ranker.model_name, "latency_ms": 38},
            })
            return
        if path.startswith("/renders/"):
            target = (DATA_ROOT / path.lstrip("/")).resolve()
            if DATA_ROOT.resolve() not in target.parents or not target.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        target = WEB_ROOT / ("index.html" if path == "/" else path.lstrip("/"))
        target = target.resolve()
        if WEB_ROOT.resolve() not in target.parents and target != (WEB_ROOT / "index.html").resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not target.exists() or not target.is_file():
            target = WEB_ROOT / "index.html"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "text/plain")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/control":
                action = body.get("action", "maintain")
                if action == "energy_up":
                    state.energy_target = min(10.0, state.energy_target + 0.5)
                    state.mode = action
                elif action == "energy_down":
                    state.energy_target = max(1.0, state.energy_target - 0.5)
                    state.mode = action
                elif action == "skip":
                    state.blocked.append(state.next_track)
                    state.current_track = state.next_track
                elif action == "emergency_fade":
                    state.mode = "emergency fade"
                else:
                    state.mode = action
                event = {"at": int(time.time()), "type": "operator", "action": action}
                state.telemetry.append(event)
                queue = ranked_queue()
                if queue:
                    state.next_track = queue[0]["title"]
                self._json({"ok": True, "state": state.to_dict(), "ranked_queue": queue})
                return
            if path == "/api/request":
                title = str(body.get("title", "")).strip()
                if title:
                    state.requests.append(title)
                    state.telemetry.append({"at": int(time.time()), "type": "request", "title": title})
                self._json({"ok": True, "state": state.to_dict()})
                return
            if path == "/api/feedback":
                source = _runtime_features(_track_by_id(str(body["source_id"])))
                if "winner_id" in body and "loser_id" in body:
                    choices = [
                        (_runtime_features(_track_by_id(str(body["winner_id"]))), 2),
                        (_runtime_features(_track_by_id(str(body["loser_id"]))), 0),
                    ]
                else:
                    candidate = _runtime_features(_track_by_id(str(body["candidate_id"])))
                    choices = [(candidate, 1 if bool(body.get("accepted")) else 0)]
                context_id = str(body.get("context_id") or f"live-{int(time.time() * 1000)}")
                count = append_feedback(
                    DATA_ROOT / "training" / "feedback.csv",
                    context_id,
                    source,
                    choices,
                    state.energy_target,
                    state.mode,
                )
                state.telemetry.append({"at": int(time.time()), "type": "feedback", "context_id": context_id})
                self._json({"ok": True, "recorded": count, "context_id": context_id})
                return
            if path == "/api/demo/render":
                source, destination = ensure_demo_audio()
                technique = body.get("technique", "bass_swap")
                plan = TransitionPlan(
                    source_track_id="midnight-circuit",
                    destination_track_id="afterimage",
                    source_start_time=4.0,
                    destination_start_time=0.0,
                    transition_bars=int(body.get("bars", 16)),
                    target_bpm=123.0,
                    technique=technique,
                )
                output = DATA_ROOT / "renders" / "transition.wav"
                metrics = render_transition(source, destination, output, plan, 122.0, 124.0)
                source_analysis = analyze_track(source, "midnight-circuit")
                destination_analysis = analyze_track(destination, "afterimage")
                state.telemetry.append({"at": int(time.time()), "type": "render", "technique": technique})
                self._json({
                    "ok": True,
                    "url": "/renders/transition.wav",
                    "plan": plan.to_dict(),
                    "metrics": metrics.to_dict(),
                    "analysis": [source_analysis.to_dict(), destination_analysis.to_dict()],
                })
                return
            self._json({"error": "Unknown endpoint"}, 404)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    ensure_demo_audio()
    print(f"Autonomous DJ operator console: http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
