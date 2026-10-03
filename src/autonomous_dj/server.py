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
from .models import OperatorState, TransitionPlan

ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "apps" / "web"
DATA_ROOT = ROOT / "data"
state = OperatorState()

TRACKS = [
    {"id": "midnight-circuit", "title": "Midnight Circuit", "artist": "Demo System", "bpm": 122.0, "key": "8A", "energy": 6.8, "duration": "04:18", "color": "violet"},
    {"id": "afterimage", "title": "Afterimage", "artist": "Demo System", "bpm": 124.0, "key": "8A", "energy": 7.6, "duration": "03:52", "color": "lime"},
    {"id": "glass-horizon", "title": "Glass Horizon", "artist": "Demo System", "bpm": 125.0, "key": "9A", "energy": 8.1, "duration": "04:04", "color": "cyan"},
    {"id": "soft-focus", "title": "Soft Focus", "artist": "Demo System", "bpm": 120.0, "key": "7A", "energy": 5.9, "duration": "03:46", "color": "amber"},
]


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
            self._json({"state": state.to_dict(), "tracks": TRACKS, "engine": {"status": "ready", "model": "heuristic-v0.1", "latency_ms": 38}})
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
                elif action == "energy_down":
                    state.energy_target = max(1.0, state.energy_target - 0.5)
                elif action == "skip":
                    state.current_track, state.next_track = state.next_track, "Glass Horizon"
                elif action == "emergency_fade":
                    state.mode = "emergency fade"
                else:
                    state.mode = action
                event = {"at": int(time.time()), "type": "operator", "action": action}
                state.telemetry.append(event)
                self._json({"ok": True, "state": state.to_dict()})
                return
            if path == "/api/request":
                title = str(body.get("title", "")).strip()
                if title:
                    state.requests.append(title)
                    state.telemetry.append({"at": int(time.time()), "type": "request", "title": title})
                self._json({"ok": True, "state": state.to_dict()})
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

