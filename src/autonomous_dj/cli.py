from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audio import analyze_track, render_transition
from .models import TransitionPlan
from .server import run


def main() -> None:
    parser = argparse.ArgumentParser(prog="autonomous-dj")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Launch the operator console")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    analyze = sub.add_parser("analyze", help="Analyze a PCM WAV track")
    analyze.add_argument("path")
    render = sub.add_parser("render", help="Render a phrase-aligned transition")
    render.add_argument("source")
    render.add_argument("destination")
    render.add_argument("output")
    render.add_argument("--source-start", type=float, default=0.0)
    render.add_argument("--destination-start", type=float, default=0.0)
    render.add_argument("--source-bpm", type=float)
    render.add_argument("--destination-bpm", type=float)
    render.add_argument("--target-bpm", type=float, default=124.0)
    render.add_argument("--bars", type=int, default=16)
    render.add_argument("--technique", choices=["bass_swap", "eq_blend", "quick_cut"], default="bass_swap")
    args = parser.parse_args()
    if args.command == "serve":
        run(args.host, args.port)
    elif args.command == "analyze":
        print(json.dumps(analyze_track(args.path).to_dict(), indent=2))
    else:
        plan = TransitionPlan(Path(args.source).stem, Path(args.destination).stem, args.source_start, args.destination_start, args.bars, args.target_bpm, args.technique)
        metrics = render_transition(args.source, args.destination, args.output, plan, args.source_bpm, args.destination_bpm)
        print(json.dumps({"plan": plan.to_dict(), "metrics": metrics.to_dict()}, indent=2))


if __name__ == "__main__":
    main()

