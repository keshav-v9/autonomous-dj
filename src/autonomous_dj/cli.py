from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .audio import analyze_track, render_transition
from .models import TransitionPlan


def _manifest_audio_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    extract = sub.add_parser("ml-extract", help="Extract ML music features from a dataset manifest")
    extract.add_argument("manifest", type=Path)
    extract.add_argument("output", type=Path)
    extract.add_argument("--limit", type=int)
    candidates = sub.add_parser("ml-candidates", help="Create a transition-candidate labeling CSV")
    candidates.add_argument("catalog", type=Path)
    candidates.add_argument("output", type=Path)
    candidates.add_argument("--target-energy", type=float, default=7.0)
    candidates.add_argument("--direction", choices=["maintain", "energy_up", "energy_down"], default="maintain")
    train = sub.add_parser("ml-train", help="Train and evaluate an XGBoost pairwise ranker")
    train.add_argument("labels", type=Path)
    train.add_argument("output_dir", type=Path)
    train.add_argument("--seed", type=int, default=42)
    train_torch = sub.add_parser("ml-train-torch", help="Train and evaluate a PyTorch RankNet model")
    train_torch.add_argument("labels", type=Path)
    train_torch.add_argument("output_dir", type=Path)
    train_torch.add_argument("--seed", type=int, default=42)
    train_torch.add_argument("--epochs", type=int, default=24)
    rank = sub.add_parser("ml-rank", help="Rank catalog tracks for a source track")
    rank.add_argument("catalog", type=Path)
    rank.add_argument("source_id")
    rank.add_argument("--model-dir", type=Path, default=Path("data/models/transition_ranker"))
    rank.add_argument("--target-energy", type=float, default=7.0)
    rank.add_argument("--direction", choices=["maintain", "energy_up", "energy_down"], default="maintain")
    rank.add_argument("--limit", type=int, default=10)
    rank_torch = sub.add_parser("ml-rank-torch", help="Rank tracks with a trained PyTorch RankNet model")
    rank_torch.add_argument("catalog", type=Path)
    rank_torch.add_argument("source_id")
    rank_torch.add_argument("--model-dir", type=Path, default=Path("data/models/transition_ranker"))
    rank_torch.add_argument("--target-energy", type=float, default=7.0)
    rank_torch.add_argument("--direction", choices=["maintain", "energy_up", "energy_down"], default="maintain")
    rank_torch.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.command == "serve":
        from .server import run

        run(args.host, args.port)
    elif args.command == "analyze":
        print(json.dumps(analyze_track(args.path).to_dict(), indent=2))
    elif args.command == "render":
        plan = TransitionPlan(Path(args.source).stem, Path(args.destination).stem, args.source_start, args.destination_start, args.bars, args.target_bpm, args.technique)
        metrics = render_transition(args.source, args.destination, args.output, plan, args.source_bpm, args.destination_bpm)
        print(json.dumps({"plan": plan.to_dict(), "metrics": metrics.to_dict()}, indent=2))
    elif args.command == "ml-extract":
        from .ml import analyze_music, write_feature_catalog

        rows = _manifest_audio_rows(args.manifest)
        if args.limit is not None:
            rows = rows[: args.limit]
        catalog = []
        for row in rows:
            wav_path = row.get("wav_path", "").strip()
            mp3_path = row.get("mp3_path", "").strip()
            path = Path(wav_path) if wav_path else Path(mp3_path)
            if not path.is_file() and mp3_path:
                path = Path(mp3_path)
            if not path.is_file():
                raise SystemExit(f"Audio file not found for {row.get('title', row.get('index'))}: {path}")
            item = analyze_music(path, track_id=path.stem)
            catalog.append(item)
            print(f"[{len(catalog)}/{len(rows)}] {row.get('title', item.track_id)}: {item.bpm:.1f} BPM, {item.camelot}")
        write_feature_catalog(catalog, args.output)
        print(json.dumps({"tracks": len(catalog), "output": str(args.output)}, indent=2))
    elif args.command == "ml-candidates":
        from .ml import build_labeling_rows, read_feature_catalog, write_labeling_csv

        rows = build_labeling_rows(read_feature_catalog(args.catalog), args.target_energy, args.direction)
        write_labeling_csv(rows, args.output)
        print(json.dumps({"candidates": len(rows), "output": str(args.output)}, indent=2))
    elif args.command == "ml-train":
        from .training import train_ranker

        print(json.dumps(train_ranker(args.labels, args.output_dir, args.seed), indent=2))
    elif args.command == "ml-train-torch":
        from .torch_ranker import train_torch_ranker

        print(json.dumps(train_torch_ranker(args.labels, args.output_dir, args.seed, args.epochs), indent=2))
    elif args.command == "ml-rank":
        from .ml import TransitionRanker, read_feature_catalog

        catalog = read_feature_catalog(args.catalog)
        by_id = {item.track_id: item for item in catalog}
        if args.source_id not in by_id:
            raise SystemExit(f"Unknown source_id {args.source_id!r}")
        ranked = TransitionRanker.load(args.model_dir).rank(
            by_id[args.source_id],
            [item for item in catalog if item.track_id != args.source_id],
            args.target_energy,
            args.direction,
        )
        print(json.dumps([item.to_dict() for item in ranked[: args.limit]], indent=2))
    elif args.command == "ml-rank-torch":
        from .ml import read_feature_catalog
        from .torch_ranker import TorchTransitionRanker

        catalog = read_feature_catalog(args.catalog)
        by_id = {item.track_id: item for item in catalog}
        if args.source_id not in by_id:
            raise SystemExit(f"Unknown source_id {args.source_id!r}")
        ranked = TorchTransitionRanker.load(args.model_dir).rank(
            by_id[args.source_id],
            [item for item in catalog if item.track_id != args.source_id],
            args.target_energy,
            args.direction,
        )
        print(json.dumps([item.to_dict() for item in ranked[: args.limit]], indent=2))


if __name__ == "__main__":
    main()
