#!/usr/bin/env python3
"""Build a CC BY house-audio dataset from a Free Music Archive album page."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen


ALBUM_URL = "https://freemusicarchive.org/music/1000-handz/cc-by-free-to-use-dancehouse-instrumentals"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
TRACK_PATTERN = re.compile(r"data-track-info='([^']+)'")


def fetch(url: str) -> str:
    request = Request(url, headers={"User-Agent": "autonomous-dj-dataset-builder/0.1"})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def parse_tracks(page: str) -> list[dict]:
    tracks: list[dict] = []
    seen: set[int] = set()
    for encoded in TRACK_PATTERN.findall(page):
        track = json.loads(html.unescape(encoded))
        if track["id"] in seen:
            continue
        seen.add(track["id"])
        tracks.append(track)
    return tracks


def safe_name(index: int, title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return f"{index:02d}-{slug}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", type=Path, help="Use a previously downloaded album page")
    parser.add_argument("--output", type=Path, default=Path("data/training/cc_by_house"))
    parser.add_argument("--download", action="store_true", help="Download the licensed MP3 files")
    parser.add_argument("--convert-wav", action="store_true", help="Also create 44.1 kHz PCM WAV files (uses much more disk space)")
    args = parser.parse_args()

    page = args.page.read_text(encoding="utf-8") if args.page else fetch(ALBUM_URL)
    tracks = parse_tracks(page)
    if not tracks:
        raise SystemExit("No tracks found; the source page format may have changed")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.csv"
    fields = ["index", "title", "artist", "album", "source_url", "audio_url", "license", "attribution", "mp3_path", "wav_path"]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, track in enumerate(tracks, 1):
            stem = safe_name(index, track["title"])
            mp3 = args.output / f"{stem}.mp3"
            wav = args.output / f"{stem}.wav"
            writer.writerow({
                "index": index,
                "title": track["title"],
                "artist": track["artistName"],
                "album": track["albumTitle"],
                "source_url": track["url"],
                "audio_url": track["fileUrl"],
                "license": LICENSE_URL,
                "attribution": "1000 Handz / 1000Handz.com",
                "mp3_path": mp3,
                "wav_path": wav,
            })
            if args.download or args.convert_wav:
                if not mp3.exists():
                    subprocess.run(["curl", "-L", "--fail", "--silent", "--show-error", track["fileUrl"], "-o", str(mp3)], check=True)
            if args.convert_wav:
                if not wav.exists():
                    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp3), "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(wav)], check=True)

    print(f"Prepared {len(tracks)} CC BY tracks in {args.output}")


if __name__ == "__main__":
    main()
