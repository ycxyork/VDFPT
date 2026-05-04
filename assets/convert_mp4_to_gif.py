"""
Batch convert MP4 demos in this folder to GIF files for GitHub README previews.

Requirements:
  - ffmpeg must be installed and available in PATH.

Usage:
  python convert_mp4_to_gif.py
  python convert_mp4_to_gif.py --fps 10 --width 360
  python convert_mp4_to_gif.py --input 1.mp4 2.mp4 --output-dir gifs
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MP4 files in assets/ to GitHub-friendly GIF previews."
    )
    parser.add_argument(
        "--input",
        nargs="*",
        default=None,
        help="MP4 files to convert. Defaults to all *.mp4 files in this folder.",
    )
    parser.add_argument(
        "--output-dir",
        default="gifs",
        help="Output directory, relative to this script folder. Defaults to gifs/.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="GIF frame rate. Lower values make smaller files. Defaults to 8.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="GIF width in pixels. Height is kept proportional. Defaults to 320.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing GIF files.",
    )
    return parser.parse_args()


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg was not found in PATH. Please install ffmpeg first, then rerun this script."
        )


def collect_inputs(script_dir: Path, names: list[str] | None) -> list[Path]:
    if names:
        videos = [script_dir / name for name in names]
    else:
        videos = sorted(script_dir.glob("*.mp4"), key=lambda p: natural_key(p.stem))

    missing = [path for path in videos if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path.name) for path in missing)
        raise SystemExit(f"Input file(s) not found: {missing_text}")

    return videos


def natural_key(text: str) -> tuple[int, str]:
    return (int(text), text) if text.isdigit() else (10**9, text)


def convert_video(video_path: Path, output_path: Path, fps: int, width: int, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        print(f"Skip existing: {output_path.name}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # palettegen/paletteuse keeps GIF colors cleaner than one-step conversion.
    filter_graph = (
        f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];"
        "[s0]palettegen=max_colors=128[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=5"
    )
    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-vf",
        filter_graph,
        "-loop",
        "0",
        str(output_path),
    ]

    print(f"Convert: {video_path.name} -> {output_path.relative_to(output_path.parent.parent)}")
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    ensure_ffmpeg()

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / args.output_dir
    videos = collect_inputs(script_dir, args.input)

    if not videos:
        raise SystemExit("No MP4 files found.")

    for video_path in videos:
        output_path = output_dir / f"{video_path.stem}.gif"
        convert_video(video_path, output_path, args.fps, args.width, args.overwrite)

    print(f"Done. GIF files are in: {output_dir}")


if __name__ == "__main__":
    main()
