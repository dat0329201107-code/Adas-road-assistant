"""Download authentic Vietnamese car dashcam videos for all 5 road categories.

All videos are real car windshield dashcams showing:
- Clear road surface (asphalt / concrete / brick)
- Lane markings (dashed / solid / median dividers)
- Real Vietnamese traffic, street signs, and vehicles
- ZERO cartoons, ZERO AI animations, ZERO talking heads
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "videos" / "raw_downloads"
FFMPEG_PATH = BASE_DIR.parent / ".venv" / "Scripts" / "ffmpeg.exe"

TARGETS: dict[str, list[tuple[str, str]]] = {
    "hem_ngo": [
        ("https://www.youtube.com/watch?v=n7XkgpXJ12I", "dashcam_ngo_ellicam"),
        ("https://www.youtube.com/watch?v=OJm1vYNbfXI", "dashcam_ngo_pandaauto"),
        ("https://www.youtube.com/watch?v=gIR4atgOEio", "dashcam_ngo_kinhnghiem"),
    ],
    "cao_toc": [
        ("https://www.youtube.com/watch?v=03HGcZzueP0", "dashcam_caotoc_hungyen"),
        ("https://www.youtube.com/watch?v=ywZ-X6wxRWQ", "dashcam_caotoc_thanglong"),
    ],
    "duong_doi_do_thi": [
        ("https://www.youtube.com/watch?v=ejy-excurzM", "dashcam_duongdoi_nguyenchithanh"),
        ("https://www.youtube.com/watch?v=YRi2JzmTTro", "dashcam_duongdoi_hotungmau"),
        ("https://www.youtube.com/watch?v=2xD8y15OXyo", "dashcam_duongdoi_saigon"),
    ],
    "pho_hai_chieu": [
        ("https://www.youtube.com/watch?v=Q6OnGSzf8zg", "dashcam_pho2chieu_hanoi"),
        ("https://www.youtube.com/watch?v=I86VjXEbgKs", "dashcam_pho2chieu_oldquarter"),
        ("https://www.youtube.com/watch?v=HHpEMzaxnP4", "dashcam_pho2chieu_street"),
    ],
    "ngoai_khu_dan_cu": [
        ("https://www.youtube.com/watch?v=fM42eW_DGOI", "dashcam_ngoaict_ql1a_1"),
        ("https://www.youtube.com/watch?v=PpuPYkilb5g", "dashcam_ngoaict_ql1a_2"),
        ("https://www.youtube.com/watch?v=hkn4brhh2Sk", "dashcam_ngoaict_thainguyen"),
    ],
}


def download_video(url: str, out_stem: Path) -> Path | None:
    for ext in [".webm", ".mp4", ".mkv"]:
        existing = out_stem.parent / f"{out_stem.name}{ext}"
        if existing.exists() and existing.stat().st_size > 1_000_000:
            logger.info("File already exists: %s (%d bytes)", existing.name, existing.stat().st_size)
            return existing

    out_pattern = str(out_stem.parent / f"{out_stem.name}.%(ext)s")
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--ffmpeg-location",
        str(FFMPEG_PATH),
        "-f",
        "best[height<=360]/bestvideo[height<=360]",
        "-o",
        out_pattern,
        url,
    ]
    logger.info("Downloading: %s -> %s", url, out_stem.name)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error("Download failed for %s:\n%s", url, res.stderr)
        return None

    for ext in [".webm", ".mp4", ".mkv"]:
        p = out_stem.parent / f"{out_stem.name}{ext}"
        if p.exists() and p.stat().st_size > 500_000:
            logger.info("Downloaded successfully: %s (%d MB)", p.name, p.stat().st_size // (1024 * 1024))
            return p
    return None


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total_downloaded = 0
    for cat, items in TARGETS.items():
        cat_dir = RAW_DIR / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        for url, name in items:
            out_stem = cat_dir / name
            p = download_video(url, out_stem)
            if p is not None:
                total_downloaded += 1

    logger.info("Finished downloading all targets! Total: %d", total_downloaded)


if __name__ == "__main__":
    main()
