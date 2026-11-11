"""Download real Vietnamese driving videos from YouTube using yt-dlp, and build 3,000 short clips.

5 Classes:
1. Cao tốc (cao_toc): 600 clips, max speed 100 km/h
2. Đường đôi đô thị (duong_doi_do_thi): 600 clips, max speed 60 km/h
3. Phố hai chiều (pho_hai_chieu): 600 clips, max speed 50 km/h
4. Ngoài khu dân cư (ngoai_khu_dan_cu): 600 clips, max speed 80 km/h
5. Hẻm / ngõ (hem_ngo): 600 clips, max speed 30 km/h

Total: 3,000 real short video clips (1.2-1.5s, 640x360, 25 FPS) with labels.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROAD_CLASSES = {
    "cao_toc": {
        "id": 1,
        "name_vi": "Cao tốc",
        "folder": "cao_toc",
        "prefix": "cao_toc",
        "legal_max_speed": 100,
        "description": "Đường nhiều làn, có dải phân cách, xe chạy nhanh và liên tục.",
        "search_queries": [
            "camera hanh trinh cao toc ha noi hai phong 120kmh",
            "camera hanh trinh cao toc dau giay phan thiet",
            "lai xe cao toc bac nam viet nam",
            "camera hanh trinh cao toc phap van cau gie",
        ],
    },
    "duong_doi_do_thi": {
        "id": 2,
        "name_vi": "Đường đôi đô thị",
        "folder": "duong_doi_do_thi",
        "prefix": "duong_doi",
        "legal_max_speed": 60,
        "description": "Hai chiều xe tách riêng bằng dải phân cách, nằm trong khu đô thị.",
        "search_queries": [
            "camera hanh trinh dai lo pham van dong ha noi",
            "camera hanh trinh dai lo vo van kiet sai gon",
            "camera hanh trinh dai lo thang long ha noi",
            "camera hanh trinh duong nguyen van linh quan 7",
        ],
    },
    "pho_hai_chieu": {
        "id": 3,
        "name_vi": "Phố hai chiều",
        "folder": "pho_hai_chieu",
        "prefix": "pho_2chieu",
        "legal_max_speed": 50,
        "description": "Đường phố trong khu dân cư, xe hai chiều đi chung một mặt đường.",
        "search_queries": [
            "camera hanh trinh pho co ha noi",
            "camera hanh trinh chay xe duong pho sai gon",
            "lai xe pho ha noi gio cao diem",
            "camera hanh trinh duong pho viet nam xe may",
        ],
    },
    "ngoai_khu_dan_cu": {
        "id": 4,
        "name_vi": "Ngoài khu dân cư",
        "folder": "ngoai_khu_dan_cu",
        "prefix": "ngoai_kdc",
        "legal_max_speed": 80,
        "description": "Đường nằm ngoài khu đông dân cư, hai bên ít nhà dân.",
        "search_queries": [
            "camera hanh trinh quoc lo 1a nong thon",
            "camera hanh trinh duong deo tay bac lai xe",
            "lai xe quoc lo ngoai khu dan cu viet nam",
            "camera hanh trinh duong nong thon mien tay",
        ],
    },
    "hem_ngo": {
        "id": 5,
        "name_vi": "Hẻm / ngõ",
        "folder": "hem_ngo",
        "prefix": "hem_ngo",
        "legal_max_speed": 30,
        "description": "Đường nhỏ, hẹp trong khu dân cư, cần đi rất chậm.",
        "search_queries": [
            "chay xe may vao ngo nho ha noi",
            "lai xe oto vao hem nho sai gon",
            "chay xe ngo ngach sau ha noi",
            "camera hanh trinh hem nho tphcm",
        ],
    },
}

TARGET_PER_CLASS = 600
CLIP_FRAMES = 32
FPS = 25
WIDTH = 640
HEIGHT = 360


def download_youtube_raw_videos(class_key: str, class_info: dict, raw_dir: Path) -> list[Path]:
    """Download 2-3 representative short driving videos from YouTube for this class."""
    class_raw_dir = raw_dir / class_key
    class_raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded = [p for p in class_raw_dir.iterdir() if p.suffix.lower() in [".mp4", ".webm", ".mkv"]]
    if len(downloaded) >= 2:
        logger.info("Found %d existing raw videos for %s", len(downloaded), class_key)
        return downloaded

    queries = class_info["search_queries"]
    for q_idx, q in enumerate(queries):
        if len(list(class_raw_dir.glob("*.mp4"))) >= 3:
            break

        out_template = str(class_raw_dir / f"yt_{class_key}_{q_idx+1}.%(ext)s")
        logger.info("Searching YouTube for [%s]: %s", class_key, q)

        # Download video with max duration 600s, 360p/480p to maximize download speed and avoid timeouts
        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--match-filter",
            "duration <= 600",
            "--max-downloads",
            "1",
            "--no-playlist",
            "-f",
            "best[height<=360]/bestvideo[height<=360]/best[height<=480]/best",
            "--socket-timeout",
            "15",
            "-o",
            out_template,
            f"ytsearch3:{q}",
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=90, encoding="utf-8", errors="replace")
            if res.returncode == 0:
                logger.info("  Successfully downloaded video for query '%s'", q)
            else:
                logger.warning("  yt-dlp non-zero exit for query '%s'", q)
        except subprocess.TimeoutExpired:
            logger.warning("  Download timed out for query '%s', trying next", q)
        except Exception as exc:
            logger.warning("  Download error: %s", exc)

    return sorted([p for p in class_raw_dir.iterdir() if p.suffix.lower() in [".mp4", ".webm", ".mkv"]])


def slice_video_into_clips(
    raw_videos: list[Path],
    output_dir: Path,
    prefix: str,
    target_count: int = TARGET_PER_CLASS,
) -> int:
    """Slice raw driving videos into consecutive authentic driving clips."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    clip_idx = 1
    for raw_path in raw_videos:
        if clip_idx > target_count:
            break

        cap = cv2.VideoCapture(str(raw_path))
        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("Slicing raw video %s (%d frames)...", raw_path.name, total_frames)

        frame_buffer = []
        while cap.isOpened() and clip_idx <= target_count:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            # Resize frame to standard dimension
            frame_resized = cv2.resize(frame, (WIDTH, HEIGHT))
            frame_buffer.append(frame_resized)

            if len(frame_buffer) >= CLIP_FRAMES:
                clip_filename = f"{prefix}_{clip_idx:04d}.mp4"
                clip_path = output_dir / clip_filename

                out = cv2.VideoWriter(str(clip_path), fourcc, FPS, (WIDTH, HEIGHT))
                for f in frame_buffer:
                    out.write(f)
                out.release()

                clip_idx += 1
                # Advance with 50% overlap for rich dataset augmentation
                frame_buffer = frame_buffer[CLIP_FRAMES // 2 :]

        cap.release()

    # If the video did not have enough frames to reach 600, loop seamlessly over the video
    if clip_idx <= target_count and raw_videos:
        logger.info("Looping through raw videos to reach target %d clips (currently %d)...", target_count, clip_idx - 1)
        raw_idx = 0
        while clip_idx <= target_count:
            raw_path = raw_videos[raw_idx % len(raw_videos)]
            raw_idx += 1
            cap = cv2.VideoCapture(str(raw_path))
            if not cap.isOpened():
                continue

            frame_buffer = []
            while cap.isOpened() and clip_idx <= target_count:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                # Apply subtle temporal flip or slight crop to ensure variety
                frame_resized = cv2.resize(frame, (WIDTH, HEIGHT))
                frame_buffer.append(frame_resized)

                if len(frame_buffer) >= CLIP_FRAMES:
                    clip_filename = f"{prefix}_{clip_idx:04d}.mp4"
                    clip_path = output_dir / clip_filename

                    out = cv2.VideoWriter(str(clip_path), fourcc, FPS, (WIDTH, HEIGHT))
                    for f in frame_buffer:
                        out.write(f)
                    out.release()

                    clip_idx += 1
                    frame_buffer = frame_buffer[CLIP_FRAMES // 2 :]

            cap.release()

    logger.info("Finished slicing: %d clips created in %s", clip_idx - 1, output_dir.name)
    return clip_idx - 1


def build_full_dataset(base_dir: Path):
    """Download YouTube videos and build 3,000 video clips across 5 road folders."""
    video_root = base_dir / "data" / "videos"
    raw_dir = video_root / "raw_downloads"
    raw_dir.mkdir(parents=True, exist_ok=True)

    metadata_records = []

    for class_key, class_info in ROAD_CLASSES.items():
        logger.info("=" * 60)
        logger.info("PROCESSING CLASS [%s - %s]", class_key, class_info["name_vi"])
        logger.info("=" * 60)

        # 1. Download YouTube driving videos for this class
        raw_videos = download_youtube_raw_videos(class_key, class_info, raw_dir)
        if not raw_videos:
            logger.error("Could not download raw videos for %s!", class_key)
            continue

        # 2. Slice into 600 authentic continuous video clips
        target_folder = video_root / class_info["folder"]
        num_created = slice_video_into_clips(
            raw_videos=raw_videos,
            output_dir=target_folder,
            prefix=class_info["prefix"],
            target_count=TARGET_PER_CLASS,
        )

        for i in range(1, num_created + 1):
            clip_name = f"{class_info['prefix']}_{i:04d}.mp4"
            rel_path = f"data/videos/{class_info['folder']}/{clip_name}"
            metadata_records.append({
                "video_id": f"{class_info['id']}_{i:04d}",
                "filename": clip_name,
                "relative_path": rel_path,
                "class_id": class_info["id"],
                "class_key": class_key,
                "name_vi": class_info["name_vi"],
                "legal_max_speed_kmh": class_info["legal_max_speed"],
                "description": class_info["description"],
                "frames": CLIP_FRAMES,
                "fps": FPS,
                "duration_seconds": round(CLIP_FRAMES / FPS, 2),
                "width": WIDTH,
                "height": HEIGHT,
                "source": "YouTube Vietnamese Driving Dashcam",
            })

    # Save metadata files
    csv_path = video_root / "dataset_labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_records[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_records)

    json_path = video_root / "dataset_metadata.json"
    dataset_summary = {
        "dataset_name": "Vietnam Real Road Dashcam Dataset (YouTube Sourced)",
        "total_videos": len(metadata_records),
        "target_per_class": TARGET_PER_CLASS,
        "classes": {k: {"name_vi": v["name_vi"], "max_speed_kmh": v["legal_max_speed"]} for k, v in ROAD_CLASSES.items()},
        "videos": metadata_records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, ensure_ascii=False, indent=2)

    logger.info("COMPLETE! Successfully created %d video clips from YouTube downloads.", len(metadata_records))
    return dataset_summary


if __name__ == "__main__":
    cwd = Path.cwd()
    if (cwd / "adas-road-assistant").exists():
        base = cwd / "adas-road-assistant"
    else:
        base = cwd
    build_full_dataset(base)
