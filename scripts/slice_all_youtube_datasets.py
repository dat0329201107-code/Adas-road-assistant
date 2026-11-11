"""Slice downloaded real YouTube driving videos into exactly 600 clips per road class (total 3,000 clips)."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import cv2

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
    },
    "duong_doi_do_thi": {
        "id": 2,
        "name_vi": "Đường đôi đô thị",
        "folder": "duong_doi_do_thi",
        "prefix": "duong_doi",
        "legal_max_speed": 60,
        "description": "Hai chiều xe tách riêng bằng dải phân cách, nằm trong khu đô thị.",
    },
    "pho_hai_chieu": {
        "id": 3,
        "name_vi": "Phố hai chiều",
        "folder": "pho_hai_chieu",
        "prefix": "pho_2chieu",
        "legal_max_speed": 50,
        "description": "Đường phố trong khu dân cư, xe hai chiều đi chung một mặt đường.",
    },
    "ngoai_khu_dan_cu": {
        "id": 4,
        "name_vi": "Ngoài khu dân cư",
        "folder": "ngoai_khu_dan_cu",
        "prefix": "ngoai_kdc",
        "legal_max_speed": 80,
        "description": "Đường nằm ngoài khu đông dân cư, hai bên ít nhà dân.",
    },
    "hem_ngo": {
        "id": 5,
        "name_vi": "Hẻm / ngõ",
        "folder": "hem_ngo",
        "prefix": "hem_ngo",
        "legal_max_speed": 30,
        "description": "Đường nhỏ, hẹp trong khu dân cư, cần đi rất chậm.",
    },
}

TARGET_PER_CLASS = 600
CLIP_FRAMES = 32
FPS = 25
WIDTH = 640
HEIGHT = 360


def slice_class(raw_videos: list[Path], output_dir: Path, prefix: str, target: int = TARGET_PER_CLASS) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Clean existing clips in the output dir first
    for f in output_dir.glob("*.mp4"):
        try:
            f.unlink()
        except Exception:
            pass

    clip_idx = 1
    for raw_path in raw_videos:
        if clip_idx > target:
            break

        cap = cv2.VideoCapture(str(raw_path))
        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("  Slicing raw video %s (%d frames)...", raw_path.name, total_frames)

        frame_buffer = []
        while cap.isOpened() and clip_idx <= target:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_resized = cv2.resize(frame, (WIDTH, HEIGHT))
            frame_buffer.append(frame_resized)

            if len(frame_buffer) >= CLIP_FRAMES:
                clip_path = output_dir / f"{prefix}_{clip_idx:04d}.mp4"
                out = cv2.VideoWriter(str(clip_path), fourcc, FPS, (WIDTH, HEIGHT))
                for f in frame_buffer:
                    out.write(f)
                out.release()

                clip_idx += 1
                frame_buffer = frame_buffer[CLIP_FRAMES // 2 :]

        cap.release()

    # Loop over the videos if needed until exact target count is reached
    if clip_idx <= target and raw_videos:
        logger.info("  Looping over raw videos to reach target %d clips (currently %d)...", target, clip_idx - 1)
        raw_idx = 0
        while clip_idx <= target:
            raw_path = raw_videos[raw_idx % len(raw_videos)]
            raw_idx += 1
            cap = cv2.VideoCapture(str(raw_path))
            if not cap.isOpened():
                continue

            frame_buffer = []
            while cap.isOpened() and clip_idx <= target:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                frame_resized = cv2.resize(frame, (WIDTH, HEIGHT))
                frame_buffer.append(frame_resized)

                if len(frame_buffer) >= CLIP_FRAMES:
                    clip_path = output_dir / f"{prefix}_{clip_idx:04d}.mp4"
                    out = cv2.VideoWriter(str(clip_path), fourcc, FPS, (WIDTH, HEIGHT))
                    for f in frame_buffer:
                        out.write(f)
                    out.release()

                    clip_idx += 1
                    frame_buffer = frame_buffer[CLIP_FRAMES // 2 :]

            cap.release()

    count = clip_idx - 1
    logger.info("  Done: %d clips created in %s", count, output_dir.name)
    return count


def slice_all(base_dir: Path):
    video_root = base_dir / "data" / "videos"
    raw_dir = video_root / "raw_downloads"

    metadata_records = []

    for class_key, class_info in ROAD_CLASSES.items():
        logger.info("Processing [%s - %s]...", class_key, class_info["name_vi"])
        class_raw_dir = raw_dir / class_key
        raw_videos = sorted([p for p in class_raw_dir.iterdir() if p.suffix.lower() in [".mp4", ".webm", ".mkv"]])

        if not raw_videos:
            logger.error("No raw videos found in %s", class_raw_dir)
            continue

        target_folder = video_root / class_info["folder"]
        num_created = slice_class(
            raw_videos=raw_videos,
            output_dir=target_folder,
            prefix=class_info["prefix"],
            target=TARGET_PER_CLASS,
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

    # Save CSV
    csv_path = video_root / "dataset_labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_records[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_records)

    # Save JSON
    json_path = video_root / "dataset_metadata.json"
    summary = {
        "dataset_name": "Vietnam Real Road Dashcam Dataset (YouTube Sourced)",
        "total_videos": len(metadata_records),
        "target_per_class": TARGET_PER_CLASS,
        "classes": {k: {"name_vi": v["name_vi"], "max_speed_kmh": v["legal_max_speed"]} for k, v in ROAD_CLASSES.items()},
        "videos": metadata_records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("ALL DONE! Created %d video clips from YouTube downloads.", len(metadata_records))
    return summary


if __name__ == "__main__":
    cwd = Path.cwd()
    if (cwd / "adas-road-assistant").exists():
        base = cwd / "adas-road-assistant"
    else:
        base = cwd
    slice_all(base)
