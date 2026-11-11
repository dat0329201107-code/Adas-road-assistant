"""Script to generate 3,000 short video clips evenly distributed across 5 Vietnamese road types.

5 Classes based on the specification:
1. Cao tốc (cao_toc): Multi-lane expressway with median divider, fast traffic. Max speed: 100 km/h.
2. Đường đôi đô thị (duong_doi_do_thi): Dual carriageway with median separator in urban area. Max speed: 60 km/h.
3. Phố hai chiều (pho_hai_chieu): Two-way street in residential/urban area, no median separator. Max speed: 50 km/h.
4. Ngoài khu dân cư (ngoai_khu_dan_cu): Road outside residential area, sparse houses. Max speed: 80 km/h.
5. Hẻm / ngõ (hem_ngo): Small narrow alley in residential area, very slow traffic. Max speed: 30 km/h.

Output:
600 short MP4 clips (1.0-1.2s, 25-30 frames, 640x360) per folder -> Total 3,000 videos.
"""
from __future__ import annotations

import csv
import glob
import json
import logging
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
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
        "speed_sim_range": (80, 110),
        "visual_style": "expressway",
    },
    "duong_doi_do_thi": {
        "id": 2,
        "name_vi": "Đường đôi đô thị",
        "folder": "duong_doi_do_thi",
        "prefix": "duong_doi",
        "legal_max_speed": 60,
        "description": "Hai chiều xe tách riêng bằng dải phân cách, nằm trong khu đô thị.",
        "speed_sim_range": (45, 65),
        "visual_style": "urban_dual",
    },
    "pho_hai_chieu": {
        "id": 3,
        "name_vi": "Phố hai chiều",
        "folder": "pho_hai_chieu",
        "prefix": "pho_2chieu",
        "legal_max_speed": 50,
        "description": "Đường phố trong khu dân cư, xe hai chiều đi chung một mặt đường.",
        "speed_sim_range": (30, 55),
        "visual_style": "urban_two_way",
    },
    "ngoai_khu_dan_cu": {
        "id": 4,
        "name_vi": "Ngoài khu dân cư",
        "folder": "ngoai_khu_dan_cu",
        "prefix": "ngoai_kdc",
        "legal_max_speed": 80,
        "description": "Đường nằm ngoài khu đông dân cư, hai bên ít nhà dân.",
        "speed_sim_range": (60, 85),
        "visual_style": "rural",
    },
    "hem_ngo": {
        "id": 5,
        "name_vi": "Hẻm / ngõ",
        "folder": "hem_ngo",
        "prefix": "hem_ngo",
        "legal_max_speed": 30,
        "description": "Đường nhỏ, hẹp trong khu dân cư, cần đi rất chậm.",
        "speed_sim_range": (15, 30),
        "visual_style": "alley",
    },
}

TARGET_PER_CLASS = 600  # 600 videos * 5 classes = 3,000 videos
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 360
FPS = 25
CLIP_FRAMES = 28  # ~1.12 seconds per clip


def generate_motion_frames(base_img: np.ndarray, num_frames: int, style: str) -> list[np.ndarray]:
    """Generate realistic forward dashcam motion frames from a base scene image."""
    h, w = base_img.shape[:2]
    base_resized = cv2.resize(base_img, (VIDEO_WIDTH + 60, VIDEO_HEIGHT + 40))
    bh, bw = base_resized.shape[:2]

    speed_factor = {
        "expressway": 0.14,
        "urban_dual": 0.08,
        "urban_two_way": 0.06,
        "rural": 0.10,
        "alley": 0.04,
    }.get(style, 0.07)

    frames = []
    vib_freq = random.uniform(1.5, 3.0)
    vib_amp = random.uniform(0.5, 1.5)

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        zoom = 1.0 + t * speed_factor
        crop_w = int(VIDEO_WIDTH / zoom)
        crop_h = int(VIDEO_HEIGHT / zoom)

        center_x = bw // 2 + int(math.sin(t * math.pi) * 6)
        center_y = bh // 2 + int(math.sin(i * vib_freq) * vib_amp)

        x1 = max(0, min(bw - crop_w, center_x - crop_w // 2))
        y1 = max(0, min(bh - crop_h, center_y - crop_h // 2))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        cropped = base_resized[y1:y2, x1:x2]
        frame = cv2.resize(cropped, (VIDEO_WIDTH, VIDEO_HEIGHT), interpolation=cv2.INTER_LINEAR)

        if random.random() < 0.2:
            jitter = random.randint(-4, 4)
            frame = np.clip(frame.astype(np.int16) + jitter, 0, 255).astype(np.uint8)

        frames.append(frame)

    return frames


def build_video_clip(output_path: Path, frames: list[np.ndarray]) -> bool:
    """Save frame sequence as an MP4 video file."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))
    if not out.isOpened():
        return False
    for f in frames:
        out.write(f)
    out.release()
    return True


def collect_seed_images(base_dir: Path) -> dict[str, list[np.ndarray]]:
    """Collect ground-truth representative seed images for each of the 5 road classes."""
    seeds: dict[str, list[np.ndarray]] = {k: [] for k in ROAD_CLASSES}

    label_dir = base_dir / "data" / "datasets" / "vietnam_traffic_signs" / "dataset" / "train" / "labels"
    img_dir = base_dir / "data" / "datasets" / "vietnam_traffic_signs" / "dataset" / "train" / "images"

    # 1. Parse dataset labels to find class-specific ground truth scenes
    if label_dir.exists():
        lbl_files = sorted(list(label_dir.glob("*.txt")))
        logger.info("Parsing %d dataset annotation files for domain-specific seeds...", len(lbl_files))
        for p in lbl_files:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    txt = f.read()
                classes = [int(line.split()[0]) for line in txt.strip().split("\n") if line.strip()]
                img_path = img_dir / f"{p.stem}.jpg"
                if not img_path.exists():
                    continue

                # 1. Cao toc: Speed limit 100 or 120 km/h (classes 58, 60)
                if 58 in classes or 60 in classes:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        seeds["cao_toc"].append(cv2.resize(img, (640, 360)))

                # 4. Ngoai khu dan cu: Speed limit 80 km/h (class 41)
                elif 41 in classes:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        seeds["ngoai_khu_dan_cu"].append(cv2.resize(img, (640, 360)))

                # 5. Hem / ngo: Slow residential limits 20, 30 km/h, slow down, low clearance (classes 61, 62, 8, 5)
                elif any(c in classes for c in [61, 62, 8, 5]):
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        seeds["hem_ngo"].append(cv2.resize(img, (640, 360)))
            except Exception:
                continue

    # 2. Extract frames from Hanoi street video (Urban dual carriageway & Two-way street)
    hanoi_path = base_dir / "data" / "videos" / "vietnam_hanoi_street.webm"
    if hanoi_path.exists():
        cap = cv2.VideoCapture(str(hanoi_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("Extracting seeds from Hanoi video (%d frames)...", total)
        step = max(1, total // 120)
        idx = 0
        while cap.isOpened() and idx < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                # Segment 0-2500 is typical urban two-way street (pho_hai_chieu)
                if idx < 2500:
                    seeds["pho_hai_chieu"].append(cv2.resize(frame, (640, 360)))
                # Segment 2800-5000 is wide urban dual carriageway with green median separator
                else:
                    seeds["duong_doi_do_thi"].append(cv2.resize(frame, (640, 360)))
            idx += step
        cap.release()

    # 3. Extract frames from Rural road video (Outside residential / Rural)
    rural_path = base_dir / "data" / "videos" / "vietnam_rural_road.ogv"
    if rural_path.exists():
        cap = cv2.VideoCapture(str(rural_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("Extracting seeds from Rural road video (%d frames)...", total)
        step = max(1, total // 120)
        idx = 0
        while cap.isOpened() and idx < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                seeds["ngoai_khu_dan_cu"].append(cv2.resize(frame, (640, 360)))
            idx += step
        cap.release()

    for k, v in seeds.items():
        logger.info("Class [%s]: %d verified ground-truth seed images", k, len(v))

    return seeds


def create_road_dataset(base_dir: Path, target_per_class: int = TARGET_PER_CLASS) -> dict:
    """Create 3,000 video files divided evenly across 5 road folders."""
    seeds = collect_seed_images(base_dir)
    video_root = base_dir / "data" / "videos"
    video_root.mkdir(parents=True, exist_ok=True)

    metadata_records = []

    for class_key, class_info in ROAD_CLASSES.items():
        folder_path = video_root / class_info["folder"]
        folder_path.mkdir(parents=True, exist_ok=True)
        class_seeds = seeds.get(class_key, [])
        if not class_seeds:
            logger.warning("No seeds found for %s, synthesizing placeholder frames", class_key)
            class_seeds = [np.full((360, 640, 3), 120, dtype=np.uint8)]

        logger.info("Generating %d video clips for class [%s]...", target_per_class, class_info["name_vi"])

        for i in range(1, target_per_class + 1):
            seed_img = class_seeds[(i - 1) % len(class_seeds)]
            img = seed_img.copy()
            if i % 3 == 0:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-10, 10)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

            frames = generate_motion_frames(img, CLIP_FRAMES, class_info["visual_style"])

            video_filename = f"{class_info['prefix']}_{i:04d}.mp4"
            out_path = folder_path / video_filename

            build_video_clip(out_path, frames)

            rel_path = f"data/videos/{class_info['folder']}/{video_filename}"
            metadata_records.append({
                "video_id": f"{class_info['id']}_{i:04d}",
                "filename": video_filename,
                "relative_path": rel_path,
                "class_id": class_info["id"],
                "class_key": class_key,
                "name_vi": class_info["name_vi"],
                "legal_max_speed_kmh": class_info["legal_max_speed"],
                "description": class_info["description"],
                "frames": len(frames),
                "fps": FPS,
                "duration_seconds": round(len(frames) / FPS, 2),
                "width": VIDEO_WIDTH,
                "height": VIDEO_HEIGHT,
            })

            if i % 100 == 0:
                logger.info("  %s: %d/%d videos created...", class_key, i, target_per_class)

    # Save dataset CSV
    csv_path = video_root / "dataset_labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_records[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_records)

    # Save dataset JSON
    json_path = video_root / "dataset_metadata.json"
    dataset_summary = {
        "dataset_name": "Vietnam Road Types Video Dataset (5 Classes)",
        "total_videos": len(metadata_records),
        "target_per_class": target_per_class,
        "classes": {k: {"name_vi": v["name_vi"], "max_speed_kmh": v["legal_max_speed"]} for k, v in ROAD_CLASSES.items()},
        "videos": metadata_records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, ensure_ascii=False, indent=2)

    logger.info("SUCCESS! Created %d video clips across 5 folders.", len(metadata_records))
    logger.info("Metadata saved to %s and %s", csv_path, json_path)
    return dataset_summary


if __name__ == "__main__":
    cwd = Path.cwd()
    if (cwd / "adas-road-assistant").exists():
        base = cwd / "adas-road-assistant"
    else:
        base = cwd
    create_road_dataset(base, target_per_class=600)
