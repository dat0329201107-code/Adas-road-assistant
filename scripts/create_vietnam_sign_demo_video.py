"""Create a realistic demo video from verified Vietnamese traffic sign images.

Compiles scenes containing actual Vietnamese speed limit signs and regulatory signs
into a 1280x720 video preserving aspect ratio (letterboxing), allowing complete
verification of automatic sign detection and speed limit updates.
"""
from __future__ import annotations

import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "vietnam_traffic_signs" / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "data" / "videos"
OUTPUT_VIDEO = OUTPUT_DIR / "vietnam_signs_demo.mp4"

# Selected images verified to have strong detections:
VERIFIED_IMAGES = [
    ("val/images/2613.jpg", "Cấm đỗ xe & Biển phân làn"),
    ("val/images/3222.jpg", "Biển Giới hạn tốc độ 20km/h"),
    ("val/images/1535.jpg", "Biển Hết giới hạn tốc độ 50km/h"),
    ("val/images/3242.jpg", "Biển Giới hạn tốc độ 100km/h"),
    ("val/images/0591.jpg", "Cấm dừng, đỗ xe & Cấm xe tải"),
    ("val/images/0829.jpg", "Cấm đỗ xe trên đường"),
]


def letterbox_image(image: np.ndarray, target_w: int = 1280, target_h: int = 720) -> np.ndarray:
    """Resize image to fit within target dimensions preserving aspect ratio with black padding."""
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = (target_h - nh) // 2
    left = (target_w - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def create_demo_video(
    output_path: Path = OUTPUT_VIDEO,
    width: int = 1280,
    height: int = 720,
    fps: int = 25,
    frames_per_scene: int = 50,
) -> Path | None:
    """Generate driving demo video from actual sign images."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    valid_scenes: list[tuple[np.ndarray, str]] = []
    for rel_path, desc in VERIFIED_IMAGES:
        full_path = DATASET_DIR / rel_path
        if full_path.exists():
            img = cv2.imread(str(full_path))
            if img is not None:
                boxed = letterbox_image(img, width, height)
                valid_scenes.append((boxed, desc))

    if not valid_scenes:
        print("LỖI: Không tìm thấy ảnh trong dataset để tạo video!", file=sys.stderr)
        return None

    print(f"Tạo video demo từ {len(valid_scenes)} cảnh biển báo thực tế...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = 0
    for idx, (scene, desc) in enumerate(valid_scenes, 1):
        print(f"  [{idx}/{len(valid_scenes)}] Cảnh: {desc} ({frames_per_scene} frames)")
        # Slight zoom in to simulate moving forward
        h, w = scene.shape[:2]
        for f in range(frames_per_scene):
            # Write frames
            writer.write(scene)
            total_frames += 1

    writer.release()
    print(f"\n✅ Đã tạo thành công video demo: {output_path}")
    print(f"   Tổng số khung hình: {total_frames} ({total_frames / fps:.1f} giây)")
    return output_path


def main() -> None:
    create_demo_video()


if __name__ == "__main__":
    main()
