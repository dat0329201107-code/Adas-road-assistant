"""Train YOLO26 on Vietnamese Traffic Sign Detection Dataset.

This script fine-tunes a YOLO26 model on the 82-class Vietnam traffic
sign dataset, and optionally saves the resulting weights to
``models/traffic_signs_yolo26n.pt`` for immediate use by the ADAS pipeline.

Usage:
    python scripts/train_yolo26_signs.py --epochs 30 --batch 16
    python scripts/train_yolo26_signs.py --device cpu --epochs 5 --imgsz 320  # quick CPU test
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_YAML = PROJECT_ROOT / "data" / "datasets" / "vietnam_traffic_signs" / "data.yaml"
OUTPUT_MODEL_DIR = PROJECT_ROOT / "models"
TARGET_MODEL_PATH = OUTPUT_MODEL_DIR / "traffic_signs_yolo26n.pt"


def train(
    data_yaml: Path = DEFAULT_DATA_YAML,
    base_model: str = "yolo26n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "auto",
    project_dir: str = "runs/train",
    name: str = "vietnam_signs_yolo26",
) -> Path | None:
    """Train YOLO26 on the dataset."""
    if not data_yaml.exists():
        print(f"LỖI: Không tìm thấy file data.yaml tại: {data_yaml}", file=sys.stderr)
        print("Hãy chạy script tải dataset trước: python scripts/download_dataset.py", file=sys.stderr)
        return None

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        print("LỖI: Chưa cài đặt ultralytics. Hãy chạy:", file=sys.stderr)
        print("  pip install ultralytics", file=sys.stderr)
        return None

    print("=" * 60)
    print("  Huấn luyện mô hình YOLO26 — Biển báo giao thông Việt Nam")
    print("=" * 60)
    print(f"  Dataset:      {data_yaml}")
    print(f"  Base Model:   {base_model}")
    print(f"  Epochs:       {epochs}")
    print(f"  Image Size:   {imgsz}")
    print(f"  Batch Size:   {batch}")
    print(f"  Device:       {device}")
    print("=" * 60)

    # Initialize model
    model = YOLO(base_model)

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": project_dir,
        "name": name,
        "exist_ok": True,
    }
    if device != "auto":
        train_kwargs["device"] = device

    results = model.train(**train_kwargs)

    # Copy best weights to models/traffic_signs_yolo26n.pt
    best_weights = Path(project_dir) / name / "weights" / "best.pt"
    if best_weights.exists():
        OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, TARGET_MODEL_PATH)
        print(f"\n✅ Đã lưu weights tốt nhất vào: {TARGET_MODEL_PATH}")
        print("Hệ thống ADAS sẽ tự động nhận diện và sử dụng model này!")
        return TARGET_MODEL_PATH
    else:
        print(f"\nHoàn thành training. Kết quả lưu tại: {Path(project_dir) / name}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình YOLO26 nhận diện biển báo Việt Nam"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="Đường dẫn tới file data.yaml của dataset",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolo26n.pt",
        help="Model YOLO gốc (mặc định: yolo26n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Số epoch huấn luyện (mặc định: 50)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Kích thước ảnh đầu vào (mặc định: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (mặc định: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Thiết bị chạy (auto, cpu, 0, cuda:0)",
    )
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
