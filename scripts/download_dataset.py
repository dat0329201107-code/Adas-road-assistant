"""Download traffic sign datasets for ADAS Road Assistant.

Downloads two datasets:
1. Vietnam Traffic Sign Detection (YOLO format) from Hugging Face
2. GTSDB - German Traffic Sign Detection Benchmark (speed-limit signs)

Usage:
    python scripts/download_dataset.py              # Download all
    python scripts/download_dataset.py --vietnam     # Vietnam dataset only
    python scripts/download_dataset.py --gtsdb       # GTSDB dataset only
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import struct
import sys
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"

VIETNAM_DIR = DATASETS_DIR / "vietnam_traffic_signs"
GTSDB_DIR = DATASETS_DIR / "gtsdb"


# ---------------------------------------------------------------------------
# Hugging Face – Vietnam Traffic Sign Detection (YOLO)
# ---------------------------------------------------------------------------
_HF_DATASET_REPO = "star092304/Traffic-sign-detection-VietNam"
_HF_API_BASE = "https://huggingface.co/api/datasets"


def _hf_list_files(repo_id: str) -> list[dict]:
    """List files in a HF dataset repo via the REST API."""
    url = f"{_HF_API_BASE}/{repo_id}"
    req = Request(url, headers={"User-Agent": "adas-road-assistant/0.1"})
    with urlopen(req, timeout=30) as resp:
        meta = json.loads(resp.read())
    # siblings contains file list
    return meta.get("siblings", [])


def _hf_download_file(repo_id: str, filename: str, dest: Path) -> None:
    """Download a single file from a HF dataset repo."""
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    req = Request(url, headers={"User-Agent": "adas-road-assistant/0.1"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def download_vietnam_dataset() -> None:
    """Download the Vietnam traffic sign dataset in YOLO format."""
    marker = VIETNAM_DIR / ".download_complete"
    if marker.exists():
        print(f"[Vietnam] Đã tải trước đó: {VIETNAM_DIR}")
        return

    print(f"[Vietnam] Đang tải dataset từ HuggingFace: {_HF_DATASET_REPO} ...")
    try:
        files = _hf_list_files(_HF_DATASET_REPO)
    except Exception as exc:
        print(f"[Vietnam] Không thể liệt kê file: {exc}", file=sys.stderr)
        print("[Vietnam] Thử phương án dự phòng: tải trực tiếp ...")
        _download_vietnam_fallback()
        return

    if not files:
        print("[Vietnam] Repo rỗng, thử phương án dự phòng ...", file=sys.stderr)
        _download_vietnam_fallback()
        return

    total = len(files)
    for i, finfo in enumerate(files, 1):
        rfilename = finfo.get("rfilename", "")
        if not rfilename:
            continue
        dest = VIETNAM_DIR / rfilename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [{i}/{total}] Bỏ qua (đã có): {rfilename}")
            continue
        print(f"  [{i}/{total}] Tải: {rfilename}")
        try:
            _hf_download_file(_HF_DATASET_REPO, rfilename, dest)
        except Exception as exc:
            print(f"    CẢNH BÁO: Lỗi tải {rfilename}: {exc}", file=sys.stderr)

    # Giải nén dataset zip nếu có
    zip_path = VIETNAM_DIR / "traffic-sign-detection-vietnam.zip"
    if zip_path.exists():
        dataset_extracted = VIETNAM_DIR / "dataset"
        if not dataset_extracted.exists():
            print("[Vietnam] Đang giải nén traffic-sign-detection-vietnam.zip ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(VIETNAM_DIR)
            print("[Vietnam] Giải nén thành công!")

    # Tạo data.yaml từ classid.csv cho YOLO training
    _generate_vietnam_yaml()

    marker.touch()
    _print_dataset_stats(VIETNAM_DIR, "Vietnam")


def _generate_vietnam_yaml() -> None:
    """Generate YOLO data.yaml and classes mapping from classid.csv."""
    import csv

    csv_path = VIETNAM_DIR / "classid.csv"
    if not csv_path.exists():
        return

    classes: dict[int, str] = {}
    vn_names: dict[int, str] = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Clean keys in case of whitespace or BOM
            clean_row = {k.strip().lstrip("\ufeff"): v for k, v in row.items() if k}
            if "ID" not in clean_row or not clean_row["ID"]:
                continue
            cid = int(clean_row["ID"])
            classes[cid] = clean_row.get("classes-en", f"class_{cid}").strip()
            vn_names[cid] = clean_row.get("classes-vn", classes[cid]).strip()

    data_dir = VIETNAM_DIR / "dataset"
    yaml_path = VIETNAM_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {data_dir.as_posix()}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write("names:\n")
        for cid in sorted(classes):
            f.write(f'  {cid}: "{classes[cid]}"\n')

    # Also save bilingual class dictionary for ADAS inference
    bilingual_path = VIETNAM_DIR / "classes_bilingual.json"
    bilingual_data = {
        str(cid): {"en": classes[cid], "vi": vn_names.get(cid, classes[cid])}
        for cid in sorted(classes)
    }
    with open(bilingual_path, "w", encoding="utf-8") as f:
        json.dump(bilingual_data, f, ensure_ascii=False, indent=2)

    print(f"[Vietnam] Đã tạo data.yaml và classes_bilingual.json ({len(classes)} classes)!")


def _download_vietnam_fallback() -> None:
    """Fallback: try snapshot_download via huggingface_hub if installed."""
    try:
        from huggingface_hub import snapshot_download  # type: ignore

        print("[Vietnam] Dùng huggingface_hub.snapshot_download ...")
        snapshot_download(
            repo_id=_HF_DATASET_REPO,
            repo_type="dataset",
            local_dir=str(VIETNAM_DIR),
        )
        (VIETNAM_DIR / ".download_complete").touch()
        _print_dataset_stats(VIETNAM_DIR, "Vietnam")
    except ImportError:
        print(
            "[Vietnam] LỖI: Cần cài `huggingface_hub` hoặc kiểm tra kết nối mạng.",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# GTSDB – German Traffic Sign Detection Benchmark
# ---------------------------------------------------------------------------
_GTSDB_URL = "https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip"

# GTSDB class mapping (43 classes). Speed-limit classes: 0-8
_GTSDB_CLASSES = {
    0: "speed_limit_20",
    1: "speed_limit_30",
    2: "speed_limit_50",
    3: "speed_limit_60",
    4: "speed_limit_70",
    5: "speed_limit_80",
    6: "end_speed_limit_80",
    7: "speed_limit_100",
    8: "speed_limit_120",
    9: "no_overtaking",
    10: "no_overtaking_trucks",
    11: "priority_road_crossroad",
    12: "priority_road",
    13: "yield",
    14: "stop",
    15: "no_vehicles",
    16: "no_trucks",
    17: "no_entry",
    18: "danger",
    19: "left_curve",
    20: "right_curve",
    21: "double_curve",
    22: "bumpy_road",
    23: "slippery_road",
    24: "narrowing_road",
    25: "construction",
    26: "traffic_signals",
    27: "pedestrian_crossing",
    28: "children_crossing",
    29: "bicycle_crossing",
    30: "ice_snow",
    31: "wild_animals",
    32: "end_restrictions",
    33: "right_turn_ahead",
    34: "left_turn_ahead",
    35: "ahead_only",
    36: "ahead_or_right",
    37: "ahead_or_left",
    38: "keep_right",
    39: "keep_left",
    40: "roundabout",
    41: "end_no_overtaking",
    42: "end_no_overtaking_trucks",
}


def _convert_ppm_to_jpg(ppm_path: Path, jpg_path: Path) -> tuple[int, int]:
    """Convert a PPM (P6 binary) image to JPEG using Pillow or raw fallback.

    Returns (width, height).
    """
    try:
        from PIL import Image  # type: ignore

        img = Image.open(ppm_path)
        img.save(jpg_path, "JPEG", quality=95)
        return img.size  # (w, h)
    except ImportError:
        pass

    # Fallback: read PPM header for dimensions, save via OpenCV
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        img = cv2.imread(str(ppm_path))
        if img is not None:
            cv2.imwrite(str(jpg_path), img)
            h, w = img.shape[:2]
            return w, h
    except ImportError:
        pass

    raise RuntimeError("Cần Pillow hoặc OpenCV để chuyển đổi PPM → JPEG")


def _parse_gtsdb_gt(gt_path: Path) -> dict[str, list[tuple]]:
    """Parse GTSDB gt.txt into {filename: [(x1,y1,x2,y2,class_id), ...]}."""
    annotations: dict[str, list[tuple]] = {}
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(";")
            if len(parts) < 6:
                continue
            fname = parts[0]
            x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            cls_id = int(parts[5])
            annotations.setdefault(fname, []).append((x1, y1, x2, y2, cls_id))
    return annotations


def _to_yolo_label(
    x1: int, y1: int, x2: int, y2: int, cls_id: int, img_w: int, img_h: int
) -> str:
    """Convert absolute bbox to YOLO normalized format."""
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def download_gtsdb_dataset() -> None:
    """Download GTSDB and convert to YOLO format."""
    marker = GTSDB_DIR / ".download_complete"
    if marker.exists():
        print(f"[GTSDB] Đã tải trước đó: {GTSDB_DIR}")
        return

    print("[GTSDB] Đang tải GTSDB (~1.6 GB) ...")
    GTSDB_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = GTSDB_DIR / "FullIJCNN2013.zip"

    # Download
    if not zip_path.exists():
        try:
            req = Request(_GTSDB_URL, headers={"User-Agent": "adas-road-assistant/0.1"})
            total_downloaded = 0
            with urlopen(req, timeout=300) as resp, open(zip_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
                    total_downloaded += len(chunk)
                    mb = total_downloaded / (1024 * 1024)
                    print(f"\r  Đã tải: {mb:.1f} MB", end="", flush=True)
            print()
        except (URLError, OSError) as exc:
            print(f"\n[GTSDB] LỖI tải: {exc}", file=sys.stderr)
            zip_path.unlink(missing_ok=True)
            return

    # Extract
    print("[GTSDB] Giải nén ...")
    raw_dir = GTSDB_DIR / "_raw"
    raw_dir.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
    except zipfile.BadZipFile:
        print("[GTSDB] LỖI: File zip bị hỏng. Xóa và thử lại.", file=sys.stderr)
        zip_path.unlink(missing_ok=True)
        return

    # Find gt.txt
    gt_candidates = list(raw_dir.rglob("gt.txt"))
    if not gt_candidates:
        print("[GTSDB] LỖI: Không tìm thấy gt.txt trong archive.", file=sys.stderr)
        return
    gt_path = gt_candidates[0]
    ppm_root = gt_path.parent

    # Parse annotations
    annotations = _parse_gtsdb_gt(gt_path)
    print(f"[GTSDB] Tìm thấy {len(annotations)} ảnh có annotation.")

    # Convert to YOLO format
    images_dir = GTSDB_DIR / "images"
    labels_dir = GTSDB_DIR / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    ppm_files = sorted(ppm_root.glob("*.ppm"))
    if not ppm_files:
        # Try jpg/png
        ppm_files = sorted(ppm_root.glob("*.jpg")) + sorted(ppm_root.glob("*.png"))

    total = len(ppm_files)
    converted = 0
    for i, ppm in enumerate(ppm_files, 1):
        stem = ppm.stem
        jpg_dest = images_dir / f"{stem}.jpg"

        if not jpg_dest.exists():
            try:
                img_w, img_h = _convert_ppm_to_jpg(ppm, jpg_dest)
            except Exception as exc:
                print(f"  CẢNH BÁO: Bỏ qua {ppm.name}: {exc}")
                continue
        else:
            # Get image size from existing file
            try:
                from PIL import Image

                with Image.open(jpg_dest) as img:
                    img_w, img_h = img.size
            except ImportError:
                import cv2

                tmp = cv2.imread(str(jpg_dest))
                if tmp is None:
                    continue
                img_h, img_w = tmp.shape[:2]

        # Write YOLO label
        label_dest = labels_dir / f"{stem}.txt"
        bboxes = annotations.get(ppm.name, [])
        if bboxes:
            with open(label_dest, "w") as lf:
                for x1, y1, x2, y2, cls_id in bboxes:
                    lf.write(_to_yolo_label(x1, y1, x2, y2, cls_id, img_w, img_h) + "\n")
        else:
            # Negative sample (no signs) — empty label file
            label_dest.touch()

        converted += 1
        if i % 100 == 0 or i == total:
            print(f"\r  Chuyển đổi: {i}/{total}", end="", flush=True)

    print(f"\n[GTSDB] Đã chuyển đổi {converted} ảnh sang YOLO format.")

    # Write data.yaml
    data_yaml = GTSDB_DIR / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {GTSDB_DIR.as_posix()}\n")
        f.write("train: images\n")
        f.write("val: images\n\n")
        f.write(f"nc: {len(_GTSDB_CLASSES)}\n")
        f.write("names:\n")
        for cid in sorted(_GTSDB_CLASSES):
            f.write(f"  {cid}: {_GTSDB_CLASSES[cid]}\n")

    # Write class_names.txt for reference
    names_file = GTSDB_DIR / "class_names.txt"
    with open(names_file, "w", encoding="utf-8") as f:
        for cid in sorted(_GTSDB_CLASSES):
            f.write(f"{cid}: {_GTSDB_CLASSES[cid]}\n")

    # Clean up raw extracted files to save space
    print("[GTSDB] Dọn dẹp file tạm ...")
    shutil.rmtree(raw_dir, ignore_errors=True)
    zip_path.unlink(missing_ok=True)

    marker.touch()
    _print_dataset_stats(GTSDB_DIR, "GTSDB")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _print_dataset_stats(dataset_dir: Path, name: str) -> None:
    """Print summary statistics about a downloaded dataset."""
    print(f"\n{'='*50}")
    print(f"  Dataset: {name}")
    print(f"  Vị trí:  {dataset_dir}")

    # Count images
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    for ext in img_exts:
        images.extend(dataset_dir.rglob(f"*{ext}"))
    print(f"  Ảnh:     {len(images)}")

    # Count labels
    labels = list(dataset_dir.rglob("*.txt"))
    labels = [l for l in labels if l.name not in ("class_names.txt", "README.md")]
    print(f"  Labels:  {len(labels)}")

    # Check data.yaml
    yaml_files = list(dataset_dir.rglob("data.yaml")) + list(
        dataset_dir.rglob("*.yaml")
    )
    if yaml_files:
        print(f"  Config:  {yaml_files[0].relative_to(dataset_dir)}")

    # Total size
    total_bytes = sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file())
    mb = total_bytes / (1024 * 1024)
    print(f"  Dung lượng: {mb:.1f} MB")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tải dataset biển báo giao thông cho ADAS Road Assistant"
    )
    parser.add_argument(
        "--vietnam", action="store_true", help="Chỉ tải dataset biển báo Việt Nam"
    )
    parser.add_argument(
        "--gtsdb",
        action="store_true",
        help="Chỉ tải dataset GTSDB (Đức, quốc tế)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ADAS Road Assistant — Dataset Downloader")
    print("=" * 60)

    download_all = not args.vietnam and not args.gtsdb

    if download_all or args.vietnam:
        download_vietnam_dataset()

    if download_all or args.gtsdb:
        download_gtsdb_dataset()

    print("\n✅ Hoàn tất tải dataset!")
    print(f"📁 Thư mục dataset: {DATASETS_DIR}")
    print("\nHướng dẫn train model YOLO26 với dataset:")
    print("  from ultralytics import YOLO")
    print("  model = YOLO('yolo26n.pt')")
    print(f"  model.train(data='{GTSDB_DIR / 'data.yaml'}', epochs=50, imgsz=640)")


if __name__ == "__main__":
    main()
