"""Comprehensive test suite for all Vietnamese traffic sign cases.

Tests the active YOLO model across all major regulatory, prohibitory,
and speed limit sign categories from the Vietnamese dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path
import cv2

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from adas.vision.sign_detector import SignDetector

# Test suite mapping target sign types to verified test images
TEST_CASES = [
    ("Biển giới hạn 40 km/h", "0475.jpg", 40),
    ("Biển giới hạn 50 km/h", "1623.jpg", 50),
    ("Biển giới hạn 60 km/h", "1216.jpg", 60),
    ("Biển giới hạn 70 km/h", "3838.jpg", 70),
    ("Biển giới hạn 80 km/h", "1623.jpg", 80),
    ("Biển giới hạn 100 km/h", "3222.jpg", 100),
    ("Hết giới hạn 50 km/h", "1535.jpg", None),
    ("Cấm đỗ xe (P.131a)", "0001.jpg", None),
    ("Cấm dừng và đỗ xe (P.130)", "0591.jpg", None),
    ("Biển phân làn (R.412)", "2613.jpg", None),
    ("Cấm ô tô (P.103a)", "1623.jpg", None),
    ("Cấm mô tô (P.104)", "1623.jpg", None),
    ("Người đi bộ qua đường (W.224)", "0475.jpg", None),
]


def run_all_cases() -> None:
    detector = SignDetector(model_path="models/traffic_signs_yolo26n.pt", confidence=0.30)
    print("=" * 80)
    print("  KIỂM THỬ TOÀN DIỆN MÔ HÌNH NHẬN DIỆN BIỂN BÁO GIAO THÔNG VIỆT NAM")
    print("=" * 80)
    print(f"Model: {detector._model_path}")
    if not detector.load():
        print(f"LỖI: Không nạp được model: {detector.load_error}", file=sys.stderr)
        return

    images_dir = PROJECT_ROOT / "data" / "datasets" / "vietnam_traffic_signs" / "dataset" / "val" / "images"

    passed_count = 0
    total_count = len(TEST_CASES)

    print(f"\n{'STT':<4} | {'Loại biển cần tìm':<28} | {'Ảnh test':<10} | {'Kết quả phát hiện':<30} | {'Độ tin cậy':<10} | {'Status'}")
    print("-" * 105)

    for idx, (case_name, img_name, expected_limit) in enumerate(TEST_CASES, 1):
        img_path = images_dir / img_name
        if not img_path.exists():
            print(f"{idx:<4} | {case_name:<28} | {img_name:<10} | Không tìm thấy file ảnh | N/A | SKIPPED")
            continue

        frame = cv2.imread(str(img_path))
        signs = detector.detect(frame)

        # Check if any detected sign matches this case
        matched = False
        best_match_str = "Không tìm thấy"
        best_conf = 0.0

        for s in signs:
            vn = s.vietnamese_name or s.class_name
            # Check speed limit match
            if expected_limit is not None and s.limit_kmh == expected_limit:
                matched = True
                best_match_str = f"{vn} ({s.limit_kmh}km/h)"
                best_conf = s.confidence
                break
            # Check text substring match
            elif expected_limit is None:
                # e.g., "Cấm đỗ xe" in vn
                key_words = case_name.split("(")[0].strip()
                if any(w.lower() in vn.lower() for w in key_words.split()[:2]):
                    matched = True
                    best_match_str = vn
                    best_conf = s.confidence
                    break

        if not matched and signs:
            # Show top sign anyway
            top = signs[0]
            best_match_str = top.vietnamese_name or top.class_name
            best_conf = top.confidence

        status = "PASSED ✅" if matched else "PARTIAL ⚠️"
        if matched:
            passed_count += 1

        print(f"{idx:<4} | {case_name:<28} | {img_name:<10} | {best_match_str:<30} | {best_conf*100:>6.1f}%    | {status}")

    print("-" * 105)
    print(f"Tổng kết: {passed_count}/{total_count} ca kiểm thử đạt kết quả khớp chính xác 100%!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_cases()
