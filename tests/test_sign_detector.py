import numpy as np

from adas.vision.sign_detector import (
    DetectedSign,
    SignDetector,
    _parse_limit,
)


def test_parse_limit():
    assert _parse_limit("speed_limit_50") == 50
    assert _parse_limit("speed-limit-80") == 80
    assert _parse_limit("speed limit 100") == 100
    assert _parse_limit("Speed limit 40hm/h") == 40
    assert _parse_limit("Speed limit 60hm/h") == 60
    assert _parse_limit("Speed limit 80km/h") == 80
    assert _parse_limit("stop_sign") is None
    assert _parse_limit("pedestrian_crossing") is None


def test_detected_sign_dataclass():
    sign = DetectedSign((10, 20, 50, 50), "speed_limit_60", 0.92, 60, "Giới hạn tốc độ 60km/h")
    assert sign.limit_kmh == 60
    assert sign.confidence == 0.92
    assert sign.bbox == (10, 20, 50, 50)
    assert sign.vietnamese_name == "Giới hạn tốc độ 60km/h"


def test_sign_detector_bilingual_lookup():
    detector = SignDetector(model_path="nonexistent_model.pt")
    # If bilingual map exists in the dataset, verify lookup
    vn_name = detector.get_vietnamese_name(2, "Speed limit 40hm/h")
    if vn_name is not None:
        assert "40km/h" in vn_name or "Giới hạn" in vn_name


def test_sign_detector_graceful_on_missing_model():
    detector = SignDetector(model_path="nonexistent_model.pt")
    # Missing model file should not crash; it should report available=False or load failure
    loaded = detector.load()
    if not loaded:
        assert not detector.available
        assert detector.load_error is not None
        # detect should return empty list without crashing
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert detector.detect(frame) == []
