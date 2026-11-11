import numpy as np
import pytest
from pathlib import Path

from adas.core.speed_limit_tracker import SpeedLimitTracker
from adas.vision.vietnam_road_classifier import (
    VIETNAM_ROAD_RULES,
    VietnamRoadClassifier,
)


def test_vietnam_road_rules_definitions():
    assert "cao_toc" in VIETNAM_ROAD_RULES
    assert VIETNAM_ROAD_RULES["cao_toc"]["legal_max_speed"] == 100
    assert VIETNAM_ROAD_RULES["duong_doi_do_thi"]["legal_max_speed"] == 60
    assert VIETNAM_ROAD_RULES["pho_hai_chieu"]["legal_max_speed"] == 50
    assert VIETNAM_ROAD_RULES["ngoai_khu_dan_cu"]["legal_max_speed"] == 80
    assert VIETNAM_ROAD_RULES["hem_ngo"]["legal_max_speed"] == 30
    assert VIETNAM_ROAD_RULES["khong_xac_dinh"]["legal_max_speed"] is None


def test_speed_limit_tracker_uses_road_limit():
    tracker = SpeedLimitTracker()
    # Initially no limit
    assert tracker.active().value_kmh is None

    # Road type observed: Đường đôi đô thị (60 km/h)
    tracker.observe_road_limit(60, road_name="Đường đôi đô thị")
    active = tracker.active()
    assert active.value_kmh == 60
    assert "Đường đôi đô thị" in active.source
    assert active.confirmed

    # Now observe a specific speed sign across 2 frames: 40 km/h (sign must override general road type limit)
    tracker.observe([40], now=10.0)
    tracker.observe([40], now=10.5)
    active_with_sign = tracker.active(now=10.5)
    assert active_with_sign.value_kmh == 40
    assert "BIỂN TỐC ĐỘ" in active_with_sign.source

    # After sign expires (TTL = 45s), fallback seamlessly to road type limit (60 km/h)
    active_after_expire = tracker.active(now=60.0)
    assert active_after_expire.value_kmh == 60
    assert "Đường đôi đô thị" in active_after_expire.source


def test_vietnam_road_classifier_load_and_predict():
    classifier = VietnamRoadClassifier(model_path="models/vietnam_road_classifier.pt")
    if Path("models/vietnam_road_classifier.pt").exists():
        assert classifier.load()
        # Create a synthetic test frame
        dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        res = classifier.classify(dummy_frame)
        assert "class_key" in res
        assert "label" in res
        assert "score" in res
