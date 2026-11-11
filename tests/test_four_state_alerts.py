"""Unit tests for the 4 Alert States and Explainable Warnings (VisionDrive AI)."""
from __future__ import annotations

import pytest
from adas.core.alert_manager import (
    AlertDecision,
    AlertStatus,
    FourStateAlertManager,
    OverspeedAlert,
)


def test_state_1_dang_xac_dinh_insufficient_frames():
    """State 1: Chưa đủ khung hình khởi động camera."""
    mgr = FourStateAlertManager(min_warmup_frames=5)
    decision = mgr.decide(
        speed_kmh=50.0,
        limit_kmh=60,
        road_name="Đường đôi đô thị",
        road_confidence=0.95,
        frame_index=2,  # < 5
    )
    assert decision.status == AlertStatus.DANG_XAC_DINH
    assert decision.status_text == "Đang xác định"
    assert not decision.trigger_sound
    assert "Chưa đủ điều kiện đưa ra cảnh báo" in decision.explanation


def test_state_1_dang_xac_dinh_low_confidence():
    """State 1: Độ tin cậy nhận diện loại đường còn thấp (< min_confidence)."""
    mgr = FourStateAlertManager(min_confidence=0.60, min_warmup_frames=1)
    decision = mgr.decide(
        speed_kmh=50.0,
        limit_kmh=60,
        road_name="Đường đôi đô thị",
        road_confidence=0.42,  # 42% < 60%
        frame_index=10,
    )
    assert decision.status == AlertStatus.DANG_XAC_DINH
    assert not decision.trigger_sound
    assert "độ tin cậy thấp" in decision.explanation


def test_state_1_dang_xac_dinh_none_limit():
    """State 1: Chưa xác định được giới hạn tốc độ hợp pháp."""
    mgr = FourStateAlertManager(min_warmup_frames=1)
    decision = mgr.decide(
        speed_kmh=50.0,
        limit_kmh=None,
        road_name="Không xác định",
        road_confidence=0.85,
        frame_index=10,
    )
    assert decision.status == AlertStatus.DANG_XAC_DINH
    assert not decision.trigger_sound


def test_state_2_an_toan():
    """State 2: An toàn - Tốc độ nằm trong giới hạn của loại đường hiện tại."""
    mgr = FourStateAlertManager(caution_margin=5.0, min_confidence=0.60, min_warmup_frames=1)
    decision = mgr.decide(
        speed_kmh=40.0,
        limit_kmh=50,
        road_name="Phố hai chiều",
        road_confidence=0.92,
        frame_index=10,
    )
    assert decision.status == AlertStatus.AN_TOAN
    assert decision.status_text == "An toàn"
    assert decision.is_safe
    assert not decision.trigger_sound
    assert "Đang ở Phố hai chiều" in decision.explanation
    assert "giới hạn 50 km/h" in decision.explanation
    assert "Tốc độ an toàn" in decision.explanation


def test_state_3_chu_y_approaching_limit():
    """State 3: Chú ý - Tốc độ gần chạm giới hạn (trong ngưỡng chú ý)."""
    mgr = FourStateAlertManager(caution_margin=5.0, min_confidence=0.60, min_warmup_frames=1)
    decision = mgr.decide(
        speed_kmh=48.0,  # 50 - 5 = 45 <= 48 <= 50
        limit_kmh=50,
        road_name="Phố hai chiều",
        road_confidence=0.94,
        frame_index=10,
    )
    assert decision.status == AlertStatus.CHU_Y
    assert decision.status_text == "Chú ý"
    assert decision.is_caution
    assert decision.trigger_sound
    assert decision.sound_severity == "caution"
    assert "gần chạm giới hạn" in decision.explanation
    assert "Đang ở Phố hai chiều" in decision.explanation


def test_state_4_vuot_toc_overspeed():
    """State 4: Vượt tốc - Tốc độ vượt giới hạn; cảnh báo mạnh kèm giải thích."""
    mgr = FourStateAlertManager(confirm_seconds=0.0, min_confidence=0.60, min_warmup_frames=1)
    decision = mgr.decide(
        speed_kmh=58.0,
        limit_kmh=50,
        road_name="Phố hai chiều",
        road_confidence=0.92,
        frame_index=10,
    )
    assert decision.status == AlertStatus.VUOT_TOC
    assert decision.status_text == "Vượt tốc"
    assert decision.is_overspeed
    assert decision.trigger_sound
    assert decision.sound_severity == "alarm"
    assert "VƯỢT QUÁ +8 km/h" in decision.explanation
    assert "Đang ở Phố hai chiều" in decision.explanation


def test_configurable_caution_margin():
    """Ngưỡng 'chú ý' có thể chỉnh trong cài đặt."""
    # Khi ngưỡng là 10 km/h: tốc độ 72 km/h trên đường 80 km/h sẽ là 'Chú ý'
    mgr_wide = FourStateAlertManager(caution_margin=10.0, min_warmup_frames=1)
    d_wide = mgr_wide.decide(speed_kmh=72.0, limit_kmh=80, road_name="Ngoài khu dân cư", road_confidence=0.95)
    assert d_wide.status == AlertStatus.CHU_Y

    # Khi ngưỡng là 3 km/h: tốc độ 72 km/h trên đường 80 km/h sẽ là 'An toàn'
    mgr_narrow = FourStateAlertManager(caution_margin=3.0, min_warmup_frames=1)
    d_narrow = mgr_narrow.decide(speed_kmh=72.0, limit_kmh=80, road_name="Ngoài khu dân cư", road_confidence=0.95)
    assert d_narrow.status == AlertStatus.AN_TOAN


def test_overspeed_alert_backwards_compatible():
    """Lớp OverspeedAlert vẫn tương thích 100% với code cũ."""
    alert = OverspeedAlert(buffer_kmh=0.0, confirm_seconds=0.5)
    assert not alert.update(55, 50, now=10.0)
    assert alert.update(55, 50, now=10.6)
    assert not alert.update(48, 50, now=10.7)
