from adas.core.alert_manager import OverspeedAlert
from adas.core.speed_limit_tracker import SpeedLimitTracker


def test_manual_limit_is_fallback_and_road_type_is_not_used():
    tracker = SpeedLimitTracker(manual_limit=50)
    active = tracker.active(now=100.0)
    assert active.value_kmh == 50
    assert active.source == "GIỚI HẠN CẤU HÌNH THỦ CÔNG"
    assert not active.confirmed


def test_sign_limit_requires_repeated_hits():
    tracker = SpeedLimitTracker(manual_limit=60, sign_ttl_seconds=45)
    tracker.observe([50, 50, 50], now=100.0)
    assert tracker.active(now=100.0).value_kmh == 60
    tracker.observe([50], now=100.5)
    active = tracker.active(now=100.5)
    assert active.value_kmh == 50
    assert active.confirmed


def test_sign_limit_expires_to_manual_fallback():
    tracker = SpeedLimitTracker(manual_limit=60, sign_ttl_seconds=10)
    tracker.observe([50], now=100.0)
    tracker.observe([50], now=101.0)
    assert tracker.active(now=105.0).value_kmh == 50
    assert tracker.active(now=112.0).value_kmh == 60


def test_overspeed_requires_persistence_and_clears():
    alert = OverspeedAlert(buffer_kmh=0.0, confirm_seconds=0.5)
    assert not alert.update(55, 50, now=10.0)
    assert alert.update(55, 50, now=10.6)
    assert not alert.update(48, 50, now=10.7)
