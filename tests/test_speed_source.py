from adas.core.speed_source import (
    GPSSpeedSource,
    OBDSpeedSource,
    SimulatedSpeed,
)


def test_simulated_initial_none():
    s = SimulatedSpeed()
    assert s.read_kmh() is None
    assert s.is_available()


def test_simulated_initial_value():
    s = SimulatedSpeed(initial_kmh=50.0)
    assert s.read_kmh() == 50.0


def test_simulated_increase_from_none():
    s = SimulatedSpeed()
    s.increase()
    assert s.read_kmh() == 30.0


def test_simulated_increase_from_value():
    s = SimulatedSpeed(initial_kmh=50.0)
    s.increase(10.0)
    assert s.read_kmh() == 60.0


def test_simulated_decrease_from_none():
    s = SimulatedSpeed()
    s.decrease()
    assert s.read_kmh() == 25.0


def test_simulated_decrease_floor_zero():
    s = SimulatedSpeed(initial_kmh=3.0)
    s.decrease()
    assert s.read_kmh() == 0.0


def test_simulated_set_speed():
    s = SimulatedSpeed()
    s.set_speed(88.0)
    assert s.read_kmh() == 88.0
    s.set_speed(None)
    assert s.read_kmh() is None


def test_simulated_toggle():
    s = SimulatedSpeed()
    s.toggle()  # None -> 40
    assert s.read_kmh() == 40.0
    s.toggle()  # 40 -> None
    assert s.read_kmh() is None


def test_gps_source_unavailable_without_device():
    """GPSSpeedSource should handle non-existent port gracefully."""
    gps = GPSSpeedSource(port="COM999_NONEXISTENT")
    assert not gps.is_available()
    assert gps.read_kmh() is None
    gps.close()


def test_obd_source_unavailable_without_device():
    """OBDSpeedSource should handle non-existent OBD gracefully."""
    obd_src = OBDSpeedSource(port="NONEXISTENT")
    assert not obd_src.is_available()
    assert obd_src.read_kmh() is None
    obd_src.close()
