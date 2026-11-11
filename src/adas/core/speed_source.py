"""Pluggable speed sources: simulated, GPS (NMEA), and OBD-II."""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SpeedSource(ABC):
    """Abstract base for anything that can supply vehicle speed."""

    @abstractmethod
    def read_kmh(self) -> float | None:
        """Return current speed in km/h, or ``None`` if unavailable."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` when the source is connected and producing data."""

    def close(self) -> None:
        """Release underlying resources (serial port, socket, etc.)."""


# ---------------------------------------------------------------------------
# Simulated (keyboard-driven) speed
# ---------------------------------------------------------------------------

class SimulatedSpeed(SpeedSource):
    """Keyboard-driven simulated speed for demo / testing.

    Call :meth:`increase`, :meth:`decrease`, :meth:`set_speed`, or
    :meth:`toggle` from a key-handler to change the reported speed.
    """

    def __init__(self, initial_kmh: float | None = None) -> None:
        self.speed_kmh = initial_kmh

    def read_kmh(self) -> float | None:
        return self.speed_kmh

    def is_available(self) -> bool:
        return True

    def increase(self, delta: float = 5.0) -> None:
        """Increase speed by *delta* km/h (starts at 30 if currently ``None``)."""
        self.speed_kmh = 30.0 if self.speed_kmh is None else self.speed_kmh + delta

    def decrease(self, delta: float = 5.0) -> None:
        """Decrease speed by *delta* km/h (starts at 25 if currently ``None``)."""
        self.speed_kmh = 25.0 if self.speed_kmh is None else max(0.0, self.speed_kmh - delta)

    def set_speed(self, value: float | None) -> None:
        """Set speed to an exact value (or ``None`` to clear)."""
        self.speed_kmh = value

    def toggle(self) -> None:
        """Toggle between a default speed (40 km/h) and ``None``."""
        self.speed_kmh = 40.0 if self.speed_kmh is None else None


# ---------------------------------------------------------------------------
# GPS via serial NMEA sentences
# ---------------------------------------------------------------------------

_KNOTS_TO_KMH = 1.852


class GPSSpeedSource(SpeedSource):
    """Read speed from a GPS receiver via serial port (NMEA sentences).

    Parses ``$GPRMC`` and ``$GPVTG`` sentences for speed-over-ground in
    knots and converts to km/h.  A background thread continuously reads
    the serial port so :meth:`read_kmh` never blocks.

    Requires the optional ``pyserial`` package.
    """

    def __init__(self, port: str = "COM3", baudrate: int = 9600) -> None:
        self._port = port
        self._baudrate = baudrate

        self._lock = threading.Lock()
        self._speed_kmh: float | None = None
        self._available = False
        self._serial: object | None = None  # serial.Serial when connected
        self._running = False
        self._thread: threading.Thread | None = None

        self._connect()

    # -- public interface ---------------------------------------------------

    def read_kmh(self) -> float | None:
        with self._lock:
            return self._speed_kmh

    def is_available(self) -> bool:
        with self._lock:
            return self._available

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._serial is not None:
            try:
                self._serial.close()  # type: ignore[union-attr]
            except Exception:
                pass

    # -- internals ----------------------------------------------------------

    def _connect(self) -> None:
        try:
            import serial  # pyserial
        except ImportError:
            logger.warning("pyserial is not installed – GPSSpeedSource disabled")
            return

        try:
            self._serial = serial.Serial(self._port, self._baudrate, timeout=1)
            with self._lock:
                self._available = True
            self._running = True
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()
            logger.info("GPS connected on %s @ %d baud", self._port, self._baudrate)
        except serial.SerialException as exc:
            logger.error("Failed to open GPS serial port %s: %s", self._port, exc)

    def _reader_loop(self) -> None:
        """Background loop that reads NMEA sentences from the serial port."""
        assert self._serial is not None
        while self._running:
            try:
                raw = self._serial.readline()  # type: ignore[union-attr]
                line = raw.decode("ascii", errors="replace").strip()
                if not line:
                    continue
                self._parse_nmea(line)
            except Exception as exc:
                logger.debug("GPS read error: %s", exc)
                time.sleep(0.5)

    def _parse_nmea(self, sentence: str) -> None:
        """Extract speed from ``$GPRMC`` or ``$GPVTG`` sentences."""
        parts = sentence.split(",")
        try:
            if sentence.startswith("$GPRMC") and len(parts) >= 8:
                # Field 7 = speed over ground in knots
                knots_str = parts[7]
                if knots_str:
                    speed = float(knots_str) * _KNOTS_TO_KMH
                    with self._lock:
                        self._speed_kmh = speed
            elif sentence.startswith("$GPVTG") and len(parts) >= 8:
                # Field 7 = speed in km/h (some receivers)
                kmh_str = parts[7]
                if kmh_str:
                    with self._lock:
                        self._speed_kmh = float(kmh_str)
        except (ValueError, IndexError) as exc:
            logger.debug("NMEA parse error in '%s': %s", sentence, exc)


# ---------------------------------------------------------------------------
# OBD-II adapter
# ---------------------------------------------------------------------------

class OBDSpeedSource(SpeedSource):
    """Read vehicle speed from an OBD-II adapter (PID 0x0D).

    Requires the optional ``obd`` package (``python-obd``).  The adapter
    is queried in a background thread so :meth:`read_kmh` never blocks.
    """

    _POLL_INTERVAL = 0.25  # seconds between OBD queries

    def __init__(self, port: str | None = None) -> None:
        self._port = port

        self._lock = threading.Lock()
        self._speed_kmh: float | None = None
        self._available = False
        self._connection: object | None = None  # obd.OBD when connected
        self._running = False
        self._thread: threading.Thread | None = None

        self._connect()

    # -- public interface ---------------------------------------------------

    def read_kmh(self) -> float | None:
        with self._lock:
            return self._speed_kmh

    def is_available(self) -> bool:
        with self._lock:
            return self._available

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._connection is not None:
            try:
                self._connection.close()  # type: ignore[union-attr]
            except Exception:
                pass

    # -- internals ----------------------------------------------------------

    def _connect(self) -> None:
        try:
            import obd  # python-obd
        except ImportError:
            logger.warning("python-obd is not installed – OBDSpeedSource disabled")
            return

        try:
            self._connection = obd.OBD(self._port) if self._port else obd.OBD()
            if not self._connection.is_connected():  # type: ignore[union-attr]
                logger.error("OBD-II adapter not connected")
                return
            with self._lock:
                self._available = True
            self._running = True
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
            logger.info("OBD-II connected via %s", self._port or "auto-detect")
        except Exception as exc:
            logger.error("Failed to connect OBD-II: %s", exc)

    def _poll_loop(self) -> None:
        """Background loop that queries PID 0x0D (Vehicle Speed)."""
        import obd  # guaranteed importable if we got here

        cmd = obd.commands.SPEED  # PID 0x0D
        while self._running:
            try:
                response = self._connection.query(cmd)  # type: ignore[union-attr]
                if not response.is_null():
                    with self._lock:
                        self._speed_kmh = response.value.to("kph").magnitude
            except Exception as exc:
                logger.debug("OBD query error: %s", exc)
            time.sleep(self._POLL_INTERVAL)
