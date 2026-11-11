"""Session logging and FPS telemetry for ADAS runs."""
from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)


class FPSTracker:
    """Tracks frames per second with a rolling-window average.

    Call :meth:`tick` once per frame.  Read :attr:`fps` or
    :attr:`avg_frame_time_ms` at any time.
    """

    def __init__(self, window_size: int = 30) -> None:
        self._window_size = max(1, window_size)
        self._timestamps: deque[float] = deque(maxlen=self._window_size + 1)

    def tick(self) -> None:
        """Record the current timestamp (call once per frame)."""
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        """Current rolling-average frames per second."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    @property
    def avg_frame_time_ms(self) -> float:
        """Average time per frame in milliseconds."""
        current_fps = self.fps
        if current_fps <= 0:
            return 0.0
        return 1000.0 / current_fps


class SessionLogger:
    """Logs ADAS session data to a timestamped file for post-run analysis.

    Creates a log file under *log_dir* named
    ``adas_session_YYYYMMDD_HHMMSS.log``.  Optionally mirrors output to
    the console.
    """

    def __init__(
        self,
        log_dir: str | Path = "outputs/logs",
        enable_console: bool = True,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._enable_console = enable_console
        self._fps_tracker = FPSTracker()
        self._log_file: Path | None = None
        self._handlers: list[logging.Handler] = []

    # -- setup --------------------------------------------------------------

    def setup_logging(self, level: int = logging.INFO) -> None:
        """Configure the root logger with file and (optionally) console handlers."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._log_file = self._log_dir / f"adas_session_{timestamp}.log"

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        root = logging.getLogger()
        root.setLevel(level)

        file_handler = logging.FileHandler(str(self._log_file), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        self._handlers.append(file_handler)

        if self._enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(fmt)
            root.addHandler(console_handler)
            self._handlers.append(console_handler)

        logger.info("Session log: %s", self._log_file)

    def close(self) -> None:
        """Flush, close, and remove session handlers."""
        root = logging.getLogger()
        for handler in self._handlers:
            try:
                handler.flush()
                handler.close()
                root.removeHandler(handler)
            except Exception:
                pass
        self._handlers.clear()

    # -- structured log helpers --------------------------------------------

    def log_frame_stats(
        self,
        frame_index: int,
        fps: float,
        road_label: str | None,
        speed: float | None,
        limit: int | None,
        overspeed: bool,
    ) -> None:
        """Write a structured log line summarising one processed frame."""
        logger.info(
            "FRAME #%05d | FPS %.1f | road=%s | speed=%s km/h | limit=%s km/h | overspeed=%s",
            frame_index,
            fps,
            road_label or "N/A",
            f"{speed:.1f}" if speed is not None else "N/A",
            str(limit) if limit is not None else "N/A",
            overspeed,
        )

    def log_detection(self, detection_type: str, details: dict) -> None:
        """Log a sign, lane, or object detection event."""
        detail_str = " | ".join(f"{k}={v}" for k, v in details.items())
        logger.info("DETECT [%s] %s", detection_type, detail_str)

    def log_alert(self, alert_type: str, speed: float | None, limit: int | None) -> None:
        """Log an overspeed or other alert event."""
        logger.warning(
            "ALERT  [%s] speed=%s km/h limit=%s km/h",
            alert_type,
            f"{speed:.1f}" if speed is not None else "N/A",
            str(limit) if limit is not None else "N/A",
        )

    # -- accessors ----------------------------------------------------------

    def get_fps_tracker(self) -> FPSTracker:
        """Return the :class:`FPSTracker` instance owned by this session."""
        return self._fps_tracker
