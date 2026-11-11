import tempfile
import time
from pathlib import Path

from adas.core.logger import FPSTracker, SessionLogger


def test_fps_tracker_initial():
    tracker = FPSTracker(window_size=10)
    assert tracker.fps == 0.0
    assert tracker.avg_frame_time_ms == 0.0


def test_fps_tracker_after_ticks():
    tracker = FPSTracker(window_size=5)
    for _ in range(6):
        tracker.tick()
        time.sleep(0.005)
    assert tracker.fps > 0.0
    assert tracker.avg_frame_time_ms > 0.0


def test_session_logger_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SessionLogger(log_dir=tmpdir, enable_console=False)
        try:
            logger.setup_logging()
            log_files = list(Path(tmpdir).glob("adas_session_*.log"))
            assert len(log_files) == 1
        finally:
            logger.close()


def test_session_logger_log_methods():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SessionLogger(log_dir=tmpdir, enable_console=False)
        try:
            logger.setup_logging()
            logger.log_frame_stats(1, 30.0, "Đường đô thị", 50.0, 50, False)
            logger.log_detection("sign", {"limit": 50, "confidence": 0.85})
            logger.log_alert("overspeed", 65.0, 50)
            assert logger.get_fps_tracker() is not None
        finally:
            logger.close()
