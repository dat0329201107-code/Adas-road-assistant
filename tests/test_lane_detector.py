import cv2
import numpy as np

from adas.vision.lane_detector import LaneDetector, LaneResult


def test_lane_result_dataclass():
    r = LaneResult(
        left_line=None,
        right_line=None,
        center_offset_px=0.0,
        lane_detected=False,
        overlay_points=[],
    )
    assert r.lane_detected is False
    assert r.center_offset_px == 0.0


def test_no_lanes_on_blank_image():
    detector = LaneDetector()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(blank)
    assert not result.lane_detected
    assert result.left_line is None
    assert result.right_line is None


def test_detect_with_synthetic_lanes():
    detector = LaneDetector(roi_top_ratio=0.3)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw left lane line (negative slope in image coordinates: goes from (150, 470) to (280, 200))
    cv2.line(frame, (150, 470), (280, 200), (255, 255, 255), 4)
    # Draw right lane line (positive slope: goes from (360, 200) to (490, 470))
    cv2.line(frame, (490, 470), (360, 200), (255, 255, 255), 4)

    result = detector.detect(frame)
    assert isinstance(result.center_offset_px, float)
