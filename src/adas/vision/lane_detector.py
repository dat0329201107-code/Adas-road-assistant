"""Simple Hough-line lane detector for the ADAS prototype."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class LaneResult:
    """Output of a single lane-detection pass."""

    left_line: tuple[int, int, int, int] | None
    right_line: tuple[int, int, int, int] | None
    center_offset_px: float
    lane_detected: bool
    overlay_points: list[tuple[int, int]] = field(default_factory=list)


class LaneDetector:
    """Detect left / right lane lines with Canny + probabilistic Hough.

    Parameters
    ----------
    roi_top_ratio:
        Fraction of the image height where the region-of-interest begins
        (0.0 = top of image, 1.0 = bottom).
    canny_low:
        Low threshold for the Canny edge detector.
    canny_high:
        High threshold for the Canny edge detector.
    """

    def __init__(
        self,
        roi_top_ratio: float = 0.6,
        canny_low: int = 50,
        canny_high: int = 150,
    ) -> None:
        self.roi_top_ratio = roi_top_ratio
        self.canny_low = canny_low
        self.canny_high = canny_high

    def detect(self, frame: NDArray[np.uint8]) -> LaneResult:
        """Run lane detection on a BGR *frame* and return a :class:`LaneResult`."""
        h, w = frame.shape[:2]
        roi_top = int(h * self.roi_top_ratio)

        # Pre-process: grayscale -> blur -> Canny
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)

        # Mask to ROI
        mask = np.zeros_like(edges)
        roi_vertices = np.array(
            [[(0, h), (0, roi_top), (w, roi_top), (w, h)]],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, roi_vertices, 255)
        masked = cv2.bitwise_and(edges, mask)

        # Hough lines
        lines = cv2.HoughLinesP(
            masked,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=40,
            maxLineGap=100,
        )

        left_segments: list[tuple[int, int, int, int]] = []
        right_segments: list[tuple[int, int, int, int]] = []

        if lines is not None:
            for seg in lines:
                x1, y1, x2, y2 = seg[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.3:
                    continue  # nearly horizontal – skip
                if slope < 0:
                    left_segments.append((x1, y1, x2, y2))
                else:
                    right_segments.append((x1, y1, x2, y2))

        left_line = self._average_line(left_segments, h, roi_top)
        right_line = self._average_line(right_segments, h, roi_top)

        center_offset = 0.0
        overlay: list[tuple[int, int]] = []
        if left_line and right_line:
            lane_centre = (left_line[0] + right_line[0]) / 2
            center_offset = float(w / 2 - lane_centre)
            overlay = [
                (left_line[0], left_line[1]),
                (left_line[2], left_line[3]),
                (right_line[2], right_line[3]),
                (right_line[0], right_line[1]),
            ]

        return LaneResult(
            left_line=left_line,
            right_line=right_line,
            center_offset_px=center_offset,
            lane_detected=left_line is not None and right_line is not None,
            overlay_points=overlay,
        )

    @staticmethod
    def _average_line(
        segments: list[tuple[int, int, int, int]],
        img_h: int,
        roi_top: int,
    ) -> tuple[int, int, int, int] | None:
        """Fit an average line through *segments* spanning from *img_h* to *roi_top*."""
        if not segments:
            return None
        xs, ys = [], []
        for x1, y1, x2, y2 in segments:
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        try:
            poly = np.polyfit(ys, xs, 1)
        except (np.linalg.LinAlgError, ValueError):
            return None
        x_bottom = int(np.polyval(poly, img_h))
        x_top = int(np.polyval(poly, roi_top))
        return (x_bottom, img_h, x_top, roi_top)
