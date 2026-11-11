"""Candidate detection of red circular speed-limit signs with optional Tesseract OCR.

This is a prototype heuristic, not a validated traffic-sign detector. Small, distant,
blurred, dirty, non-red or partially occluded signs may be missed, and red circular
objects can create false positives. OCR output should be confirmed before real-world use.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import pytesseract
except ImportError:  # OCR is optional; the visual candidate detector can still run.
    pytesseract = None


@dataclass(frozen=True)
class SignCandidate:
    bbox: tuple[int, int, int, int]  # x, y, width, height
    limit_kmh: int | None
    ocr_text: str = ""


class SpeedSignRecognizer:
    def __init__(self, enable_ocr: bool = True, tesseract_cmd: str | None = None) -> None:
        self.enable_ocr = enable_ocr and pytesseract is not None
        self.ocr_error: str | None = None
        if self.enable_ocr:
            binary = tesseract_cmd or os.environ.get("TESSERACT_CMD")
            if binary:
                pytesseract.pytesseract.tesseract_cmd = binary
            try:
                # Accessing tesseract_version verifies that the system executable exists.
                pytesseract.get_tesseract_version()
            except Exception as exc:
                self.enable_ocr = False
                self.ocr_error = str(exc)

    @staticmethod
    def _red_mask(frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        low_red = cv2.inRange(hsv, np.array([0, 75, 45]), np.array([12, 255, 255]))
        high_red = cv2.inRange(hsv, np.array([168, 75, 45]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(low_red, high_red)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    @staticmethod
    def _looks_like_sign(roi_bgr: np.ndarray, red_mask: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        if w < 22 or h < 22 or w > 0.35 * red_mask.shape[1] or h > 0.5 * red_mask.shape[0]:
            return False
        ratio = w / max(h, 1)
        if not 0.70 <= ratio <= 1.30:
            return False
        contour_mask = red_mask[y:y + h, x:x + w]
        if contour_mask.size == 0:
            return False
        red_fraction = float(np.count_nonzero(contour_mask)) / float(contour_mask.size)
        if not 0.08 <= red_fraction <= 0.72:
            return False
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        # A typical speed-limit sign has a low-saturation light center surrounded by red.
        center = hsv[int(h * 0.22):max(int(h * 0.78), 1), int(w * 0.22):max(int(w * 0.78), 1)]
        if center.size == 0:
            return False
        light_center = (center[:, :, 2] > 125) & (center[:, :, 1] < 130)
        return float(np.mean(light_center)) >= 0.12

    def _read_digits(self, roi_bgr: np.ndarray) -> tuple[int | None, str]:
        if not self.enable_ocr:
            return None, ""
        try:
            # Crop toward the center to reduce influence from the red border.
            h, w = roi_bgr.shape[:2]
            inner = roi_bgr[int(h * 0.16):int(h * 0.84), int(w * 0.16):int(w * 0.84)]
            if inner.size == 0:
                inner = roi_bgr
            gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            config = "--psm 8 -c tessedit_char_whitelist=0123456789"
            raw = pytesseract.image_to_string(binary, config=config).strip()  # type: ignore[union-attr]
            digits = "".join(ch for ch in raw if ch.isdigit())
            if not digits:
                return None, raw
            value = int(digits)
            # Broad range for detector filtering only; applicable limits depend on local law/signage.
            if 5 <= value <= 130:
                return value, raw
            return None, raw
        except Exception as exc:
            self.ocr_error = str(exc)
            return None, ""

    def detect(self, frame_bgr: np.ndarray) -> list[SignCandidate]:
        mask = self._red_mask(frame_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[SignCandidate] = []
        frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max(120.0, frame_area * 0.000025):
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            if circularity < 0.42:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame_bgr[y:y + h, x:x + w]
            if not self._looks_like_sign(roi, mask, x, y, w, h):
                continue
            limit, text = self._read_digits(roi)
            candidates.append(SignCandidate((x, y, w, h), limit, text))
        # Put the largest candidates first; likely nearer to the camera and easier to read.
        candidates.sort(key=lambda item: item.bbox[2] * item.bbox[3], reverse=True)
        return candidates[:8]
