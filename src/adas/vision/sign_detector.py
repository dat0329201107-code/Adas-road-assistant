"""Model-based traffic sign detection using YOLO26 (ultralytics).

Detects traffic signs in video frames and maps recognised class names to
speed-limit values where possible.  Supports both ``.pt`` and ``.onnx``
model files.  The NMS-free end-to-end design of YOLO26 provides faster
inference on edge devices with streamlined export paths.

If the ``ultralytics`` package is not installed the detector will report
itself as unavailable rather than raising.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/traffic_signs_yolo26n.pt")
DEFAULT_BILINGUAL_MAP_PATH = Path("data/datasets/vietnam_traffic_signs/classes_bilingual.json")

# ── speed-limit class-name mapping ──────────────────────────────────────
# YOLO class names such as "speed_limit_30" or "speed-limit-50" are mapped
# to the numeric km/h value.  Unknown names yield ``None``.
_SPEED_RE = re.compile(r"speed[_\-\s]*limit[_\-\s]*(\d+)", re.IGNORECASE)


def _parse_limit(class_name: str) -> int | None:
    """Extract a km/h speed-limit value from a YOLO class name, if any."""
    m = _SPEED_RE.search(class_name)
    return int(m.group(1)) if m else None


@dataclass(frozen=True)
class DetectedSign:
    """A single traffic sign detected in a frame."""

    bbox: tuple[int, int, int, int]  # x, y, width, height
    class_name: str
    confidence: float
    limit_kmh: int | None = field(default=None)
    vietnamese_name: str | None = field(default=None)


class SignDetector:
    """YOLO-based traffic sign detector with lazy model loading.

    Parameters
    ----------
    model_path:
        Path to a YOLO26 ``.pt`` or ``.onnx`` model trained on traffic
        signs.  *None* uses :data:`DEFAULT_MODEL_PATH`.
    confidence:
        Minimum detection confidence threshold.
    bilingual_map_path:
        Path to JSON mapping class ID/name to Vietnamese translations.
    """

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float = 0.4,
        bilingual_map_path: str | Path | None = None,
    ) -> None:
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._confidence = confidence
        self._bilingual_path = Path(bilingual_map_path) if bilingual_map_path else DEFAULT_BILINGUAL_MAP_PATH
        self._bilingual_map: dict[str, str] = {}
        self._load_bilingual_map()
        self._model: Any | None = None
        self._load_error: str | None = None

    def _load_bilingual_map(self) -> None:
        """Load Vietnamese translations for sign classes if file exists."""
        if not self._bilingual_path.exists():
            return
        try:
            with open(self._bilingual_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Map both class ID and English name (lowercase) to Vietnamese
            for cid, info in data.items():
                if isinstance(info, dict):
                    vi = info.get("vi", "")
                    en = info.get("en", "").lower()
                    if vi:
                        self._bilingual_map[str(cid)] = vi
                        if en:
                            self._bilingual_map[en] = vi
            logger.debug("Loaded %d bilingual sign translations", len(self._bilingual_map))
        except Exception as exc:
            logger.warning("Could not load bilingual map from %s: %s", self._bilingual_path, exc)

    def get_vietnamese_name(self, cls_id: int | str, class_name: str) -> str | None:
        """Look up the Vietnamese name for a detected sign."""
        if str(cls_id) in self._bilingual_map:
            return self._bilingual_map[str(cls_id)]
        return self._bilingual_map.get(class_name.lower())

    # ── properties ──────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """``True`` when the model has been loaded successfully."""
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        """Human-readable error message if loading failed, else *None*."""
        return self._load_error

    # ── loading ─────────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load the YOLO model on demand.

        Returns ``True`` on success.  Subsequent calls are no-ops if the
        model has already been loaded (or a previous attempt failed).
        """
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]

            logger.info("Loading sign-detection model from %s", self._model_path)
            self._model = YOLO(str(self._model_path))
            return True
        except ImportError:
            self._load_error = (
                "ultralytics is not installed – run `pip install ultralytics`"
            )
            logger.warning(self._load_error)
            return False
        except Exception as exc:
            self._load_error = str(exc)
            logger.exception("Could not load sign-detection model")
            return False

    # ── inference ───────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedSign]:
        """Run YOLO inference on *frame_bgr* and return detected signs.

        Returns an empty list when the model is unavailable or inference
        fails.
        """
        if not self.load():
            return []
        try:
            results = self._model(frame_bgr, conf=self._confidence, verbose=False)  # type: ignore[misc]
            detections: list[DetectedSign] = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    w, h = int(x2 - x1), int(y2 - y1)
                    cls_id = int(box.cls[0])
                    class_name = result.names.get(cls_id, str(cls_id))
                    conf = float(box.conf[0])
                    vn_name = self.get_vietnamese_name(cls_id, class_name)
                    detections.append(
                        DetectedSign(
                            bbox=(int(x1), int(y1), w, h),
                            class_name=class_name,
                            confidence=conf,
                            limit_kmh=_parse_limit(class_name),
                            vietnamese_name=vn_name,
                        )
                    )
            # Largest detections first (closer to camera).
            detections.sort(
                key=lambda d: d.bbox[2] * d.bbox[3], reverse=True
            )
            return detections
        except Exception:
            logger.exception("Sign-detection inference failed")
            return []
