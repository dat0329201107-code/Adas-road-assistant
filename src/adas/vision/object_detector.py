"""General object detection (vehicles, pedestrians) using YOLO26.

Uses a standard COCO-pretrained YOLO26n model to detect road-relevant
objects such as cars, trucks, buses, motorcycles, bicycles, and
pedestrians.  YOLO26's NMS-free end-to-end design and DFL-free
detection head provide up to 43% faster CPU inference compared to
older YOLO generations.  An optional rough distance estimate is
computed from the bounding-box height using a simplified pinhole
camera model.

If the ``ultralytics`` package is not installed the detector reports
itself as unavailable rather than raising.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/yolo26n.pt")

# Default COCO classes relevant to driving scenarios.
_DEFAULT_TARGET_CLASSES: list[str] = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "person",
]

# ── rough distance estimation constants ─────────────────────────────────
# These are ballpark values for a generic dash-cam.  Override via
# ``focal_length_px`` and ``real_heights`` for better accuracy.
_DEFAULT_FOCAL_LENGTH_PX: float = 800.0
_DEFAULT_REAL_HEIGHTS: dict[str, float] = {
    "car": 1.5,
    "truck": 3.0,
    "bus": 3.2,
    "motorcycle": 1.2,
    "bicycle": 1.1,
    "person": 1.7,
}


@dataclass(frozen=True)
class DetectedObject:
    """A single object detected in a frame."""

    bbox: tuple[int, int, int, int]  # x, y, width, height
    class_name: str
    confidence: float
    estimated_distance_m: float | None = field(default=None)


class ObjectDetector:
    """YOLO-based general object detector with lazy model loading.

    Parameters
    ----------
    model_path:
        Path to a YOLO26 ``.pt`` or ``.onnx`` model.  *None* uses
        :data:`DEFAULT_MODEL_PATH`.
    confidence:
        Minimum detection confidence threshold.
    target_classes:
        COCO class names to keep.  *None* uses the default driving-related
        subset.
    focal_length_px:
        Approximate focal length in pixels for the pinhole distance model.
    real_heights:
        Mapping from class name to estimated real-world height in metres.
    """

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float = 0.35,
        target_classes: list[str] | None = None,
        focal_length_px: float = _DEFAULT_FOCAL_LENGTH_PX,
        real_heights: dict[str, float] | None = None,
    ) -> None:
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._confidence = confidence
        self._target_classes = [
            c.lower() for c in (target_classes or _DEFAULT_TARGET_CLASSES)
        ]
        self._focal_length_px = focal_length_px
        self._real_heights = real_heights or dict(_DEFAULT_REAL_HEIGHTS)
        self._model: Any | None = None
        self._load_error: str | None = None

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

            logger.info("Loading object-detection model from %s", self._model_path)
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
            logger.exception("Could not load object-detection model")
            return False

    # ── distance estimation ─────────────────────────────────────────────

    def _estimate_distance(self, class_name: str, bbox_height: int) -> float | None:
        """Rough distance from a pinhole camera model: d = f·H / h.

        Parameters
        ----------
        class_name:
            Detected class (used to look up real-world height).
        bbox_height:
            Height of the bounding box in pixels.

        Returns
        -------
        Estimated distance in metres, or *None* if the class height is
        unknown or the bbox is degenerate.
        """
        real_h = self._real_heights.get(class_name.lower())
        if real_h is None or bbox_height <= 0:
            return None
        return round(self._focal_length_px * real_h / bbox_height, 1)

    # ── inference ───────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedObject]:
        """Run YOLO inference on *frame_bgr* and return detected objects.

        Only objects whose class name appears in ``target_classes`` are
        returned.  Results are sorted by bounding-box area (largest first).
        Returns an empty list when the model is unavailable or inference
        fails.
        """
        if not self.load():
            return []
        try:
            results = self._model(frame_bgr, conf=self._confidence, verbose=False)  # type: ignore[misc]
            detections: list[DetectedObject] = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = result.names.get(cls_id, str(cls_id))
                    if class_name.lower() not in self._target_classes:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    w, h = int(x2 - x1), int(y2 - y1)
                    conf = float(box.conf[0])
                    distance = self._estimate_distance(class_name, h)
                    detections.append(
                        DetectedObject(
                            bbox=(int(x1), int(y1), w, h),
                            class_name=class_name,
                            confidence=conf,
                            estimated_distance_m=distance,
                        )
                    )
            # Largest detections first (closer to camera).
            detections.sort(
                key=lambda d: d.bbox[2] * d.bbox[3], reverse=True
            )
            return detections
        except Exception:
            logger.exception("Object-detection inference failed")
            return []
