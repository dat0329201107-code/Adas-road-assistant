"""Zero-shot road-scene classification using CLIP through Transformers.

This describes visual scene context. It does NOT determine the legal speed limit.
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
from PIL import Image

from adas.config import ROAD_PROMPTS

logger = logging.getLogger(__name__)


class RoadSceneClassifier:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self.model_name = model_name
        self._pipeline: Any | None = None
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        return self._pipeline is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> bool:
        """Load model on demand. The weights are downloaded and cached on first use."""
        if self._pipeline is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() else -1
            logger.info("Loading road-scene model %s (device=%s)", self.model_name, device)
            self._pipeline = pipeline(
                task="zero-shot-image-classification",
                model=self.model_name,
                device=device,
            )
            return True
        except Exception as exc:  # Dependency/model download failures should not crash video playback.
            self._load_error = str(exc)
            logger.exception("Could not load the road-scene model")
            return False

    def classify(self, frame_bgr) -> dict[str, object] | None:
        """Return {'label', 'score', 'prompt'} or None when inference is unavailable."""
        if not self.load():
            return None
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            # Keep inference cost bounded for large dash-camera frames.
            image.thumbnail((640, 384))
            results = self._pipeline(image, candidate_labels=list(ROAD_PROMPTS.keys()), hypothesis_template="{}")
            if not results:
                return None
            best = results[0]
            prompt = str(best["label"])
            return {
                "label": ROAD_PROMPTS.get(prompt, "Chưa xác định"),
                "score": float(best.get("score", 0.0)),
                "prompt": prompt,
            }
        except Exception:
            logger.exception("Road-scene inference failed")
            return None
