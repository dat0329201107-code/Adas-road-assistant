"""Road Scene Classifier Service using MobileNetV3 for 5 Vietnamese Road Types."""
from __future__ import annotations

import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

CLASS_NAMES = ["cao_toc", "duong_doi_do_thi", "pho_hai_chieu", "ngoai_khu_dan_cu", "hem_ngo"]

CLASS_VI_MAP = {
    "cao_toc": "Cao tốc",
    "duong_doi_do_thi": "Đường đôi đô thị",
    "pho_hai_chieu": "Phố hai chiều",
    "ngoai_khu_dan_cu": "Ngoài khu dân cư",
    "hem_ngo": "Hẻm / ngõ",
}

CLASS_ID_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Default simulated boxes for UI demonstration when YOLO is not active
MOCK_BOXES_BY_CLASS = {
    "cao_toc": [
        {"x": 360, "y": 232, "w": 58, "h": 42, "label": "ô tô 96%", "confidence": 0.96},
        {"x": 474, "y": 206, "w": 32, "h": 24, "label": "xe tải 88%", "confidence": 0.88},
    ],
    "duong_doi_do_thi": [
        {"x": 322, "y": 240, "w": 62, "h": 44, "label": "ô tô 94%", "confidence": 0.94},
        {"x": 478, "y": 252, "w": 34, "h": 46, "label": "xe máy 89%", "confidence": 0.89},
        {"x": 438, "y": 208, "w": 26, "h": 22, "label": "ô tô 81%", "confidence": 0.81},
    ],
    "pho_hai_chieu": [
        {"x": 340, "y": 230, "w": 55, "h": 40, "label": "ô tô 93%", "confidence": 0.93},
        {"x": 460, "y": 245, "w": 30, "h": 42, "label": "xe máy 90%", "confidence": 0.90},
    ],
    "ngoai_khu_dan_cu": [
        {"x": 370, "y": 216, "w": 36, "h": 28, "label": "ô tô 92%", "confidence": 0.92},
    ],
    "hem_ngo": [
        {"x": 380, "y": 262, "w": 42, "h": 60, "label": "xe máy 87%", "confidence": 0.87},
        {"x": 306, "y": 244, "w": 22, "h": 70, "label": "người 79%", "confidence": 0.79},
    ],
}


def build_mobilenet_model(num_classes: int = 5) -> Any:
    """Build MobileNetV3-Small architecture matching the trained checkpoint."""
    if not _TORCH_AVAILABLE:
        return None
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes),
    )
    return model


class RoadClassifierService:
    """Service that handles road classification inference on CPU/GPU."""

    def __init__(self, model_path: str | Path | None = None, device: str | None = None) -> None:
        self.model_path = self._resolve_model_path(model_path)
        self.device = self._resolve_device(device)
        self.model: Any = None
        self.loaded = False
        self.load_error: str | None = None

        if _TORCH_AVAILABLE:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = None

        self.load()

    def _resolve_model_path(self, model_path: str | Path | None) -> Path | None:
        """Find model path among standard locations."""
        if model_path and Path(model_path).exists():
            return Path(model_path)

        env_path = os.getenv("MODEL_PATH")
        if env_path and Path(env_path).exists():
            return Path(env_path)

        candidates = [
            Path("models/vietnam_road_classifier.pt"),
            Path("backend/models/vietnam_road_classifier.pt"),
            Path("../models/vietnam_road_classifier.pt"),
            Path("../../models/vietnam_road_classifier.pt"),
            Path("adas-road-assistant/models/vietnam_road_classifier.pt"),
            Path(__file__).resolve().parents[2] / "models" / "vietnam_road_classifier.pt",
            Path(__file__).resolve().parents[3] / "models" / "vietnam_road_classifier.pt",
            Path(__file__).resolve().parents[4] / "models" / "vietnam_road_classifier.pt",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _resolve_device(self, device: str | None) -> str:
        """Pick optimal inference device (CUDA if available, else CPU)."""
        if device:
            return device
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load(self) -> bool:
        """Load PyTorch model weights onto target device."""
        if not _TORCH_AVAILABLE:
            self.load_error = "PyTorch is not installed"
            logger.warning("Classifier running in mock fallback mode: %s", self.load_error)
            return False

        if not self.model_path or not self.model_path.exists():
            self.load_error = f"Model checkpoint not found. Searched candidates."
            logger.warning("Classifier running in mock fallback mode: %s", self.load_error)
            return False

        try:
            logger.info("Loading Vietnam Road Classifier from %s (device=%s)", self.model_path, self.device)
            net = build_mobilenet_model(num_classes=5)
            checkpoint = torch.load(str(self.model_path), map_location=self.device, weights_only=False)
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            net.load_state_dict(state_dict)
            net.to(self.device)
            net.eval()
            self.model = net
            self.loaded = True
            logger.info("Successfully loaded Vietnam Road Classifier on %s", self.device)
            return True
        except Exception as exc:
            self.load_error = str(exc)
            logger.error("Failed to load model from %s: %s", self.model_path, exc)
            return False

    def predict_image(self, image: Image.Image | np.ndarray | bytes | str) -> dict[str, Any]:
        """Classify a road scene image into 5 Vietnamese road types.

        Accepts:
          - PIL.Image.Image
          - numpy.ndarray (BGR from OpenCV or RGB)
          - bytes (raw JPEG/PNG file bytes)
          - str (base64 data URL or pure base64)

        Returns dictionary with predicted class, Vietnamese name, confidence,
        probabilities dict, latency in ms, and candidate boxes.
        """
        t0 = time.perf_counter()

        # 1. Parse image into PIL Image in RGB
        pil_img = self._to_pil_image(image)

        # 2. If model is loaded, run real PyTorch GPU/CPU inference
        if self.loaded and self.model is not None and _TORCH_AVAILABLE:
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)  # type: ignore
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            best_idx = int(np.argmax(probs))
            conf = float(probs[best_idx])
            pred_key = CLASS_NAMES[best_idx]
            prob_dict = {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}
        else:
            # Fallback mock prediction when running without weights
            pred_key = "duong_doi_do_thi"
            best_idx = 1
            conf = 0.91
            prob_dict = {
                "cao_toc": 0.02,
                "duong_doi_do_thi": 0.91,
                "pho_hai_chieu": 0.05,
                "ngoai_khu_dan_cu": 0.01,
                "hem_ngo": 0.01,
            }

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Retrieve mock bounding boxes for UI rendering
        boxes = MOCK_BOXES_BY_CLASS.get(pred_key, [])

        return {
            "road_type": pred_key,
            "road_name_vi": CLASS_VI_MAP[pred_key],
            "road_class_id": best_idx,
            "confidence": conf,
            "confidence_percent": round(conf * 100, 1),
            "probabilities": prob_dict,
            "boxes": boxes,
            "latency_ms": round(latency_ms, 2),
        }

    @staticmethod
    def _to_pil_image(img_input: Any) -> Image.Image:
        """Convert various input formats to PIL Image in RGB format."""
        if isinstance(img_input, Image.Image):
            return img_input.convert("RGB")

        if isinstance(img_input, np.ndarray):
            # Assume BGR from OpenCV
            if img_input.ndim == 3 and img_input.shape[2] == 3:
                rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb)
            elif img_input.ndim == 2:
                return Image.fromarray(img_input).convert("RGB")
            return Image.fromarray(img_input)

        if isinstance(img_input, (bytes, bytearray)):
            return Image.open(io.BytesIO(img_input)).convert("RGB")

        if isinstance(img_input, str):
            # Base64 string
            raw_b64 = img_input
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            raw_bytes = base64.b64decode(raw_b64)
            return Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        raise ValueError(f"Unsupported image type: {type(img_input)}")
