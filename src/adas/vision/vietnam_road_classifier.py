"""Deep learning road scene classifier trained on 5 Vietnamese road types.

Based on the 5-class Vietnamese Road Specification:
1. Cao tốc (cao_toc): Đường nhiều làn, có dải phân cách, xe chạy nhanh. Tốc độ tối đa: 100 km/h (hoặc 120 km/h).
2. Đường đôi đô thị (duong_doi_do_thi): Hai chiều xe tách riêng bằng dải phân cách, trong đô thị. Tốc độ tối đa: 60 km/h.
3. Phố hai chiều (pho_hai_chieu): Đường phố trong khu dân cư, xe hai chiều đi chung một mặt đường. Tốc độ tối đa: 50 km/h.
4. Ngoài khu dân cư (ngoai_khu_dan_cu): Đường nằm ngoài khu đông dân cư, hai bên ít nhà dân. Tốc độ tối đa: 80 km/h.
5. Hẻm / ngõ (hem_ngo): Đường nhỏ, hẹp trong khu dân cư, cần đi rất chậm. Tốc độ tối đa an toàn: 30 km/h.
*. Không xác định (khong_xac_dinh): Ảnh khó phân biệt / độ tin cậy thấp, để mô hình không bị ép đoán.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# Legal and standard limits per Vietnamese Road Traffic Law (Circular 31/2019/TT-BGTVT)
VIETNAM_ROAD_RULES = {
    "cao_toc": {
        "id": 1,
        "class_key": "cao_toc",
        "name_vi": "Cao tốc",
        "legal_max_speed": 100,
        "description": "Đường nhiều làn, có dải phân cách, xe chạy nhanh và liên tục.",
    },
    "duong_doi_do_thi": {
        "id": 2,
        "class_key": "duong_doi_do_thi",
        "name_vi": "Đường đôi đô thị",
        "legal_max_speed": 60,
        "description": "Hai chiều xe tách riêng bằng dải phân cách, nằm trong khu đô thị.",
    },
    "pho_hai_chieu": {
        "id": 3,
        "class_key": "pho_hai_chieu",
        "name_vi": "Phố hai chiều",
        "legal_max_speed": 50,
        "description": "Đường phố trong khu dân cư, xe hai chiều đi chung một mặt đường.",
    },
    "ngoai_khu_dan_cu": {
        "id": 4,
        "class_key": "ngoai_khu_dan_cu",
        "name_vi": "Ngoài khu dân cư",
        "legal_max_speed": 80,
        "description": "Đường nằm ngoài khu đông dân cư, hai bên ít nhà dân.",
    },
    "hem_ngo": {
        "id": 5,
        "class_key": "hem_ngo",
        "name_vi": "Hẻm / ngõ",
        "legal_max_speed": 30,
        "description": "Đường nhỏ, hẹp trong khu dân cư, cần đi rất chậm.",
    },
    "khong_xac_dinh": {
        "id": 0,
        "class_key": "khong_xac_dinh",
        "name_vi": "Không xác định",
        "legal_max_speed": None,
        "description": "Ảnh khó phân biệt được gán nhãn không xác định.",
    },
}

CLASS_KEYS = ["cao_toc", "duong_doi_do_thi", "pho_hai_chieu", "ngoai_khu_dan_cu", "hem_ngo"]


def build_model(num_classes: int = 5) -> nn.Module:
    """Build MobileNetV3 architecture matching the trained weights."""
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes),
    )
    return model


class VietnamRoadClassifier:
    """Classifies video frames into 5 Vietnamese road types and determines legal speed limits."""

    def __init__(
        self,
        model_path: str | Path = "models/vietnam_road_classifier.pt",
        confidence_threshold: float = 0.40,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._model: nn.Module | None = None
        self._load_error: str | None = None
        self._classes = CLASS_KEYS

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> bool:
        """Load trained weights if available."""
        if self._model is not None:
            return True

        actual_path = self.model_path
        if not actual_path.exists():
            candidates = [
                Path("adas-road-assistant") / self.model_path,
                Path(__file__).resolve().parents[3] / self.model_path,
                Path(__file__).resolve().parents[2] / self.model_path,
                Path(__file__).resolve().parents[3] / "models" / self.model_path.name,
            ]
            for c in candidates:
                if c.exists():
                    actual_path = c
                    break

        if not actual_path.exists():
            self._load_error = f"Model weights not found at {self.model_path}"
            logger.warning(self._load_error)
            return False

        try:
            logger.info("Loading Vietnam Road Classifier from %s (device=%s)", actual_path, self.device)
            model = build_model(num_classes=len(self._classes))
            checkpoint = torch.load(actual_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
                if "classes" in checkpoint:
                    self._classes = checkpoint["classes"]
            else:
                model.load_state_dict(checkpoint)

            model.to(self.device)
            model.eval()
            self._model = model
            logger.info("Vietnam Road Classifier loaded successfully on %s", self.device)
            return True
        except Exception as exc:
            self._load_error = str(exc)
            logger.exception("Failed to load Vietnam Road Classifier weights")
            return False

    def classify(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        """Classify a single BGR frame.

        Returns:
            dict containing:
                - class_key: e.g. 'duong_doi_do_thi'
                - label: e.g. 'Đường đôi đô thị'
                - score: float confidence (0.0 to 1.0)
                - legal_max_speed: int km/h (e.g. 60)
                - description: description string
                - is_confident: bool (False if below threshold -> 'Không xác định')
        """
        if not self.load() or self._model is None:
            rule = VIETNAM_ROAD_RULES["khong_xac_dinh"]
            return {
                "class_key": "khong_xac_dinh",
                "label": rule["name_vi"],
                "score": 0.0,
                "legal_max_speed": None,
                "description": rule["description"],
                "is_confident": False,
            }

        try:
            # Convert BGR OpenCV image to PIL Image
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self._model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            best_idx = int(np.argmax(probs))
            score = float(probs[best_idx])
            best_key = self._classes[best_idx]

            # Rule for ambiguous / low confidence images: tag as "Không xác định"
            if score < self.confidence_threshold:
                rule = VIETNAM_ROAD_RULES["khong_xac_dinh"]
                return {
                    "class_key": "khong_xac_dinh",
                    "label": rule["name_vi"],
                    "score": score,
                    "legal_max_speed": None,
                    "description": rule["description"],
                    "is_confident": False,
                    "probabilities": {k: float(p) for k, p in zip(self._classes, probs)},
                }

            rule = VIETNAM_ROAD_RULES.get(best_key, VIETNAM_ROAD_RULES["khong_xac_dinh"])
            return {
                "class_key": best_key,
                "label": rule["name_vi"],
                "score": score,
                "legal_max_speed": rule["legal_max_speed"],
                "description": rule["description"],
                "is_confident": True,
                "probabilities": {k: float(p) for k, p in zip(self._classes, probs)},
            }
        except Exception:
            logger.exception("Inference error in Vietnam Road Classifier")
            rule = VIETNAM_ROAD_RULES["khong_xac_dinh"]
            return {
                "class_key": "khong_xac_dinh",
                "label": rule["name_vi"],
                "score": 0.0,
                "legal_max_speed": None,
                "description": rule["description"],
                "is_confident": False,
            }
