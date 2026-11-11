"""Temporal Smoothing Service across video frames to prevent label flicker."""
from __future__ import annotations

from collections import deque
from typing import Any
import numpy as np

CLASS_NAMES = ["cao_toc", "duong_doi_do_thi", "pho_hai_chieu", "ngoai_khu_dan_cu", "hem_ngo"]
CLASS_VI_MAP = {
    "cao_toc": "Cao tốc",
    "duong_doi_do_thi": "Đường đôi đô thị",
    "pho_hai_chieu": "Phố hai chiều",
    "ngoai_khu_dan_cu": "Ngoài khu dân cư",
    "hem_ngo": "Hẻm / ngõ",
}


class TemporalSmoother:
    """Maintains a rolling window of predictions and applies exponential moving average (EMA)
    smoothing over class probability distributions to eliminate flicker.
    """

    def __init__(self, window_size: int = 7, alpha: float = 0.40, switch_threshold: float = 0.15) -> None:
        self.window_size = max(1, int(window_size))
        self.alpha = float(alpha)  # Weight for current frame (0 < alpha <= 1)
        self.switch_threshold = float(switch_threshold)

        self.history: deque[dict[str, Any]] = deque(maxlen=self.window_size)
        self.ema_probs: np.ndarray | None = None
        self.current_class: str | None = None
        self.frame_count = 0

    def update(self, prediction: dict[str, Any]) -> dict[str, Any]:
        """Update smoother with a new frame prediction and return stabilized result."""
        self.frame_count += 1
        self.history.append(prediction)

        # Extract current probability vector
        prob_dict = prediction.get("probabilities", {})
        curr_vec = np.array([prob_dict.get(c, 0.0) for c in CLASS_NAMES], dtype=np.float32)

        # Normalize in case probabilities don't sum to exactly 1.0
        s = curr_vec.sum()
        if s > 0:
            curr_vec /= s

        # Update Exponential Moving Average (EMA)
        if self.ema_probs is None:
            self.ema_probs = curr_vec
            self.current_class = prediction["road_type"]
        else:
            self.ema_probs = self.alpha * curr_vec + (1.0 - self.alpha) * self.ema_probs

        best_idx = int(np.argmax(self.ema_probs))
        best_class = CLASS_NAMES[best_idx]
        best_prob = float(self.ema_probs[best_idx])

        # Hysteresis: only switch if the new class probability is clearly higher
        if self.current_class is not None and best_class != self.current_class:
            curr_idx = CLASS_NAMES.index(self.current_class)
            curr_prob = float(self.ema_probs[curr_idx])
            if best_prob > curr_prob + self.switch_threshold or self.frame_count < self.window_size:
                self.current_class = best_class
        else:
            self.current_class = best_class

        active_idx = CLASS_NAMES.index(self.current_class)
        active_prob = float(self.ema_probs[active_idx])

        smoothed_probs = {c: float(self.ema_probs[i]) for i, c in enumerate(CLASS_NAMES)}

        result = dict(prediction)
        result["road_type"] = self.current_class
        result["road_name_vi"] = CLASS_VI_MAP[self.current_class]
        result["road_class_id"] = active_idx
        result["confidence"] = active_prob
        result["confidence_percent"] = round(active_prob * 100, 1)
        result["probabilities"] = smoothed_probs
        result["smoothed"] = True
        return result

    def reset(self) -> None:
        """Reset historical buffer (e.g. when switching video sources)."""
        self.history.clear()
        self.ema_probs = None
        self.current_class = None
        self.frame_count = 0
