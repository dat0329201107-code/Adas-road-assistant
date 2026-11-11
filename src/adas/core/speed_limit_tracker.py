"""Tracks OCR/YOLO sign candidates and resolves the limit source shown by the UI."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ActiveSpeedLimit:
    value_kmh: int | None
    source: str
    confirmed: bool


class SpeedLimitTracker:
    """Require repeated sign hits before accepting a sign; expire detections after a TTL.

    Priority hierarchy:
    1. AI/OCR Speed Sign (highest priority, legally overrides general road type rules).
    2. Explicit manual configuration (if set by user).
    3. Road type legal speed limit (Circular 31/2019/TT-BGTVT for Vietnam roads).
    4. Unknown fallback.
    """

    def __init__(
        self,
        manual_limit: int | None = None,
        sign_ttl_seconds: float = 45.0,
        apply_road_limit: bool = True,
    ) -> None:
        if manual_limit is not None and not 5 <= manual_limit <= 130:
            raise ValueError("manual_limit must be between 5 and 130 km/h")
        self.manual_limit = manual_limit
        self.sign_ttl_seconds = sign_ttl_seconds
        self.apply_road_limit = apply_road_limit

        self._hits: deque[tuple[float, int]] = deque(maxlen=10)
        self._confirmed_sign_limit: int | None = None
        self._confirmed_at: float | None = None

        self._road_limit: int | None = None
        self._road_name: str | None = None
        self._road_updated_at: float | None = None

    def observe(self, candidate_limits: list[int], now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        # Keep only recent hits to avoid carrying old evidence into a new scene.
        while self._hits and now - self._hits[0][0] > 5.0:
            self._hits.popleft()
        # Duplicate boxes in one frame count as one observation, not repeated evidence.
        for value in set(candidate_limits):
            if 5 <= value <= 130:
                self._hits.append((now, int(value)))
        counts: dict[int, int] = {}
        for _, value in self._hits:
            counts[value] = counts.get(value, 0) + 1
        confirmed = [value for value, count in counts.items() if count >= 2]
        if confirmed:
            # Most frequently repeated candidate wins; ties use the latest candidate.
            winner = max(confirmed, key=lambda value: (counts[value], max(t for t, v in self._hits if v == value)))
            self._confirmed_sign_limit = winner
            self._confirmed_at = now

    def observe_road_limit(
        self,
        road_max_speed: int | None,
        road_name: str | None = None,
        now: float | None = None,
    ) -> None:
        """Update active legal speed limit derived from recognized road type."""
        self._road_limit = road_max_speed
        self._road_name = road_name
        self._road_updated_at = time.monotonic() if now is None else now

    def active(self, now: float | None = None) -> ActiveSpeedLimit:
        now = time.monotonic() if now is None else now

        # Priority 1: Speed sign detected on road
        if self._confirmed_sign_limit is not None and self._confirmed_at is not None:
            if now - self._confirmed_at <= self.sign_ttl_seconds:
                return ActiveSpeedLimit(self._confirmed_sign_limit, "BIỂN TỐC ĐỘ (AI)", True)

        # Priority 2: Manual fallback configured explicitly
        if self.manual_limit is not None:
            return ActiveSpeedLimit(self.manual_limit, "GIỚI HẠN CẤU HÌNH THỦ CÔNG", False)

        # Priority 3: Legal speed limit based on recognized road type (Circular 31/2019/TT-BGTVT)
        if self.apply_road_limit and self._road_limit is not None:
            source_desc = f"LOẠI ĐƯỜNG: {self._road_name}" if self._road_name else "QUY ĐỊNH LOẠI ĐƯỜNG"
            return ActiveSpeedLimit(self._road_limit, source_desc, True)

        return ActiveSpeedLimit(None, "CHƯA XÁC ĐỊNH", False)

    def reset_detected_sign(self) -> None:
        self._hits.clear()
        self._confirmed_sign_limit = None
        self._confirmed_at = None
