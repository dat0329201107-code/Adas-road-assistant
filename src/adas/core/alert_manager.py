"""Four-state alert decision logic and explainable warning system for ADAS Road Assistant.

Implements the 4 distinct alert states defined by VisionDrive AI:
1. Đang xác định (Determining / Low confidence / Warmup):
   Chưa đủ khung hình hoặc độ tin cậy còn thấp, chưa đưa ra cảnh báo.
2. An toàn (Safe):
   Tốc độ nằm trong giới hạn của loại đường hiện tại.
3. Chú ý (Caution):
   Tốc độ gần chạm giới hạn; ngưỡng "chú ý" có thể chỉnh trong cài đặt (mặc định 5 km/h).
4. Vượt tốc (Overspeed):
   Tốc độ vượt giới hạn; cảnh báo mạnh kèm câu giải thích rõ ràng.

Mỗi cảnh báo đều có lời giải thích:
Ví dụ: "Đang ở Phố hai chiều (độ tin cậy 92%), giới hạn 50 km/h." Người dùng biết vì sao hệ thống cảnh báo.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class AlertStatus(str, Enum):
    """The 4 Alert States defined for ADAS Road Assistant."""

    DANG_XAC_DINH = "ĐANG XÁC ĐỊNH"
    AN_TOAN = "AN TOÀN"
    CHU_Y = "CHÚ Ý"
    VUOT_TOC = "VƯỢT TỐC"


# Color palettes (BGR for OpenCV and RGB for Pillow/UI)
ALERT_COLORS_BGR = {
    AlertStatus.DANG_XAC_DINH: (170, 160, 150),  # Neutral slate/gray
    AlertStatus.AN_TOAN: (70, 215, 80),          # Fresh emerald green
    AlertStatus.CHU_Y: (30, 200, 255),           # Vivid amber / warning yellow
    AlertStatus.VUOT_TOC: (40, 40, 245),         # High-visibility crimson red
}

ALERT_COLORS_RGB = {
    AlertStatus.DANG_XAC_DINH: (150, 160, 170),
    AlertStatus.AN_TOAN: (80, 215, 70),
    AlertStatus.CHU_Y: (255, 200, 30),
    AlertStatus.VUOT_TOC: (245, 40, 40),
}


@dataclass
class AlertDecision:
    """Rich decision outcome containing alert status, explainable text, and UI styling."""

    status: AlertStatus
    status_text: str
    explanation: str
    speed_kmh: float | None
    limit_kmh: int | None
    diff_kmh: float | None
    color_bgr: tuple[int, int, int]
    color_rgb: tuple[int, int, int]
    is_overspeed: bool
    is_caution: bool
    is_safe: bool
    is_determining: bool
    trigger_sound: bool
    sound_severity: str  # "none", "caution", "alarm"


class FourStateAlertManager:
    """Manages evaluation of the 4 alert states with explainable messages."""

    def __init__(
        self,
        buffer_kmh: float = 0.0,
        confirm_seconds: float = 0.5,
        caution_margin: float = 5.0,
        min_confidence: float = 0.60,
        min_warmup_frames: int = 5,
    ) -> None:
        self.buffer_kmh = max(0.0, float(buffer_kmh))
        self.confirm_seconds = max(0.0, float(confirm_seconds))
        self.caution_margin = max(0.0, float(caution_margin))
        self.min_confidence = max(0.0, float(min_confidence))
        self.min_warmup_frames = max(1, int(min_warmup_frames))

        self._over_since: float | None = None
        self._caution_since: float | None = None
        self.is_over = False
        self.last_decision: AlertDecision | None = None

    def decide(
        self,
        speed_kmh: float | None,
        limit_kmh: int | None,
        road_name: str | None = None,
        road_confidence: float | None = None,
        limit_source: str | None = None,
        frame_index: int = 10,
        now: float | None = None,
    ) -> AlertDecision:
        """Evaluate scene data and produce a comprehensive AlertDecision."""
        now = time.monotonic() if now is None else now

        confidence_val = float(road_confidence) if road_confidence is not None else 0.0
        conf_pct = round(confidence_val * 100, 1)
        r_name = road_name if road_name and road_name != "Không xác định" else "Đang nhận diện"

        # -------------------------------------------------------------
        # STATE 1: ĐANG XÁC ĐỊNH
        # Điều kiện: Chưa đủ khung hình hoặc độ tin cậy còn thấp / chưa có giới hạn
        # -------------------------------------------------------------
        is_determining = False
        determining_reason = ""

        if frame_index < self.min_warmup_frames:
            is_determining = True
            determining_reason = f"Đang khởi động camera ({frame_index}/{self.min_warmup_frames} frames)."
        elif road_confidence is not None and confidence_val < self.min_confidence:
            is_determining = True
            determining_reason = (
                f"Đang ở {r_name} (độ tin cậy thấp: {conf_pct:.0f}% < {self.min_confidence * 100:.0f}%)."
            )
        elif limit_kmh is None:
            is_determining = True
            determining_reason = f"Đang ở {r_name} ({conf_pct:.0f}%). Chưa xác định được giới hạn tốc độ."
        elif speed_kmh is None:
            is_determining = True
            determining_reason = f"Đang ở {r_name} (độ tin cậy {conf_pct:.0f}%). Chưa nhận được tín hiệu tốc độ xe."

        if is_determining:
            self._over_since = None
            self.is_over = False
            explanation = (
                f"{determining_reason} Chưa đủ điều kiện đưa ra cảnh báo. Lái xe thận trọng."
            )
            decision = AlertDecision(
                status=AlertStatus.DANG_XAC_DINH,
                status_text="Đang xác định",
                explanation=explanation,
                speed_kmh=speed_kmh,
                limit_kmh=limit_kmh,
                diff_kmh=None,
                color_bgr=ALERT_COLORS_BGR[AlertStatus.DANG_XAC_DINH],
                color_rgb=ALERT_COLORS_RGB[AlertStatus.DANG_XAC_DINH],
                is_overspeed=False,
                is_caution=False,
                is_safe=False,
                is_determining=True,
                trigger_sound=False,
                sound_severity="none",
            )
            self.last_decision = decision
            return decision

        # -------------------------------------------------------------
        # SPEED EVALUATION WITH CONFIRMED LIMIT
        # -------------------------------------------------------------
        diff = speed_kmh - limit_kmh
        source_note = f" [{limit_source}]" if limit_source else ""

        # STATE 4: VƯỢT TỐC
        # Tốc độ vượt giới hạn; cảnh báo mạnh kèm câu giải thích
        if speed_kmh > limit_kmh + self.buffer_kmh:
            if self._over_since is None:
                self._over_since = now
            self.is_over = now - self._over_since >= self.confirm_seconds

            over_diff = speed_kmh - limit_kmh
            explanation = (
                f"Đang ở {r_name} (độ tin cậy {conf_pct:.0f}%), giới hạn {limit_kmh} km/h{source_note}. "
                f"Tốc độ {speed_kmh:.0f} km/h VƯỢT QUÁ +{over_diff:.0f} km/h! Cảnh báo nguy hiểm, giảm tốc ngay!"
            )
            decision = AlertDecision(
                status=AlertStatus.VUOT_TOC,
                status_text="Vượt tốc",
                explanation=explanation,
                speed_kmh=speed_kmh,
                limit_kmh=limit_kmh,
                diff_kmh=over_diff,
                color_bgr=ALERT_COLORS_BGR[AlertStatus.VUOT_TOC],
                color_rgb=ALERT_COLORS_RGB[AlertStatus.VUOT_TOC],
                is_overspeed=self.is_over,
                is_caution=False,
                is_safe=False,
                is_determining=False,
                trigger_sound=self.is_over,
                sound_severity="alarm",
            )
            self.last_decision = decision
            return decision

        # Reset overspeed timer if speed dropped below limit
        self._over_since = None
        self.is_over = False

        # STATE 3: CHÚ Ý
        # Tốc độ gần chạm giới hạn; ngưỡng "chú ý" có thể chỉnh trong cài đặt
        if speed_kmh >= limit_kmh - self.caution_margin:
            remaining = limit_kmh - speed_kmh
            explanation = (
                f"Đang ở {r_name} (độ tin cậy {conf_pct:.0f}%), giới hạn {limit_kmh} km/h{source_note}. "
                f"Tốc độ {speed_kmh:.0f} km/h gần chạm giới hạn (còn {remaining:.0f} km/h). Chú ý duy trì tốc độ."
            )
            decision = AlertDecision(
                status=AlertStatus.CHU_Y,
                status_text="Chú ý",
                explanation=explanation,
                speed_kmh=speed_kmh,
                limit_kmh=limit_kmh,
                diff_kmh=diff,
                color_bgr=ALERT_COLORS_BGR[AlertStatus.CHU_Y],
                color_rgb=ALERT_COLORS_RGB[AlertStatus.CHU_Y],
                is_overspeed=False,
                is_caution=True,
                is_safe=False,
                is_determining=False,
                trigger_sound=True,
                sound_severity="caution",
            )
            self.last_decision = decision
            return decision

        # STATE 2: AN TOÀN
        # Tốc độ nằm trong giới hạn của loại đường hiện tại
        explanation = (
            f"Đang ở {r_name} (độ tin cậy {conf_pct:.0f}%), giới hạn {limit_kmh} km/h{source_note}. "
            f"Tốc độ an toàn ({speed_kmh:.0f} km/h nằm trong giới hạn)."
        )
        decision = AlertDecision(
            status=AlertStatus.AN_TOAN,
            status_text="An toàn",
            explanation=explanation,
            speed_kmh=speed_kmh,
            limit_kmh=limit_kmh,
            diff_kmh=diff,
            color_bgr=ALERT_COLORS_BGR[AlertStatus.AN_TOAN],
            color_rgb=ALERT_COLORS_RGB[AlertStatus.AN_TOAN],
            is_overspeed=False,
            is_caution=False,
            is_safe=True,
            is_determining=False,
            trigger_sound=False,
            sound_severity="none",
        )
        self.last_decision = decision
        return decision


class OverspeedAlert(FourStateAlertManager):
    """Backwards-compatible wrapper matching original OverspeedAlert API.

    Keeps the exact `update(speed_kmh, limit_kmh, now=None) -> bool` signature
    while providing access to the full 4-state alert logic.
    """

    def __init__(
        self,
        buffer_kmh: float = 0.0,
        confirm_seconds: float = 0.5,
        caution_margin: float = 5.0,
        min_confidence: float = 0.60,
    ) -> None:
        super().__init__(
            buffer_kmh=buffer_kmh,
            confirm_seconds=confirm_seconds,
            caution_margin=caution_margin,
            min_confidence=min_confidence,
            min_warmup_frames=1,
        )

    def update(
        self,
        speed_kmh: float | None,
        limit_kmh: int | None,
        now: float | None = None,
    ) -> bool:
        """Original API: returns True only when overspeed is confirmed."""
        decision = self.decide(
            speed_kmh=speed_kmh,
            limit_kmh=limit_kmh,
            road_name="Tuyến đường hiện tại",
            road_confidence=1.0,
            frame_index=10,
            now=now,
        )
        return decision.is_overspeed
