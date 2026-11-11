"""Warning Engine implementing the 4 Alert States and Explainable Messages (VisionDrive AI)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.schemas import AlertLevelEnum, VehicleTypeEnum

logger = logging.getLogger(__name__)

# Default fallback rules matching Thông tư 31/2019/TT-BGTVT
DEFAULT_SPEED_LIMITS = {
    "cao_toc": 100,
    "duong_doi_do_thi": 60,
    "pho_hai_chieu": 50,
    "ngoai_khu_dan_cu": 80,
    "hem_ngo": 30,
}


class WarningEngine:
    """Evaluates vehicle speed against legal road limits and produces 4-state decisions with
    explicit explanations for the driver.
    """

    def __init__(
        self,
        rules_path: str | Path | None = None,
        min_confidence: float = 0.60,
        default_caution_margin: float = 5.0,
        min_warmup_frames: int = 5,
    ) -> None:
        self.rules_path = Path(rules_path) if rules_path else self._resolve_rules_path()
        self.min_confidence = float(min_confidence)
        self.default_caution_margin = float(default_caution_margin)
        self.min_warmup_frames = int(min_warmup_frames)
        self.rules_data: dict[str, Any] = self._load_rules()

    def _resolve_rules_path(self) -> Path:
        """Find speed_rules.json location."""
        candidates = [
            Path("rules/speed_rules.json"),
            Path("backend/rules/speed_rules.json"),
            Path(__file__).resolve().parents[2] / "rules" / "speed_rules.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        return Path("rules/speed_rules.json")

    def _load_rules(self) -> dict[str, Any]:
        """Load speed rules from JSON file or return built-in defaults."""
        if self.rules_path.exists():
            try:
                with open(self.rules_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("Could not read rules file %s: %s", self.rules_path, exc)
        return {"road_classes": {}, "alert_levels": {}}

    def reload_rules(self) -> None:
        """Reload speed rules from disk."""
        self.rules_data = self._load_rules()

    def get_speed_limit(
        self,
        road_type: str,
        vehicle_type: VehicleTypeEnum | str = VehicleTypeEnum.CAR,
    ) -> int | None:
        """Resolve legal speed limit for given road type and vehicle."""
        v_key = vehicle_type.value if isinstance(vehicle_type, VehicleTypeEnum) else str(vehicle_type).lower()
        road_classes = self.rules_data.get("road_classes", {})

        if road_type in road_classes:
            r_info = road_classes[road_type]
            v_limits = r_info.get("vehicle_limits", {})
            if v_key in v_limits and v_limits[v_key] > 0:
                return int(v_limits[v_key])
            return int(r_info.get("legal_max_speed", DEFAULT_SPEED_LIMITS.get(road_type, 50)))

        return DEFAULT_SPEED_LIMITS.get(road_type)

    def evaluate(
        self,
        road_type: str,
        road_name_vi: str,
        confidence: float,
        current_speed: float | None = None,
        vehicle_type: VehicleTypeEnum | str = VehicleTypeEnum.CAR,
        caution_margin: float | None = None,
        frame_index: int = 10,
    ) -> dict[str, Any]:
        """Produce the 4-state alert decision and explanation text.

        The 4 States:
          1. idle (ĐANG XÁC ĐỊNH): low confidence (< min_confidence) or warmup frames.
          2. ok (AN TOÀN): current_speed <= limit - caution_margin.
          3. warn (CHÚ Ý): limit - caution_margin < current_speed <= limit.
          4. bad (VƯỢT TỐC): current_speed > limit.
        """
        margin = float(caution_margin) if caution_margin is not None else self.default_caution_margin
        speed_limit = self.get_speed_limit(road_type, vehicle_type)
        conf_pct = round(confidence * 100, 1)

        # -------------------------------------------------------------
        # STATE 1: ĐANG XÁC ĐỊNH (idle)
        # -------------------------------------------------------------
        if frame_index < self.min_warmup_frames:
            return {
                "alert_level": AlertLevelEnum.IDLE,
                "alert_label": "ĐANG XÁC ĐỊNH",
                "speed_limit": speed_limit,
                "current_speed": current_speed,
                "diff_speed": None,
                "explanation": (
                    f"Đang khởi động camera và phân tích khung hình ({frame_index}/{self.min_warmup_frames} frames). "
                    f"Chưa đủ điều kiện đưa ra cảnh báo. Lái xe thận trọng."
                ),
                "rule_description": "Chưa đủ khung hình tích luỹ để xác nhận loại đường ổn định.",
            }

        if confidence < self.min_confidence:
            return {
                "alert_level": AlertLevelEnum.IDLE,
                "alert_label": "ĐANG XÁC ĐỊNH",
                "speed_limit": speed_limit,
                "current_speed": current_speed,
                "diff_speed": None,
                "explanation": (
                    f"Đang phân tích loại đường (độ tin cậy thấp: {conf_pct:.0f}% < {self.min_confidence*100:.0f}%). "
                    f"Chưa đủ độ tin cậy để đưa ra cảnh báo tốc độ. Lái xe thận trọng."
                ),
                "rule_description": "Hệ thống chỉ cảnh báo khi độ tin cậy phân loại đường đạt từ 60% trở lên.",
            }

        if speed_limit is None:
            return {
                "alert_level": AlertLevelEnum.IDLE,
                "alert_label": "ĐANG XÁC ĐỊNH",
                "speed_limit": None,
                "current_speed": current_speed,
                "diff_speed": None,
                "explanation": (
                    f"Đang ở {road_name_vi} (độ tin cậy {conf_pct:.0f}%). Chưa xác định được giới hạn tốc độ pháp lý."
                ),
                "rule_description": "Chưa có quy định tốc độ cụ thể cho phân loại này.",
            }

        if current_speed is None:
            return {
                "alert_level": AlertLevelEnum.OK,
                "alert_label": "AN TOÀN",
                "speed_limit": speed_limit,
                "current_speed": None,
                "diff_speed": None,
                "explanation": (
                    f"Đang ở {road_name_vi} (độ tin cậy {conf_pct:.0f}%), giới hạn {speed_limit} km/h. "
                    f"Chưa có tín hiệu tốc độ xe (GPS/OBD/mô phỏng)."
                ),
                "rule_description": f"Giới hạn tốc độ theo Thông tư 31/2019/TT-BGTVT là {speed_limit} km/h.",
            }

        # -------------------------------------------------------------
        # SPEED EVALUATION WITH CONFIRMED LIMIT
        # -------------------------------------------------------------
        diff = current_speed - speed_limit

        # STATE 4: VƯỢT TỐC (bad)
        if current_speed > speed_limit:
            over = current_speed - speed_limit
            explanation = (
                f"Đang ở {road_name_vi} (độ tin cậy {conf_pct:.0f}%), giới hạn {speed_limit} km/h. "
                f"Tốc độ {current_speed:.0f} km/h VƯỢT QUÁ +{over:.0f} km/h! Cảnh báo nguy hiểm, giảm tốc ngay!"
            )
            return {
                "alert_level": AlertLevelEnum.BAD,
                "alert_label": "VƯỢT TỐC",
                "speed_limit": speed_limit,
                "current_speed": current_speed,
                "diff_speed": round(diff, 1),
                "explanation": explanation,
                "rule_description": f"Vượt quá tốc độ tối đa cho phép {speed_limit} km/h trên {road_name_vi}.",
            }

        # STATE 3: CHÚ Ý (warn)
        if current_speed >= (speed_limit - margin):
            remaining = speed_limit - current_speed
            explanation = (
                f"Đang ở {road_name_vi} (độ tin cậy {conf_pct:.0f}%), giới hạn {speed_limit} km/h. "
                f"Tốc độ {current_speed:.0f} km/h gần chạm giới hạn (còn {remaining:.0f} km/h). Chú ý duy trì tốc độ."
            )
            return {
                "alert_level": AlertLevelEnum.WARN,
                "alert_label": "CHÚ Ý",
                "speed_limit": speed_limit,
                "current_speed": current_speed,
                "diff_speed": round(diff, 1),
                "explanation": explanation,
                "rule_description": f"Tốc độ tiệm cận mức tối đa {speed_limit} km/h (ngưỡng chú ý: {margin:.0f} km/h).",
            }

        # STATE 2: AN TOÀN (ok)
        explanation = (
            f"Đang ở {road_name_vi} (độ tin cậy {conf_pct:.0f}%), giới hạn {speed_limit} km/h. "
            f"Tốc độ an toàn ({current_speed:.0f} km/h nằm trong giới hạn)."
        )
        return {
            "alert_level": AlertLevelEnum.OK,
            "alert_label": "AN TOÀN",
            "speed_limit": speed_limit,
            "current_speed": current_speed,
            "diff_speed": round(diff, 1),
            "explanation": explanation,
            "rule_description": f"Tốc độ xe tuân thủ đúng quy định tối đa {speed_limit} km/h.",
        }
