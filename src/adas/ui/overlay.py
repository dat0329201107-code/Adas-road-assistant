"""Dashboard overlay for ADAS Road Assistant with full Vietnamese Unicode support and 4 Alert States."""
from __future__ import annotations

import unicodedata
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from adas.core.alert_manager import AlertDecision, AlertStatus, FourStateAlertManager


# Font candidates for Windows / Linux / macOS
FONT_CANDIDATES = [
    "arial.ttf",
    "segoeui.ttf",
    "tahoma.ttf",
    "calibri.ttf",
    "DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

FONT_BOLD_CANDIDATES = [
    "arialbd.ttf",
    "segoeuib.ttf",
    "tahomabd.ttf",
    "calibrib.ttf",
    "DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


@lru_cache(maxsize=32)
def _get_font(size: int = 16, bold: bool = False) -> Any:
    """Load and cache a TrueType font that supports Vietnamese Unicode."""
    if not _PIL_AVAILABLE:
        return None

    candidates = FONT_BOLD_CANDIDATES if bold else FONT_CANDIDATES
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue

    # Fallback to general list if bold variant was not found
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue

    try:
        return ImageFont.load_default()
    except Exception:
        return None


def strip_accents(text: str) -> str:
    """Convert Vietnamese accented text to clean ASCII to prevent '???' in OpenCV putText."""
    replacements = {
        "đ": "d",
        "Đ": "D",
        "ư": "u",
        "Ư": "U",
        "ơ": "o",
        "Ơ": "O",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text)


def wrap_text(text: str, max_chars: int = 58) -> list[str]:
    """Wrap long Vietnamese sentences into multiple lines for clean HUD rendering."""
    words = text.split(" ")
    lines: list[str] = []
    curr: list[str] = []
    curr_len = 0
    for w in words:
        if curr_len + len(w) + 1 > max_chars and curr:
            lines.append(" ".join(curr))
            curr = [w]
            curr_len = len(w)
        else:
            curr.append(w)
            curr_len += len(w) + 1
    if curr:
        lines.append(" ".join(curr))
    return lines


def _render_text_overlay(
    frame: np.ndarray,
    text_items: list[tuple[str, tuple[int, int], int, tuple[int, int, int], bool]],
) -> None:
    """Render a batch of text items onto frame.

    Uses PIL TrueType for crisp Vietnamese with diacritics. If PIL fails, falls
    back to OpenCV with stripped accents (never prints '???').
    """
    if _PIL_AVAILABLE:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(pil_img)

            for text, (x, y), font_size, color_bgr, bold in text_items:
                font = _get_font(font_size, bold=bold)
                # BGR to RGB color
                color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
                # Draw text with 1px black outline for contrast on any background
                draw.text(
                    (x, y),
                    text,
                    font=font,
                    fill=color_rgb,
                    stroke_width=1,
                    stroke_fill=(0, 0, 0),
                )

            frame[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
            return
        except Exception:
            pass

    # Fallback: strip accents so cv2.putText does not output '???'
    for text, (x, y), font_size, color_bgr, _ in text_items:
        safe_text = strip_accents(text)
        scale = font_size / 28.0
        origin = (x, y + font_size)
        cv2.putText(
            frame,
            safe_text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            safe_text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color_bgr,
            1,
            cv2.LINE_AA,
        )


def _put_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Backwards-compatible helper: strips accents to avoid '???' if called directly."""
    safe_text = strip_accents(text)
    cv2.putText(
        frame,
        safe_text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        safe_text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_dashboard(
    frame: np.ndarray,
    road_result: dict[str, object] | None,
    active_limit: Any,
    speed_kmh: float | None,
    overspeed: bool = False,
    sign_candidates: list[Any] | None = None,
    ocr_enabled: bool = True,
    road_ai_enabled: bool = True,
    frame_index: int = 0,
    fps: float | None = None,
    lane_result: Any | None = None,
    detected_objects: list[Any] | None = None,
    speed_source_name: str = "giả lập",
    alert_decision: AlertDecision | None = None,
) -> None:
    """Render ADAS HUD dashboard with 4 Alert States and dedicated explanation card."""
    h, w = frame.shape[:2]
    sign_candidates = sign_candidates or []
    text_items: list[
        tuple[str, tuple[int, int], int, tuple[int, int, int], bool]
    ] = []

    # If alert_decision not explicitly supplied, synthesize one for backwards compatibility
    if alert_decision is None:
        mgr = FourStateAlertManager()
        r_name = road_result.get("label") if road_result else None
        r_score = road_result.get("score") if road_result else None
        limit_val = getattr(active_limit, "value_kmh", None)
        limit_src = getattr(active_limit, "source", None)
        alert_decision = mgr.decide(
            speed_kmh=speed_kmh,
            limit_kmh=limit_val,
            road_name=r_name,
            road_confidence=r_score,
            limit_source=limit_src,
            frame_index=frame_index,
        )

    # 1. Draw lane overlay if available
    if lane_result is not None and getattr(lane_result, "lane_detected", False):
        overlay_pts = getattr(lane_result, "overlay_points", [])
        if len(overlay_pts) == 4:
            lane_mask = frame.copy()
            pts = np.array(overlay_pts, dtype=np.int32)
            cv2.fillPoly(lane_mask, [pts], (0, 200, 100))
            cv2.addWeighted(lane_mask, 0.25, frame, 0.75, 0, frame)

        if lane_result.left_line:
            x1, y1, x2, y2 = lane_result.left_line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        if lane_result.right_line:
            x1, y1, x2, y2 = lane_result.right_line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

    # 2. Draw detected objects (vehicles, pedestrians)
    if detected_objects:
        for obj in detected_objects:
            x, y, bw, bh = obj.bbox
            color = (255, 180, 0) if obj.class_name == "person" else (255, 120, 50)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            dist_str = f" {obj.estimated_distance_m:.1f}m" if obj.estimated_distance_m else ""
            label = f"{obj.class_name} {obj.confidence:.2f}{dist_str}"
            text_items.append((label, (x, max(4, y - 18)), 13, color, False))

    # 3. Draw sign candidates
    for candidate in sign_candidates:
        x, y, bw, bh = candidate.bbox
        limit = getattr(candidate, "limit_kmh", None)
        vn_name = getattr(candidate, "vietnamese_name", None)
        class_name = getattr(candidate, "class_name", None)
        conf = getattr(candidate, "confidence", None)
        conf_str = f" ({conf*100:.0f}%)" if conf is not None else ""
        color = (80, 255, 100) if limit is not None else (0, 235, 255)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        if vn_name:
            label = f"{vn_name}{conf_str}"
        elif limit is not None:
            label = f"Tốc độ: {limit} km/h{conf_str}"
        elif class_name:
            label = f"{class_name}{conf_str}"
        else:
            label = "Biển báo (ứng viên)"
        text_items.append((label, (x, max(4, y - 18)), 14, color, True))

    # 4. Main Dashboard HUD Panel
    panel_w = min(w - 20, 600)
    panel_h = 165
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (8, 22, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    # Panel border with state accent
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), alert_decision.color_bgr, 2)

    # Header with FPS & VisionDrive branding
    title = "VISIONDRIVE AI | ADAS ROAD ASSIST"
    if fps is not None and fps > 0:
        title += f"  (FPS: {fps:.1f})"
    text_items.append((title, (22, 18), 17, (80, 255, 190), True))

    # Road type
    if road_result:
        road_name = str(road_result["label"])
        score = float(road_result.get("score", 0.0))
        road_text = f"LOẠI ĐƯỜNG: {road_name} ({score * 100:.0f}%)"
    elif road_ai_enabled:
        road_text = "LOẠI ĐƯỜNG: đang tải model / chưa có kết quả"
    else:
        road_text = "LOẠI ĐƯỜNG: AI đang tắt"
    text_items.append((road_text, (22, 45), 15, (230, 245, 240), False))

    # Speed limit
    limit_value = active_limit.value_kmh
    if limit_value is not None:
        source_label = active_limit.source
        limit_text = f"GIỚI HẠN: {limit_value} km/h [{source_label}]"
    else:
        limit_text = "GIỚI HẠN: chưa xác định"
    text_items.append((limit_text, (22, 72), 16, (80, 220, 255), True))

    # Vehicle speed
    if speed_kmh is None:
        speed_text = "TỐC ĐỘ XE: -- km/h (chưa có nguồn tốc độ)"
        speed_color = (230, 245, 240)
    else:
        speed_text = f"TỐC ĐỘ XE: {speed_kmh:.0f} km/h ({speed_source_name})"
        speed_color = alert_decision.color_bgr
    text_items.append((speed_text, (22, 100), 16, speed_color, True))

    # State Badge (One of the 4 States: Đang xác định, An toàn, Chú ý, Vượt tốc)
    status_str = f"TRẠNG THÁI: [ {alert_decision.status_text.upper()} ]"
    # Draw badge highlight rectangle
    badge_x = 22
    badge_y = 126
    cv2.rectangle(frame, (badge_x - 4, badge_y), (badge_x + 310, badge_y + 26), (20, 30, 30), -1)
    cv2.rectangle(frame, (badge_x - 4, badge_y), (badge_x + 310, badge_y + 26), alert_decision.color_bgr, 1)
    text_items.append((status_str, (badge_x, badge_y + 4), 15, alert_decision.color_bgr, True))

    # 5. Dedicated Explanation Box ("Mỗi cảnh báo đều có lời giải thích")
    # Wrap explanation text into max 55 chars per line
    explanation_lines = wrap_text(alert_decision.explanation, max_chars=55)
    box_h = 36 + len(explanation_lines) * 22
    box_top = 10 + panel_h + 8

    # Explanation card background
    box_overlay = frame.copy()
    cv2.rectangle(box_overlay, (10, box_top), (10 + panel_w, box_top + box_h), (6, 18, 16), -1)
    frame[:] = cv2.addWeighted(box_overlay, 0.88, frame, 0.12, 0)
    # Border
    cv2.rectangle(frame, (10, box_top), (10 + panel_w, box_top + box_h), alert_decision.color_bgr, 2)

    # Explanation Title
    text_items.append(
        ("LỜI GIẢI THÍCH CẢNH BÁO:", (22, box_top + 8), 13, (100, 240, 255), True)
    )

    # Render wrapped explanation lines
    line_y = box_top + 30
    for eline in explanation_lines:
        text_items.append((eline, (22, line_y), 14, (245, 250, 248), False))
        line_y += 22

    # 6. Controls footer
    footer = "q: thoát | +/-: tốc độ | 0: dừng | v: bật/tắt tốc độ | r: xóa biển"
    text_items.append((footer, (14, h - 25), 13, (245, 245, 245), False))

    if not ocr_enabled:
        text_items.append(
            (
                "OCR chưa sẵn sàng: cài Tesseract binary và pytesseract",
                (14, h - 46),
                13,
                (0, 190, 255),
                False,
            )
        )

    # Render all text with Unicode Pillow TrueType
    _render_text_overlay(frame, text_items)
