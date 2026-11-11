"""Classify road scene of any input video and display the legal speed limit with 4 Alert States.

4 Alert States (VisionDrive AI):
1. Đang xác định (Determining / Low confidence): Chưa đủ khung hình hoặc độ tin cậy còn thấp.
2. An toàn (Safe): Tốc độ nằm trong giới hạn của loại đường hiện tại.
3. Chú ý (Caution): Tốc độ gần chạm giới hạn; ngưỡng "chú ý" có thể chỉnh trong cài đặt.
4. Vượt tốc (Overspeed): Tốc độ vượt giới hạn; cảnh báo mạnh kèm câu giải thích rõ ràng.

Usage:
  python scripts/classify_video_road.py --video "data/videos/cao_toc/cao_toc_0001.mp4" --speed 90 --play
  python scripts/classify_video_road.py --video "data/videos/hem_ngo/hem_ngo_0001.mp4" --speed 45 --play
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adas.core.alert_manager import AlertDecision, AlertStatus, FourStateAlertManager
from adas.vision.vietnam_road_classifier import VIETNAM_ROAD_RULES, VietnamRoadClassifier

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def wrap_text(text: str, max_chars: int = 58) -> list[str]:
    """Wrap long Vietnamese text into multiple lines for clean HUD display."""
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


def draw_road_hud(
    frame: np.ndarray,
    result: dict,
    frame_idx: int,
    total_frames: int,
    alert_decision: AlertDecision,
) -> np.ndarray:
    """Draw clean road type, legal speed limit, and 4-state alert box on frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent main panel
    panel_w = min(w - 20, 620)
    panel_h = 165
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (8, 22, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    # State border accent
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), alert_decision.color_bgr, 2)

    # Title
    cv2.putText(
        frame,
        "VISIONDRIVE AI | NHAN DIEN DUONG & 4 TRANG THAI CANH BAO",
        (22, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 255, 200),
        2,
        cv2.LINE_AA,
    )

    # Road Type
    label_vi = result.get("label", "Đang nhận diện")
    score = result.get("score", 0.0)
    conf_pct = score * 100
    cv2.putText(
        frame,
        f"LOAI DUONG: {label_vi.upper()} ({conf_pct:.1f}%)",
        (22, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Legal Speed Limit
    max_speed = result.get("legal_max_speed")
    if max_speed is not None:
        speed_text = f"TOC DO TOI DA CHO PHEP: {max_speed} km/h"
        color = (50, 230, 255)
    else:
        speed_text = "TOC DO TOI DA: Khong xac dinh (Lai xe than trong)"
        color = (180, 180, 180)

    cv2.putText(
        frame,
        speed_text,
        (22, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        color,
        2,
        cv2.LINE_AA,
    )

    # Vehicle Speed
    if alert_decision.speed_kmh is not None:
        v_speed_text = f"TOC DO XE HIEN TAI: {alert_decision.speed_kmh:.0f} km/h"
    else:
        v_speed_text = "TOC DO XE HIEN TAI: -- km/h (Chua co nguon toc do)"
    cv2.putText(
        frame,
        v_speed_text,
        (22, 126),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        alert_decision.color_bgr,
        2,
        cv2.LINE_AA,
    )

    # Alert State Badge
    badge_str = f"TRANG THAI: [ {alert_decision.status_text.upper()} ]"
    cv2.putText(
        frame,
        badge_str,
        (22, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        alert_decision.color_bgr,
        2,
        cv2.LINE_AA,
    )

    # Dedicated Explanation Card ("Mỗi cảnh báo đều có lời giải thích")
    explanation_lines = wrap_text(alert_decision.explanation, max_chars=55)
    card_top = 10 + panel_h + 8
    card_h = 32 + len(explanation_lines) * 22

    card_overlay = frame.copy()
    cv2.rectangle(card_overlay, (10, card_top), (10 + panel_w, card_top + card_h), (6, 18, 16), -1)
    frame[:] = cv2.addWeighted(card_overlay, 0.88, frame, 0.12, 0)
    cv2.rectangle(frame, (10, card_top), (10 + panel_w, card_top + card_h), alert_decision.color_bgr, 2)

    cv2.putText(
        frame,
        "LOI GIAI THICH CANH BAO:",
        (22, card_top + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (100, 240, 255),
        1,
        cv2.LINE_AA,
    )

    line_y = card_top + 46
    for eline in explanation_lines:
        cv2.putText(
            frame,
            eline,
            (22, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (240, 250, 245),
            1,
            cv2.LINE_AA,
        )
        line_y += 22

    return frame


def analyze_video(
    video_path: str | Path,
    model_path: str = "models/vietnam_road_classifier.pt",
    play: bool = False,
    save_path: str | None = None,
    speed: float | None = None,
    caution_margin: float = 5.0,
) -> dict:
    video_path = Path(video_path)
    if not video_path.exists():
        candidates = [
            Path("adas-road-assistant") / video_path,
            Path.cwd() / video_path,
        ]
        for c in candidates:
            if c.exists():
                video_path = c
                break

    if not video_path.exists():
        print(f"[LOI] Khong tim thay file video: {video_path}", file=sys.stderr)
        return {}

    classifier = VietnamRoadClassifier(model_path=model_path)
    if not classifier.load():
        print(f"[LOI] Khong the tai mo hinh tu: {model_path}", file=sys.stderr)
        print(f"Chi tiet: {classifier.load_error}", file=sys.stderr)
        return {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[LOI] OpenCV khong doc duoc video: {video_path}", file=sys.stderr)
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360

    writer = None
    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))

    alert_mgr = FourStateAlertManager(caution_margin=caution_margin)
    current_speed = speed

    print("=" * 68)
    print(f"DANG PHAN TICH VIDEO: {video_path.name}")
    print(f"Tong so frames: {total_frames} | FPS: {fps:.1f} | Do phan giai: {width}x{height}")
    if current_speed is not None:
        print(f"Toc do xe kiem thu: {current_speed} km/h (Nguong chu y: {caution_margin} km/h)")
    print("=" * 68)

    frame_idx = 0
    class_votes: dict[str, list[float]] = {}
    last_result = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_idx += 1

        # Sample every 5 frames for speed and stability
        if frame_idx % 5 == 1 or last_result is None:
            last_result = classifier.classify(frame)
            k = last_result["class_key"]
            if k not in class_votes:
                class_votes[k] = []
            class_votes[k].append(last_result["score"])

        # Decide 4-state alert
        r_name = last_result.get("label") if last_result else None
        r_score = last_result.get("score") if last_result else None
        r_limit = last_result.get("legal_max_speed") if last_result else None

        alert_dec = alert_mgr.decide(
            speed_kmh=current_speed,
            limit_kmh=r_limit,
            road_name=r_name,
            road_confidence=r_score,
            frame_index=frame_idx,
        )

        # Render HUD if playing or saving
        if play or writer is not None:
            vis_frame = draw_road_hud(frame.copy(), last_result, frame_idx, total_frames, alert_dec)
            if writer is not None:
                writer.write(vis_frame)
            if play:
                cv2.imshow("Vietnam Road Scene Classification & 4-State Alerts", vis_frame)
                key = cv2.waitKey(int(1000 / fps)) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key in (ord("+"), ord("=")):
                    current_speed = (current_speed or 0) + 5
                elif key in (ord("-"), ord("_")):
                    current_speed = max(0, (current_speed or 0) - 5)

    cap.release()
    if writer is not None:
        writer.release()
        print(f"\nDa luu video kem HUD vao: {save_path}")
    if play:
        cv2.destroyAllWindows()

    # Aggregate overall video classification
    summary = {}
    if class_votes:
        best_class = max(class_votes.keys(), key=lambda c: (len(class_votes[c]), np.mean(class_votes[c])))
        avg_score = float(np.mean(class_votes[best_class]))
        rule = VIETNAM_ROAD_RULES.get(best_class, VIETNAM_ROAD_RULES["khong_xac_dinh"])

        final_dec = alert_mgr.decide(
            speed_kmh=current_speed,
            limit_kmh=rule["legal_max_speed"],
            road_name=rule["name_vi"],
            road_confidence=avg_score,
            frame_index=10,
        )

        summary = {
            "video_path": str(video_path),
            "total_frames_analyzed": frame_idx,
            "detected_class_key": best_class,
            "detected_road_name": rule["name_vi"],
            "confidence_percent": round(avg_score * 100, 2),
            "legal_max_speed_kmh": rule["legal_max_speed"],
            "alert_status": final_dec.status.value,
            "alert_explanation": final_dec.explanation,
            "description": rule["description"],
            "legal_rule": "Thông tư 31/2019/TT-BGTVT",
        }

        print("\n" + "=" * 68)
        print("          KET QUA NHAN DIEN LOAI DUONG & 4 TRANG THAI CANH BAO")
        print("=" * 68)
        print(f"▶ File video           : {video_path.name}")
        print(f"▶ LOAI DUONG NHAN DIEN : {rule['name_vi'].upper()}")
        print(f"▶ Do tin cay (Accuracy): {summary['confidence_percent']}%")
        if rule['legal_max_speed'] is not None:
            print(f"▶ TOC DO TOI DA CHO PHEP: {rule['legal_max_speed']} km/h")
        else:
            print(f"▶ TOC DO TOI DA CHO PHEP: Khong xac dinh (Lai xe than trong)")
        if current_speed is not None:
            print(f"▶ TOC DO XE KIEM THU   : {current_speed} km/h")
            print(f"▶ TRANG THAI CANH BAO  : [ {final_dec.status_text.upper()} ]")
            print(f"▶ LOI GIAI THICH       : {final_dec.explanation}")
        print(f"▶ Dac diem loai duong  : {rule['description']}")
        print(f"▶ Can cu phap ly       : Thong tu 31/2019/TT-BGTVT")
        print("=" * 68)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Classify Vietnamese road type of a video and evaluate 4 alert states.")
    parser.add_argument("--video", "-v", required=True, help="Path to input video file")
    parser.add_argument("--speed", "-s", type=float, default=None, help="Simulated vehicle speed in km/h to test 4 alert states")
    parser.add_argument("--caution-margin", type=float, default=5.0, help="Caution margin in km/h below speed limit (default 5.0)")
    parser.add_argument("--model", default="models/vietnam_road_classifier.pt", help="Path to trained classifier model")
    parser.add_argument("--play", action="store_true", help="Display video playback with live HUD overlay")
    parser.add_argument("--save", default=None, help="Save annotated video with HUD overlay to path (.mp4)")
    args = parser.parse_args()

    analyze_video(
        video_path=args.video,
        model_path=args.model,
        play=args.play,
        save_path=args.save,
        speed=args.speed,
        caution_margin=args.caution_margin,
    )


if __name__ == "__main__":
    main()
