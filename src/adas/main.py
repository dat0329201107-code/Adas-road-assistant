"""Run the ADAS road-scene and speed-limit alert demo on a video or webcam."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from adas.config import AppConfig
from adas.core.alert_manager import AlertStatus, FourStateAlertManager, OverspeedAlert
from adas.core.audio_alert import AudioAlert
from adas.core.logger import SessionLogger
from adas.core.speed_limit_tracker import SpeedLimitTracker
from adas.core.speed_source import (
    GPSSpeedSource,
    OBDSpeedSource,
    SimulatedSpeed,
    SpeedSource,
)
from adas.ui.overlay import draw_dashboard
from adas.vision.lane_detector import LaneDetector
from adas.vision.object_detector import ObjectDetector
from adas.vision.road_classifier import RoadSceneClassifier
from adas.vision.sign_detector import SignDetector
from adas.vision.speed_sign import SpeedSignRecognizer

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logger = logging.getLogger(__name__)


def parse_source(value: str):
    """Treat a digit-only source as a webcam index; all other values are paths/URLs."""
    return int(value) if value.isdigit() else value


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ADAS demo: classify road scene and alert when a supplied speed exceeds a limit."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml file")
    parser.add_argument("--source", default="0", help="Webcam index (default 0), video path, or supported stream URL")
    parser.add_argument("--manual-limit", type=int, default=None,
                        help="Optional fallback limit in km/h. Does not come from the road-scene label.")
    parser.add_argument("--demo-speed", type=float, default=None,
                        help="Initial simulated vehicle speed in km/h, e.g. 58; for testing only")
    parser.add_argument("--overspeed-buffer", type=float, default=0.0,
                        help="Extra km/h above the displayed limit before alerting (default 0)")
    parser.add_argument("--road-every", type=positive_int, default=20,
                        help="Run CLIP road-scene inference every N frames (default 20)")
    parser.add_argument("--sign-every", type=positive_int, default=5,
                        help="Run red-sign/OCR detection every N frames (default 5)")
    parser.add_argument("--limit-ttl", type=float, default=45.0,
                        help="Seconds to retain a repeatedly OCR-confirmed sign (default 45)")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32",
                        help="Hugging Face zero-shot image model id")
    parser.add_argument("--no-road-ai", action="store_true", help="Disable CLIP road-scene classification")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR, but still draw red-circle candidates")
    parser.add_argument("--no-sound", action="store_true", help="Disable desktop audio alerts")
    parser.add_argument("--tesseract-cmd", default=None,
                        help="Full path to tesseract executable if it is not on PATH")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # New options for enhanced features
    parser.add_argument("--speed-source", choices=["simulated", "gps", "obd"], default=None,
                        help="Speed source: simulated (default), gps, or obd")
    parser.add_argument("--gps-port", default="COM3", help="Serial port for GPS receiver")
    parser.add_argument("--gps-baudrate", type=int, default=9600, help="Baud rate for GPS receiver")
    parser.add_argument("--obd-port", default=None, help="Port or Bluetooth name for OBD-II adapter")
    parser.add_argument("--enable-lane", action="store_true", help="Enable lane detection")
    parser.add_argument("--enable-objects", action="store_true", help="Enable YOLO object detection (cars, people)")
    parser.add_argument("--use-yolo-sign", action="store_true", help="Enable YOLO-based traffic sign detector")
    parser.add_argument("--record", default=None, help="Path to save processed video recording with HUD (.mp4)")
    parser.add_argument("--caution-margin", type=float, default=None,
                        help="Speed margin (km/h) below limit for 'Chú ý' caution state (default 5.0)")
    parser.add_argument("--min-confidence", type=float, default=None,
                        help="Min road confidence threshold before leaving 'Đang xác định' (default 0.60)")
    parser.add_argument("--max-frames", type=positive_int, default=None, help="Process at most N frames and exit")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI window")

    return parser


def create_speed_source(cfg: AppConfig) -> SpeedSource:
    """Instantiate the speed source specified by configuration."""
    source_type = cfg.speed_source.lower()
    if source_type == "gps":
        logger.info("Initializing GPS speed source on %s (%d baud)", cfg.gps_port, cfg.gps_baudrate)
        return GPSSpeedSource(port=cfg.gps_port, baudrate=cfg.gps_baudrate)
    elif source_type == "obd":
        logger.info("Initializing OBD-II speed source on %s", cfg.obd_port or "auto-detect")
        return OBDSpeedSource(port=cfg.obd_port)
    else:
        return SimulatedSpeed(initial_kmh=cfg.initial_demo_speed)


def run(args: argparse.Namespace) -> int:
    cfg = AppConfig.from_args_and_yaml(args, args.config)

    # CLI explicit overrides for new flags
    if args.speed_source:
        cfg.speed_source = args.speed_source
    if args.enable_lane:
        cfg.lane_enabled = True
    if args.enable_objects:
        cfg.object_detection_enabled = True
    if args.use_yolo_sign:
        cfg.use_yolo = True
    if args.caution_margin is not None:
        cfg.caution_margin = args.caution_margin
    if args.min_confidence is not None:
        cfg.min_confidence = args.min_confidence

    # Setup session logging
    session_logger = SessionLogger(log_dir=cfg.log_dir, enable_console=True)
    log_level = getattr(logging, cfg.log_level, logging.INFO)
    session_logger.setup_logging(level=log_level)
    fps_tracker = session_logger.get_fps_tracker()

    if cfg.manual_limit is not None and not 5 <= cfg.manual_limit <= 130:
        print("Lỗi: manual_limit phải trong khoảng 5..130 km/h", file=sys.stderr)
        return 2

    source = parse_source(str(cfg.camera_source))
    if isinstance(source, str) and not Path(source).exists():
        candidates = [
            Path("adas-road-assistant") / source,
            Path(__file__).resolve().parents[2] / source,
            Path(__file__).resolve().parents[3] / source,
        ]
        for c in candidates:
            if c.exists():
                source = str(c)
                break

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Không mở được nguồn video/webcam: {source}", file=sys.stderr)
        print("Kiểm tra đường dẫn video, quyền camera, hoặc thử --source 0.", file=sys.stderr)
        session_logger.close()
        return 2

    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera_height)

    # Initialize vision and core components
    classifier = None
    if cfg.road_ai_enabled:
        vn_candidates = [
            Path("models/vietnam_road_classifier.pt"),
            Path("adas-road-assistant/models/vietnam_road_classifier.pt"),
            Path(__file__).resolve().parents[2] / "models" / "vietnam_road_classifier.pt",
            Path(__file__).resolve().parents[3] / "models" / "vietnam_road_classifier.pt",
        ]
        vn_road_model = next((p for p in vn_candidates if p.exists()), None)
        if vn_road_model is not None:
            from adas.vision.vietnam_road_classifier import VietnamRoadClassifier
            classifier = VietnamRoadClassifier(model_path=str(vn_road_model))
            if not classifier.load():
                logger.warning("Could not load VietnamRoadClassifier (%s), falling back to CLIP", classifier.load_error)
                classifier = RoadSceneClassifier(cfg.road_model_name)
            else:
                logger.info("Vietnam Road Classifier (MobileNetV3 on GPU) active: %s", vn_road_model)
        else:
            classifier = RoadSceneClassifier(cfg.road_model_name)
    sign_recognizer = SpeedSignRecognizer(enable_ocr=cfg.enable_ocr, tesseract_cmd=cfg.tesseract_cmd)
    yolo_sign_detector = None
    if cfg.use_yolo:
        yolo_sign_detector = SignDetector(
            model_path=cfg.yolo_model_path,
            confidence=cfg.yolo_confidence,
        )
        if not yolo_sign_detector.load():
            logger.warning("Could not load YOLO sign detector: %s", yolo_sign_detector.load_error)
        else:
            logger.info("YOLO sign detector loaded successfully on GPU/CPU.")

    lane_detector = LaneDetector(roi_top_ratio=cfg.lane_roi_top) if cfg.lane_enabled else None
    object_detector = None
    if cfg.object_detection_enabled:
        object_detector = ObjectDetector(
            model_path=cfg.object_model_path,
            confidence=cfg.object_confidence,
            target_classes=cfg.object_target_classes,
        )
        object_detector.load()

    tracker = SpeedLimitTracker(manual_limit=cfg.manual_limit, sign_ttl_seconds=cfg.sign_ttl_seconds)
    alert = FourStateAlertManager(
        buffer_kmh=cfg.overspeed_buffer,
        confirm_seconds=cfg.confirm_seconds,
        caution_margin=cfg.caution_margin,
        min_confidence=cfg.min_confidence,
        min_warmup_frames=cfg.min_warmup_frames,
    )
    audio = AudioAlert(enabled=cfg.enable_sound, repeat_seconds=cfg.sound_repeat_seconds)
    speed_source = create_speed_source(cfg)

    # Video recording setup if requested
    video_writer = None
    if args.record:
        record_path = Path(args.record)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cfg.camera_width
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cfg.camera_height
        video_writer = cv2.VideoWriter(str(record_path), fourcc, fps_in, (frame_w, frame_h))
        logger.info("Recording video output to: %s", record_path)

    print("\nADAS ROAD ASSIST — DEMO")
    print(f"- Nguon toc do: {cfg.speed_source.upper()}")
    if isinstance(speed_source, SimulatedSpeed):
        print("- Dung +/- de tang/giam toc do; 0 de dung; v de bat/tat.")
    print("- Nhan dien bien toc do: Heuristic + OCR" + (" + YOLO Detector" if cfg.use_yolo else ""))
    if cfg.lane_enabled:
        print("- Phat hien lan duong: BAT")
    if cfg.object_detection_enabled:
        print("- Phat hien xe & nguoi: BAT (YOLO)")
    if args.record:
        print(f"- Dang ghi hinh video ra: {args.record}")
    print("Nhan 'q' de thoat.\n")

    road_result = None
    sign_candidates = []
    lane_result = None
    detected_objects = []
    frame_index = 0
    previous_overspeed = False
    previous_alert_status: AlertStatus | None = None
    last_road_log = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                # If playing a video file without a max-frame limit, loop back to beginning
                if not isinstance(source, int) and cfg.max_frames is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                else:
                    break
            frame_index += 1
            fps_tracker.tick()

            # 1. Road scene classification & legal speed limit enforcement
            if classifier is not None and frame_index % cfg.road_every == 1:
                road_result = classifier.classify(frame)
                if road_result:
                    max_speed = road_result.get("legal_max_speed")
                    road_name = road_result.get("label")
                    if max_speed is not None:
                        tracker.observe_road_limit(max_speed, road_name=road_name)

                    if time.monotonic() - last_road_log > 5:
                        logger.info(
                            "Road scene: %s (score %.2f) | Quy định tốc độ: %s km/h",
                            road_result["label"],
                            road_result.get("score", 0.0),
                            max_speed or "Chưa rõ",
                        )
                        last_road_log = time.monotonic()

            # 2. Traffic sign detection (YOLO Detector + Heuristic OCR)
            if frame_index % cfg.sign_every == 1:
                sign_candidates = []
                limits = []

                # A. YOLO Deep Learning sign detector (runs on GPU, detects 82 Vietnamese sign classes)
                if yolo_sign_detector is not None and yolo_sign_detector.available:
                    yolo_signs = yolo_sign_detector.detect(frame)
                    sign_candidates.extend(yolo_signs)
                    for ys in yolo_signs:
                        if ys.limit_kmh is not None:
                            limits.append(ys.limit_kmh)

                # B. OpenCV Heuristic + OCR (complementary fallback)
                if sign_recognizer is not None:
                    # If YOLO is active, only run heuristic if YOLO found no signs (or keep only OCR-confirmed signs)
                    if not sign_candidates:
                        heuristic_signs = sign_recognizer.detect(frame)
                        if not cfg.use_yolo:
                            sign_candidates.extend(heuristic_signs)
                        else:
                            # Only include heuristic candidate if OCR actually read a speed limit
                            sign_candidates.extend([h for h in heuristic_signs if h.limit_kmh is not None])
                        for hs in heuristic_signs:
                            if hs.limit_kmh is not None and hs.limit_kmh not in limits:
                                limits.append(hs.limit_kmh)

                if limits:
                    session_logger.log_detection("sign", {"limits": limits})
                tracker.observe([int(x) for x in limits])

            # 3. Lane detection
            if lane_detector is not None and frame_index % 2 == 0:
                lane_result = lane_detector.detect(frame)

            # 4. Object detection
            if object_detector is not None and frame_index % 4 == 0:
                detected_objects = object_detector.detect(frame)

            # 5. Speed and alert logic (Four Alert States + Explainable Message)
            speed_kmh = speed_source.read_kmh()
            active_limit = tracker.active()
            road_name = road_result["label"] if road_result else None
            road_conf = road_result.get("score") if road_result else None

            alert_decision = alert.decide(
                speed_kmh=speed_kmh,
                limit_kmh=active_limit.value_kmh,
                road_name=road_name,
                road_confidence=road_conf,
                limit_source=active_limit.source,
                frame_index=frame_index,
            )

            overspeed = alert_decision.is_overspeed

            # Multi-level audio: alarm for overspeed, gentle chime for caution
            if alert_decision.trigger_sound and alert_decision.sound_severity != "none":
                audio.trigger(severity=alert_decision.sound_severity)

            if alert_decision.status != previous_alert_status:
                session_logger.log_alert(alert_decision.status.value, speed_kmh, active_limit.value_kmh)
                logger.info(
                    "TRẠNG THÁI CẢNH BÁO: [%s] | %s",
                    alert_decision.status_text.upper(),
                    alert_decision.explanation,
                )
                previous_alert_status = alert_decision.status

            previous_overspeed = overspeed

            # Periodic frame stats logging
            if frame_index % cfg.log_frame_stats_every == 0:
                session_logger.log_frame_stats(
                    frame_index=frame_index,
                    fps=fps_tracker.fps,
                    road_label=str(road_result["label"]) if road_result else None,
                    speed=speed_kmh,
                    limit=active_limit.value_kmh,
                    overspeed=overspeed,
                )

            # 6. UI HUD rendering
            speed_src_name = "giả lập" if isinstance(speed_source, SimulatedSpeed) else cfg.speed_source.upper()
            draw_dashboard(
                frame=frame,
                road_result=road_result,
                active_limit=active_limit,
                speed_kmh=speed_kmh,
                overspeed=overspeed,
                sign_candidates=sign_candidates,
                ocr_enabled=sign_recognizer.enable_ocr or (yolo_sign_detector is not None and yolo_sign_detector.available),
                road_ai_enabled=classifier is not None,
                frame_index=frame_index,
                fps=fps_tracker.fps,
                lane_result=lane_result,
                detected_objects=detected_objects,
                speed_source_name=speed_src_name,
                alert_decision=alert_decision,
            )

            if video_writer is not None:
                video_writer.write(frame)

            if not cfg.headless:
                cv2.imshow("ADAS Road Assistant - Demo", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
            else:
                key = -1

            if cfg.max_frames and frame_index >= cfg.max_frames:
                logger.info("Reached target limit of %d frames. Exiting loop.", cfg.max_frames)
                break

            # Handle interactive simulated speed keys
            if isinstance(speed_source, SimulatedSpeed):
                if key in (ord("+"), ord("=")):
                    speed_source.increase(5.0)
                elif key in (ord("-"), ord("_")):
                    speed_source.decrease(5.0)
                elif key == ord("0"):
                    speed_source.set_speed(0.0)
                elif key == ord("v"):
                    speed_source.toggle()

            if key == ord("r"):
                tracker.reset_detected_sign()
                logger.info("Cleared OCR-confirmed speed sign")

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        speed_source.close()
        session_logger.close()

    return 0


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
