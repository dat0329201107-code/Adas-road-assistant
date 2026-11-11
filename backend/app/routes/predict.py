"""Prediction routes for VisionDrive AI Backend API."""
from __future__ import annotations

import io
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from app.schemas import (
    AlertLevelEnum,
    BoundingBox,
    PredictFrameRequest,
    PredictResponse,
    VehicleTypeEnum,
    VideoPredictSummary,
)
from app.services.classifier import RoadClassifierService
from app.services.smoothing import TemporalSmoother
from app.services.warning import WarningEngine

router = APIRouter(prefix="/predict", tags=["Prediction"])

# In-memory smoother for default streaming session
_default_smoother = TemporalSmoother(window_size=7, alpha=0.40)


def get_classifier(request: Request) -> RoadClassifierService:
    """Dependency: retrieve classifier service from app state."""
    return request.app.state.classifier


def get_warning_engine(request: Request) -> WarningEngine:
    """Dependency: retrieve warning engine from app state."""
    return request.app.state.warning_engine


@router.post("/frame", response_model=PredictResponse, summary="Predict road type from base64 image frame")
async def predict_frame(
    payload: PredictFrameRequest,
    classifier: RoadClassifierService = Depends(get_classifier),
    warning_engine: WarningEngine = Depends(get_warning_engine),
) -> PredictResponse:
    """Classify a single base64 video frame and evaluate the 4 alert states."""
    if not payload.image_base64:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp chuỗi base64 của ảnh qua trường 'image_base64'.")

    try:
        # Run vision classification
        raw_pred = classifier.predict_image(payload.image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Không thể giải mã hoặc xử lý ảnh: {exc}")

    # Optional temporal smoothing across stream frames
    if payload.apply_smoothing:
        smoothed_pred = _default_smoother.update(raw_pred)
    else:
        smoothed_pred = raw_pred

    # Evaluate 4-state alert and explainable message
    alert_info = warning_engine.evaluate(
        road_type=smoothed_pred["road_type"],
        road_name_vi=smoothed_pred["road_name_vi"],
        confidence=smoothed_pred["confidence"],
        current_speed=payload.current_speed,
        vehicle_type=payload.vehicle_type,
        caution_margin=payload.caution_margin,
        frame_index=_default_smoother.frame_count if payload.apply_smoothing else 10,
    )

    boxes = [BoundingBox(**b) for b in smoothed_pred.get("boxes", [])]

    return PredictResponse(
        road_type=smoothed_pred["road_type"],
        road_name_vi=smoothed_pred["road_name_vi"],
        road_class_id=smoothed_pred["road_class_id"],
        confidence=smoothed_pred["confidence"],
        confidence_percent=smoothed_pred["confidence_percent"],
        probabilities=smoothed_pred["probabilities"],
        speed_limit=alert_info["speed_limit"],
        current_speed=alert_info["current_speed"],
        diff_speed=alert_info["diff_speed"],
        alert_level=alert_info["alert_level"],
        alert_label=alert_info["alert_label"],
        explanation=alert_info["explanation"],
        rule_description=alert_info["rule_description"],
        legal_citation="Thông tư 31/2019/TT-BGTVT",
        boxes=boxes,
        latency_ms=smoothed_pred.get("latency_ms", 0.0),
        smoothed=smoothed_pred.get("smoothed", False),
    )


@router.post("/image", response_model=PredictResponse, summary="Predict road type from uploaded image file")
async def predict_image_file(
    file: UploadFile = File(..., description="JPEG/PNG/WebP image file"),
    speed: Optional[float] = Form(None, description="Vehicle speed in km/h"),
    vehicle_type: VehicleTypeEnum = Form(VehicleTypeEnum.CAR, description="Type of vehicle"),
    caution_margin: Optional[float] = Form(5.0, description="Caution speed margin in km/h"),
    apply_smoothing: bool = Form(False, description="Apply temporal smoothing"),
    classifier: RoadClassifierService = Depends(get_classifier),
    warning_engine: WarningEngine = Depends(get_warning_engine),
) -> PredictResponse:
    """Classify an uploaded image file (multipart/form-data) and output the 4 alert states."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Tệp tin ảnh rỗng.")

    try:
        raw_pred = classifier.predict_image(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Lỗi đọc file ảnh: {exc}")

    if apply_smoothing:
        pred = _default_smoother.update(raw_pred)
    else:
        pred = raw_pred

    alert_info = warning_engine.evaluate(
        road_type=pred["road_type"],
        road_name_vi=pred["road_name_vi"],
        confidence=pred["confidence"],
        current_speed=speed,
        vehicle_type=vehicle_type,
        caution_margin=caution_margin,
        frame_index=10,
    )

    boxes = [BoundingBox(**b) for b in pred.get("boxes", [])]

    return PredictResponse(
        road_type=pred["road_type"],
        road_name_vi=pred["road_name_vi"],
        road_class_id=pred["road_class_id"],
        confidence=pred["confidence"],
        confidence_percent=pred["confidence_percent"],
        probabilities=pred["probabilities"],
        speed_limit=alert_info["speed_limit"],
        current_speed=alert_info["current_speed"],
        diff_speed=alert_info["diff_speed"],
        alert_level=alert_info["alert_level"],
        alert_label=alert_info["alert_label"],
        explanation=alert_info["explanation"],
        rule_description=alert_info["rule_description"],
        legal_citation="Thông tư 31/2019/TT-BGTVT",
        boxes=boxes,
        latency_ms=pred.get("latency_ms", 0.0),
        smoothed=pred.get("smoothed", False),
    )


@router.post("/video", response_model=VideoPredictSummary, summary="Process short dashcam video clip")
async def predict_video_file(
    file: UploadFile = File(..., description="MP4/WebM dashcam video file"),
    speed: Optional[float] = Form(None, description="Vehicle speed in km/h for testing alerts"),
    sample_every_n_frames: int = Form(10, description="Process every N frames for fast evaluation"),
    classifier: RoadClassifierService = Depends(get_classifier),
    warning_engine: WarningEngine = Depends(get_warning_engine),
) -> VideoPredictSummary:
    """Analyze a dashcam video clip and determine the dominant road class and speed alert."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Không thể mở hoặc giải mã tệp video.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        dur_s = round(total_frames / fps, 2)

        votes: dict[str, list[float]] = {}
        frame_idx = 0
        latencies = []

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx += 1

            if frame_idx % sample_every_n_frames == 1:
                t0 = time.perf_counter()
                pred = classifier.predict_image(frame)
                latencies.append((time.perf_counter() - t0) * 1000.0)
                k = pred["road_type"]
                if k not in votes:
                    votes[k] = []
                votes[k].append(pred["confidence"])

        cap.release()

        if not votes:
            raise HTTPException(status_code=400, detail="Không trích xuất được khung hình nào từ video.")

        # Determine dominant road class
        dominant_class = max(votes.keys(), key=lambda c: (len(votes[c]), sum(votes[c])))
        avg_conf = float(sum(votes[dominant_class]) / len(votes[dominant_class]))
        road_name_vi = classifier.CLASS_VI_MAP.get(dominant_class, dominant_class)

        alert_info = warning_engine.evaluate(
            road_type=dominant_class,
            road_name_vi=road_name_vi,
            confidence=avg_conf,
            current_speed=speed,
            frame_index=10,
        )

        avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

        return VideoPredictSummary(
            filename=file.filename or "video.mp4",
            total_frames=total_frames,
            fps=round(fps, 1),
            duration_seconds=dur_s,
            dominant_road_type=dominant_class,
            dominant_road_name_vi=road_name_vi,
            average_confidence=round(avg_conf, 3),
            legal_speed_limit=alert_info["speed_limit"],
            tested_speed=speed,
            final_alert_level=alert_info["alert_level"],
            final_alert_label=alert_info["alert_label"],
            final_explanation=alert_info["explanation"],
            latency_ms_per_frame=avg_lat,
        )
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


@router.post("/reset-stream", summary="Reset temporal smoother buffer")
async def reset_stream_smoother() -> dict[str, Any]:
    """Reset the rolling buffer of predictions for starting a new camera/video session."""
    _default_smoother.reset()
    return {"status": "ok", "message": "Đã đặt lại bộ đệm làm mượt thời gian (temporal smoother)."}


@router.get("/demo-scenarios", summary="Get pre-configured demo scenarios")
async def get_demo_scenarios() -> list[dict[str, Any]]:
    """Return the 5 interactive demo scenarios matching the VisionDrive AI dashboard."""
    return [
        {
            "chip": "Cao tốc",
            "cls": 0,
            "road_type": "cao_toc",
            "name_vi": "Cao tốc",
            "limit": 100,
            "max_limit": 120,
            "main": 93,
            "probs": [93, 3, 1, 3, 0],
            "rule": "Cao tốc: giới hạn theo biển báo cao tốc, tối đa 100 - 120 km/h.",
            "speeds": [88, 98, 104, 110, 115, 122, 126, 118, 105, 95],
            "boxes": [
                {"x": 360, "y": 232, "w": 58, "h": 42, "label": "ô tô 96%"},
                {"x": 474, "y": 206, "w": 32, "h": 24, "label": "xe tải 88%"},
            ],
        },
        {
            "chip": "Đô thị đôi",
            "cls": 1,
            "road_type": "duong_doi_do_thi",
            "name_vi": "Đường đôi đô thị",
            "limit": 60,
            "max_limit": 60,
            "main": 91,
            "probs": [2, 91, 5, 1, 1],
            "rule": "Khu đông dân cư, đường đôi có dải phân cách: tối đa 60 km/h.",
            "speeds": [48, 52, 55, 58, 62, 66, 61, 56, 52, 50],
            "boxes": [
                {"x": 322, "y": 240, "w": 62, "h": 44, "label": "ô tô 94%"},
                {"x": 478, "y": 252, "w": 34, "h": 46, "label": "xe máy 89%"},
            ],
        },
        {
            "chip": "Phố 2 chiều",
            "cls": 2,
            "road_type": "pho_hai_chieu",
            "name_vi": "Phố hai chiều",
            "limit": 50,
            "max_limit": 50,
            "main": 92,
            "probs": [1, 4, 92, 2, 1],
            "rule": "Đường phố 2 chiều không có dải phân cách giữa: tối đa 50 km/h.",
            "speeds": [38, 42, 47, 49, 53, 58, 51, 45, 40, 36],
            "boxes": [
                {"x": 340, "y": 230, "w": 55, "h": 40, "label": "ô tô 93%"},
                {"x": 460, "y": 245, "w": 30, "h": 42, "label": "xe máy 90%"},
            ],
        },
        {
            "chip": "Ngoài KDC",
            "cls": 3,
            "road_type": "ngoai_khu_dan_cu",
            "name_vi": "Ngoài khu dân cư",
            "limit": 80,
            "max_limit": 90,
            "main": 89,
            "probs": [3, 1, 5, 89, 2],
            "rule": "Ngoài khu đông dân cư: 80 km/h (đường 2 chiều) hoặc 90 km/h (đường đôi).",
            "speeds": [70, 75, 78, 82, 88, 94, 85, 80, 74, 72],
            "boxes": [
                {"x": 370, "y": 216, "w": 36, "h": 28, "label": "ô tô 92%"},
            ],
        },
        {
            "chip": "Hẻm / ngõ",
            "cls": 4,
            "road_type": "hem_ngo",
            "name_vi": "Hẻm / ngõ",
            "limit": 30,
            "max_limit": 30,
            "main": 95,
            "probs": [0, 1, 2, 2, 95],
            "rule": "Hẻm, ngõ nhỏ: ngưỡng an toàn khuyến nghị 30 km/h.",
            "speeds": [20, 24, 27, 29, 34, 38, 31, 26, 22, 18],
            "boxes": [
                {"x": 380, "y": 262, "w": 42, "h": 60, "label": "xe máy 87%"},
                {"x": 306, "y": 244, "w": 22, "h": 70, "label": "người 79%"},
            ],
        },
    ]
