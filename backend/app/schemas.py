"""Pydantic schemas for VisionDrive AI Backend API."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class RoadTypeEnum(str, Enum):
    """5 Vietnamese Road Types defined in VisionDrive AI."""

    CAO_TOC = "cao_toc"
    DUONG_DOI_DO_THI = "duong_doi_do_thi"
    PHO_HAI_CHIEU = "pho_hai_chieu"
    NGOAI_KHU_DAN_CU = "ngoai_khu_dan_cu"
    HEM_NGO = "hem_ngo"


class AlertLevelEnum(str, Enum):
    """4 Alert States defined in VisionDrive AI."""

    IDLE = "idle"  # ĐANG XÁC ĐỊNH
    OK = "ok"      # AN TOÀN
    WARN = "warn"  # CHÚ Ý
    BAD = "bad"    # VƯỢT TỐC


class VehicleTypeEnum(str, Enum):
    """Vehicle types supported for differential speed limits."""

    CAR = "car"                # Ô tô con
    MOTORCYCLE = "motorcycle"  # Xe mô tô, xe gắn máy
    TRUCK = "truck"            # Xe tải
    BUS = "bus"                # Xe buýt / xe khách


class BoundingBox(BaseModel):
    """Bounding box detected in road scene."""

    x: int = Field(..., description="Top-left X coordinate in pixels")
    y: int = Field(..., description="Top-left Y coordinate in pixels")
    w: int = Field(..., description="Box width in pixels")
    h: int = Field(..., description="Box height in pixels")
    label: str = Field(..., description="Object name with confidence, e.g. 'ô tô 95%'")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)


class PredictFrameRequest(BaseModel):
    """Request payload for single frame prediction."""

    image_base64: Optional[str] = Field(None, description="Base64 encoded JPEG or PNG image frame")
    current_speed: Optional[float] = Field(None, description="Current vehicle speed in km/h")
    vehicle_type: VehicleTypeEnum = Field(default=VehicleTypeEnum.CAR, description="Type of vehicle")
    caution_margin: Optional[float] = Field(default=5.0, description="Margin in km/h below speed limit for 'Chú ý' state")
    apply_smoothing: bool = Field(default=True, description="Whether to apply temporal smoothing across frames")


class PredictResponse(BaseModel):
    """Comprehensive response for road type classification and 4-state alert."""

    road_type: str = Field(..., description="Class key (e.g. cao_toc, duong_doi_do_thi)")
    road_name_vi: str = Field(..., description="Vietnamese road name (e.g. Cao tốc, Hẻm / ngõ)")
    road_class_id: int = Field(..., description="Class integer ID (0 to 4)")
    confidence: float = Field(..., description="Model prediction confidence (0.0 to 1.0)")
    confidence_percent: float = Field(..., description="Confidence as percentage (0 to 100)")
    probabilities: dict[str, float] = Field(..., description="Probability distribution across all 5 classes")
    speed_limit: Optional[int] = Field(None, description="Legal maximum speed limit in km/h")
    current_speed: Optional[float] = Field(None, description="Current vehicle speed in km/h")
    diff_speed: Optional[float] = Field(None, description="Speed difference compared to speed limit (+/- km/h)")
    alert_level: AlertLevelEnum = Field(..., description="One of the 4 alert states: idle, ok, warn, bad")
    alert_label: str = Field(..., description="Vietnamese alert label: ĐANG XÁC ĐỊNH, AN TOÀN, CHÚ Ý, VƯỢT TỐC")
    explanation: str = Field(..., description="Clear explanation sentence for the alert")
    rule_description: str = Field(..., description="Description of the legal rule")
    legal_citation: str = Field(default="Thông tư 31/2019/TT-BGTVT", description="Legal regulatory reference")
    boxes: list[BoundingBox] = Field(default_factory=list, description="Detected objects (vehicles, signs, pedestrians)")
    latency_ms: float = Field(default=0.0, description="Processing latency in milliseconds")
    smoothed: bool = Field(default=False, description="Whether result was smoothed temporally")


class VideoPredictSummary(BaseModel):
    """Summary of processing an entire video clip."""

    filename: str
    total_frames: int
    fps: float
    duration_seconds: float
    dominant_road_type: str
    dominant_road_name_vi: str
    average_confidence: float
    legal_speed_limit: Optional[int]
    tested_speed: Optional[float]
    final_alert_level: AlertLevelEnum
    final_alert_label: str
    final_explanation: str
    latency_ms_per_frame: float


class SpeedRuleItem(BaseModel):
    """Speed rule definition for a road class."""

    class_id: int
    class_key: str
    name_vi: str
    name_en: str
    legal_max_speed: int
    max_allowed_speed: int
    min_allowed_speed: Optional[int]
    default_caution_margin: float
    description: str
    rule_detail: str
    vehicle_limits: dict[str, int]


class SpeedRulesResponse(BaseModel):
    """Response containing all legal speed rules."""

    legal_framework: str
    version: str
    road_classes: dict[str, SpeedRuleItem]
    alert_levels: dict[str, Any]


class FeedbackCreate(BaseModel):
    """Payload for user flagging an incorrect classification."""

    predicted_class: str = Field(..., description="Class predicted by model")
    correct_class: str = Field(..., description="Correct class specified by user")
    current_speed: Optional[float] = Field(None, description="Vehicle speed at time of error")
    timestamp_str: Optional[str] = Field(None, description="Time formatted as mm:ss or ISO")
    notes: Optional[str] = Field(None, description="Optional user note")
    image_base64: Optional[str] = Field(None, description="Optional frame image thumbnail")


class FeedbackItem(FeedbackCreate):
    """Stored feedback item with assigned ID and creation timestamp."""

    id: int
    created_at: str


class FeedbackListResponse(BaseModel):
    """List of all submitted feedback items."""

    total: int
    items: list[FeedbackItem]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    project: str
    version: str
    model_loaded: bool
    device: str
    gpu_available: bool
