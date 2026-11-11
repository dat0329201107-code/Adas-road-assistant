"""VisionDrive AI Backend - Main FastAPI Application.

Course Project: Đề tài 6 - Notification for Driver based on type of road (VisionDrive AI)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.feedback import router as feedback_router
from app.routes.predict import router as predict_router
from app.routes.rules import router as rules_router
from app.schemas import HealthResponse
from app.services.classifier import RoadClassifierService
from app.services.warning import WarningEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("visiondrive.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle manager."""
    logger.info("=" * 65)
    logger.info("   KHOI DONG VISIONDRIVE AI BACKEND API (DE TAI 6)")
    logger.info("   Nhan dien 5 loai duong Viet Nam + 4 trang thai canh bao")
    logger.info("=" * 65)

    # 1. Initialize classifier service
    classifier = RoadClassifierService()
    app.state.classifier = classifier

    # 2. Initialize warning engine
    warning_engine = WarningEngine()
    app.state.warning_engine = warning_engine

    logger.info("Classifier Status: loaded=%s, device=%s", classifier.loaded, classifier.device)
    logger.info("Speed Rules Status: %d road classes loaded", len(warning_engine.rules_data.get("road_classes", {})))

    yield

    logger.info("VisionDrive AI Backend shutting down...")


app = FastAPI(
    title="VisionDrive AI Backend API",
    description=(
        "Backend API hỗ trợ đồ án môn học: Hệ thống AI nhận diện 5 loại đường Việt Nam "
        "(Cao tốc, Đường đôi đô thị, Phố hai chiều, Ngoài khu dân cư, Hẻm / ngõ) và "
        "thực thi 4 trạng thái cảnh báo tốc độ cho tài xế (Đang xác định, An toàn, Chú ý, Vượt tốc) "
        "kèm lời giải thích minh bạch theo Thông tư 31/2019/TT-BGTVT."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for Next.js frontend / web UI / mobile clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(predict_router, prefix="/api")
app.include_router(rules_router, prefix="/api")
app.include_router(feedback_router, prefix="/api")

# Pre-initialize state for testing and synchronous access
app.state.classifier = RoadClassifierService()
app.state.warning_engine = WarningEngine()


@app.get("/", summary="Root API Information")
async def root():
    """Root endpoint with project information and API documentation links."""
    return {
        "project": "VisionDrive AI",
        "description": "Notification for Driver based on type of road (Đề tài 6)",
        "version": "1.0.0",
        "documentation": "/docs",
        "api_endpoints": {
            "predict_frame": "POST /api/predict/frame",
            "predict_image": "POST /api/predict/image",
            "predict_video": "POST /api/predict/video",
            "demo_scenarios": "GET /api/predict/demo-scenarios",
            "rules": "GET /api/rules",
            "feedback": "GET /api/feedback, POST /api/feedback",
            "health": "GET /health",
        },
    }


@app.get("/health", response_model=HealthResponse, summary="Service Healthcheck")
async def health_check():
    """Verify backend status and model readiness."""
    classifier: RoadClassifierService = getattr(app.state, "classifier", None)
    is_loaded = classifier.loaded if classifier else False
    dev = classifier.device if classifier else "unknown"

    import torch
    gpu = torch.cuda.is_available() if hasattr(torch, "cuda") else False

    return HealthResponse(
        status="ok",
        project="VisionDrive AI",
        version="1.0.0",
        model_loaded=is_loaded,
        device=dev,
        gpu_available=gpu,
    )
