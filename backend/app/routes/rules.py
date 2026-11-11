"""Speed Rules API endpoints for VisionDrive AI."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.schemas import SpeedRuleItem, SpeedRulesResponse
from app.services.warning import WarningEngine

router = APIRouter(prefix="/rules", tags=["Speed Rules"])


class RuleUpdatePayload(BaseModel):
    """Payload to update custom speed limit for a road type."""

    legal_max_speed: int = Field(..., ge=10, le=140, description="Legal maximum speed limit in km/h")
    default_caution_margin: float = Field(default=5.0, ge=1.0, le=30.0, description="Caution margin in km/h")
    rule_detail: str | None = Field(None, description="Optional custom rule notes")


@router.get("", response_model=SpeedRulesResponse, summary="Get all road speed rules")
async def get_all_rules(request: Request) -> SpeedRulesResponse:
    """Retrieve full speed limits and legal regulations per Thông tư 31/2019/TT-BGTVT."""
    warning_engine: WarningEngine = request.app.state.warning_engine
    data = warning_engine.rules_data

    road_classes: dict[str, SpeedRuleItem] = {}
    for k, v in data.get("road_classes", {}).items():
        road_classes[k] = SpeedRuleItem(**v)

    return SpeedRulesResponse(
        legal_framework=data.get("legal_framework", "Thông tư 31/2019/TT-BGTVT"),
        version=data.get("version", "1.0.0"),
        road_classes=road_classes,
        alert_levels=data.get("alert_levels", {}),
    )


@router.get("/{road_type}", response_model=SpeedRuleItem, summary="Get rule for specific road type")
async def get_rule_by_type(road_type: str, request: Request) -> SpeedRuleItem:
    """Get speed rule for one specific road type (cao_toc, duong_doi_do_thi, etc.)."""
    warning_engine: WarningEngine = request.app.state.warning_engine
    classes = warning_engine.rules_data.get("road_classes", {})
    if road_type not in classes:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy loại đường '{road_type}'. Danh sách hợp lệ: {list(classes.keys())}",
        )
    return SpeedRuleItem(**classes[road_type])


@router.put("/{road_type}", response_model=SpeedRuleItem, summary="Update speed limit for road type")
async def update_rule_by_type(road_type: str, payload: RuleUpdatePayload, request: Request) -> SpeedRuleItem:
    """Update custom speed limit or caution margin for a specific road type."""
    warning_engine: WarningEngine = request.app.state.warning_engine
    classes = warning_engine.rules_data.get("road_classes", {})
    if road_type not in classes:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy loại đường '{road_type}'")

    classes[road_type]["legal_max_speed"] = payload.legal_max_speed
    classes[road_type]["default_caution_margin"] = payload.default_caution_margin
    if payload.rule_detail:
        classes[road_type]["rule_detail"] = payload.rule_detail

    # Persist updated rules to disk if path is writable
    try:
        if warning_engine.rules_path.exists():
            with open(warning_engine.rules_path, "w", encoding="utf-8") as f:
                json.dump(warning_engine.rules_data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        pass

    return SpeedRuleItem(**classes[road_type])
