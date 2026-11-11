"""Feedback API endpoints for VisionDrive AI (Báo sai & Gắn nhãn bổ sung)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from fastapi import APIRouter, HTTPException

from app.schemas import FeedbackCreate, FeedbackItem, FeedbackListResponse

router = APIRouter(prefix="/feedback", tags=["Feedback & Data Flagging"])

# In-memory store for user feedback flags
_feedback_store: list[FeedbackItem] = []
_feedback_file = Path("data/feedback_flags.json")


def _load_stored_feedback():
    global _feedback_store
    if _feedback_file.exists():
        try:
            with open(_feedback_file, "r", encoding="utf-8") as f:
                items = json.load(f)
                _feedback_store = [FeedbackItem(**it) for it in items]
        except Exception:
            _feedback_store = []


def _save_feedback():
    try:
        _feedback_file.parent.mkdir(parents=True, exist_ok=True)
        with open(_feedback_file, "w", encoding="utf-8") as f:
            json.dump([it.model_dump() for it in _feedback_store], f, ensure_ascii=False, indent=2)
    except Exception:
        pass


_load_stored_feedback()


@router.post("", response_model=FeedbackItem, summary="Submit misclassified sample feedback")
async def submit_feedback(payload: FeedbackCreate) -> FeedbackItem:
    """Submit a user correction for an incorrectly classified road frame.
    Matches the 'Báo sai' feature in VisionDrive AI dashboard.
    """
    new_id = len(_feedback_store) + 1
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    item = FeedbackItem(
        id=new_id,
        created_at=now_str,
        predicted_class=payload.predicted_class,
        correct_class=payload.correct_class,
        current_speed=payload.current_speed,
        timestamp_str=payload.timestamp_str or now_str,
        notes=payload.notes,
        image_base64=payload.image_base64,
    )
    _feedback_store.append(item)
    _save_feedback()
    return item


@router.get("", response_model=FeedbackListResponse, summary="List all submitted feedback items")
async def list_feedback() -> FeedbackListResponse:
    """List all user-flagged misclassification samples for model retraining."""
    return FeedbackListResponse(total=len(_feedback_store), items=_feedback_store)


@router.delete("", summary="Clear all feedback items")
async def clear_feedback() -> dict[str, Any]:
    """Clear all stored feedback items."""
    _feedback_store.clear()
    _save_feedback()
    return {"status": "ok", "message": "Đã xóa toàn bộ phản hồi."}
