"""Unit and Integration Tests for VisionDrive AI Warning System and Backend API."""
from __future__ import annotations

import base64
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.schemas import AlertLevelEnum, VehicleTypeEnum
from app.services.classifier import RoadClassifierService
from app.services.smoothing import TemporalSmoother
from app.services.warning import WarningEngine


@pytest.fixture
def warning_engine() -> WarningEngine:
    return WarningEngine(min_confidence=0.60, default_caution_margin=5.0, min_warmup_frames=3)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_warning_state_1_idle_warmup(warning_engine: WarningEngine):
    """Trạng thái 1: ĐANG XÁC ĐỊNH khi chưa đủ khung hình camera."""
    res = warning_engine.evaluate(
        road_type="duong_doi_do_thi",
        road_name_vi="Đường đôi đô thị",
        confidence=0.92,
        current_speed=50.0,
        frame_index=1,  # < min_warmup_frames (3)
    )
    assert res["alert_level"] == AlertLevelEnum.IDLE
    assert res["alert_label"] == "ĐANG XÁC ĐỊNH"
    assert "Đang khởi động camera" in res["explanation"]


def test_warning_state_1_idle_low_confidence(warning_engine: WarningEngine):
    """Trạng thái 1: ĐANG XÁC ĐỊNH khi độ tin cậy thấp (< 60%)."""
    res = warning_engine.evaluate(
        road_type="duong_doi_do_thi",
        road_name_vi="Đường đôi đô thị",
        confidence=0.45,  # 45% < 60%
        current_speed=50.0,
        frame_index=10,
    )
    assert res["alert_level"] == AlertLevelEnum.IDLE
    assert "độ tin cậy thấp" in res["explanation"]


def test_warning_state_2_ok_safe(warning_engine: WarningEngine):
    """Trạng thái 2: AN TOÀN khi xe chạy trong giới hạn tốc độ."""
    res = warning_engine.evaluate(
        road_type="pho_hai_chieu",
        road_name_vi="Phố hai chiều",
        confidence=0.92,
        current_speed=40.0,  # Limit is 50, caution is 45-50
        frame_index=10,
    )
    assert res["alert_level"] == AlertLevelEnum.OK
    assert res["alert_label"] == "AN TOÀN"
    assert res["speed_limit"] == 50
    assert "Tốc độ an toàn" in res["explanation"]
    assert "Phố hai chiều" in res["explanation"]


def test_warning_state_3_warn_caution(warning_engine: WarningEngine):
    """Trạng thái 3: CHÚ Ý khi xe chạy gần chạm giới hạn (50 - 5 = 45 đến 50 km/h)."""
    res = warning_engine.evaluate(
        road_type="pho_hai_chieu",
        road_name_vi="Phố hai chiều",
        confidence=0.92,
        current_speed=48.0,
        caution_margin=5.0,
        frame_index=10,
    )
    assert res["alert_level"] == AlertLevelEnum.WARN
    assert res["alert_label"] == "CHÚ Ý"
    assert "gần chạm giới hạn" in res["explanation"]
    assert "Phố hai chiều" in res["explanation"]


def test_warning_state_4_bad_overspeed(warning_engine: WarningEngine):
    """Trạng thái 4: VƯỢT TỐC khi xe chạy quá giới hạn cho phép."""
    res = warning_engine.evaluate(
        road_type="pho_hai_chieu",
        road_name_vi="Phố hai chiều",
        confidence=0.92,
        current_speed=58.0,  # 58 > 50 (+8 km/h)
        frame_index=10,
    )
    assert res["alert_level"] == AlertLevelEnum.BAD
    assert res["alert_label"] == "VƯỢT TỐC"
    assert res["diff_speed"] == 8.0
    assert "VƯỢT QUÁ +8 km/h" in res["explanation"]
    assert "giảm tốc ngay" in res["explanation"]


def test_custom_caution_margin(warning_engine: WarningEngine):
    """Ngưỡng 'chú ý' có thể điều chỉnh trong cài đặt."""
    # Khi ngưỡng là 10 km/h: 73 km/h trên đường 80 km/h sẽ là CHÚ Ý
    res_wide = warning_engine.evaluate(
        road_type="ngoai_khu_dan_cu",
        road_name_vi="Ngoài khu dân cư",
        confidence=0.90,
        current_speed=73.0,
        caution_margin=10.0,
        frame_index=10,
    )
    assert res_wide["alert_level"] == AlertLevelEnum.WARN

    # Khi ngưỡng là 3 km/h: 73 km/h trên đường 80 km/h sẽ là AN TOÀN
    res_narrow = warning_engine.evaluate(
        road_type="ngoai_khu_dan_cu",
        road_name_vi="Ngoài khu dân cư",
        confidence=0.90,
        current_speed=73.0,
        caution_margin=3.0,
        frame_index=10,
    )
    assert res_narrow["alert_level"] == AlertLevelEnum.OK


def test_temporal_smoother():
    """Kiểm tra dịch vụ làm mượt thời gian (TemporalSmoother)."""
    smoother = TemporalSmoother(window_size=5, alpha=0.50)

    # Khung hình ban đầu phân loại là cao_toc
    sample = {
        "road_type": "cao_toc",
        "road_name_vi": "Cao tốc",
        "road_class_id": 0,
        "confidence": 0.90,
        "probabilities": {"cao_toc": 0.90, "duong_doi_do_thi": 0.05, "pho_hai_chieu": 0.02, "ngoai_khu_dan_cu": 0.02, "hem_ngo": 0.01},
    }
    out1 = smoother.update(sample)
    assert out1["road_type"] == "cao_toc"
    assert out1["smoothed"] is True

    # 1 khung hình bị nhiễu nhảy sang hem_ngo với xác suất thấp
    noise = {
        "road_type": "hem_ngo",
        "road_name_vi": "Hẻm / ngõ",
        "road_class_id": 4,
        "confidence": 0.55,
        "probabilities": {"cao_toc": 0.40, "duong_doi_do_thi": 0.02, "pho_hai_chieu": 0.01, "ngoai_khu_dan_cu": 0.02, "hem_ngo": 0.55},
    }
    out2 = smoother.update(noise)
    # Nhờ EMA và hysteresis, kết quả vẫn giữ vững cao_toc, không bị nhấp nháy!
    assert out2["road_type"] == "cao_toc"


def test_api_health(client: TestClient):
    """Kiểm tra endpoint /health."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["project"] == "VisionDrive AI"


def test_api_rules(client: TestClient):
    """Kiểm tra endpoint /api/rules."""
    res = client.get("/api/rules")
    assert res.status_code == 200
    data = res.json()
    assert "cao_toc" in data["road_classes"]
    assert data["road_classes"]["cao_toc"]["legal_max_speed"] == 100
    assert data["road_classes"]["duong_doi_do_thi"]["legal_max_speed"] == 60


def test_api_demo_scenarios(client: TestClient):
    """Kiểm tra endpoint /api/predict/demo-scenarios."""
    res = client.get("/api/predict/demo-scenarios")
    assert res.status_code == 200
    scenarios = res.json()
    assert len(scenarios) == 5
    assert scenarios[0]["road_type"] == "cao_toc"


def test_api_predict_frame(client: TestClient):
    """Kiểm tra endpoint POST /api/predict/frame với ảnh base64."""
    # Tạo 1 ảnh nhỏ 100x100 RGB
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "image_base64": b64_str,
        "current_speed": 45.0,
        "caution_margin": 5.0,
        "apply_smoothing": False,
    }
    res = client.post("/api/predict/frame", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "road_type" in data
    assert "alert_level" in data
    assert "explanation" in data
    assert data["current_speed"] == 45.0


def test_api_feedback_flow(client: TestClient):
    """Kiểm tra quy trình Báo sai (Feedback) từ người dùng."""
    payload = {
        "predicted_class": "Đường đôi đô thị",
        "correct_class": "Cao tốc",
        "current_speed": 90.0,
        "notes": "Có dải phân cách cứng bê tông và 3 làn xe chạy nhanh",
    }
    post_res = client.post("/api/feedback", json=payload)
    assert post_res.status_code == 200
    item = post_res.json()
    assert item["predicted_class"] == "Đường đôi đô thị"
    assert item["correct_class"] == "Cao tốc"

    get_res = client.get("/api/feedback")
    assert get_res.status_code == 200
    list_data = get_res.json()
    assert list_data["total"] >= 1
