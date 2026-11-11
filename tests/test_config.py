import tempfile
from pathlib import Path

from adas.config import ROAD_LABELS, ROAD_PROMPTS, AppConfig


def test_road_prompts_not_empty():
    assert len(ROAD_PROMPTS) > 0
    assert len(ROAD_LABELS) > 0
    assert len(ROAD_LABELS) == len(ROAD_PROMPTS)


def test_default_config():
    cfg = AppConfig()
    assert cfg.camera_source == 0
    assert cfg.camera_width == 1280
    assert cfg.road_ai_enabled is True
    assert cfg.manual_limit is None
    assert cfg.log_level == "INFO"


def test_config_from_yaml():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("camera:\n  source: 1\n  width: 640\nspeed:\n  manual_limit: 60\n")
        f.flush()
        cfg = AppConfig.from_yaml(f.name)
    assert cfg.camera_source == 1
    assert cfg.camera_width == 640
    assert cfg.manual_limit == 60
    assert cfg.road_ai_enabled is True
    Path(f.name).unlink(missing_ok=True)


def test_config_from_missing_yaml():
    cfg = AppConfig.from_yaml("nonexistent_config_file.yaml")
    assert cfg.camera_source == 0


def test_config_from_empty_yaml():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("")
        f.flush()
        cfg = AppConfig.from_yaml(f.name)
    assert cfg.camera_source == 0
    Path(f.name).unlink(missing_ok=True)
