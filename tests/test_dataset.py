"""Tests for traffic sign dataset integrity and configuration."""
import json
from pathlib import Path
import pytest
import yaml

DATASETS_DIR = Path(__file__).resolve().parent.parent / "data" / "datasets"
VIETNAM_DIR = DATASETS_DIR / "vietnam_traffic_signs"


def test_vietnam_dataset_files_exist():
    """Verify essential dataset files are present."""
    if not VIETNAM_DIR.exists():
        pytest.skip("Vietnam dataset directory not downloaded yet")

    assert (VIETNAM_DIR / "classid.csv").exists()
    assert (VIETNAM_DIR / "data.yaml").exists()
    assert (VIETNAM_DIR / "classes_bilingual.json").exists()


def test_vietnam_data_yaml_structure():
    """Verify data.yaml has proper YOLO format and classes."""
    yaml_path = VIETNAM_DIR / "data.yaml"
    if not yaml_path.exists():
        pytest.skip("data.yaml does not exist")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "train" in data
    assert "val" in data
    assert "nc" in data
    assert data["nc"] == 82
    assert "names" in data
    assert len(data["names"]) == 82


def test_vietnam_bilingual_json():
    """Verify bilingual JSON contains speed limit translations."""
    json_path = VIETNAM_DIR / "classes_bilingual.json"
    if not json_path.exists():
        pytest.skip("classes_bilingual.json does not exist")

    with open(json_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    assert len(mapping) == 82
    # Verify ID 2 is 40km/h
    assert "2" in mapping
    assert "40" in mapping["2"]["vi"]
