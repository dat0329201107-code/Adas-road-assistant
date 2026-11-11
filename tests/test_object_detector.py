import numpy as np

from adas.vision.object_detector import DetectedObject, ObjectDetector


def test_detected_object_dataclass():
    obj = DetectedObject((100, 150, 80, 120), "car", 0.88, 15.2)
    assert obj.class_name == "car"
    assert obj.confidence == 0.88
    assert obj.estimated_distance_m == 15.2


def test_object_detector_graceful_on_missing_model():
    detector = ObjectDetector(model_path="nonexistent_model.pt")
    loaded = detector.load()
    if not loaded:
        assert not detector.available
        assert detector.load_error is not None
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert detector.detect(frame) == []
