"""Export trained PyTorch MobileNetV3 model to ONNX format for web/edge inference."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("visiondrive.export")


def build_model(num_classes: int = 5) -> nn.Module:
    """Build MobileNetV3-Small classifier."""
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes),
    )
    return model


def export_to_onnx(
    checkpoint_path: Path = Path("models/vietnam_road_classifier.pt"),
    output_onnx: Path = Path("models/vietnam_road_classifier.onnx"),
    opset_version: int = 14,
):
    """Export checkpoint to ONNX."""
    logger.info("Loading PyTorch model from %s", checkpoint_path)
    model = build_model(num_classes=5)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    st = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(st)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting to ONNX format at %s...", output_onnx)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_onnx),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    logger.info("ONNX model successfully exported: %s (size: %.2f MB)", output_onnx, output_onnx.stat().st_size / (1024 * 1024))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--checkpoint", default="models/vietnam_road_classifier.pt", help="Path to .pt model")
    parser.add_argument("--output", default="models/vietnam_road_classifier.onnx", help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=Path(args.checkpoint),
        output_onnx=Path(args.output),
        opset_version=args.opset,
    )
