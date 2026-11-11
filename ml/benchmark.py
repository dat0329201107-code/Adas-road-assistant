"""Benchmark latency (ms) and throughput (FPS) of VisionDrive AI model."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("visiondrive.benchmark")


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


def run_benchmark(
    checkpoint_path: Path = Path("models/vietnam_road_classifier.pt"),
    num_runs: int = 200,
    device_name: str | None = None,
):
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Running benchmark on %s for %d iterations...", device, num_runs)

    model = build_model(num_classes=5)
    if checkpoint_path.exists():
        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        st = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(st)
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # Warmup
    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark loop
    latencies = []
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
    total_time = time.perf_counter() - t_start

    fps = num_runs / total_time
    avg_lat = float(np.mean(latencies))
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))

    print("\n" + "=" * 60)
    print("       BENCHMARK HIỆU NĂNG MÔ HÌNH VISIONDRIVE AI")
    print("=" * 60)
    print(f"Thiết bị thử nghiệm : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Số lượt suy luận   : {num_runs}")
    print(f"Độ trễ trung bình   : {avg_lat:.2f} ms")
    print(f"Độ trễ P50 (median) : {p50:.2f} ms")
    print(f"Độ trễ P95          : {p95:.2f} ms")
    print(f"Độ trễ P99          : {p99:.2f} ms")
    print(f"Tốc độ xử lý (FPS)  : {fps:.1f} FPS")
    print("=" * 60)

    return {"device": str(device), "avg_ms": avg_lat, "p50_ms": p50, "p95_ms": p95, "fps": fps}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark model latency and FPS")
    parser.add_argument("--checkpoint", default="models/vietnam_road_classifier.pt", help="Model checkpoint path")
    parser.add_argument("--runs", type=int, default=200, help="Number of benchmark iterations")
    parser.add_argument("--device", default=None, help="Device (cpu or cuda)")
    args = parser.parse_args()

    run_benchmark(checkpoint_path=Path(args.checkpoint), num_runs=args.runs, device_name=args.device)
