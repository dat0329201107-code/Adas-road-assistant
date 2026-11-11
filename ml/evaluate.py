"""Evaluation script generating Confusion Matrix, Accuracy, and F1-Scores (VisionDrive AI)."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from ml.train import CLASS_NAMES, CLASS_VI_MAP, MemoryRoadDataset, build_model, load_dataset_keyframes

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("visiondrive.eval")


def evaluate_model(
    model_path: Path = Path("models/vietnam_road_classifier.pt"),
    data_dir: Path = Path("data/videos"),
    output_report: Path = Path("models/road_classifier_eval.json"),
    device_name: str | None = None,
) -> dict[str, Any]:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Evaluating on device: %s using model %s", device, model_path)

    images, labels = load_dataset_keyframes(data_dir)

    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = MemoryRoadDataset(images, labels, transform=val_tf)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    model = build_model(num_classes=5)
    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    st = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(st)
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_b, y_b in loader:
            x_b = x_b.to(device)
            out = model(x_b)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_b.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 5x5 Confusion Matrix: rows = actual, columns = predicted
    num_classes = len(CLASS_NAMES)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        matrix[t, p] += 1

    total_samples = len(all_targets)
    total_correct = int(np.trace(matrix))
    overall_acc = (total_correct / total_samples) * 100

    row_sums = matrix.sum(axis=1)  # actual counts
    col_sums = matrix.sum(axis=0)  # predicted counts

    class_metrics = []
    f1_list = []

    for i, name in enumerate(CLASS_NAMES):
        tp = matrix[i, i]
        precision = (tp / col_sums[i]) * 100 if col_sums[i] > 0 else 0.0
        recall = (tp / row_sums[i]) * 100 if row_sums[i] > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_list.append(f1)
        class_metrics.append({
            "class_id": i,
            "class_key": name,
            "name_vi": CLASS_VI_MAP[name],
            "total_samples": int(row_sums[i]),
            "precision_pct": round(precision, 2),
            "recall_pct": round(recall, 2),
            "f1_score_pct": round(f1, 2),
        })

    macro_f1 = sum(f1_list) / len(f1_list)

    report = {
        "dataset": "Vietnam Real Road Dashcam Dataset",
        "total_evaluated_samples": total_samples,
        "overall_accuracy_pct": round(overall_acc, 2),
        "macro_f1_score_pct": round(macro_f1, 2),
        "confusion_matrix": matrix.tolist(),
        "class_names": [CLASS_VI_MAP[c] for c in CLASS_NAMES],
        "class_keys": CLASS_NAMES,
        "per_class_metrics": class_metrics,
    }

    output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 65)
    print("          KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH VISIONDRIVE AI")
    print("=" * 65)
    print(f"Tổng số mẫu đánh giá : {total_samples}")
    print(f"Độ chính xác tổng thể : {overall_acc:.2f}%")
    print(f"Macro F1-Score        : {macro_f1:.2f}%")
    print("-" * 65)
    print(f"{'Loại đường':<22} {'Mẫu':<6} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 65)
    for cm in class_metrics:
        print(f"{cm['name_vi']:<22} {cm['total_samples']:<6} {cm['precision_pct']:>6.1f}%      {cm['recall_pct']:>6.1f}%    {cm['f1_score_pct']:>6.1f}%")
    print("-" * 65)
    print("Ma trận nhầm lẫn (Confusion Matrix):")
    print(matrix)
    print("=" * 65)
    logger.info("Saved evaluation report to: %s", output_report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Road Classifier")
    parser.add_argument("--model", default="models/vietnam_road_classifier.pt", help="Model weights path")
    parser.add_argument("--data-dir", default="data/videos", help="Data directory")
    parser.add_argument("--output", default="models/road_classifier_eval.json", help="Output report JSON")
    args = parser.parse_args()

    evaluate_model(
        model_path=Path(args.model),
        data_dir=Path(args.data_dir),
        output_report=Path(args.output),
    )
