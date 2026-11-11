"""Train MobileNetV3 road scene classifier on 5 Vietnamese road categories (VisionDrive AI).

5 Classes:
0: cao_toc (Cao tốc - Max 100 km/h)
1: duong_doi_do_thi (Đường đôi đô thị - Max 60 km/h)
2: pho_hai_chieu (Phố hai chiều - Max 50 km/h)
3: ngoai_khu_dan_cu (Ngoài khu dân cư - Max 80 km/h)
4: hem_ngo (Hẻm / ngõ - Max 30 km/h)
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("visiondrive.train")

CLASS_NAMES = ["cao_toc", "duong_doi_do_thi", "pho_hai_chieu", "ngoai_khu_dan_cu", "hem_ngo"]
CLASS_VI_MAP = {
    "cao_toc": "Cao tốc",
    "duong_doi_do_thi": "Đường đôi đô thị",
    "pho_hai_chieu": "Phố hai chiều",
    "ngoai_khu_dan_cu": "Ngoài khu dân cư",
    "hem_ngo": "Hẻm / ngõ",
}


def extract_keyframe(video_path: Path) -> np.ndarray:
    """Extract and pre-resize middle keyframe to 224x224."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 25
    mid_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
    else:
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class MemoryRoadDataset(Dataset):
    """Dataset with all pre-resized images in RAM for high-throughput GPU training."""

    def __init__(self, images: list[np.ndarray], labels: list[int], transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img_arr = self.images[idx]
        label = self.labels[idx]
        pil_img = Image.fromarray(img_arr)
        if self.transform:
            tensor = self.transform(pil_img)
        else:
            tensor = transforms.functional.to_tensor(pil_img)
        return tensor, label


def load_dataset_keyframes(data_dir: Path) -> tuple[list[np.ndarray], list[int]]:
    """Scan and load keyframes from 5 category directories."""
    all_items = []
    for idx, class_name in enumerate(CLASS_NAMES):
        folder = data_dir / class_name
        if not folder.exists():
            folder = data_dir / "videos" / class_name
        videos = sorted(list(folder.glob("*.mp4"))) + sorted(list(folder.glob("*.webm")))
        logger.info("Found %d clips for class [%s]", len(videos), class_name)
        for v in videos:
            all_items.append((v, idx))

    if not all_items:
        raise FileNotFoundError(f"No video files found in {data_dir}")

    logger.info("Loading %d keyframes into RAM with ThreadPoolExecutor...", len(all_items))
    t0 = time.time()

    def worker(item):
        v_path, label_idx = item
        frame = extract_keyframe(v_path)
        return frame, label_idx

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(worker, all_items))

    images = [r[0] for r in results]
    labels = [r[1] for r in results]
    logger.info("Loaded %d keyframes in %.1f seconds", len(images), time.time() - t0)
    return images, labels


def build_model(num_classes: int = 5) -> nn.Module:
    """Build MobileNetV3-Small classifier."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes),
    )
    return model


def train_model(
    data_dir: Path,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
    output_model: Path = Path("models/vietnam_road_classifier.pt"),
    device_name: str | None = None,
):
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Training on device: %s", device)

    images, labels = load_dataset_keyframes(data_dir)

    # 80/20 train/val split
    combined = list(zip(images, labels))
    random.seed(42)
    random.shuffle(combined)
    split_idx = int(0.80 * len(combined))
    train_data = combined[:split_idx]
    val_data = combined[split_idx:]

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = MemoryRoadDataset([x[0] for x in train_data], [x[1] for x in train_data], transform=train_tf)
    val_ds = MemoryRoadDataset([x[0] for x in val_data], [x[1] for x in val_data], transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    model = build_model(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    output_model.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_b)
            preds = out.argmax(dim=1)
            train_correct += (preds == y_b).sum().item()
            total_train += len(y_b)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        class_correct = [0] * 5
        class_total = [0] * 5

        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                out = model(x_v)
                loss = criterion(out, y_v)
                val_loss += loss.item() * len(y_v)
                preds = out.argmax(dim=1)
                val_correct += (preds == y_v).sum().item()
                total_val += len(y_v)
                for p, y in zip(preds.cpu().numpy(), y_v.cpu().numpy()):
                    class_total[y] += 1
                    if p == y:
                        class_correct[y] += 1

        train_acc = (train_correct / total_train) * 100
        val_acc = (val_correct / total_val) * 100
        dur = time.time() - t_ep

        logger.info(
            "Epoch %d/%d (%.2fs) | Train Loss: %.4f Acc: %.2f%% | Val Loss: %.4f Acc: %.2f%%",
            epoch, epochs, dur, train_loss / total_train, train_acc, val_loss / total_val, val_acc
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(output_model))
            logger.info("Saved best model (Val Acc: %.2f%%) to %s", best_val_acc, output_model)

    logger.info("Training complete! Best Validation Accuracy: %.2f%%", best_val_acc)
    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV3 Road Classifier")
    parser.add_argument("--data-dir", default="data/videos", help="Directory with categorized video folders")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", default="models/vietnam_road_classifier.pt", help="Output model path")
    args = parser.parse_args()

    train_model(
        data_dir=Path(args.data_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_model=Path(args.output),
    )
