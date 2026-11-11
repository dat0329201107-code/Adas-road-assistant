"""Train a deep neural network to classify 5 Vietnamese road types on RTX 4050 GPU.

5 Classes:
0: cao_toc (Cao tốc - Max 100 km/h)
1: duong_doi_do_thi (Đường đôi đô thị - Max 60 km/h)
2: pho_hai_chieu (Phố hai chiều - Max 50 km/h)
3: ngoai_khu_dan_cu (Ngoài khu dân cư - Max 80 km/h)
4: hem_ngo (Hẻm / ngõ - Max 30 km/h)

Optimized for ultra-fast GPU training via memory-cached keyframes.
Saves weights to: models/vietnam_road_classifier.pt
"""
from __future__ import annotations

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

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
    """Dataset with all images pre-loaded into memory for blazing fast GPU training."""

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


def load_all_video_keyframes(video_root: Path):
    """Load middle keyframes of all 3,000 videos using ThreadPoolExecutor."""
    logger.info("Pre-loading keyframes from 3,000 videos into memory...")
    t0 = time.time()

    all_items = []
    for idx, class_name in enumerate(CLASS_NAMES):
        folder = video_root / class_name
        videos = sorted(list(folder.glob("*.mp4")))
        logger.info("Found %d videos for class %s", len(videos), class_name)
        for v in videos:
            all_items.append((v, idx))

    # Multithreaded frame extraction
    def worker(item):
        v_path, label_idx = item
        frame = extract_keyframe(v_path)
        return frame, label_idx

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker, all_items))

    images = [r[0] for r in results]
    labels = [r[1] for r in results]

    elapsed = time.time() - t0
    logger.info("Loaded %d keyframes into RAM in %.1f seconds!", len(images), elapsed)
    return images, labels


def train_model(
    base_dir: Path,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    video_root = base_dir / "data" / "videos"
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info("Training on device: %s (%s)", device, gpu_name)

    images, labels = load_all_video_keyframes(video_root)

    # 80/20 train/val split with stratification
    indices_by_class: dict[int, list[int]] = {i: [] for i in range(len(CLASS_NAMES))}
    for idx, l in enumerate(labels):
        indices_by_class[l].append(idx)

    train_indices = []
    val_indices = []
    for c, idxs in indices_by_class.items():
        random.seed(42 + c)
        random.shuffle(idxs)
        split = int(len(idxs) * 0.8)
        train_indices.extend(idxs[:split])
        val_indices.extend(idxs[split:])

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    train_imgs = [images[i] for i in train_indices]
    train_lbls = [labels[i] for i in train_indices]
    val_imgs = [images[i] for i in val_indices]
    val_lbls = [labels[i] for i in val_indices]

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MemoryRoadDataset(train_imgs, train_lbls, transform=train_transform)
    val_dataset = MemoryRoadDataset(val_imgs, val_lbls, transform=val_transform)

    # num_workers=0 avoids Windows multiprocessing overhead and guarantees 0 deadlocks
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info("Initializing MobileNetV3-Small architecture with pretrained weights...")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(128, len(CLASS_NAMES)),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_state = None
    history = []

    logger.info("Training on %d train samples, %d val samples...", len(train_imgs), len(val_imgs))
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for batch_imgs, batch_lbls in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_lbls = batch_lbls.to(device)

            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_lbls)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == batch_lbls).sum().item()
            total_train += batch_imgs.size(0)

        scheduler.step()
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = train_correct / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        class_correct = [0] * len(CLASS_NAMES)
        class_total = [0] * len(CLASS_NAMES)

        with torch.no_grad():
            for batch_imgs, batch_lbls in val_loader:
                batch_imgs = batch_imgs.to(device)
                batch_lbls = batch_lbls.to(device)

                outputs = model(batch_imgs)
                loss = criterion(outputs, batch_lbls)

                val_loss += loss.item() * batch_imgs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == batch_lbls).sum().item()
                total_val += batch_imgs.size(0)

                for p, l in zip(preds, batch_lbls):
                    if p == l:
                        class_correct[l.item()] += 1
                    class_total[l.item()] += 1

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = val_correct / total_val
        elapsed = time.time() - t0

        logger.info(
            "Epoch %d/%d (%.2fs) | Train Loss: %.4f Acc: %.2f%% | Val Loss: %.4f Acc: %.2f%%",
            epoch, epochs, elapsed, epoch_train_loss, epoch_train_acc * 100, epoch_val_loss, epoch_val_acc * 100
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 4),
            "val_loss": round(epoch_val_loss, 4),
            "val_acc": round(epoch_val_acc, 4),
        })

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save trained model weights
    output_model_path = models_dir / "vietnam_road_classifier.pt"
    logger.info("Saving best model (Val Acc: %.2f%%) to %s", best_val_acc * 100, output_model_path)
    torch.save({
        "state_dict": best_state if best_state else model.state_dict(),
        "classes": CLASS_NAMES,
        "classes_vi": CLASS_VI_MAP,
        "val_acc": best_val_acc,
        "architecture": "mobilenet_v3_small",
    }, output_model_path)

    # Save evaluation report
    report_path = models_dir / "road_classifier_eval.json"
    per_class_acc = {
        CLASS_NAMES[i]: round(class_correct[i] / max(1, class_total[i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }
    eval_report = {
        "best_val_accuracy_percent": round(best_val_acc * 100, 2),
        "per_class_accuracy_percent": per_class_acc,
        "total_train_samples": len(train_imgs),
        "total_val_samples": len(val_imgs),
        "history": history,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)

    logger.info("Evaluation report saved to: %s", report_path)
    logger.info("Per-class Accuracy: %s", per_class_acc)
    return eval_report


if __name__ == "__main__":
    cwd = Path.cwd()
    if (cwd / "adas-road-assistant").exists():
        base = cwd / "adas-road-assistant"
    else:
        base = cwd
    train_model(base, epochs=8, batch_size=64)
