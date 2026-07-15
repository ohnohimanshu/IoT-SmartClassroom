"""
Fine-tune mc3_18 for fight detection.

Produces a checkpoint file that fight_detection_3dcnn.py's
FightDetector3DCNN._load_weights() will load correctly (it accepts a dict
with a 'model_state_dict' key, which is exactly what this script saves).

BEFORE RUNNING:
1. Get a dataset of short video clips labeled fight / no-fight
   (RWF-2000 or the Kaggle "Real Life Violence Situations" dataset — see
   the chat message this came with for links).
2. Organize your downloaded clips into this folder structure:

     dataset/
       train/
         fight/       <- video files (.mp4/.avi) of fighting
         nofight/     <- video files of normal activity
       val/
         fight/
         nofight/

3. Install the one extra dependency this needs:
     pip install opencv-python torchvision torch

Usage:
    python train_fight_detector.py --data_dir ./dataset --epochs 20

Output:
    model_weights/fight_mc3_18_finetuned.pth
    (copy this into classroom_monitor/../model_weights/ on your server,
    matching FIGHT_MODEL_WEIGHTS_PATH / the default path your file expects)
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Must match fight_detection_3dcnn.py exactly ──────────────────────────────
SEQUENCE_LENGTH = 16     # frames per clip fed to the model
INPUT_SIZE      = 112    # frame resize (H=W=112)
# Effective sample rate your live system feeds the model at (~2fps heavy
# detection tier). We sample training clips at roughly the same rate so the
# model isn't learning on smoother/choppier motion than it'll see live.
TARGET_SAMPLE_FPS = 2.0

MEAN = [0.43216, 0.394666, 0.37645]   # Kinetics-400 normalization,
STD  = [0.22803, 0.22145, 0.216989]   # matches the mc3_18 pretrained backbone


def sample_clip_frames(video_path: str, num_frames: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Read a video file and return `num_frames` frames sampled at
    TARGET_SAMPLE_FPS, resized to INPUT_SIZE x INPUT_SIZE, RGB."""
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(1, int(round(src_fps / TARGET_SAMPLE_FPS)))
    frame_indices = list(range(0, total_frames, step))[:num_frames]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        frames.append(frame)
    cap.release()

    # Pad by repeating the last frame if the clip was too short
    while len(frames) < num_frames and frames:
        frames.append(frames[-1])
    if not frames:
        frames = [np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)] * num_frames

    return np.stack(frames[:num_frames])  # (T, H, W, C)


class FightClipDataset(Dataset):
    def __init__(self, root_dir: str, augment: bool = False):
        self.samples = []  # list of (path, label)
        root = Path(root_dir)
        for label_name, label in (('fight', 1), ('nofight', 0)):
            folder = root / label_name
            if not folder.exists():
                continue
            for f in folder.iterdir():
                if f.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
                    self.samples.append((str(f), label))
        self.augment = augment
        if not self.samples:
            raise RuntimeError(
                f"No video files found under {root_dir}/fight or {root_dir}/nofight — "
                f"check your dataset folder structure.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = sample_clip_frames(path)  # (T, H, W, C), uint8

        if self.augment and random.random() < 0.5:
            frames = frames[:, :, ::-1, :].copy()  # horizontal flip

        clip = frames.astype(np.float32) / 255.0
        for c in range(3):
            clip[..., c] = (clip[..., c] - MEAN[c]) / STD[c]

        # (T, H, W, C) -> (C, T, H, W) for torchvision's video models
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).float()
        return clip, label


def build_model(num_classes: int = 2) -> nn.Module:
    from torchvision.models.video import mc3_18, MC3_18_Weights
    model = mc3_18(weights=MC3_18_Weights.KINETICS400_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[TRAIN] Using device: {device}')

    train_ds = FightClipDataset(os.path.join(args.data_dir, 'train'), augment=True)
    val_ds   = FightClipDataset(os.path.join(args.data_dir, 'val'),   augment=False)
    print(f'[TRAIN] {len(train_ds)} training clips, {len(val_ds)} validation clips')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    out_path = Path(args.out_dir) / 'fight_mc3_18_finetuned.pth'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * clips.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / max(total, 1)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for clips, labels in val_loader:
                clips, labels = clips.to(device), labels.to(device)
                outputs = model(clips)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / max(val_total, 1)

        print(f'[TRAIN] Epoch {epoch+1}/{args.epochs} — '
              f'loss={running_loss/max(total,1):.4f} '
              f'train_acc={train_acc:.3f} val_acc={val_acc:.3f}')

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(),
                        'val_acc': val_acc,
                        'num_classes': 2}, out_path)
            print(f'[TRAIN] Saved new best checkpoint (val_acc={val_acc:.3f}) -> {out_path}')

    print(f'[TRAIN] Done. Best val_acc={best_val_acc:.3f}. Checkpoint at {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                         help='Folder containing train/ and val/ subfolders, each with fight/ and nofight/')
    parser.add_argument('--out_dir', type=str, default='./model_weights')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
