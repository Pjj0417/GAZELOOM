import os
import glob
import copy
import random
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


# =========================================================
# Utils
# =========================================================
def get_head_box_channel(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    size: Tuple[int, int],
    coordconv: bool = False,
):
    h, w = size
    head = torch.zeros(h, w)

    x_min = int(max(0, x_min * w))
    y_min = int(max(0, y_min * h))
    x_max = int(min(w, x_max * w))
    y_max = int(min(h, y_max * h))

    head[y_min:y_max, x_min:x_max] = 1.0

    if not coordconv:
        return head.unsqueeze(0)

    y_channel = torch.linspace(0, 1, h).view(-1, 1).repeat(1, w)
    x_channel = torch.linspace(0, 1, w).view(1, -1).repeat(h, 1)

    return torch.stack([head, y_channel, x_channel], dim=0)


def draw_labelmap(
    pt,
    sigma,
    output_size=(64, 64),
    htype="Gaussian",
):
    img = np.zeros(output_size, dtype=np.float32)

    if pt[0] <= 0 or pt[1] <= 0:
        return img

    x, y = pt
    ul = [int(x - 3 * sigma), int(y - 3 * sigma)]
    br = [int(x + 3 * sigma + 1), int(y + 3 * sigma + 1)]

    if ul[0] >= output_size[1] or ul[1] >= output_size[0] or br[0] < 0 or br[1] < 0:
        return img

    size = 6 * sigma + 1
    xx = np.arange(0, size, 1, float)
    yy = xx[:, np.newaxis]
    center = size // 2

    if htype == "Gaussian":
        g = np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma**2))
    else:
        g = sigma / (((xx - center) ** 2 + (yy - center) ** 2 + sigma**2) ** 1.5)

    g_x = max(0, -ul[0]), min(br[0], output_size[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], output_size[0]) - ul[1]

    img_x = max(0, ul[0]), min(br[0], output_size[1])
    img_y = max(0, ul[1]), min(br[1], output_size[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] += g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    if img.max() > 0:
        img = img / img.max()

    return img


# =========================================================
# Dataset
# =========================================================
class ChildPlay(Dataset):
    def __init__(
        self,
        data_dir: str,
        ann_dir: str,
        transform,
        input_size_scene=(448, 448),
        input_size_human=(224, 224),
        output_size=(64, 64),
        skip_frame=1,
        mode="train",
        subset="full",
        max_samples=None,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.ann_dir = Path(ann_dir)
        self.transform = transform

        self.input_size_scene = input_size_scene
        self.input_size_human = input_size_human
        self.output_size = output_size

        self.mode = mode
        self.subset = subset
        self.max_samples = max_samples

        self.annotations = self._load_annotations(skip_frame)

    # -----------------------------------------------------
    def _load_annotations(self, stride=1):
        files = glob.glob(str(self.ann_dir / "**/*.csv"), recursive=True)

        if len(files) == 0:
            raise RuntimeError(f"No annotation files found in {self.ann_dir}")

        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

        if self.mode == "train" and stride > 1:
            df = df.iloc[::stride]

        # remove known bad samples
        df = df.drop(
            df[(df["clip"] == "4yWavYq9_Ks_405-451") & (df.frame == 48)].index
        )

        df = df.drop(
            df[(df["gaze_class"] == "inside_visible") & (df["gaze_x"] == -1)].index
        )

        if self.subset == "child":
            df = df[df["is_child"] == 1]
        elif self.subset == "adult":
            df = df[df["is_child"] == 0]

        if self.max_samples is not None:
            df = df[: self.max_samples]

        print(f"Loaded {len(df)} samples")

        return df.reset_index(drop=True)

    # -----------------------------------------------------
    def __getitem__(self, idx):
        item = self.annotations.iloc[idx]

        # --- load image ---
        clip = item["clip"]
        video_id, interval = clip.replace("-downsample", "").rsplit("_", 1)
        offset = int(interval.split("-")[0])

        frame_nb = item["frame"]
        img_name = f"{video_id}_{offset + frame_nb - 1}.jpg"
        path = os.path.join(clip, img_name)

        img = Image.open(self.data_dir / "images" / path).convert("RGB")

        width, height = img.size

        # --- bbox (pixel) ---
        x_min = item["bbox_x"]
        y_min = item["bbox_y"]
        x_max = x_min + item["bbox_width"]
        y_max = y_min + item["bbox_height"]

        # normalize bbox
        x_min /= width
        x_max /= width
        y_min /= height
        y_max /= height

        # --- gaze ---
        gaze_x = item["gaze_x"]
        gaze_y = item["gaze_y"]
        gaze_inside = item["gaze_class"] == "inside_visible"

        if gaze_inside:
            gaze_x /= width
            gaze_y /= height
        else:
            gaze_x, gaze_y = -1.0, -1.0

        # -------------------------------------------------
        # augmentation
        if self.mode == "train" and random.random() < 0.5:
            img = TF.hflip(img)
            x_min, x_max = 1 - x_max, 1 - x_min
            if gaze_inside:
                gaze_x = 1 - gaze_x

        # -------------------------------------------------
        # resize
        img = img.resize(self.input_size_scene)

        # face crop
        face = img.crop(
            (
                int(x_min * self.input_size_scene[0]),
                int(y_min * self.input_size_scene[1]),
                int(x_max * self.input_size_scene[0]),
                int(y_max * self.input_size_scene[1]),
            )
        ).resize(self.input_size_human)

        # head channel
        head_channel = get_head_box_channel(
            x_min,
            y_min,
            x_max,
            y_max,
            self.input_size_scene,
        )

        # -------------------------------------------------
        # heatmap
        heatmap = np.zeros(self.output_size, dtype=np.float32)

        if gaze_inside:
            px = gaze_x * self.output_size[1]
            py = gaze_y * self.output_size[0]

            heatmap = draw_labelmap(
                [px, py],
                sigma=3,
                output_size=self.output_size,
            )

        heatmap = torch.from_numpy(heatmap).float()

        # -------------------------------------------------
        # tensor transform
        if self.transform:
            img = self.transform(img)
            face = self.transform(face)

        bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

        return (
            img,
            bbox,
            torch.tensor(gaze_x, dtype=torch.float32),
            torch.tensor(gaze_y, dtype=torch.float32),
            torch.tensor(float(gaze_inside)),
            height,
            width,
            heatmap,
        )

    # -----------------------------------------------------
    def __len__(self):
        return len(self.annotations)
