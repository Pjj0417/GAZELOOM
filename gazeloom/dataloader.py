import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import gazeloom.utils as utils


def load_data_vat(file_path: str, sample_rate: int = 1) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        sequences = json.load(f)

    data = []
    for sequence in sequences:
        frames = sequence["frames"]
        for idx in range(0, len(frames), sample_rate):
            data.append(frames[idx])

    return data


def load_data_gazefollow(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


class GazeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        path: str,
        split: str,
        transform,
        in_frame_only: bool = True,
        sample_rate: int = 1,
        depth_resize: Tuple[int, int] = (256, 256),
        heatmap_size: Tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()

        self.dataset_name = dataset_name.lower()
        self.path = Path(path)
        self.split = split
        self.transform = transform
        self.in_frame_only = in_frame_only
        self.sample_rate = sample_rate
        self.depth_resize = depth_resize
        self.heatmap_size = heatmap_size
        self.aug = split == "train"

        self.depth_transform = transforms.ToTensor()

        self.data = self._load_annotations()
        self.data_idxs = self._build_index()

    def _load_annotations(self) -> List[Dict[str, Any]]:
        annotation_file = self.path / f"{self.split}_preprocessed.json"

        if self.dataset_name == "gazefollow":
            return load_data_gazefollow(str(annotation_file))

        if self.dataset_name == "videoattentiontarget":
            return load_data_vat(str(annotation_file), sample_rate=self.sample_rate)

        raise ValueError(f"Invalid dataset name: {self.dataset_name}")

    def _build_index(self) -> List[Tuple[int, int]]:
        data_idxs = []

        for image_idx, image_data in enumerate(self.data):
            for head_idx, head_data in enumerate(image_data["heads"]):
                inout = head_data["inout"]

                if self.in_frame_only and inout != 1:
                    continue

                data_idxs.append((image_idx, head_idx))

        return data_idxs

    def _get_image_path(self, relative_path: str) -> Path:
        return self.path / self.dataset_name / relative_path

    def _get_depth_path(self, relative_path: str) -> Path:
        return self.path / "depth" / relative_path

    def _load_rgb_image(self, relative_path: str) -> Image.Image:
        image_path = self._get_image_path(relative_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        return Image.open(image_path).convert("RGB")

    def _load_depth_image(self, relative_path: str) -> Image.Image:
        depth_path = self._get_depth_path(relative_path)

        if not depth_path.exists():
            raise FileNotFoundError(f"Depth file not found: {depth_path}")

        return Image.open(depth_path).convert("L")

    def __getitem__(self, idx: int):
        image_idx, head_idx = self.data_idxs[idx]

        image_data = self.data[image_idx]
        head_data = copy.deepcopy(image_data["heads"][head_idx])

        relative_path = image_data["path"]

        image = self._load_rgb_image(relative_path)
        depth_image = self._load_depth_image(relative_path)

        width, height = image.size

        bbox_norm = head_data["bbox_norm"]
        gazex_norm = head_data["gazex_norm"]
        gazey_norm = head_data["gazey_norm"]
        inout = head_data["inout"]

        if self.aug:
            image, depth_image, bbox_norm, gazex_norm, gazey_norm, width, height = (
                self._apply_train_augmentation(
                    image=image,
                    depth_image=depth_image,
                    head_data=head_data,
                    inout=inout,
                )
            )

        image_tensor = self.transform(image)
        depth_tensor = self._process_depth(depth_image)

        inout_tensor = torch.tensor(inout, dtype=torch.float32)

        if self.split == "train":
            heatmap = utils.get_heatmap(
                gazex_norm[0],
                gazey_norm[0],
                self.heatmap_size[0],
                self.heatmap_size[1],
            )

            return (
                image_tensor,
                depth_tensor,
                bbox_norm,
                gazex_norm,
                gazey_norm,
                inout_tensor,
                height,
                width,
                heatmap,
            )

        return (
            image_tensor,
            depth_tensor,
            bbox_norm,
            gazex_norm,
            gazey_norm,
            inout_tensor,
            height,
            width,
        )

    def _apply_train_augmentation(
        self,
        image: Image.Image,
        depth_image: Image.Image,
        head_data: Dict[str, Any],
        inout: int,
    ):
        bbox = head_data["bbox"]
        gazex = head_data["gazex"]
        gazey = head_data["gazey"]

        if np.random.rand() <= 0.5:
            image, depth_image, bbox, gazex, gazey = utils.random_crop_with_depth(
                image,
                depth_image,
                bbox,
                gazex,
                gazey,
                inout,
            )

        if np.random.rand() <= 0.5:
            image, depth_image, bbox, gazex, gazey = utils.horiz_flip_with_depth(
                image,
                depth_image,
                bbox,
                gazex,
                gazey,
                inout,
            )

        if np.random.rand() <= 0.5:
            bbox = utils.random_bbox_jitter(image, bbox)

        width, height = image.size

        bbox_norm = [
            bbox[0] / width,
            bbox[1] / height,
            bbox[2] / width,
            bbox[3] / height,
        ]

        gazex_norm = [x / float(width) for x in gazex]
        gazey_norm = [y / float(height) for y in gazey]

        return image, depth_image, bbox_norm, gazex_norm, gazey_norm, width, height

    def _process_depth(self, depth_image: Image.Image) -> torch.Tensor:
        depth_image = depth_image.resize(
            self.depth_resize,
            Image.Resampling.BILINEAR,
        )
        return self.depth_transform(depth_image)

    def __len__(self) -> int:
        return len(self.data_idxs)


def collate_fn(batch: Sequence[Tuple[Any, ...]]) -> Tuple[Any, ...]:
    transposed = list(zip(*batch))

    output = []
    for items in transposed:
        first_item = items[0]

        if isinstance(first_item, torch.Tensor):
            output.append(torch.stack(items))
        else:
            output.append(list(items))

    return tuple(output)
