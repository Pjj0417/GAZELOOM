
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torchvision.transforms as transforms
from PIL import Image

import gazeloom.utils as utils


class GOORealDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pickle_path: str,
        image_dir: str,
        transform=None,
        split: str = "train",
        heatmap_size: Tuple[int, int] = (64, 64),
        image_size: Optional[Tuple[int, int]] = (640, 480),
        inout: float = 1.0,
        remove_invalid_files: bool = False,
    ) -> None:
        super().__init__()

        self.pickle_path = Path(pickle_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.split = split
        self.heatmap_size = heatmap_size
        self.image_size = image_size
        self.inout = inout
        self.remove_invalid_files = remove_invalid_files

        with open(self.pickle_path, "rb") as f:
            self.data = pickle.load(f)

        self.data = self._filter_valid_samples(self.data)

        if len(self.data) == 0:
            raise RuntimeError("No valid samples found in the dataset.")

    def __len__(self) -> int:
        return len(self.data)

    def _filter_valid_samples(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid_data = []

        for sample in data:
            image_path = self.image_dir / sample["filename"]

            if self._is_valid_image(image_path):
                valid_data.append(sample)
                continue

            if self.remove_invalid_files and image_path.exists():
                image_path.unlink()

            print(f"[GOORealDataset] Skipped invalid image: {image_path}")

        return valid_data

    @staticmethod
    def _is_valid_image(path: Union[str, Path]) -> bool:
        try:
            with Image.open(path) as image:
                image.verify()
            return True
        except Exception:
            return False

    @staticmethod
    def _get_first_value(value):
        if isinstance(value, list):
            return value[0]
        return value

    def _load_image(self, filename: str) -> Image.Image:
        image_path = self.image_dir / filename

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        return Image.open(image_path).convert("RGB")

    def _resize_with_annotations(
        self,
        image: Image.Image,
        bbox: List[float],
        gazex: float,
        gazey: float,
    ):
        if self.image_size is None:
            return image, bbox, gazex, gazey

        old_width, old_height = image.size
        new_width, new_height = self.image_size

        scale_x = new_width / float(old_width)
        scale_y = new_height / float(old_height)

        image = image.resize(self.image_size, Image.Resampling.BILINEAR)

        bbox = [
            bbox[0] * scale_x,
            bbox[1] * scale_y,
            bbox[2] * scale_x,
            bbox[3] * scale_y,
        ]

        gazex = gazex * scale_x
        gazey = gazey * scale_y

        return image, bbox, gazex, gazey

    def __getitem__(self, idx: int):
        sample = self.data[idx]

        filename = sample["filename"]
        bboxes = sample["ann"]["bboxes"]
        head_bbox = list(bboxes[-1])

        gaze_x = self._get_first_value(sample["gaze_cx"])
        gaze_y = self._get_first_value(sample["gaze_cy"])

        image = self._load_image(filename)
        image, head_bbox, gaze_x, gaze_y = self._resize_with_annotations(
            image=image,
            bbox=head_bbox,
            gazex=gaze_x,
            gazey=gaze_y,
        )

        if self.split == "train":
            if torch.rand(1).item() < 0.5:
                image, head_bbox, gaze_x_list, gaze_y_list = utils.random_crop(
                    image,
                    head_bbox,
                    [gaze_x],
                    [gaze_y],
                    int(self.inout),
                )
                gaze_x = gaze_x_list[0]
                gaze_y = gaze_y_list[0]

            if torch.rand(1).item() < 0.5:
                image, head_bbox, gaze_x_list, gaze_y_list = utils.horiz_flip(
                    image,
                    head_bbox,
                    [gaze_x],
                    [gaze_y],
                    int(self.inout),
                )
                gaze_x = gaze_x_list[0]
                gaze_y = gaze_y_list[0]

        width, height = image.size

        bbox_norm = [
            head_bbox[0] / width,
            head_bbox[1] / height,
            head_bbox[2] / width,
            head_bbox[3] / height,
        ]

        gaze_x_norm = [gaze_x / width]
        gaze_y_norm = [gaze_y / height]
        inout = [self.inout]

        if self.transform is not None:
            image = self.transform(image)

        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)

        heatmap = utils.get_heatmap(
            gaze_x_norm[0],
            gaze_y_norm[0],
            self.heatmap_size[0],
            self.heatmap_size[1],
        ).float()

        return (
            image,
            torch.tensor(bbox_norm, dtype=torch.float32),
            torch.tensor(gaze_x_norm, dtype=torch.float32),
            torch.tensor(gaze_y_norm, dtype=torch.float32),
            torch.tensor(inout, dtype=torch.float32),
            height,
            width,
            heatmap,
        )
