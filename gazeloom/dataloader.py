import copy
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import gazeloom.utils as utils


ImageSize = Tuple[int, int]


DATASET_ALIASES = {
    "gazefollow": "gazefollow",
    "gaze_follow": "gazefollow",

    "videoattentiontarget": "videoattentiontarget",
    "video_attention_target": "videoattentiontarget",
    "vat": "videoattentiontarget",

    "gooreal": "gooreal",
    "goo_real": "gooreal",
    "goo-real": "gooreal",

    "goosynth": "goosynth",
    "goo_synth": "goosynth",
    "goo-synth": "goosynth",
}


DEFAULT_IMAGE_DIRS = {
    "gazefollow": "gazefollow",
    "videoattentiontarget": "videoattentiontarget",
    "gooreal": "gooreal",
    "goosynth": "goosynth",
}


class GazeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        path: str,
        split: str,
        transform=None,
        annotation_file: Optional[str] = None,
        image_root: Optional[str] = None,
        depth_root: Optional[str] = None,
        in_frame_only: bool = True,
        sample_rate: int = 1,
        depth_resize: ImageSize = (256, 256),
        heatmap_size: ImageSize = (64, 64),
        image_resize: Optional[ImageSize] = None,
        use_depth: bool = True,
        missing_depth: str = "zeros",
    ) -> None:
        super().__init__()

        dataset_key = dataset_name.lower()
        if dataset_key not in DATASET_ALIASES:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self.dataset_name = DATASET_ALIASES[dataset_key]
        self.path = Path(path)
        self.split = split
        self.transform = transform or transforms.ToTensor()
        self.annotation_file = Path(annotation_file) if annotation_file else None

        self.image_root = Path(image_root) if image_root else self.path / DEFAULT_IMAGE_DIRS[self.dataset_name]
        self.depth_root = Path(depth_root) if depth_root else self.path / "depth"

        self.in_frame_only = in_frame_only
        self.sample_rate = sample_rate
        self.depth_resize = depth_resize
        self.heatmap_size = heatmap_size
        self.image_resize = image_resize
        self.use_depth = use_depth
        self.missing_depth = missing_depth
        self.aug = split == "train"

        if self.missing_depth not in {"zeros", "raise"}:
            raise ValueError("missing_depth must be either 'zeros' or 'raise'.")

        self.depth_transform = transforms.ToTensor()

        raw_data = self._load_annotations()
        self.records = self._build_records(raw_data)

        if len(self.records) == 0:
            raise RuntimeError(f"No valid samples found for dataset: {self.dataset_name}")

    def _find_annotation_file(self) -> Path:
        if self.annotation_file is not None:
            return self.annotation_file

        candidates = [
            self.path / f"{self.split}_preprocessed.json",
            self.path / f"{self.split}_preprocessed.pkl",
            self.path / f"{self.split}.json",
            self.path / f"{self.split}.pkl",
            self.path / f"{self.dataset_name}_{self.split}.json",
            self.path / f"{self.dataset_name}_{self.split}.pkl",
        ]

        for file_path in candidates:
            if file_path.exists():
                return file_path

        raise FileNotFoundError(
            f"Annotation file not found. Tried: {[str(p) for p in candidates]}"
        )

    def _load_annotations(self) -> Any:
        file_path = self._find_annotation_file()

        if file_path.suffix.lower() == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)

        if file_path.suffix.lower() in {".pkl", ".pickle"}:
            with open(file_path, "rb") as f:
                return pickle.load(f)

        raise ValueError(f"Unsupported annotation format: {file_path.suffix}")

    def _build_records(self, raw_data: Any) -> List[Dict[str, Any]]:
        if self.dataset_name in {"gazefollow", "videoattentiontarget"}:
            return self._build_records_from_gazefollow_style(raw_data)

        if self.dataset_name in {"gooreal", "goosynth"}:
            return self._build_records_from_goo_style(raw_data)

        raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def _build_records_from_gazefollow_style(self, raw_data: Any) -> List[Dict[str, Any]]:
        frames = []

        if self.dataset_name == "videoattentiontarget":
            for sequence in raw_data:
                sequence_frames = sequence.get("frames", [])
                for idx in range(0, len(sequence_frames), self.sample_rate):
                    frames.append(sequence_frames[idx])
        else:
            frames = raw_data

        records = []

        for image_data in frames:
            image_path = image_data["path"]

            for head_data in image_data["heads"]:
                inout = int(head_data.get("inout", 1))

                if self.in_frame_only and inout != 1:
                    continue

                records.append(
                    {
                        "path": image_path,
                        "bbox": copy.deepcopy(head_data.get("bbox")),
                        "bbox_norm": copy.deepcopy(head_data.get("bbox_norm")),
                        "gazex": self._as_list(head_data.get("gazex")),
                        "gazey": self._as_list(head_data.get("gazey")),
                        "gazex_norm": self._as_list(head_data.get("gazex_norm")),
                        "gazey_norm": self._as_list(head_data.get("gazey_norm")),
                        "inout": inout,
                    }
                )

        return records

    def _build_records_from_goo_style(self, raw_data: Any) -> List[Dict[str, Any]]:
        records = []

        for sample in raw_data:
            filename = sample.get("filename") or sample.get("path") or sample.get("image")
            if filename is None:
                continue

            ann = sample.get("ann", {})
            bboxes = ann.get("bboxes") or sample.get("bboxes")
            if not bboxes:
                continue

            bbox = list(bboxes[-1])

            gaze_x = sample.get("gaze_cx", sample.get("gazex", sample.get("gaze_x", -1)))
            gaze_y = sample.get("gaze_cy", sample.get("gazey", sample.get("gaze_y", -1)))

            gaze_x = self._first_value(gaze_x)
            gaze_y = self._first_value(gaze_y)

            inout = int(sample.get("inout", 1))

            if self.in_frame_only and inout != 1:
                continue

            records.append(
                {
                    "path": filename,
                    "bbox": bbox,
                    "bbox_norm": None,
                    "gazex": [gaze_x],
                    "gazey": [gaze_y],
                    "gazex_norm": None,
                    "gazey_norm": None,
                    "inout": inout,
                }
            )

        return records

    @staticmethod
    def _as_list(value: Any) -> List[float]:
        if value is None:
            return [-1.0]
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @staticmethod
    def _first_value(value: Any) -> float:
        if isinstance(value, (list, tuple)):
            return float(value[0])
        return float(value)

    def _resolve_path(self, root: Path, relative_path: str) -> Path:
        relative_path = Path(relative_path)

        candidates = [
            root / relative_path,
            root / relative_path.name,
            self.path / relative_path,
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]

    def _load_rgb_image(self, relative_path: str) -> Image.Image:
        image_path = self._resolve_path(self.image_root, relative_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        return Image.open(image_path).convert("RGB")

    def _load_depth_image(self, relative_path: str, image_size: ImageSize) -> Image.Image:
        depth_path = self._resolve_path(self.depth_root, relative_path)

        if depth_path.exists():
            return Image.open(depth_path).convert("L")

        if self.missing_depth == "raise":
            raise FileNotFoundError(f"Depth file not found: {depth_path}")

        return Image.new("L", image_size, 0)

    def __getitem__(self, idx: int):
        record = copy.deepcopy(self.records[idx])

        image = self._load_rgb_image(record["path"])
        original_width, original_height = image.size

        if self.use_depth:
            depth_image = self._load_depth_image(record["path"], image.size)
        else:
            depth_image = Image.new("L", image.size, 0)

        bbox = record["bbox"]
        gazex = record["gazex"]
        gazey = record["gazey"]
        inout = record["inout"]

        if self.image_resize is not None:
            image, depth_image, bbox, gazex, gazey = self._resize_sample(
                image=image,
                depth_image=depth_image,
                bbox=bbox,
                gazex=gazex,
                gazey=gazey,
                target_size=self.image_resize,
            )

        width, height = image.size

        if record["bbox_norm"] is None:
            bbox_norm = self._normalize_bbox(bbox, width, height)
        else:
            bbox_norm = record["bbox_norm"]

        if record["gazex_norm"] is None or record["gazey_norm"] is None:
            gazex_norm = self._normalize_points(gazex, width)
            gazey_norm = self._normalize_points(gazey, height)
        else:
            gazex_norm = record["gazex_norm"]
            gazey_norm = record["gazey_norm"]

        if self.aug:
            image, depth_image, bbox_norm, gazex_norm, gazey_norm, width, height = (
                self._apply_train_augmentation(
                    image=image,
                    depth_image=depth_image,
                    bbox=bbox,
                    gazex=gazex,
                    gazey=gazey,
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
            ).float()

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

    def _resize_sample(
        self,
        image: Image.Image,
        depth_image: Image.Image,
        bbox: Sequence[float],
        gazex: Sequence[float],
        gazey: Sequence[float],
        target_size: ImageSize,
    ):
        old_width, old_height = image.size
        new_width, new_height = target_size

        scale_x = new_width / float(old_width)
        scale_y = new_height / float(old_height)

        image = image.resize(target_size, Image.Resampling.BILINEAR)
        depth_image = depth_image.resize(target_size, Image.Resampling.BILINEAR)

        bbox = [
            bbox[0] * scale_x,
            bbox[1] * scale_y,
            bbox[2] * scale_x,
            bbox[3] * scale_y,
        ]

        gazex = [x * scale_x for x in gazex]
        gazey = [y * scale_y for y in gazey]

        return image, depth_image, bbox, gazex, gazey

    def _apply_train_augmentation(
        self,
        image: Image.Image,
        depth_image: Image.Image,
        bbox: Sequence[float],
        gazex: Sequence[float],
        gazey: Sequence[float],
        inout: int,
    ):
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

        bbox_norm = self._normalize_bbox(bbox, width, height)
        gazex_norm = self._normalize_points(gazex, width)
        gazey_norm = self._normalize_points(gazey, height)

        return image, depth_image, bbox_norm, gazex_norm, gazey_norm, width, height

    @staticmethod
    def _normalize_bbox(
        bbox: Sequence[float],
        width: int,
        height: int,
    ) -> List[float]:
        return [
            bbox[0] / float(width),
            bbox[1] / float(height),
            bbox[2] / float(width),
            bbox[3] / float(height),
        ]

    @staticmethod
    def _normalize_points(points: Sequence[float], size: int) -> List[float]:
        return [p / float(size) for p in points]

    def _process_depth(self, depth_image: Image.Image) -> torch.Tensor:
        depth_image = depth_image.resize(
            self.depth_resize,
            Image.Resampling.BILINEAR,
        )
        return self.depth_transform(depth_image)

    def __len__(self) -> int:
        return len(self.records)


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
