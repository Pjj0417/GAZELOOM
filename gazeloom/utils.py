import random
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score


Tensor = torch.Tensor
BBox = Sequence[float]


def repeat_tensors(tensor: Tensor, repeat_counts: Sequence[int]) -> Tensor:
    repeated = [
        tensor[i : i + 1].repeat(repeat, *([1] * (tensor.ndim - 1)))
        for i, repeat in enumerate(repeat_counts)
    ]
    return torch.cat(repeated, dim=0)


def split_tensors(tensor: Tensor, split_counts: Sequence[int]) -> List[Tensor]:
    indices = torch.cumsum(
        torch.tensor([0] + list(split_counts), device=tensor.device),
        dim=0,
    )
    return [
        tensor[indices[i] : indices[i + 1]]
        for i in range(len(split_counts))
    ]


def stack_and_pad(tensor_list: Sequence[Tensor]) -> Tensor:
    if len(tensor_list) == 0:
        raise ValueError("tensor_list must not be empty.")

    max_size = max(t.shape[0] for t in tensor_list)
    padded_list = []

    for tensor in tensor_list:
        if tensor.shape[0] == max_size:
            padded_list.append(tensor)
            continue

        pad_shape = (max_size - tensor.shape[0], *tensor.shape[1:])
        padding = torch.zeros(
            pad_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padded_list.append(torch.cat([tensor, padding], dim=0))

    return torch.stack(padded_list, dim=0)


def visualize_heatmap(
    pil_image: Image.Image,
    heatmap: Union[Tensor, np.ndarray],
    bbox: BBox = None,
    alpha: int = 128,
) -> Image.Image:
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()

    heatmap = np.asarray(heatmap)
    heatmap = np.clip(heatmap, 0.0, 1.0)

    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize(pil_image.size, Image.Resampling.BILINEAR)

    import matplotlib.pyplot as plt

    heatmap_color = plt.cm.jet(np.asarray(heatmap_img) / 255.0)
    heatmap_color = (heatmap_color[:, :, :3] * 255).astype(np.uint8)

    overlay = Image.fromarray(heatmap_color).convert("RGBA")
    overlay.putalpha(alpha)

    output = Image.alpha_composite(pil_image.convert("RGBA"), overlay)

    if bbox is not None:
        width, height = pil_image.size
        xmin, ymin, xmax, ymax = bbox

        draw = ImageDraw.Draw(output)
        draw.rectangle(
            [
                xmin * width,
                ymin * height,
                xmax * width,
                ymax * height,
            ],
            outline="green",
            width=3,
        )

    return output


def random_crop(
    image: Image.Image,
    bbox: BBox,
    gazex: Sequence[float],
    gazey: Sequence[float],
    inout: int,
):
    width, height = image.size
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox

    if inout:
        crop_xmin = min(bbox_xmin, min(gazex))
        crop_ymin = min(bbox_ymin, min(gazey))
        crop_xmax = max(bbox_xmax, max(gazex))
        crop_ymax = max(bbox_ymax, max(gazey))
    else:
        crop_xmin = bbox_xmin
        crop_ymin = bbox_ymin
        crop_xmax = bbox_xmax
        crop_ymax = bbox_ymax

    crop_xmin = max(0, int(crop_xmin))
    crop_ymin = max(0, int(crop_ymin))
    crop_xmax = min(width, int(crop_xmax))
    crop_ymax = min(height, int(crop_ymax))

    left = random.randint(0, crop_xmin)
    top = random.randint(0, crop_ymin)
    right = random.randint(crop_xmax, width)
    bottom = random.randint(crop_ymax, height)

    image = TF.crop(image, top, left, bottom - top, right - left)

    bbox = [
        bbox_xmin - left,
        bbox_ymin - top,
        bbox_xmax - left,
        bbox_ymax - top,
    ]

    gazex = [x - left for x in gazex]
    gazey = [y - top for y in gazey]

    return image, bbox, gazex, gazey


def horiz_flip(
    image: Image.Image,
    bbox: BBox,
    gazex: Sequence[float],
    gazey: Sequence[float],
    inout: int,
):
    width, _ = image.size
    image = TF.hflip(image)

    xmin, ymin, xmax, ymax = bbox
    bbox = [width - xmax, ymin, width - xmin, ymax]

    if inout:
        gazex = [width - x for x in gazex]

    return image, bbox, gazex, gazey


def random_bbox_jitter(
    image: Image.Image,
    bbox: BBox,
    jitter: float = 0.2,
) -> List[float]:
    width, height = image.size
    xmin, ymin, xmax, ymax = bbox

    box_w = xmax - xmin
    box_h = ymax - ymin

    xmin_j = (np.random.random_sample() * 2.0 - 1.0) * jitter * box_w
    xmax_j = (np.random.random_sample() * 2.0 - 1.0) * jitter * box_w
    ymin_j = (np.random.random_sample() * 2.0 - 1.0) * jitter * box_h
    ymax_j = (np.random.random_sample() * 2.0 - 1.0) * jitter * box_h

    return [
        max(0.0, xmin + xmin_j),
        max(0.0, ymin + ymin_j),
        min(float(width), xmax + xmax_j),
        min(float(height), ymax + ymax_j),
    ]


def get_heatmap(
    gazex: float,
    gazey: float,
    height: int,
    width: int,
    sigma: int = 3,
    htype: str = "Gaussian",
) -> Tensor:
    heatmap = torch.zeros(height, width)

    if gazex < 0 or gazey < 0:
        return heatmap

    gazex = int(gazex * width)
    gazey = int(gazey * height)

    ul = [int(gazex - 3 * sigma), int(gazey - 3 * sigma)]
    br = [int(gazex + 3 * sigma + 1), int(gazey + 3 * sigma + 1)]

    if (
        ul[0] >= width
        or ul[1] >= height
        or br[0] < 0
        or br[1] < 0
    ):
        return heatmap

    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = size // 2
    y0 = size // 2

    if htype == "Gaussian":
        kernel = np.exp(
            -((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2)
        )
    elif htype == "Cauchy":
        kernel = sigma / (
            ((x - x0) ** 2 + (y - y0) ** 2 + sigma**2) ** 1.5
        )
    else:
        raise ValueError(f"Unsupported heatmap type: {htype}")

    g_x = max(0, -ul[0]), min(br[0], width) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], height) - ul[1]

    img_x = max(0, ul[0]), min(br[0], width)
    img_y = max(0, ul[1]), min(br[1], height)

    heatmap[
        img_y[0] : img_y[1],
        img_x[0] : img_x[1],
    ] += torch.from_numpy(
        kernel[g_y[0] : g_y[1], g_x[0] : g_x[1]]
    ).float()

    max_value = heatmap.max()
    if max_value > 0:
        heatmap = heatmap / max_value

    return heatmap


def _safe_auc(target_map: np.ndarray, score_map: np.ndarray) -> float:
    target = target_map.flatten()
    score = score_map.flatten()

    if len(np.unique(target)) < 2:
        return float("nan")

    return float(roc_auc_score(target, score))


def gazefollow_auc(
    heatmap: Tensor,
    gt_gazex: Sequence[float],
    gt_gazey: Sequence[float],
    height: int,
    width: int,
) -> float:
    target_map = np.zeros((height, width), dtype=np.float32)

    for x_norm, y_norm in zip(gt_gazex, gt_gazey):
        if x_norm < 0 or y_norm < 0:
            continue

        x = int(x_norm * float(width))
        y = int(y_norm * float(height))

        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)

        target_map[y, x] = 1.0

    resized_heatmap = torch.nn.functional.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    score_map = resized_heatmap.detach().cpu().numpy()
    return _safe_auc(target_map, score_map)


def gazefollow_l2(
    heatmap: Tensor,
    gt_gazex: Sequence[float],
    gt_gazey: Sequence[float],
) -> Tuple[float, float]:
    argmax = heatmap.flatten().argmax().item()
    pred_y, pred_x = np.unravel_index(
        argmax,
        (heatmap.shape[0], heatmap.shape[1]),
    )

    pred_x = pred_x / float(heatmap.shape[1])
    pred_y = pred_y / float(heatmap.shape[0])

    gazex = np.asarray(gt_gazex, dtype=np.float32)
    gazey = np.asarray(gt_gazey, dtype=np.float32)

    valid = (gazex >= 0) & (gazey >= 0)
    gazex = gazex[valid]
    gazey = gazey[valid]

    if len(gazex) == 0:
        return float("nan"), float("nan")

    avg_l2 = np.sqrt(
        (pred_x - gazex.mean()) ** 2 + (pred_y - gazey.mean()) ** 2
    )

    all_l2s = np.sqrt((pred_x - gazex) ** 2 + (pred_y - gazey) ** 2)
    min_l2 = all_l2s.min().item()

    return float(avg_l2), float(min_l2)


def vat_auc(
    heatmap: Tensor,
    gt_gazex: float,
    gt_gazey: float,
    res: int = 64,
    sigma: int = 3,
) -> float:
    if heatmap.shape[0] != res or heatmap.shape[1] != res:
        raise ValueError(f"Expected heatmap shape ({res}, {res}), got {heatmap.shape}")

    target_map = np.zeros((res, res), dtype=np.float32)

    gazex = gt_gazex * res
    gazey = gt_gazey * res

    ul = [
        max(0, int(gazex - 3 * sigma)),
        max(0, int(gazey - 3 * sigma)),
    ]
    br = [
        min(int(gazex + 3 * sigma + 1), res),
        min(int(gazey + 3 * sigma + 1), res),
    ]

    target_map[ul[1] : br[1], ul[0] : br[0]] = 1.0

    score_map = heatmap.detach().cpu().numpy()
    return _safe_auc(target_map, score_map)


def vat_l2(
    heatmap: Tensor,
    gt_gazex: float,
    gt_gazey: float,
    res: int = 64,
) -> float:
    argmax = heatmap.flatten().argmax().item()
    pred_y, pred_x = np.unravel_index(argmax, (res, res))

    pred_x = pred_x / float(res)
    pred_y = pred_y / float(res)

    l2 = np.sqrt((pred_x - gt_gazex) ** 2 + (pred_y - gt_gazey) ** 2)
    return float(l2)


def random_crop_with_depth(
    image: Image.Image,
    depth_image: Image.Image,
    bbox: BBox,
    gazex: Sequence[float],
    gazey: Sequence[float],
    inout: int,
):
    width, height = image.size
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox

    if inout:
        crop_xmin = min(bbox_xmin, min(gazex))
        crop_ymin = min(bbox_ymin, min(gazey))
        crop_xmax = max(bbox_xmax, max(gazex))
        crop_ymax = max(bbox_ymax, max(gazey))
    else:
        crop_xmin = bbox_xmin
        crop_ymin = bbox_ymin
        crop_xmax = bbox_xmax
        crop_ymax = bbox_ymax

    crop_xmin = max(0, int(crop_xmin))
    crop_ymin = max(0, int(crop_ymin))
    crop_xmax = min(width, int(crop_xmax))
    crop_ymax = min(height, int(crop_ymax))

    left = random.randint(0, crop_xmin)
    top = random.randint(0, crop_ymin)
    right = random.randint(crop_xmax, width)
    bottom = random.randint(crop_ymax, height)

    image = TF.crop(image, top, left, bottom - top, right - left)
    depth_image = TF.crop(depth_image, top, left, bottom - top, right - left)

    bbox = [
        bbox_xmin - left,
        bbox_ymin - top,
        bbox_xmax - left,
        bbox_ymax - top,
    ]

    gazex = [x - left for x in gazex]
    gazey = [y - top for y in gazey]

    return image, depth_image, bbox, gazex, gazey


def horiz_flip_with_depth(
    image: Image.Image,
    depth_image: Image.Image,
    bbox: BBox,
    gazex: Sequence[float],
    gazey: Sequence[float],
    inout: int,
):
    width, _ = image.size

    image = TF.hflip(image)
    depth_image = TF.hflip(depth_image)

    xmin, ymin, xmax, ymax = bbox
    bbox = [width - xmax, ymin, width - xmin, ymax]

    if inout:
        gazex = [width - x for x in gazex]

    return image, depth_image, bbox, gazex, gazey
