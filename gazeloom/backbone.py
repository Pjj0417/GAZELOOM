# backbone.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm


Tensor = torch.Tensor
ImageSize = Tuple[int, int]


class Backbone(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_out_size(self, in_size: ImageSize) -> ImageSize:
        raise NotImplementedError

    def get_transform(self, in_size: ImageSize) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(in_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )


def _load_checkpoint(path: Union[str, Path]) -> Dict[str, Tensor]:
    checkpoint = torch.load(str(path), map_location="cpu")

    if isinstance(checkpoint, dict):
        for key in (
            "state_dict",
            "model",
            "teacher",
            "student",
            "module",
            "network",
            "backbone",
        ):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")

    state_dict = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue

        clean_key = key
        for prefix in (
            "module.",
            "model.",
            "teacher.",
            "student.",
            "backbone.",
            "encoder.",
        ):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]

        state_dict[clean_key] = value

    return state_dict


def _load_model_weights(
    model: nn.Module,
    checkpoint_path: Optional[Union[str, Path]],
    strict: bool = False,
) -> None:
    if checkpoint_path is None:
        return

    state_dict = _load_checkpoint(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if missing:
        print(f"[Backbone] Missing keys: {len(missing)}")

    if unexpected:
        print(f"[Backbone] Unexpected keys: {len(unexpected)}")


def _get_patch_size(model: nn.Module, default: int = 16) -> int:
    if hasattr(model, "patch_size"):
        patch_size = getattr(model, "patch_size")
        if isinstance(patch_size, tuple):
            return int(patch_size[0])
        return int(patch_size)

    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is not None and hasattr(patch_embed, "patch_size"):
        patch_size = patch_embed.patch_size
        if isinstance(patch_size, tuple):
            return int(patch_size[0])
        return int(patch_size)

    return default


def _get_embed_dim(model: nn.Module) -> int:
    if hasattr(model, "embed_dim"):
        return int(model.embed_dim)

    if hasattr(model, "num_features"):
        return int(model.num_features)

    if hasattr(model, "feature_info"):
        channels = model.feature_info.channels()
        if channels:
            return int(channels[-1])

    raise AttributeError("Unable to infer model embedding dimension.")


def _resolve_timm_name(model_name: str, aliases: Dict[str, str]) -> str:
    candidate = aliases.get(model_name, model_name)

    if candidate in timm.list_models(candidate):
        return candidate

    matches = timm.list_models(candidate)
    if candidate in matches:
        return candidate

    raw_matches = timm.list_models(model_name)
    if model_name in raw_matches:
        return model_name

    short_matches = timm.list_models(f"*{candidate}*")
    if len(short_matches) == 1:
        return short_matches[0]

    raise ValueError(
        f"Model '{model_name}' is not available in timm. "
        f"Resolved candidate: '{candidate}'. "
        f"Closest matches: {short_matches[:10]}"
    )


def _extract_patch_tokens(
    tokens: Tensor,
    expected_hw: Optional[ImageSize] = None,
    num_prefix_tokens: int = 1,
    debug: bool = False,
) -> Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected token tensor with shape [B, N, C], got {tokens.shape}")

    b, n, c = tokens.shape

    if expected_hw is not None:
        expected_h, expected_w = expected_hw
        expected_n = expected_h * expected_w

        if n == expected_n:
            return tokens

        if n >= num_prefix_tokens + expected_n:
            return tokens[:, num_prefix_tokens : num_prefix_tokens + expected_n, :]

        if n > expected_n:
            return tokens[:, -expected_n:, :]

    side = int(n ** 0.5)

    if side * side == n:
        return tokens

    if n > num_prefix_tokens:
        no_prefix = tokens[:, num_prefix_tokens:, :]
        side = int(no_prefix.shape[1] ** 0.5)
        usable_n = side * side

        if usable_n > 0:
            if debug:
                print(f"[Backbone] Cropping tokens from {no_prefix.shape[1]} to {usable_n}")
            return no_prefix[:, :usable_n, :]

    usable_n = side * side
    if usable_n <= 0:
        raise ValueError(f"Unable to convert tokens to a feature map. Token count: {n}")

    if debug:
        print(f"[Backbone] Cropping tokens from {n} to {usable_n}")

    return tokens[:, :usable_n, :]


def _tokens_to_feature_map(
    tokens: Tensor,
    expected_hw: Optional[ImageSize] = None,
    num_prefix_tokens: int = 1,
    debug: bool = False,
) -> Tensor:
    tokens = _extract_patch_tokens(
        tokens=tokens,
        expected_hw=expected_hw,
        num_prefix_tokens=num_prefix_tokens,
        debug=debug,
    )

    b, n, c = tokens.shape

    if expected_hw is not None and expected_hw[0] * expected_hw[1] == n:
        h, w = expected_hw
    else:
        h = w = int(n ** 0.5)

    return tokens.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class DinoV2Backbone(Backbone):
    MODEL_ALIASES = {
        "dinov2_vits14": "dinov2_vits14",
        "dinov2_vitb14": "dinov2_vitb14",
        "dinov2_vitl14": "dinov2_vitl14",
        "dinov2_vitg14": "dinov2_vitg14",
    }

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        checkpoint_path: Optional[Union[str, Path]] = None,
        strict: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__()

        model_name = self.MODEL_ALIASES.get(model_name, model_name)
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model_name = model_name
        self.patch_size = _get_patch_size(self.model, default=14)
        self.embed_dim = _get_embed_dim(self.model)
        self.debug = debug

        _load_model_weights(self.model, checkpoint_path, strict=strict)

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))

        features = self.model.forward_features(x)

        if isinstance(features, dict):
            tokens = features.get("x_norm_patchtokens", None)
            if tokens is None:
                tokens = features.get("patchtokens", None)
            if tokens is None:
                raise KeyError("DINOv2 forward_features did not return patch tokens.")
        else:
            tokens = features

        if self.debug:
            print(f"[DinoV2Backbone] tokens={tuple(tokens.shape)}")

        return _tokens_to_feature_map(
            tokens,
            expected_hw=(out_h, out_w),
            num_prefix_tokens=0,
            debug=self.debug,
        )

    def get_dimension(self) -> int:
        return self.embed_dim

    def get_out_size(self, in_size: ImageSize) -> ImageSize:
        h, w = in_size
        return h // self.patch_size, w // self.patch_size


class TimmViTBackbone(Backbone):
    MODEL_ALIASES: Dict[str, str] = {}

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        checkpoint_path: Optional[Union[str, Path]] = None,
        strict: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__()

        resolved_name = _resolve_timm_name(model_name, self.MODEL_ALIASES)

        self.model = timm.create_model(
            resolved_name,
            pretrained=pretrained,
            num_classes=0,
        )

        self.model_name = resolved_name
        self.patch_size = _get_patch_size(self.model, default=16)
        self.embed_dim = _get_embed_dim(self.model)
        self.debug = debug

        self.num_prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))

        _load_model_weights(self.model, checkpoint_path, strict=strict)

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))

        features = self.model.forward_features(x)

        if isinstance(features, dict):
            tokens = (
                features.get("x_norm_patchtokens", None)
                or features.get("patchtokens", None)
                or features.get("tokens", None)
                or features.get("features", None)
            )
            if tokens is None:
                raise KeyError("timm forward_features returned a dict without token features.")
        else:
            tokens = features

        if tokens.ndim == 4:
            return tokens

        if self.debug:
            print(f"[TimmViTBackbone] model={self.model_name}, tokens={tuple(tokens.shape)}")

        return _tokens_to_feature_map(
            tokens,
            expected_hw=(out_h, out_w),
            num_prefix_tokens=self.num_prefix_tokens,
            debug=self.debug,
        )

    def get_dimension(self) -> int:
        return self.embed_dim

    def get_out_size(self, in_size: ImageSize) -> ImageSize:
        h, w = in_size
        return h // self.patch_size, w // self.patch_size


class DinoV3Backbone(TimmViTBackbone):
    MODEL_ALIASES = {
        "vit_small_patch16_dinov3": "vit_small_patch16_dinov3.lvd1689m",
        "vit_small_plus_patch16_dinov3": "vit_small_plus_patch16_dinov3.lvd1689m",
        "vit_base_patch16_dinov3": "vit_base_patch16_dinov3.lvd1689m",
        "vit_large_patch16_dinov3": "vit_large_patch16_dinov3.lvd1689m",
        "vit_huge_plus_patch16_dinov3": "vit_huge_plus_patch16_dinov3.lvd1689m",
        "vit_7b_patch16_dinov3": "vit_7b_patch16_dinov3.lvd1689m",
    }

    def __init__(
        self,
        model_name: str = "vit_base_patch16_dinov3",
        pretrained: bool = True,
        checkpoint_path: Optional[Union[str, Path]] = None,
        strict: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            strict=strict,
            debug=debug,
        )


class DinoBackbone(TimmViTBackbone):
    MODEL_ALIASES = {
        "dino_vits16": "vit_small_patch16_224.dino",
        "dino_vitb16": "vit_base_patch16_224.dino",
        "vit_small_patch16_dino": "vit_small_patch16_224.dino",
        "vit_base_patch16_dino": "vit_base_patch16_224.dino",
    }

    def __init__(
        self,
        model_name: str = "dino_vits16",
        pretrained: bool = True,
        checkpoint_path: Optional[Union[str, Path]] = None,
        strict: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            strict=strict,
            debug=debug,
        )


class SimDINOBackbone(TimmViTBackbone):
    MODEL_ALIASES = {
        "simdino_vits16": "vit_small_patch16_224.dino",
        "simdino_vitb16": "vit_base_patch16_224.dino",
        "vit_small_patch16_simdino": "vit_small_patch16_224.dino",
        "vit_base_patch16_simdino": "vit_base_patch16_224.dino",
    }

    def __init__(
        self,
        model_name: str = "simdino_vits16",
        pretrained: bool = True,
        checkpoint_path: Optional[Union[str, Path]] = None,
        strict: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            strict=strict,
            debug=debug,
        )


class SimDINOv2Backbone(DinoV2Backbone):
    MODEL_ALIASES = {
        "simdinov2_vits14": "dinov2_vits14",
        "simdinov2_vitb14": "dinov2_vitb14",
        "simdinov2_vitl14": "dinov2_vitl14",
        "simdinov2_vitg14": "dinov2_vitg14",
    }

    def __init__(
        self,
        model_name: str = "simdinov2_vitb14",
        checkpoint_path: Optional[Union[str, Path]] = None,
        strict: bool = False,
        debug: bool = False,
    ) -> None:
        resolved_name = self.MODEL_ALIASES.get(model_name, model_name)

        super().__init__(
            model_name=resolved_name,
            checkpoint_path=checkpoint_path,
            strict=strict,
            debug=debug,
        )
