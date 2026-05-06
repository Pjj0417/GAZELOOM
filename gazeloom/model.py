
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from timm.models.vision_transformer import Block

import gazeloom.utils as utils
from gazeloom.backbone import (
    DinoV2Backbone,
    DinoV3Backbone,
    SimDINOBackbone,
    SimDINOv2Backbone,
)


# =========================================================
# Channel Attention
# =========================================================
class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        hidden_dim = max(dim // reduction, 16)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1, bias=False),
        )

    def forward(self, x):
        return self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))


# =========================================================
# Spatial Attention
# =========================================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size,
            padding=padding,
            padding_mode="reflect",
            bias=False,
        )

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_map, max_map], dim=1)
        return self.conv(x)


# =========================================================
# Pixel Group Attention
# =========================================================
class PixelGroupAttention(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2

        self.group_conv = nn.Sequential(
            nn.Conv2d(
                2 * dim,
                dim,
                kernel_size,
                padding=padding,
                padding_mode="reflect",
                groups=dim,
                bias=False,
            ),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=True),
        )

    def forward(self, x, attn):
        x = torch.cat([x, attn], dim=1)
        return self.group_conv(x)


# =========================================================
# CGF: Cross-modal Gated Fusion with Group Convolution
# =========================================================
class CGF(nn.Module):
    def __init__(self, dim, reduction=8, groups=8):
        super().__init__()

        assert dim % groups == 0, f"dim={dim} 必须能被 groups={groups} 整除"

        self.channel_attn = ChannelAttention(dim, reduction)
        self.spatial_attn = SpatialAttention()
        self.pixel_attn = PixelGroupAttention(dim)

        self.group_mixer = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, semantic_feat, geometry_feat):
        base_feat = semantic_feat + geometry_feat

        channel_attn = self.channel_attn(base_feat)
        spatial_attn = self.spatial_attn(base_feat)

        fusion_attn = channel_attn + spatial_attn
        gate = self.sigmoid(self.pixel_attn(base_feat, fusion_attn))

        fused = gate * semantic_feat + (1.0 - gate) * geometry_feat
        fused = fused + base_feat

        fused = self.group_mixer(fused) + fused
        fused = self.out_proj(fused)

        return fused


# =========================================================
# GazeLoom Model
# =========================================================
class gazeloom(nn.Module):
    def __init__(
        self,
        backbone,
        inout=False,
        dim=256,
        num_layers=6,
        in_size=(448, 448),
        out_size=(64, 64),
        cgf_groups=8,
    ):
        super().__init__()

        self.inout = inout
        self.num_layers = num_layers
        self.backbone = backbone
        self.dim = dim
        self.in_size = in_size
        self.out_size = out_size

        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)

        self.linear = nn.Conv2d(backbone.get_dimension(), dim, 1)
        self.depth_proj = nn.Conv2d(1, dim, 1)

        self.head_token = nn.Embedding(1, dim)

        if inout:
            self.inout_token = nn.Embedding(1, dim)

        self.transformer = nn.Sequential(
            *[
                Block(
                    dim=dim,
                    num_heads=8,
                    mlp_ratio=4,
                    drop_path=0.1,
                )
                for _ in range(num_layers)
            ]
        )

        if inout:
            self.inout_head = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )

        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
        )

        self.attention_fusion = CGF(
            dim=dim,
            reduction=8,
            groups=cgf_groups,
        )

        self._pos_cache = {}

    def forward(self, input):
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]

        semantic_feat = self.backbone(input["images"])
        semantic_feat = self.linear(semantic_feat)

        depth_map = input["depth_map"]
        depth_map = F.interpolate(
            depth_map,
            size=semantic_feat.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        geometry_feat = self.depth_proj(depth_map)

        x = self.attention_fusion(semantic_feat, geometry_feat)

        pos_embed = self.get_depth_guided_positional_encoding(geometry_feat)
        x = x + pos_embed

        x = utils.repeat_tensors(x, num_ppl_per_img)

        head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0)
        head_maps = head_maps.to(x.device)

        head_map_embeddings = (
            head_maps.unsqueeze(1)
            * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        )

        x = x + head_map_embeddings

        x = x.flatten(start_dim=2).permute(0, 2, 1)

        if self.inout:
            inout_token = self.inout_token.weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
            x = torch.cat([inout_token, x], dim=1)

        x = self.transformer(x)

        if self.inout:
            inout_tokens = x[:, 0, :]
            inout_preds = self.inout_head(inout_tokens).squeeze(-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            x = x[:, 1:, :]
        else:
            inout_preds = None

        x = x.reshape(
            x.shape[0],
            self.featmap_h,
            self.featmap_w,
            x.shape[2],
        ).permute(0, 3, 1, 2)

        x = self.heatmap_head(x).squeeze(1)

        x = torchvision.transforms.functional.resize(
            x,
            self.out_size,
            antialias=True,
        )

        heatmap_preds = utils.split_tensors(x, num_ppl_per_img)

        return {
            "heatmap": heatmap_preds,
            "inout": inout_preds,
        }

    def get_input_head_maps(self, bboxes):
        head_maps = []

        for bbox_list in bboxes:
            img_head_maps = []

            for bbox in bbox_list:
                head_map = torch.zeros(
                    self.featmap_h,
                    self.featmap_w,
                    device=self.linear.weight.device,
                )

                if bbox is not None:
                    xmin, ymin, xmax, ymax = bbox

                    width = self.featmap_w
                    height = self.featmap_h

                    xmin = int(round(max(0, xmin) * width))
                    ymin = int(round(max(0, ymin) * height))
                    xmax = int(round(min(1, xmax) * width))
                    ymax = int(round(min(1, ymax) * height))

                    if xmin < xmax and ymin < ymax:
                        head_map[ymin:ymax, xmin:xmax] = 1.0

                img_head_maps.append(head_map)

            head_maps.append(torch.stack(img_head_maps))

        return head_maps

    def get_depth_guided_positional_encoding(self, depth_features):
        _, _, h, w = depth_features.shape
        key = (h, w, depth_features.device)

        if key not in self._pos_cache:
            pos_embed = positionalencoding2d(self.dim, h, w)
            pos_embed = pos_embed.to(depth_features.device)
            self._pos_cache[key] = pos_embed

        return self._pos_cache[key].unsqueeze(0)


# =========================================================
# 2D Positional Encoding
# =========================================================
def positionalencoding2d(d_model, height, width):
    if d_model % 4 != 0:
        raise ValueError("d_model 必须为 4 的倍数")

    pe = torch.zeros(d_model, height, width)

    d_model_half = d_model // 2

    div_term = torch.exp(
        torch.arange(0.0, d_model_half, 2)
        * -(math.log(10000.0) / d_model_half)
    )

    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)

    pe[0:d_model_half:2, :, :] = (
        torch.sin(pos_w * div_term)
        .transpose(0, 1)
        .unsqueeze(1)
        .repeat(1, height, 1)
    )

    pe[1:d_model_half:2, :, :] = (
        torch.cos(pos_w * div_term)
        .transpose(0, 1)
        .unsqueeze(1)
        .repeat(1, height, 1)
    )

    pe[d_model_half::2, :, :] = (
        torch.sin(pos_h * div_term)
        .transpose(0, 1)
        .unsqueeze(2)
        .repeat(1, 1, width)
    )

    pe[d_model_half + 1::2, :, :] = (
        torch.cos(pos_h * div_term)
        .transpose(0, 1)
        .unsqueeze(2)
        .repeat(1, 1, width)
    )

    return pe


# =========================================================
# Model Factory
# =========================================================
def get_gazeloom_model(model_name):
    factory = {
        # DINOv2
        "gazeloom_cgf_dinov2_vitb14": gazeloom_cgf_dinov2_vitb14,
        "gazeloom_cgf_dinov2_vitl14": gazeloom_cgf_dinov2_vitl14,
        "gazeloom_cgf_dinov2_vitb14_inout": gazeloom_cgf_dinov2_vitb14_inout,
        "gazeloom_cgf_dinov2_vitl14_inout": gazeloom_cgf_dinov2_vitl14_inout,

        # DINOv3
        "gazeloom_cgf_dinov3_vits16": gazeloom_cgf_dinov3_vits16,
        "gazeloom_cgf_dinov3_vitb16": gazeloom_cgf_dinov3_vitb16,
        "gazeloom_cgf_dinov3_vitl16": gazeloom_cgf_dinov3_vitl16,
        "gazeloom_cgf_dinov3_vith16": gazeloom_cgf_dinov3_vith16,
        "gazeloom_cgf_dinov3_vit7b16": gazeloom_cgf_dinov3_vit7b16,
        "gazeloom_cgf_dinov3_vitb16_inout": gazeloom_cgf_dinov3_vitb16_inout,
        "gazeloom_cgf_dinov3_vitl16_inout": gazeloom_cgf_dinov3_vitl16_inout,

        # SimDINO
        "gazeloom_cgf_simdino_vits16": gazeloom_cgf_simdino_vits16,
        "gazeloom_cgf_simdino_vitb16": gazeloom_cgf_simdino_vitb16,
        "gazeloom_cgf_simdino_vits16_inout": gazeloom_cgf_simdino_vits16_inout,
        "gazeloom_cgf_simdino_vitb16_inout": gazeloom_cgf_simdino_vitb16_inout,

        # SimDINOv2
        "gazeloom_cgf_simdinov2_vits14": gazeloom_cgf_simdinov2_vits14,
        "gazeloom_cgf_simdinov2_vitb14": gazeloom_cgf_simdinov2_vitb14,
        "gazeloom_cgf_simdinov2_vitl14": gazeloom_cgf_simdinov2_vitl14,
        "gazeloom_cgf_simdinov2_vits14_inout": gazeloom_cgf_simdinov2_vits14_inout,
        "gazeloom_cgf_simdinov2_vitb14_inout": gazeloom_cgf_simdinov2_vitb14_inout,
        "gazeloom_cgf_simdinov2_vitl14_inout": gazeloom_cgf_simdinov2_vitl14_inout,

        # Old name compatibility
        "gazeloom_cgaf_dinov2_vitb14": gazeloom_cgf_dinov2_vitb14,
        "gazeloom_cgaf_dinov2_vitl14": gazeloom_cgf_dinov2_vitl14,
        "gazeloom_cgaf_dinov2_vitb14_inout": gazeloom_cgf_dinov2_vitb14_inout,
        "gazeloom_cgaf_dinov2_vitl14_inout": gazeloom_cgf_dinov2_vitl14_inout,

        "gazeloom_cgaf_dinov3_vits16": gazeloom_cgf_dinov3_vits16,
        "gazeloom_cgaf_dinov3_vitb16": gazeloom_cgf_dinov3_vitb16,
        "gazeloom_cgaf_dinov3_vitl16": gazeloom_cgf_dinov3_vitl16,
        "gazeloom_cgaf_dinov3_vith16": gazeloom_cgf_dinov3_vith16,
        "gazeloom_cgaf_dinov3_vit7b16": gazeloom_cgf_dinov3_vit7b16,
        "gazeloom_cgaf_dinov3_vitb16_inout": gazeloom_cgf_dinov3_vitb16_inout,
        "gazeloom_cgaf_dinov3_vitl16_inout": gazeloom_cgf_dinov3_vitl16_inout,

        "gazeloom_cgaf_simdino_vits16": gazeloom_cgf_simdino_vits16,
        "gazeloom_cgaf_simdino_vitb16": gazeloom_cgf_simdino_vitb16,
        "gazeloom_cgaf_simdino_vits16_inout": gazeloom_cgf_simdino_vits16_inout,
        "gazeloom_cgaf_simdino_vitb16_inout": gazeloom_cgf_simdino_vitb16_inout,

        "gazeloom_cgaf_simdinov2_vits14": gazeloom_cgf_simdinov2_vits14,
        "gazeloom_cgaf_simdinov2_vitb14": gazeloom_cgf_simdinov2_vitb14,
        "gazeloom_cgaf_simdinov2_vitl14": gazeloom_cgf_simdinov2_vitl14,
        "gazeloom_cgaf_simdinov2_vits14_inout": gazeloom_cgf_simdinov2_vits14_inout,
        "gazeloom_cgaf_simdinov2_vitb14_inout": gazeloom_cgf_simdinov2_vitb14_inout,
        "gazeloom_cgaf_simdinov2_vitl14_inout": gazeloom_cgf_simdinov2_vitl14_inout,
    }

    assert model_name in factory, f"无效的模型名称: {model_name}"
    return factory[model_name]()


# =========================================================
# DINOv2 Models
# =========================================================
def gazeloom_cgf_dinov2_vitb14():
    backbone = DinoV2Backbone("dinov2_vitb14")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=False), transform


def gazeloom_cgf_dinov2_vitl14():
    backbone = DinoV2Backbone("dinov2_vitl14")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=False), transform


def gazeloom_cgf_dinov2_vitb14_inout():
    backbone = DinoV2Backbone("dinov2_vitb14")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=True), transform


def gazeloom_cgf_dinov2_vitl14_inout():
    backbone = DinoV2Backbone("dinov2_vitl14")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=True), transform


# =========================================================
# DINOv3 Models
# =========================================================
def gazeloom_cgf_dinov3_vits16():
    backbone = DinoV3Backbone("vit_small_patch16_dinov3")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=False), transform


def gazeloom_cgf_dinov3_vitb16():
    backbone = DinoV3Backbone("vit_base_patch16_dinov3")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=False), transform


def gazeloom_cgf_dinov3_vitl16():
    backbone = DinoV3Backbone("vit_large_patch16_dinov3")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=False), transform


def gazeloom_cgf_dinov3_vith16():
    backbone = DinoV3Backbone("vit_huge_plus_patch16_dinov3")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=False), transform


def gazeloom_cgf_dinov3_vit7b16():
    backbone = DinoV3Backbone("vit_7b_patch16_dinov3")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=False), transform


def gazeloom_cgf_dinov3_vitb16_inout():
    backbone = DinoV3Backbone("vit_base_patch16_dinov3")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=True), transform


def gazeloom_cgf_dinov3_vitl16_inout():
    backbone = DinoV3Backbone("vit_large_patch16_dinov3")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=True), transform


# =========================================================
# SimDINO Models
# =========================================================
def gazeloom_cgf_simdino_vits16():
    backbone = SimDINOBackbone("simdino_vits16")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=False), transform


def gazeloom_cgf_simdino_vitb16():
    backbone = SimDINOBackbone("simdino_vitb16")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=False), transform


def gazeloom_cgf_simdino_vits16_inout():
    backbone = SimDINOBackbone("simdino_vits16")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=True), transform


def gazeloom_cgf_simdino_vitb16_inout():
    backbone = SimDINOBackbone("simdino_vitb16")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=True), transform


# =========================================================
# SimDINOv2 Models
# =========================================================
def gazeloom_cgf_simdinov2_vits14():
    backbone = SimDINOv2Backbone("simdinov2_vits14")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=False), transform


def gazeloom_cgf_simdinov2_vitb14():
    backbone = SimDINOv2Backbone("simdinov2_vitb14")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=False), transform


def gazeloom_cgf_simdinov2_vitl14():
    backbone = SimDINOv2Backbone("simdinov2_vitl14")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=False), transform


def gazeloom_cgf_simdinov2_vits14_inout():
    backbone = SimDINOv2Backbone("simdinov2_vits14")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=True), transform


def gazeloom_cgf_simdinov2_vitb14_inout():
    backbone = SimDINOv2Backbone("simdinov2_vitb14")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=True), transform


def gazeloom_cgf_simdinov2_vitl14_inout():
    backbone = SimDINOv2Backbone("simdinov2_vitl14")
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=True), transform
