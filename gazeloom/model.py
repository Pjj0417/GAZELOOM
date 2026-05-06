
# model_cga.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from timm.models.vision_transformer import Block
import math

import gazeloom.utils as utils
from ggazeloom.backbone import DinoV2Backbone, DinoV3Backbone


# === Spatial Attention ===
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        return self.sa(x2)


# === Channel Attention ===
class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=True),
        )

    def forward(self, x):
        return self.ca(self.gap(x))


# === Pixel Attention ===
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)

    def forward(self, x, pattn1):
        x2 = torch.cat([x, pattn1], dim=1)
        return self.pa2(x2)


# === CGA Fusion ===
class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        return self.conv(result)


# === gazeloom 改进版 ===
class gazeloom(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, num_layers=6, in_size=(448, 448), out_size=(64, 64)):
        super().__init__()
        self.inout = inout
        self.num_layers = num_layers
        self.backbone = backbone
        self.dim = dim
        self.in_size = in_size
        self.out_size = out_size
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)

        # 特征投影
        self.linear = nn.Conv2d(backbone.get_dimension(), dim, 1)
        self.depth_proj = nn.Conv2d(1, dim, 1)

        # Head guidance
        self.head_token = nn.Embedding(1, dim)
        if inout:
            self.inout_token = nn.Embedding(1, dim)

        # Transformer blocks
        self.transformer = nn.Sequential(*[
            Block(dim=dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
            for _ in range(self.num_layers)
        ])

        # Inout head
        if inout:
            self.inout_head = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )

        # Heatmap head
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 2, 2),
            nn.Conv2d(dim, 1, 1, bias=False),
        )

        # CGA Fusion
        self.attention_fusion = CGAFusion(dim)

        # cache
        self._pos_cache = {}

    def forward(self, input):
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]

        # backbone
        x = self.linear(self.backbone(input["images"]))
        depth_map = input["depth_map"]
        depth_features = self.depth_proj(F.interpolate(
            depth_map, size=x.shape[2:], mode='bilinear', align_corners=False
        ))

        # CGA 融合
        x = self.attention_fusion(x, depth_features)

        # positional encoding
        pos_embed = self.get_depth_guided_positional_encoding(depth_features)
        x = x + pos_embed

        # head map
        x = utils.repeat_tensors(x, num_ppl_per_img)
        head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
        head_map_embeddings = head_maps.unsqueeze(1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x + head_map_embeddings

        # transformer
        x = x.flatten(start_dim=2).permute(0, 2, 1)
        if self.inout:
            x = torch.cat([self.inout_token.weight.unsqueeze(0).repeat(x.shape[0], 1, 1), x], dim=1)
        x = self.transformer(x)

        # inout
        if self.inout:
            inout_tokens = x[:, 0, :]
            inout_preds = self.inout_head(inout_tokens).squeeze(-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            x = x[:, 1:, :]
        else:
            inout_preds = None

        # heatmap
        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
        x = self.heatmap_head(x).squeeze(1)
        x = torchvision.transforms.functional.resize(x, self.out_size)
        heatmap_preds = utils.split_tensors(x, num_ppl_per_img)

        return {"heatmap": heatmap_preds, "inout": inout_preds}

    def get_input_head_maps(self, bboxes):
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for bbox in bbox_list:
                if bbox is None:
                    img_head_maps.append(torch.zeros(self.featmap_h, self.featmap_w, device=self.linear.weight.device))
                else:
                    xmin, ymin, xmax, ymax = bbox
                    width, height = self.featmap_w, self.featmap_h
                    xmin = int(round(max(0, xmin) * width))
                    ymin = int(round(max(0, ymin) * height))
                    xmax = int(round(min(1, xmax) * width))
                    ymax = int(round(min(1, ymax) * height))
                    head_map = torch.zeros((height, width), device=self.linear.weight.device)
                    if xmin < xmax and ymin < ymax:
                        head_map[ymin:ymax, xmin:xmax] = 1
                    img_head_maps.append(head_map)
            head_maps.append(torch.stack(img_head_maps))
        return head_maps

    def get_depth_guided_positional_encoding(self, depth_features):
        B, _, H, W = depth_features.shape
        key = (H, W)
        if key not in self._pos_cache:
            self._pos_cache[key] = positionalencoding2d(self.dim, H, W).to(depth_features.device)
        return self._pos_cache[key]


def positionalencoding2d(d_model, height, width):
    if d_model % 4 != 0:
        raise ValueError("d_model 必须为4的倍数")
    pe = torch.zeros(d_model, height, width)
    d_model = d_model // 2
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe

# === Model Factory ===
def get_gazeloom_model(model_name):
    factory = {
        # ---- DINOv2 ----
        "gazeloom_cgaf_dinov2_vitb14": gazeloom_cgaf_dinov2_vitb14,
        "gazeloom_cgaf_dinov2_vitl14": gazeloom_cgaf_dinov2_vitl14,
        "gazeloom_cgaf_dinov2_vitb14_inout": gazeloom_cgaf_dinov2_vitb14_inout,
        "gazeloom_cgaf_dinov2_vitl14_inout": gazeloom_cgaf_dinov2_vitl14_inout,

        # ---- DINOv3 ----
        "gazeloom_cgaf_dinov3_vits16": gazeloom_cgaf_dinov3_vits16,
        "gazeloom_cgaf_dinov3_vitb16": gazeloom_cgaf_dinov3_vitb16,
        "gazeloom_cgaf_dinov3_vitl16": gazeloom_cgaf_dinov3_vitl16,
        "gazeloom_cgaf_dinov3_vith16": gazeloom_cgaf_dinov3_vith16,
        "gazeloom_cgaf_dinov3_vit7b16": gazeloom_cgaf_dinov3_vit7b16,
        "gazeloom_cgaf_dinov3_vitb16_inout": gazeloom_cgaf_dinov3_vitb16_inout,
        "gazeloom_cgaf_dinov3_vitl16_inout": gazeloom_cgaf_dinov3_vitl16_inout,
    }
    assert model_name in factory, f"无效的模型名称: {model_name}"
    return factory[model_name]()


# ---- DINOv2 ----
def gazeloom_cgaf_dinov2_vitb14():
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone), transform

def gazeloom_cgaf_dinov2_vitl14():
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone), transform

def gazeloom_cgaf_dinov2_vitb14_inout():
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=True), transform

def gazeloom_cgaf_dinov2_vitl14_inout():
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=True), transform


# ---- DINOv3 ----
def gazeloom_cgaf_dinov3_vits16():
    backbone = DinoV3Backbone('vit_small_patch16_dinov3')
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone), transform

def gazeloom_cgaf_dinov3_vitb16():
    backbone = DinoV3Backbone('vit_base_patch16_dinov3')
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone), transform

def gazeloom_cgaf_dinov3_vitl16():
    backbone = DinoV3Backbone('vit_large_patch16_dinov3')
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone), transform

def gazeloom_cgaf_dinov3_vith16():
    backbone = DinoV3Backbone('vit_huge_plus_patch16_dinov3')
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone), transform

def gazeloom_cgaf_dinov3_vit7b16():
    backbone = DinoV3Backbone('vit_7b_patch16_dinov3')
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone), transform

def gazeloom_cgaf_dinov3_vitb16_inout():
    backbone = DinoV3Backbone('vit_base_patch16_dinov3')
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=True), transform

def gazeloom_cgaf_dinov3_vitl16_inout():
    backbone = DinoV3Backbone('vit_large_patch16_dinov3')
    transform = backbone.get_transform((448, 448))
    return gazeloom(backbone, inout=True), transform




