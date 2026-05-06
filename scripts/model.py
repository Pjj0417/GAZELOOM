# 整合 GazeLLE 模型完整代码（含深度调制的位置编码）

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
import math
import gazelle.utils as utils
from gazelle.backbone import DinoV2Backbone
import torchvision


class FeatureWeightedFusion(nn.Module):
    def __init__(self, dim):
        super(FeatureWeightedFusion, self).__init__()
        self.weight_image = nn.Parameter(torch.tensor(0.5))
        self.weight_depth = nn.Parameter(torch.tensor(0.5))

    def forward(self, image_features, depth_features):
        return image_features * self.weight_image + depth_features * self.weight_depth


class FeatureCalibration(nn.Module):
    def __init__(self, dim):
        super(FeatureCalibration, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        return features * self.sigmoid(self.conv(features))


class FeatureEnhancement(nn.Module):
    def __init__(self, dim):
        super(FeatureEnhancement, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        return features + self.conv2(self.relu(self.conv1(features)))


class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, image_features, depth_features):
        att = self.attention(image_features)
        return image_features * att + depth_features * (1 - att)


class GazeLLE(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)

        self.linear = nn.Conv2d(backbone.get_dimension(), dim, 1)
        self.head_token = nn.Embedding(1, dim)
        self.relu = nn.ReLU(inplace=True)

        if inout:
            self.inout_token = nn.Embedding(1, dim)

        self.transformer = nn.Sequential(*[
            Block(dim=dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
            for _ in range(num_layers)
        ])

        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 2, 2),
            nn.Conv2d(dim, 1, 1, bias=False),
            nn.Sigmoid()
        )

        if inout:
            self.inout_head = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        class ModalityInteraction(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.image_to_depth = nn.Conv2d(dim, dim, 1)
                self.depth_to_image = nn.Conv2d(dim, dim, 1)
                self.softmax = nn.Softmax(dim=1)
                self.image_attn = nn.MultiheadAttention(dim, 8)
                self.depth_attn = nn.MultiheadAttention(dim, 8)

            def forward(self, image_features, depth_features):
                depth_attn = self.softmax(self.image_to_depth(image_features))
                image_attn = self.softmax(self.depth_to_image(depth_features))
                image_features = image_features + depth_features * image_attn
                depth_features = depth_features + image_features * depth_attn

                B, C, H, W = image_features.shape
                image_features = image_features.reshape(B, C, -1).permute(0, 2, 1)
                depth_features = depth_features.reshape(B, C, -1).permute(0, 2, 1)

                image_features = self.image_attn(image_features, image_features, image_features)[0]
                depth_features = self.depth_attn(depth_features, depth_features, depth_features)[0]

                image_features = image_features.permute(0, 2, 1).reshape(B, C, H, W)
                depth_features = depth_features.permute(0, 2, 1).reshape(B, C, H, W)

                return image_features, depth_features

        self.interaction = ModalityInteraction(dim).cuda()
        self.fusion = FeatureWeightedFusion(dim).cuda()
        self.calibration = FeatureCalibration(dim).cuda()
        self.enhancement = FeatureEnhancement(dim).cuda()
        self.attention_fusion = AttentionFusion(dim).cuda()

    def get_depth_guided_positional_encoding(self, depth_features):
        B, _, H, W = depth_features.shape
        pos_embed = positionalencoding2d(self.dim, H, W).to(depth_features.device)
        depth_attn = torch.mean(depth_features, dim=1, keepdim=True)
        depth_min = depth_attn.amin(dim=[2, 3], keepdim=True)
        depth_max = depth_attn.amax(dim=[2, 3], keepdim=True)
        depth_norm = (depth_attn - depth_min) / (depth_max - depth_min + 1e-6)
        pos_embed = pos_embed.unsqueeze(0).expand(B, -1, -1, -1)
        return pos_embed * (1 + depth_norm)

    def forward(self, input):
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        x = self.linear(self.backbone(input["images"]))
        self.depth_conv = nn.Conv2d(1, self.dim, 1).cuda()
        depth_map = input["depth_map"]
        depth_features = self.depth_conv(depth_map)
        if depth_features.shape[2:] != x.shape[2:]:
            depth_features = F.interpolate(depth_features, size=x.shape[2:], mode='bilinear', align_corners=False)

        x, depth_features = self.interaction(x, depth_features)
        x = self.fusion(x, depth_features)
        x = self.calibration(x)
        x = self.enhancement(x)
        x = self.attention_fusion(x, depth_features)

        pos_embed = self.get_depth_guided_positional_encoding(depth_features)
        x = x + pos_embed

        x = utils.repeat_tensors(x, num_ppl_per_img)
        head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
        head_map_embeddings = head_maps.unsqueeze(1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x + head_map_embeddings
        x = x.flatten(start_dim=2).permute(0, 2, 1)

        if self.inout:
            x = torch.cat([self.inout_token.weight.unsqueeze(0).repeat(x.shape[0], 1, 1), x], dim=1)

        x = self.transformer(x)

        if self.inout:
            inout_tokens = x[:, 0, :]
            inout_preds = self.inout_head(inout_tokens).squeeze(-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            x = x[:, 1:, :]

        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
        x = self.heatmap_head(x).squeeze(1)
        x = torchvision.transforms.functional.resize(x, self.out_size)
        heatmap_preds = utils.split_tensors(x, num_ppl_per_img)

        return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

    def get_input_head_maps(self, bboxes):
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for bbox in bbox_list:
                head_map = torch.zeros((self.featmap_h, self.featmap_w))
                if bbox is not None:
                    xmin, ymin, xmax, ymax = bbox
                    xmin = round(xmin * self.featmap_w)
                    ymin = round(ymin * self.featmap_h)
                    xmax = round(xmax * self.featmap_w)
                    ymax = round(ymax * self.featmap_h)
                    head_map[ymin:ymax, xmin:xmax] = 1
                img_head_maps.append(head_map)
            head_maps.append(torch.stack(img_head_maps))
        return head_maps

    def get_gazelle_state_dict(self, include_backbone=False):
        return self.state_dict() if include_backbone else {
            k: v for k, v in self.state_dict().items() if not k.startswith("backbone")
        }

    def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):
        current_state_dict = self.state_dict()
        keys1 = set(k for k in current_state_dict if include_backbone or not k.startswith("backbone"))
        keys2 = set(k for k in ckpt_state_dict if include_backbone or not k.startswith("backbone"))
        for k in keys1 & keys2:
            current_state_dict[k] = ckpt_state_dict[k]
        self.load_state_dict(current_state_dict, strict=False)


def positionalencoding2d(d_model, height, width):
    if d_model % 4 != 0:
        raise ValueError("位置编码要求d_model为4的倍数")
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


def get_gazelle_model(model_name):
    factory = {
        "gazelle_dinov2_vits14": gazelle_dinov2_vits14,
        "gazelle_dinov2_vitb14": gazelle_dinov2_vitb14,
        "gazelle_dinov2_vitl14": gazelle_dinov2_vitl14,
        "gazelle_dinov2_vitb14_inout": gazelle_dinov2_vitb14_inout,
        "gazelle_dinov2_vitl14_inout": gazelle_dinov2_vitl14_inout,
    }
    assert model_name in factory, "invalid model name"
    return factory[model_name]()


def gazelle_dinov2_vits14():
    backbone = DinoV2Backbone('dinov2_vits14')
    transform = backbone.get_transform((448, 448))
    return GazeLLE(backbone), transform


def gazelle_dinov2_vitb14():
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    return GazeLLE(backbone), transform


def gazelle_dinov2_vitl14():
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    return GazeLLE(backbone), transform


def gazelle_dinov2_vitb14_inout():
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    return GazeLLE(backbone, inout=True), transform


def gazelle_dinov2_vitl14_inout():
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    return GazeLLE(backbone, inout=True), transform
