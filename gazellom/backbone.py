
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm   # 🔹 DINOv3 集成在 timm


# ===========================
# 抽象骨干网络基类
# ===========================
class Backbone(nn.Module, ABC):
    def __init__(self):
        super(Backbone, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_dimension(self):
        pass

    @abstractmethod
    def get_out_size(self, in_size):
        pass

    def get_transform(self, in_size):
        pass


# ===========================
# DINOv2 Backbone
# ===========================
class DinoV2Backbone(Backbone):
    def __init__(self, model_name, debug=False):
        super(DinoV2Backbone, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.debug = debug

    def forward(self, x):
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        x = self.model.forward_features(x)['x_norm_patchtokens']  # (B, N, C)
        if self.debug:
            print(f"[DinoV2Backbone] 输出: B={x.shape[0]}, N={x.shape[1]}, C={x.shape[2]}")
        x = x.view(x.size(0), out_h, out_w, -1).permute(0, 3, 1, 2)  # (B, C, H, W)
        return x

    def get_dimension(self):
        return self.model.embed_dim

    def get_out_size(self, in_size):
        h, w = in_size
        return (h // self.model.patch_size, w // self.model.patch_size)

    def get_transform(self, in_size):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.Resize(in_size),
        ])


# ===========================
# DINOv3 Backbone (timm)
# ===========================
class DinoV3Backbone(Backbone):
    def __init__(self, model_name="vit_base_patch16_dinov3", debug=False):
        super(DinoV3Backbone, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)

        # 更安全的 embed_dim / patch_size
        self.embed_dim = getattr(self.model, "embed_dim", None) or self.model.num_features
        self.patch_size = getattr(self.model, "patch_size", 16)

        self.debug = debug

    def forward(self, x):
        feats = self.model.forward_features(x)  # dict or Tensor
        if isinstance(feats, dict):
            x = feats.get("x_norm_patchtokens", feats.get("features", None))
        else:
            x = feats  # (B, N, C)

        B, N, C = x.shape
        if self.debug:
            print(f"[DinoV3Backbone] forward_features: B={B}, N={N}, C={C}")

        # 去掉 CLS token（只保留 patch tokens）
        if hasattr(self.model, "no_cls") is False and N > 1:
            x = x[:, 1:, :]
            if self.debug:
                print(f"[DinoV3Backbone] 去掉 CLS token 后: N={x.shape[1]}")

        # 如果 N 不是平方数，裁剪到最近平方数
        N = x.shape[1]
        hw = int(N ** 0.5)
        if hw * hw != N:
            if self.debug:
                print(f"[DinoV3Backbone] N={N} 不是平方数，裁剪到 {hw*hw}")
            x = x[:, :hw * hw, :]

        B, N, C = x.shape
        h = w = int(N ** 0.5)
        return x.view(B, h, w, C).permute(0, 3, 1, 2)  # (B, C, H, W)

    def get_dimension(self):
        return self.embed_dim

    def get_out_size(self, in_size):
        h, w = in_size
        return (h // self.patch_size, w // self.patch_size)

    def get_transform(self, in_size):
        return transforms.Compose([
            transforms.Resize(in_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

