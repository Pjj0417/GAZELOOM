# 导入必要的库
import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import tqdm
from pathlib import Path

# 导入Depth Anything V2模型
from depth_anything_v2.dpt import DepthAnythingV2


@torch.no_grad()
def process(root_dir, subset, encoder, outdir_base):
    # 检测设备是 GPU 还是 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 定义不同模型配置
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # 创建并加载模型
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(device).eval()

    # 获取图像路径
    paths = glob.glob(os.path.join(root_dir, subset, "**", "*.jpg"), recursive=True)
    paths.sort()

    # 创建输出目录
    outdir = os.path.join(outdir_base, subset)
    os.makedirs(outdir, exist_ok=True)

    # 处理每一张图像
    for src_path in tqdm.tqdm(paths):
        # 读取图片
        img = cv2.imread(src_path)

        # 深度估计
        depth = depth_anything.infer_image(img, 518)

        # 灰度图归一化到0-255
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        # 构建输出路径
        src_path = Path(src_path)
        dst_dir = src_path.parent.relative_to(Path(root_dir) / subset)
        dst_path = os.path.join(outdir, str(dst_dir), src_path.name)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # 保存为灰度图（jpg格式，如需png可修改扩展名）
        cv2.imwrite(dst_path, depth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 深度估计工具')
    parser.add_argument('--dataset_dir', type=str, required=True, help='数据集根目录')
    parser.add_argument('--encoder', type=str, default='vitl',
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='选择视觉Transformer模型变体')
    parser.add_argument('--outdir', type=str, default='./vis_depth', help='输出结果保存目录')
    args = parser.parse_args()

    # print("Processing train")
    # process(args.dataset_dir, "train", args.encoder, args.outdir)

    print("Processing test")
    process(args.dataset_dir, "test", args.encoder, args.outdir)


# python run2.py --dataset_dir images --encoder vitb --outdir ./video