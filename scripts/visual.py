
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from gazeloom.model import get_gazelle_model
from retinaface import RetinaFace
import os
import matplotlib.pyplot as plt
import json

# ==== 配置 ====
model_name = "gazeloom_dinov2_vitb14"
ckpt_path = "./checkpoints/epoch_14.pt"
image_dir = "test"
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# ==== 加载模型 ====
print("加载模型中...")
model, transform = get_gazelle_model(model_name)
state_dict = torch.load(ckpt_path, map_location='cpu')
model.load_gazelle_state_dict(state_dict, include_backbone=False)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("模型加载成功")
print("inout 分支启用:", hasattr(model, "inout_head"))

# ==== gaze 结果收集容器 ====
all_predictions = {}

# ==== 可视化函数：框色/线色分离 ====

def visualize(image_pil, heatmaps, bboxes, inout_scores, image_name, thresh=0.7, heat_thresh=0.6):
    draw = ImageDraw.Draw(image_pil)
    width, height = image_pil.size
    results = []

    box_colors = ["orange", "green", "blue", "magenta", "yellow", "purple"]
    line_colors = ["deepskyblue", "darkorange", "lime", "violet", "turquoise", "gold"]

    for i, bbox in enumerate(bboxes):
        box_color = box_colors[i % len(box_colors)]
        line_color = line_colors[i % len(line_colors)]

        xmin, ymin, xmax, ymax = bbox
        xmin *= width
        xmax *= width
        ymin *= height
        ymax *= height

        # 扩大人脸框
        padding_ratio = 0.05
        x_pad = (xmax - xmin) * padding_ratio
        y_pad = (ymax - ymin) * padding_ratio
        xmin = max(0, xmin - x_pad)
        xmax = min(width, xmax + x_pad)
        ymin = max(0, ymin - y_pad)
        ymax = min(height, ymax + y_pad)

        draw.rectangle([xmin, ymin, xmax, ymax], outline=box_color, width=2)
        draw.text((xmin, ymax + 5), f"#{i}", fill=box_color, font=ImageFont.load_default())

        score = inout_scores[i] if inout_scores and i < len(inout_scores) else None
        if score is not None:
            draw.text((xmin, ymax + 20), f"in-frame: {score:.2f}", fill=box_color, font=ImageFont.load_default())

        gaze_point = None
        if score is None or score > thresh:
            heatmap = heatmaps[i].detach().cpu().numpy()
            h_h, h_w = heatmap.shape

            # === 1. 所有高响应点 ===
            threshold_mask = heatmap > heat_thresh
            yy, xx = np.where(threshold_mask)
            for x, y in zip(xx, yy):
                px = x / h_w * width
                py = y / h_h * height
                draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill="green")

            # === 2. 最大响应点（红色） ===
            max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            gx = max_idx[1] / h_w * width
            gy = max_idx[0] / h_h * height
            gaze_point = [round(gx, 2), round(gy, 2)]

            draw.ellipse([gx - 5, gy - 5, gx + 5, gy + 5], fill="red")

            # === 3. 连线：人脸中心 → gaze max 点
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            draw.line([cx, cy, gx, gy], fill=line_color, width=2)

            # === 保存热图 ===
            heatmap_path = os.path.join(output_dir, f"{image_name}_heatmap_{i}.jpg")
            plt.imsave(heatmap_path, heatmap, cmap='jet')

            # === 保存 overlay 图 ===
            heatmap_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
            norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-6)
            heatmap_color = plt.cm.jet(norm)[:, :, :3]
            heatmap_color = (heatmap_color * 255).astype(np.uint8)
            heatmap_overlay = Image.fromarray(heatmap_color).convert("RGBA")
            heatmap_overlay.putalpha(100)

            base_image = image_pil.convert("RGBA")
            overlay_img = Image.alpha_composite(base_image, heatmap_overlay)
            draw_overlay = ImageDraw.Draw(overlay_img)

            for x, y in zip(xx, yy):
                px = x / h_w * width
                py = y / h_h * height
                draw_overlay.ellipse([px - 2, py - 2, px + 2, py + 2], fill="green")

            draw_overlay.ellipse([gx - 2, gy - 2, gx + 2, gy + 2], fill="red")
            draw_overlay.rectangle([xmin, ymin, xmax, ymax], outline=box_color, width=2)
            draw_overlay.line([cx, cy, gx, gy], fill=line_color, width=3)

            overlay_path = os.path.join(output_dir, f"{image_name}_overlay_{i}.jpg")
            overlay_img.convert("RGB").save(overlay_path)

        results.append({
            "bbox": [round(xmin/width, 4), round(ymin/height, 4), round(xmax/width, 4), round(ymax/height, 4)],
            "inout": round(float(score), 4) if score is not None else None,
            "gaze_point": gaze_point
        })

    return image_pil, results
# ==== 主逻辑 ====
def run_folder(image_dir):
    image_list = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

    for img_name in image_list:
        image_path = os.path.join(image_dir, img_name)
        image = Image.open(image_path).convert("RGB")
        np_image = np.array(image)
        height, width = image.height, image.width

        try:
            resp = RetinaFace.detect_faces(np_image)
        except Exception as e:
            print(f"{img_name} 人脸检测失败：{e}")
            continue

        if not resp:
            print(f"{img_name} ❌ 未检测到人脸")
            continue

        bboxes = [resp[k]["facial_area"] for k in resp]
        norm_bboxes = [[np.array(b) / np.array([width, height, width, height]) for b in bboxes]]
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model({"images": img_tensor, "bboxes": norm_bboxes})

        heatmaps = output.get("heatmap", [])[0] if isinstance(output.get("heatmap"), list) else []
        inout_scores = output["inout"][0] if isinstance(output.get("inout"), list) and len(output["inout"]) > 0 else None

        norm_bboxes_flat = norm_bboxes[0]
        image_basename = os.path.splitext(img_name)[0]
        result_image, gaze_info = visualize(image, heatmaps, norm_bboxes_flat, inout_scores, image_basename)
        
        # 保存主图
        result_image.save(os.path.join(output_dir, f"{image_basename}_result.jpg"))
        all_predictions[image_basename] = gaze_info
        print(f"✅ {img_name} 处理完成")

    # 保存 json
    json_path = os.path.join(output_dir, "gaze_predictions.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)
    print(f"\n📄 gaze_predictions.json 保存至：{json_path}")

# ==== 执行 ====
if __name__ == "__main__":
    run_folder(image_dir)
