import pandas as pd
import cv2
import logging
import os
import argparse
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置命令行参数解析器
parser = argparse.ArgumentParser(description='从视频中提取片段')
parser.add_argument('--clip_csv_path', type=str, default='./clips.csv', help='包含片段信息的CSV文件路径。')
parser.add_argument('--video_folder', type=str, default='./videos', help='包含原始视频的文件夹路径。')
parser.add_argument('--clip_folder', type=str, default='./clips', help='保存提取的片段的文件夹路径。')
parser.add_argument('--image_folder', type=str, default='./images', help='保存片段帧图片的文件夹路径。')
args = parser.parse_args()

SPECIAL_CLIPS = ["smwfiZd8HLc_7508-8408-downsampled", "smwfiZd8HLc_10346-12974-downsampled", 
                 "smwfiZd8HLc_16176-16586-downsampled", "smwfiZd8HLc_20322-20668-downsampled", 
                 "31lG75MDwSA_2857-2928", "31lG75MDwSA_3180-3306", "31lG75MDwSA_4686-4764", 
                 "EuYseDl2jm8_1869-2033", "EuYseDl2jm8_3680-3791"]

def main():
    if not os.path.exists(args.clip_folder):
        os.makedirs(args.clip_folder)

    # 加载包含片段信息的CSV文件
    df_clips = pd.read_csv(args.clip_csv_path)

    # 筛选只处理特定视频ID的片段
    target_video_id = "s0BpHgMB_8o"  # 目标视频ID
    df_clips = df_clips[df_clips['video_id'] == target_video_id]

    num_clips = len(df_clips)
    print(f"在CSV文件中找到 {num_clips} 个片段，对应视频ID为 {target_video_id}。")

    # 遍历CSV中的每个片段并从原始视频中提取它们
    for ix, row in tqdm(df_clips.iterrows(), total=num_clips):
        clip_name = row['clip']
        video_name = row['video_id']
        frame_count = row['frame_count']

        clip_image_folder = os.path.join(args.image_folder, clip_name)
        os.makedirs(clip_image_folder, exist_ok=True)

        interval = clip_name.rsplit("_", 1)[1]
        downsampled = "downsampled" in interval
        interval = interval.replace("-downsampled", "")
        frame_start, frame_end = interval.split("-")
        frame_start, frame_end = int(frame_start), int(frame_end)

        video_path = os.path.join(args.video_folder, f'{video_name}.mp4')
        print(f"尝试打开的视频文件路径: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"无法打开视频文件: {video_name}.mp4")
            continue  # 跳过当前视频

        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = 30 if downsampled else int(round(cap.get(cv2.CAP_PROP_FPS)))

        ret, frame = cap.read()
        if not ret:
            logging.error(f"无法读取视频文件的第一帧: {video_name}.mp4")
            cap.release()
            continue  # 跳过当前视频

        height, width, _ = frame.shape

        if clip_name in SPECIAL_CLIPS:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start - 2)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start - 1)

        output_clip_file = os.path.join(args.clip_folder, f"{clip_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码器
        out = cv2.VideoWriter(output_clip_file, fourcc, fps, (width, height))

        clip_frame_num = 0
        ret, frame = cap.read()
        while ret:
            clip_frame_num += 1
            if downsampled:
                ret, frame = cap.read()

            out.write(frame)
            output_frame_file = os.path.join(clip_image_folder, f"{video_name}_{frame_start + clip_frame_num - 1}.jpg")
            cv2.imwrite(output_frame_file, frame)

            ret, frame = cap.read()
            if clip_frame_num == frame_count:
                break

        cap.release()
        out.release()

if __name__ == '__main__':
    main()