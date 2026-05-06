# IMPORTS
import argparse
import os
import shlex
import subprocess as sp

import cv2
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Extract clips from videos')
parser.add_argument('--clip_csv_path', type=str, default='./clips.csv', help='Path to the csv file containing information about the clips.')
parser.add_argument('--video_folder', type=str, default='./videos', help='Path to the folder containing the videos.')
parser.add_argument('--clip_folder', type=str, default='./clips', help='Path to the folder where the clips will be saved.')
parser.add_argument('--image_folder', type=str, default='./images', help='Path to the folder where the clip images will be saved.')
args = parser.parse_args()

# These clips were initially extracted with an incorrect lag of 1 frame. We maintain the same lag here to ensure the frames match the annotations
SPECIAL_CLIPS = ["smwfiZd8HLc_7508-8408-downsampled", "smwfiZd8HLc_10346-12974-downsampled", 
                 "smwfiZd8HLc_16176-16586-downsampled", "smwfiZd8HLc_20322-20668-downsampled", 
                 "31lG75MDwSA_2857-2928", "31lG75MDwSA_3180-3306", "31lG75MDwSA_4686-4764", 
                 "EuYseDl2jm8_1869-2033", "EuYseDl2jm8_3680-3791"]


def main():
    
    ## Create output clip folder if it doesn't exist
    if not os.path.exists(args.clip_folder):
        os.makedirs(args.clip_folder)

    ## Load CSV file
    df_clips = pd.read_csv(args.clip_csv_path)
    num_clips = len(df_clips)
    print(f"Found {num_clips} clips in the CSV file.")
    
    ## Iterate over clips and extract them from their original videos
    for ix, row in tqdm(df_clips.iterrows(), total = num_clips):
        clip_name = row['clip']
        video_name = row['video_id']
        frame_count = row['frame_count']
        
        clip_image_folder = os.path.join(args.image_folder, clip_name)
        os.makedirs(clip_image_folder, exist_ok=True)
        
        interval = clip_name.rsplit("_", 1)[1] # "clip-name_start-end-downsampled" -> "start-end-downsampled"
        downsampled = "downsampled" in interval
        interval = interval.replace("-downsampled", "") # "start-end-downsampled" -> "start-end"
        frame_start, frame_end = interval.split("-")
        frame_start, frame_end = int(frame_start), int(frame_end)
        
        # Read original video
        cap = cv2.VideoCapture(os.path.join(args.video_folder, f'{video_name}.mp4')) 
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = 30 if downsampled else int(round(cap.get(cv2.CAP_PROP_FPS)))
        
        # Read first frame in the video and retrieve attributes
        ret, frame = cap.read()
        height, width, _ = frame.shape
        
        # Set the video to frame start
        if clip_name in SPECIAL_CLIPS:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start - 2)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start - 1)
        
        # Create ffmpeg process
        output_clip_file = os.path.join(args.clip_folder, f"{clip_name}.mp4")
        command = f'ffmpeg -loglevel error -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p -crf 24 {output_clip_file}'
        process = sp.Popen(shlex.split(command), stdin=sp.PIPE)
        
        clip_frame_num = 0
        ret, frame = cap.read()
        while ret:
            clip_frame_num += 1
            if downsampled: # skip every other frame when FPS is 60
                ret, frame = cap.read()
                
            # Write frame
            process.stdin.write(frame.tobytes())
            output_frame_file = os.path.join(clip_image_folder, f"{video_name}_{frame_start + clip_frame_num - 1}.jpg")
            cv2.imwrite(output_frame_file, frame)
            
            ret, frame = cap.read()
            
            if clip_frame_num == frame_count:
                break

        # Release Capture Device
        cap.release()

        # Close and flush stdin
        process.stdin.close()
        # Wait for sub-process to finish
        process.wait()
        # Terminate the sub-process
        process.terminate()



if __name__ == '__main__':
    main()