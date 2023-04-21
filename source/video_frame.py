import os
import cv2
import numpy as np

def extract_frames(video_path, output_folder, target_frame_count, new_size, convert_to_grayscale=False):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        raise ValueError(f"Failed to read video or the video contains no frames: {video_path}")

    selected_indices = np.linspace(0, total_frames - 1, num=target_frame_count, dtype=int)

    os.makedirs(output_folder, exist_ok=True)
    success_count = 0
    for idx, frame_idx in enumerate(selected_indices):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()
        if ret:
            resized_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
            if convert_to_grayscale:
                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            frame_name = os.path.join(output_folder, f"frame{success_count+1}.jpg")
            cv2.imwrite(frame_name, resized_frame)
            success_count += 1
        else:
            print(f"Failed to read frame {frame_idx} from video: {video_path}")

def is_video_file(file_name):
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    _, ext = os.path.splitext(file_name)
    return ext.lower() in video_exts

convert_to_grayscale = True  # 값을 변경하여 그레이스케일 옵션을 설정하세요.
video_folder = "/home/siwon/dev/Deeplearning-6/data_video/video_frame"  # 여기에 입력 비디오 폴더 경로를 입력하세요.
output_folder = "/home/siwon/dev/Deeplearning-6/data_video/video_frame_number"       # 여기에 출력 프레임이 저장될 폴더 경로를 입력하세요.
target_frame_count = 20         # 여기에 원하는 프레임 수를 입력하세요.
new_size = (64, 64)           # 원하는 이미지 크기로 변경하세요.

for video_name in os.listdir(video_folder):
    if not is_video_file(video_name):
        continue

    video_path = os.path.join(video_folder, video_name)
    video_output_folder = os.path.join(output_folder, os.path.splitext(video_name)[0])  # 비디오 파일 확장자 제거
    extract_frames(video_path, video_output_folder, target_frame_count, new_size, convert_to_grayscale)