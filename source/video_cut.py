import os
import cv2

def split_video(video_path, output_folder, num_clips):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps

    clip_duration = duration / num_clips
    clip_frames = int(total_frames / num_clips)

    os.makedirs(output_folder, exist_ok=True)

    for i in range(num_clips):
        clip_name = os.path.join(output_folder, f"clip{i+1}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(clip_name, fourcc, fps, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        for _ in range(clip_frames):
            ret, frame = video.read()
            if not ret:
                break
            out.write(frame)

        out.release()

    video.release()

video_path = "/home/siwon/dev/Deeplearning-6/data_video/full_video/ball_video.mp4"       # 입력 비디오 파일 경로를 입력하세요.
output_folder = "/home/siwon/dev/Deeplearning-6/data_video/frame_ball_color"     # 출력 비디오 클립이 저장될 폴더 경로를 입력하세요.
num_clips = 400                               # 원하는 클립 수를 입력하세요.

split_video(video_path, output_folder, num_clips)
