# 실행 예시 : python -m 4_19_Video_Data_extraction --video_file ~/Downloads/ball.MOV 

import cv2
import argparse
import math
from ultralytics import YOLO
import supervision as sv
import csv

coords = []
traj = []
window_x = 1280
window_y = 720 
grid_size = 5

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resoluation",
        default=[1920, 1080],
        nargs = 2,
        type=int,
        help="Webcam resolution (width, height)"
    )
    parser.add_argument("--video_file", type=str, help="Path to the input video file")
    parser.add_argument("--resize", type=int, default = [window_x,window_y], nargs=2, help="Resize resolution (width, height)")

    args = parser.parse_args()
    return args


def box_gen(frame, detections, labels):
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    
    frame_box = box_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels
            )
    return frame_box

def grid_gen(frame_box):
    # 격자 무늬를 출력합니다.
    # 세로축
    for i in range(280, 1000 , grid_size):
        cv2.line(frame_box, (i, 0), (i, window_y), (255, 0, 0), 1)
    # 가로축
    for i in range(0, window_y, grid_size):
        cv2.line(frame_box, (280, i), (1000, i), (255, 0, 0), 1)

    # frame에 사각형 그리기
    frame_grid = cv2.rectangle(frame_box, (280, 0), (1000, window_y), (0, 255, 0), 2)
    return frame_grid


def coordi_gen(detections,frame):
    # Bounding BOX 좌표 중심 구하기
    append_on = False
    global coords

    if len(detections.xyxy) == 0:  # 감지된 물체가 없는 경우
        print("No objects detected.")
    else:  # 감지된 물체가 있는 경우
        x1, y1, x2, y2 = detections.xyxy[0]

        center_x = (x1+x2)//2
        center_y = (y1+y2)//2
        if center_x > 280 and center_x < 1000:
            append_on = True 
            cv2.putText(frame, "x: " + str(center_x), (int(center_x) + 10, int(center_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "y: " + str(center_y), (int(center_x) + 10, int(center_y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("mapping_x: " + str(math.ceil((center_x - 320) / grid_size)) + ", mapping_y: " + str(math.ceil(center_y / grid_size)))
            coords.append([math.ceil((center_x - 320) / grid_size), math.ceil(center_y / grid_size)])
            # 궤적 그리기
            for i in range(len(coords)):
                cv2.circle(frame, (coords[i][0] * grid_size + 320 -2, coords[i][1] * grid_size- 2), 2, (0, 0, 255), -1)
        else:
            append_on = False
            coords = []
        if len(coords) > 0:
            traj.append(coords)
        # 물체 인식에서 나온 값 : 끝에서 1250, 700 (실제는 1280,720)

def save_data():
    with open('data/trajectory_dataset.csv', 'w', newline='') as f:
        writer = csv.writer(f)

        # 헤더 행에 열 이름 작성
        writer.writerow(['Key', 'X', 'Y'])
        
        # traj 리스트 안의 각 트라젝토리에 대해 반복
        for i, point in enumerate(traj[-1]):
    
            # 트라젝토리 번호(i+1), x 좌표, y 좌표를 가지는 행 작성
            writer.writerow([i+1, point[0], point[1]])

def main():
    args = parse_arguments()
    frame_width, frame_heigh = args.webcam_resoluation
    target_width, target_height = args.resize

    cap = cv2.VideoCapture(args.video_file)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_heigh)
    
    model = YOLO("yolov8l.pt")

    while True:
        ret, frame = cap.read()
        if not ret:
            save_data()
            break
    
        #동영상 리사이즈
        frame = cv2.resize(frame, (target_width, target_height))

        ## 모델(YOLOv8l)을 영상에 삽입?
        # result = model(frame)[0]

        ## 오렌지만 잡게 하기 위해서는 다음과 같이
        result = model(source=frame, classes=[49])[0]

        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
            
        coordi_gen(detections,frame)
        frame_box = box_gen(frame, detections, labels)
        frame_grid = grid_gen(frame_box)

        cv2.imshow("yolov8", frame_grid)

        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main()