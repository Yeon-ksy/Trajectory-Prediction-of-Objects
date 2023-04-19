import cv2
import argparse
import math
from ultralytics import YOLO
import supervision as sv
import threading

coords = []
traj = []

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resoluation",
        default=[1920, 1080],
        nargs = 2,
        type=int
    )
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
    for i in range(420, 1515, 15):
        cv2.line(frame_box, (i, 0), (i, 1080), (255, 0, 0), 1)
    # 가로축
    for i in range(0, 1095, 15):
        cv2.line(frame_box, (420, i), (1500, i), (255, 0, 0), 1)

    # frame에 사각형 그리기
    frame_grid = cv2.rectangle(frame_box, (420, 0), (1500, 1080), (0, 255, 0), 2)
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
        if center_x > 420 and center_x < 1500:
            append_on = True 
            cv2.putText(frame, "x: " + str(center_x), (int(center_x) + 10, int(center_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "y: " + str(center_y), (int(center_x) + 10, int(center_y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("mapping_x: " + str(math.ceil((center_x - 420) / 15)) + ", mapping_y: " + str(math.ceil(center_y / 15)))
            coords.append([math.ceil((center_x - 420) / 15), math.ceil(center_y / 15)])
            # 궤적 그리기
            for i in range(len(coords)):
                cv2.circle(frame, (coords[i][0] * 15 + 420 - 7, coords[i][1] * 15 - 7), 5, (0, 0, 255), -1)
        else:
            append_on = False
            coords = []
        if len(coords) > 0:
            traj.append(coords)
        # 물체 인식에서 나온 값 : 끝에서 1250, 700 (실제는 1280,720)


def main():
    args = parse_arguments()
    frame_width, frame_heigh = args.webcam_resoluation

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_heigh)
    
    model = YOLO("yolov8l.pt")

    while True:
        ret, frame = cap.read()

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
        print(f"라벨 확인 : {labels}")
        print("--------------------------------- \n")
        print(f"결과 확인 : {result} \n")
        print("--------------------------------- \n")

            
        coordi_gen(detections,frame)
        frame_box = box_gen(frame, detections, labels)
        frame_grid = grid_gen(frame_box)

        cv2.imshow("yolov8", frame_grid)

        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main()