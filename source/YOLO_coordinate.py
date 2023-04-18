import cv2
import argparse

from ultralytics import YOLO
import supervision as sv

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resoluation",
        default=[1280, 720],
        nargs = 2,
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_heigh = args.webcam_resoluation

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_heigh)
    
    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        ret, frame = cap.read()

        ## 모델(YOLOv8l)을 영상에 삽입?
        # result = model(frame)[0]

        ## 오렌지만 잡게 하기 위해서는 다음과 같이
        result = model(source=frame, classes=[49])[0]

        detections = sv.Detections.from_yolov8(result)
        print("------------------------- \n")
        print(f"디텍션 정보 : {detections} \n")
        print("---------------------- \n")
        label = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        print(f"라벨 확인 : {label}")
        print("--------------------------------- \n")
        print(f"결과 확인 : {result} \n")
        print("--------------------------------- \n")

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=label
            )

        coords = []
        if len(detections.xyxy) == 0:  # 감지된 물체가 없는 경우
            print("No objects detected.")
        else:  # 감지된 물체가 있는 경우
            x1, y1, x2, y2 = detections.xyxy[0]
            coords.append(((x1+x2)//2, (y1+y2)//2))
            print(f"좌표입니다. : {coords}")
            # 물체 인식에서 나온 값 : 끝에서 1250, 700 (실제는 1280,720)

        
        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main()