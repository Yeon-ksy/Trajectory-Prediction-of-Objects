# opencv 활용 궤적 데이터셋 생성
# 카메라프레임 1920X1080
# 72X72 격자 모델
# dataset : data/ 폴더에  trajectory_dataset.csv 파일로 저장

import cv2
import numpy as np
import math
import csv

cap = cv2.VideoCapture(0)

# 궤적 리스트
position_list = []
trajectory_list = []
append_on = False

while True:
    # 프레임을 읽어옵니다.

    ret, frame = cap.read()

    # HSV 색공간으로 변환합니다.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 주황색 범위를 정의합니다.
    lower_orange = np.array([5, 70, 200])
    upper_orange = np.array([100, 255, 255])
    
    # 주황색 영역을 마스크로 만듭니다.
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # 모폴로지 연산을 사용하여 마스크를 개선합니다.
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    
    # 주황색 영역을 바탕으로 객체의 윤곽선을 추출합니다.
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 가장 큰 윤곽선을 추출합니다.
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        
        # 윤곽선을 둘러싸는 사각형을 그립니다.
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        
        # 중심 좌표를 계산합니다.
        center_x = x + w//2
        center_y = y + h//2

        '''
        # 36X36 격자 모델
        # 중심 좌표 x,y를 프레임에 출력합니다.
        if center_x > 420 and center_x < 1500:
            append_on = True

            cv2.putText(frame, "x: " + str(center_x), (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "y: " + str(center_y), (center_x + 10, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("mapping_x: " + str(math.ceil((center_x - 420) / 30)) + ", mapping_y: " + str(math.ceil(center_y / 30)))
            position_list.append([math.ceil((center_x - 420) / 30), math.ceil(center_y / 30)])
            # 궤적 그리기
            for i in range(len(position_list)):
                cv2.circle(frame, (position_list[i][0] * 30 + 420 - 15, position_list[i][1] * 30 - 15), 5, (0, 0, 255), -1)
        else:
            append_on = False
            if len(position_list) > 0:
                trajectory_list.append(position_list)
                position_list = []
        '''
        
        # 72X72 격자 모델
        # 중심 좌표 x,y를 프레임에 출력합니다.
        if center_x > 420 and center_x < 1500:
            append_on = True

            cv2.putText(frame, "x: " + str(center_x), (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "y: " + str(center_y), (center_x + 10, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("mapping_x: " + str(math.ceil((center_x - 420) / 15)) + ", mapping_y: " + str(math.ceil(center_y / 15)))
            position_list.append([math.ceil((center_x - 420) / 15), math.ceil(center_y / 15)])
            # 궤적 그리기
            for i in range(len(position_list)):
                cv2.circle(frame, (position_list[i][0] * 15 + 420 - 7, position_list[i][1] * 15 - 7), 5, (0, 0, 255), -1)
        else:
            append_on = False
            if len(position_list) > 0:
                trajectory_list.append(position_list)
                position_list = []

    '''
    # 36X36 격자 모델
    # 격자 무늬를 출력합니다.
    for i in range(420, 1530, 30):
        cv2.line(frame, (i, 0), (i, 1080), (255, 0, 0), 1)
    for i in range(0, 1110, 30):
         cv2.line(frame, (420, i), (1500, i), (255, 0, 0), 1)
    '''

    # 72X72 격자 모델
    # 격자 무늬를 출력합니다.
    for i in range(420, 1515, 15):
        cv2.line(frame, (i, 0), (i, 1080), (255, 0, 0), 1)
    for i in range(0, 1095, 15):
        cv2.line(frame, (420, i), (1500, i), (255, 0, 0), 1)

    # 사각형 그리기
    cv2.rectangle(frame, (420, 0), (1500, 1080), (0, 255, 0), 2)

    # 결과를 출력합니다.
    cv2.imshow('Orange Ball Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(trajectory_list)
        print('dataset 개수 : '+len(trajectory_list))

        # csv 파일로 저장
        with open('data/trajectory_dataset.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(trajectory_list)

        break

cap.release()
cv2.destroyAllWindows()
