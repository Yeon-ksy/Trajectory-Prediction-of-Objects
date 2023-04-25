import cv2
import numpy as np
import math
from collections import deque
import threading
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import keras
import pandas as pd
import threading
import queue
import math
import cv2
from collections import deque
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

t=0


# 쓰레드 클래스 정의
class FrameProcessingThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(FrameProcessingThread, self).__init__(*args, **kwargs)
        self.result = None

    def run(self):
        self.result = self._target(*self._args, **self._kwargs)

# 주어진 입력 데이터에서 연속적으로 다음 위치 예측 함수
def predict_next_positions(model, input_data, num_steps):
    current_input = input_data.copy()
    predicted_positions = []

    for _ in range(num_steps):
        predicted = model.predict(current_input)
        predicted_positions.append(predicted[0])

        # 처음 4개 좌표와 예측한 좌표로 새로운 입력 데이터 생성
        new_input = np.concatenate((current_input[:, 1:, :], predicted.reshape(1, 1, -1)), axis=1)
        current_input = new_input

    return np.array(predicted_positions)

def process_frame(frame, position_list, model):
    global t
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

        
        # 72X72 격자 모델
        # 중심 좌표 x,y를 프레임에 출력합니다.
        if center_x > 420 and center_x < 1500:
            cv2.putText(frame, "x: " + str(center_x), (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "y: " + str(center_y), (center_x + 10, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            new_position = [math.ceil((center_x - 420) / 15), math.ceil(center_y / 15)]
            if new_position not in position_list:
                new_position.append(t)
                position_list.append(new_position)
                print("position_list: ", position_list)
                t = t + 1
                #print(position_list)
            # model predict
            if len(position_list) == 5:
                predict_data = np.array(position_list)
                predict_data = predict_data.reshape(1, 5, 3)
                predict_result = predict_next_positions(model, predict_data, 5)
                print("predict_result: ", predict_result)
                ## marker predict
                for i in range(len(predict_result)):
                    cv2.circle(frame, ((15*(round(predict_result[i][0])))+420, 15*(round(predict_result[i][1]))), 10, (0, 0, 255), -1)
                
        else:
            if len(position_list) > 0:
                position_list.clear()
                
        if t >= 50:
            position_list.clear()
            t = 0

        # 72X72 격자 모델
    # 격자 무늬를 출력합니다.
    for i in range(420, 1515, 15):
        cv2.line(frame, (i, 0), (i, 1080), (255, 0, 0), 1)
    for i in range(0, 1095, 15):
        cv2.line(frame, (420, i), (1500, i), (255, 0, 0), 1)

    # 사각형 그리기
    cv2.rectangle(frame, (420, 0), (1500, 1080), (0, 255, 0), 2)

    return frame


# model load
model = keras.models.load_model('/home/siwon/dev/Deeplearning-6/model/first_model.h5')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# 궤적 리스트
position_list = deque(maxlen=5)
trajectory_list = []
append_on = False

# 메인 루프
while True:
    # 프레임을 읽어옵니다.
    ret, frame = cap.read()

    frame_processing_thread = FrameProcessingThread(target=process_frame, args=(frame, position_list, model))
    frame_processing_thread.start()
    frame_processing_thread.join()

    result_frame = frame_processing_thread.result

    if result_frame is not None:
        cv2.imshow('Orange Ball Tracker', result_frame)
    else:
        print("Error: Result frame is empty.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

