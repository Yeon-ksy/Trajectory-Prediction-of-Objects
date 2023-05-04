# deeplearning-repo-6
Trajectory prediction for avoidance driving
---
![TeamGPTlogo-removebg-preview](https://user-images.githubusercontent.com/61589097/236110681-c9707bf8-d356-4709-9f30-059d79b5ff4f.png)
# GPT
## Good Prediction of Trajectory

<p align="center">
       This project is not Open AI's GPT. However, due to the influence of the name, it was produced as GPT (Good Prediction of Trajectory).
This project aims to predict the trajectory of objects for avoidance driving.

---

## Ball Trajectory
![제목 없는 동영상 (12)](https://user-images.githubusercontent.com/61589097/236112647-aac3b09d-e4e1-448b-a1e3-27e8dc8ee77c.gif)

---
## Carla Simulator
![Carla](https://user-images.githubusercontent.com/61589097/236113503-ba0305b5-eaf5-4f58-902a-44832407afaf.gif)

---
## Env.
Ubuntu 22.04
python 3.10.8
tensorflow 2.6.0
numpy 1.23.5
ultralytics
supervision 0.3.0
openCV 4.7.0

---
## Introduce
Hi! We carried out this project assuming the LIDAR situation as a deep learning project. We first show trajectory predictions for the ball and then use Carla Simulator to predict trajectories for the car.

---

## Datasets
If you want to learn, use the trajectory dataset for the [ball](https://github.com/addinedu-amr-2th/deeplearning-repo-6/files/11392528/trajectory_dataset_transformed.csv) dataset and the [carla](https://github.com/addinedu-amr-2th/deeplearning-repo-6/files/11392522/carla_data.csv) dataset.

---
## Usage
If you want to predict the trajectory for the ball in real time?

``` bash
    python3 source/mutiple_prediction.py
```
If you want to predict trajectories in real time using a Carla simulator?


* If so, look at carla_trajectory_prediction.ipynb
---

## Model
The first model.h5 is a predictive model for the trajectory of the ball. And GRU_model.h5 is a model for better ball trajectories. The carla_GRU_model.h5 model is a predictive model for the trajectory of a car.

---
## Credits

- Thanks to [PinkWink](https://github.com/PinkWink)
