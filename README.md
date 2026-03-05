# Autonomous Food Service Robot System

시스템은 환경 인식, 경로 계획, 작업 수행의 세 가지 주요 모듈로 구성되며, 안전성과 정밀성을 동시에 고려합니다.

 ---

## 1. Introduction

본 프로젝트는 **Doosan M1013** 로봇팔을 기반으로 하여, 튀김/구이 조리, 컵 파지 및 배차 작업을 자동화하는 비대면 로봇 시스템을 개발하였습니다.
로봇은 **자체 제작한 ATC**(Automatic Tool Changer)를 통해 작업에 맞는 도구를 스스로 교체하며, **범용 그리퍼와 트레이**를 직접 설계·제작하여 컵과 트레이 모두 정밀하게 파지할 수 있도록 구현하였습니다.
작업 중 주변 사람의 존재는 카메라를 통해 실시간으로 감지되며, 사람 근접 시 로봇 속도를 자동으로 조절하여 안전성을 확보합니다.

---

## 2. System Architecture

본 시스템은 Perception-Planning-Action 루프를 통해 비대면 환경에서 자율적인 서비스를 수행합니다.

- **Perception**: RealSense 및 YOLOv5를 통한 공간 인지 및 객체 탐지
- **Planning**: MoveIt 환경 내 PRM/A* 기반의 최적 경로 설계
- **Execution**: 자체 제작 ATC 및 전용 툴을 활용한 조리 및 배차(Dispatch)
  
<img width="440" height="240" alt="image" src="https://github.com/user-attachments/assets/0c883216-6d86-4913-8979-87074e92da82" />
<img width="408" height="93" alt="image" src="https://github.com/user-attachments/assets/50e070a0-b1ae-45e2-91af-ac578e48af30" />

---

## 3. Key Features

- **Optimized Path Planning**: PRM + A* 기반 충돌 없는 경로 계획
- **Vision-Guided Robust Manipulation**: YOLOv5 기반 객체 탐지 및 실시간 피드백을 통한 안정적인 물체 파지 구현
- **Autonomous Tool Exchange System**: 자체 제작 ATC를 통한 자율적 도구 교체 시스템
- **Integrated Workflow**: 조리부터 픽업 존 배차(Placement)까지의 전 공정 자동화
- **Reactive Safety Control**: 사람과의 거리에 따른 실시간 속도 제어 로직

---

## 4. System Details

시스템은 **Perception-Planning-Action** 구조를 기반으로 하며, 각 단계는 다음과 같이 구성됩니다.

### 4.1 Environment Perception
- **Initial Data Acquisition**: 작업 환경은 사람이 촬영하며, Intel RealSense D435i를 통해 RGB-D 데이터를 수집
- **Hand-Eye-Callibration**: 방법을 기반으로 카메라와 로봇 베이스 간의 좌표 변환 행렬을 산출
  
  <img width="313" height="170" alt="image" src="https://github.com/user-attachments/assets/e78d5e68-0773-4630-b32f-4711597b9fae" />
- **Mapping**: 수집된 데이터를 기반으로 **Octomap**을 생성하고, 로봇이 공간을 모델링하도록 처리
- **Remarks**: 로봇은 초기 데이터에 기반하여 환경을 이해하며, 작업 중 사람의 근접 여부는 카메라를 통해 실시간 감지
  
  <img width="322" height="219" alt="image" src="https://github.com/user-attachments/assets/3969981e-e740-4d5d-9ec6-6c104538e7cc" />

### 4.2 Path Planning
- **Algorithm**: Probabilistic Roadmap (PRM)과 A* 알고리즘을 사용하여 경로를 계획
- **Sampling Method**: Halton Sequence 기반 준난수 샘플링(Quasi-random Sampling) 적용
- **Constraints**: Self-Collision, 작업 영역 제한, 안전 거리 준수
- **Output**: 5cm 단위로 세분화된 Cartesian 좌표 경로를 ROS 서비스('/dsr01m1013/motion/move-line')에 전달
  
  <img width="225" height="311" alt="image" src="https://github.com/user-attachments/assets/2f16928f-f9a9-404a-94dd-1386d00fff56" />
  <img width="225" height="311" alt="image" src="https://github.com/user-attachments/assets/b2f729da-7ed8-4b8a-ae73-a46eae4bbd57" />
  
### 4.3 Action and Tool Management
- **Object Manipulation**: YOLOv5 기반 객체 검출로 컵 및 조리 도구 위치 파악 후 안정적인 물체 파지
  
  <img width="316" height="189" alt="image" src="https://github.com/user-attachments/assets/2e9e5e98-90d9-4bb2-8c5c-5df69876ef86" />
- **Workflow Execution**: 실제 조리 동작 시퀀스를 모델링하여 튀김/구이 작업 수행
- **2-Function Gripper & Tray**: 컵과 트레이를 동시에 파지할 수 있는 다기능 그리퍼 설계

  <img width="433" height="170" alt="image" src="https://github.com/user-attachments/assets/bd3535e4-db83-4023-b516-3ef893e5ef12" />
  <img width="306" height="273" alt="image" src="https://github.com/user-attachments/assets/82f9842f-347f-48ee-9b7e-a9910b5071e3" />
  <img width="320" height="188" alt="image" src="https://github.com/user-attachments/assets/a16ca48f-88a4-4f6d-848e-bb629536cbfc" />
- **Tool Changer (ATC)**: 자동 도구 교체를 통해 튀김기, 스크래퍼 등 다양한 작업 수행
  
  <img width="324" height="275" alt="image" src="https://github.com/user-attachments/assets/bb1aa41a-bc42-40d8-9171-315d7c5e0016" />
  <img width="313" height="177" alt="image" src="https://github.com/user-attachments/assets/4435c176-f6cf-4c9c-a4dd-c655b083d3fa" />
- **Design-to-Real Integration**: 자체 설계한 STP 데이터를 기반으로 로봇 가동 범위와 도구 간 간섭을 사전 시뮬레이션하여 하드웨어 안전성 확보
- **Hardware Analysis**:
  | Component | Analysis Type | Result |
  |-------------------|------|--------|
  | Passive ATC (v1.2)| Structural Stress (ANSYS) | 조리 모션(11.438Nm) 중 안전계수 3.32 확보|
  | Master/Tool Unit | Static Load Analysis | 12kg 하중 인가 시 안전계수 15 이상 달성|
  | Full System Integration | Joint Torque Analysis | 1kg 부하 조리 시 안전계수 2.54 확보 (기준치 2.0 상회)|
  | Tool Exchange Interface | Statistical Repeatability | 3σ 기준 $\pm 0.010 mm$의 반복 정밀도 달성|

  <img width="323" height="258" alt="image" src="https://github.com/user-attachments/assets/663b78ec-50fc-481f-a966-aa06623dd8d7" />

### 4.4 Safety Measures
- **Adaptive Speed Control**: 실시간 충돌을 검출하고 사람 거리를 감지하여 작업 속도 조정으로 안전 확보
  
  <img width="310" height="210" alt="image" src="https://github.com/user-attachments/assets/d8b7a3f6-e37f-47a7-b8d3-8923b68c097e" />

---

## 5. Demo

### 5.1 Frying & Pouring
![frying](images/demo/fying_motion.gif)
![pouring](images/demo/puring_motion.gif)

### 5.2 Cup  Detection & Grasp
![cup_grab](images/demo/cup_detection_grasp.gif)

### 5.3 Tray Handling
![tray_handling](images/demo/path_execution.gif)

### 5.4 Perception & Path Generation
![perception_path](images/demo/environment_path_planning.gif)

### 5.5 Reactive Velocity Control
![reactive_velocity_control](images/demo/safety_speed_control.gif)

### 5.6 Full Demo Video
[![Google Drive Badge](https://img.shields.io/badge/Google%20Drive-Full%20Demo%20Video-blue?logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1moUarLyRawE7uuQXFMHs0Go0cn7Y1omA/view?usp=sharing)
> *위 배지를 클릭하면 전체 구동 영상으로 이동합니다.*

---

## 6. Technical Specifications

| Component            | Description                       |
|----------------------|-----------------------------------|
| Robot                | Doosan M1013                      |
| Vision / Sensor      | Intel RealSense D435i, YOLOv5     |
| Mapping / Planning   | Octomap, PRM, A*                  |
| Middleware           | ROS1 Noetic                       |
| Design Tools         | SolidWorks, Autodesk Fusion 360 (ATC & Custom Tools) |

---

## 7. Project Structure

```
2025capstonedesign/
├── src/
│   ├── system/                        # 통합 제어 및 작업 시퀀스
│   ├── path/                          # PRM & A* 경로 계획 알고리즘
│   ├── realsense_obstacle_detector/   # RealSense 기반 환경 인지 및 Octomap
│   └── yolov5/                        # 객체 인식 및 좌표 변환 모듈
├── images/                            # 시스템 구동 시뮬레이션 및 실험 사진/GIF
├── models/                            # ATC, 그리퍼, 트레이, 어댑터 설계 데이터 (STP/STEP)
├── README.md
├── .gitignore
└── .gitattributes
```

---

## 8. Prerequisites & Installation

본 시스템은 **ROS Noetic** 환경에서 최적화되어 있습니다.

### 8.1 System & Hardware Requirements
- **OS**: Ubuntu 20.04 LTS
- **Middleware**: ROS Noetic
- **Sensors**: Intel RealSense D435i
- **Robot**: Doosan Robotics M1013 (Doosan ROS 패키지 필요)

### 8.2 Software Dependencies
이 프로젝트는 다음과 같은 외부 라이브러리 및 패키지에 의존합니다.

* **Motion Planning**: [MoveIt](https://moveit.ros.org/), [Octomap](http://octomap.github.io/)
* **Vision**: [YOLOv5](https://github.com/ultralytics/yolov5) (PyTorch Environment), RealSense SDK 2.0
* **Drivers**: `doosan-robot`, `realsense2_description`

### 8.3 Installation & Build
터미널에서 아래 명령어를 순서대로 입력하여 작업 공간을 설정하고 프로젝트를 빌드합니다.

```bash
# 1. Workspace 생성 및 소스 코드 클론
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone [본 프로젝트 저장소 URL]

# 2. 의존성 패키지 설치
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# 3. 프로젝트 빌드
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

---

## 9. Experimental Results

|Task                        | Performance | Remarks|
|----------------------------|-------------|----|
| Calibration Precision (RMSE)| 1.21mm | Tsai 방법을 활용한 Hand-Eye Calibration 및 좌표계 정합 정밀도| 
| Mapping Accuracy (Voxel) | 95.8% | Digital Twin 환경 대비 실제 실험실 환경의 OctoMap 정합 일치도|
| Path Planning Success Rate | 98%         | Halton Sequence 기반 균일 샘플링을 통한 고신뢰성 최적화된 경로 생성 |
| Safety Reliability         | 100%        | PRM+A 알고리즘으로 생성된 경로 실행 시 충돌 검증 |
| Frying / Grilling Accuracy | 95%         | 실제 조리 동작 시퀀스 모델링 및 100회 이상 반복 실험 성공률 |
| Cup Grasping Success Rate  | 98%         | YOLOv5 + ArUco Marker 기반 정밀 좌표 추출 및 파지 신뢰도 확보 |
| Human Distance Response    | 0.2s        | YOLOv5 기반 실시간 위치 추적 및 거리에 따른 2단계 능동 감속 제어 (0.5x, 0.2x) 수행 |

> *평가 지표: 총 100회 이상의 반복 실험을 통해 작업 성공률, 경로 효율성, 충돌 여부, 안전 반응 시간 검증.*

<img width="324" height="225" alt="image" src="https://github.com/user-attachments/assets/216612e7-f68b-4e0c-b686-04a722bb969a" />
<img width="336" height="299" alt="image" src="https://github.com/user-attachments/assets/7ab6a811-e7d3-441f-b8e8-27304eb8a5f4" />
<img width="831" height="182" alt="image" src="https://github.com/user-attachments/assets/4f83c5cb-2f8d-4852-a223-c04338c08f00" />
<img width="589" height="206" alt="image" src="https://github.com/user-attachments/assets/98335845-76a5-420e-a952-5c63dc7098ce" />

---

## 10. Team Contribution

| Member | Role                                          |
|--------|-----------------------------------------------|
| 고영웅 | Environment Perception & 3D Mapping            |
| 김현수 | Cooking Trajectory Design & System Integration |
| 나현수 | Gripper & Tray Design                          |
| 서영채 | Collision Avoidance & Path Planning            |
| 신원빈 | Object Detection & Speed Control               |
| 정재승 | ATC & Adapter Design                           |
