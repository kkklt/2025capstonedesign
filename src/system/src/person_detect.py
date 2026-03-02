import rospy
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from std_msgs.msg import Float32

class PersonDetector:
    def __init__(self):
        rospy.init_node('person_detector_node', anonymous=True)

        # --- YOLOv5 모델 로드 ---
        
        try:
            self.model = torch.hub.load('/home/hyunsoo/capstone2025/src/yolov5', 'custom',
                       path='/home/hyunsoo/capstone2025/src/yolov5/train/person/weights/best.pt',
                       source='local')
            self.model.classes = [0]  # 'person' 클래스(ID 0)만 감지하도록 설정
            self.model.conf = 0.6  # 최소 신뢰도 설정 (필요시 조절)
            rospy.loginfo("YOLOv5 모델 로드 완료.")
        except Exception as e:
            rospy.logerr(f"YOLOv5 모델 로드 실패: {e}")
            return

        # --- RealSense 파이프라인 설정 ---
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 해상도 및 프레임 설정 (D435i에 맞춰 조절 가능)
        W, H = 640, 480
        FPS = 30
        self.config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
        self.config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
        self.config.enable_device("138422078353")

        # 깊이-컬러 정렬 (Align) 객체 생성
        self.align = rs.align(rs.stream.color)

        # 파이프라인 시작
        self.profile = self.pipeline.start(self.config)
        rospy.loginfo("RealSense 카메라 시작.")

        # --- ROS Publisher 설정 ---
        # 속도 배율을 발행할 퍼블리셔
        self.speed_pub = rospy.Publisher('/speed_multiplier', Float32, queue_size=10)

        # 루프 속도 설정
        self.rate = rospy.Rate(10)  # 10Hz

    def run(self):
        while not rospy.is_shutdown():
            # 프레임 대기 및 정렬
            try:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    rospy.logwarn("프레임을 가져올 수 없습니다.")
                    continue

                # 이미지를 numpy 배열로 변환
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # --- YOLOv5 추론 ---
                # YOLOv5는 RGB 이미지를 기대하므로 BGR -> RGB 변환
                img_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                results = self.model(img_rgb)
                detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, class]

                min_depth = float('inf')  # 감지된 사람 중 가장 가까운 거리
                person_detected = False

                for *box, conf, cls in detections:
                    if int(cls) == 0:  # 'person' 클래스
                        person_detected = True
                        
                        # 바운딩 박스 중심점 계산
                        x1, y1, x2, y2 = map(int, box)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # --- 깊이 값 추출 ---
                        # 노이즈를 줄이기 위해 중심점 주변 5x5 픽셀의 중앙값(median) 사용
                        box_size = 5
                        depth_region = depth_image[max(0, center_y - box_size):min(479, center_y + box_size),
                                                   max(0, center_x - box_size):min(639, center_x + box_size)]
                        
                        # 유효한 깊이 값(0이 아닌 값)만 필터링
                        non_zero_depths = depth_region[depth_region > 0]
                        
                        if non_zero_depths.size > 0:
                            # mm 단위를 m 단위로 변환
                            current_depth = np.median(non_zero_depths) / 1000.0
                            
                            if current_depth < min_depth:
                                min_depth = current_depth

                # --- 로직에 따른 속도 배율 결정 ---
                speed_multiplier = 1.0  # 기본값 (사람 감지 안됨)

                if person_detected and min_depth < float('inf'):
                    if min_depth < 2.0:
                        speed_multiplier = 5
                        rospy.loginfo(f"사람 감지 (위험): {min_depth:.2f}m < 2.0m -> 배율 5.0")
                    else:
                        speed_multiplier = 2
                        rospy.loginfo(f"사람 감지 (주의): {min_depth:.2f}m >= 2.0m -> 배율 2.0")
                else:
                    rospy.loginfo("사람 감지되지 않음 -> 배율 1.0")

                # 속도 배율 발행
                self.speed_pub.publish(Float32(speed_multiplier))

                # (디버깅용) 화면에 결과 표시
                results.render() # 바운딩 박스 그리기
                debug_image = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
                cv2.imshow("Person Detection", debug_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                self.rate.sleep()

            except Exception as e:
                rospy.logerr(f"메인 루프 오류: {e}")

        # 종료 시 리소스 해제
        self.pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        detector = PersonDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
