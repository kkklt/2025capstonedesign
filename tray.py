#!/usr/bin/env python3

import xmlrpc.client
import rospy
import time
import math
import sys
from dsr_msgs.srv import MoveLine, MoveJointx, MoveJoint, MoveJointRequest, MoveJointxRequest, MoveLineRequest
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32

# ========================
# 🔧 상수 설정
# ========================
# 로봇 TCP 위치 오프셋 (mm): 로봇 베이스 좌표계 기준 목표 위치 계산 시 사용
BASE_X_OFFSET = 400.0
BASE_Y_OFFSET = 400.0
APPROACH_Y_OFFSET = -5.0 # 접근 시 Y축 미세 조정 오프셋

# Z축 높이 (mm): 로봇의 작업 높이를 정의
APPROACH_Z = 400.0 # 물체에 접근하거나 회피할 때의 Z축 높이
PICK_Z = 254.0     # 물체를 집거나 내려놓을 때의 Z축 높이

# 로봇 TCP 기본 방향 (Euler angles, degrees): 로봇 툴의 기본자세
DEFAULT_RX = 0.0
DEFAULT_RY = 180.0
DEFAULT_RZ = 180.0

# 로봇 모션 기본 속도/가속도 (MoveLine용)
MOVE_VEL = [50, 50] # [선속도, 각속도]
MOVE_ACC = [100, 100] # [선가속도, 각가속도]

class TrayController:
    """
    ArUco 마커의 위치와 각도 정보를 ROS 토픽으로 구독하고, 
    이를 기반으로 로봇 팔의 물체 집기/놓기 동작 시퀀스를 제어하는 클래스.
    """
    def __init__(self):
        """TrayController 클래스의 생성자. ROS 노드, 구독자, 서비스 프록시를 초기화합니다."""
        rospy.init_node('tray_controller_node', anonymous=True)

        # --- 수신 데이터 저장을 위한 멤버 변수 ---
        self.aruco_center_x = 0.0
        self.aruco_center_y = 0.0
        self.aruco_center_z = 0.0
        self.aruco_rotation_angle = 0.0
        self.data_received = False # ArUco 데이터 수신 완료 여부를 나타내는 플래그

        # --- ROS 토픽 구독자 설정 ---
        # 마커 중심 좌표(/aruco/center_tool)와 회전 각도(/aruco/rotation_angle)를 구독
        rospy.Subscriber("/aruco/center_tool", PointStamped, self.center_callback)
        rospy.Subscriber("/aruco/rotation_angle", Float32, self.angle_callback)

        # --- 로봇 제어 서비스 프록시 초기화 ---
        try:
            # 사용할 DSR 로봇 모션 서비스에 대한 연결 시도 (2초 타임아웃)
            rospy.wait_for_service('/dsr01m1013/motion/move_joint', timeout=2.0)
            self.move_joint_srv = rospy.ServiceProxy('/dsr01m1013/motion/move_joint', MoveJoint)
            self.move_line_srv = rospy.ServiceProxy('/dsr01m1013/motion/move_line', MoveLine)
            self.move_jointx_srv = rospy.ServiceProxy('/dsr01m1013/motion/move_jointx', MoveJointx)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr(f"Failed to connect to DSR services: {e}")
            rospy.signal_shutdown("Service connection failed")
        
        # --- Gripper 제어용 XML-RPC 프록시 초기화 ---
        try:
            server_url = "http://192.168.137.101:41414/" # 그리퍼 제어 서버 주소
            self.gripper_proxy = xmlrpc.client.ServerProxy(server_url)
        except Exception as e:
            rospy.logerr(f"Failed to connect to gripper XML-RPC server: {e}")
            self.gripper_proxy = None

    def center_callback(self, msg):
        """'/aruco/center_tool' 토픽 콜백 함수. 마커의 중심 좌표를 수신하여 멤버 변수에 저장합니다."""
        self.aruco_center_x = msg.point.x
        self.aruco_center_y = msg.point.y
        self.aruco_center_z = msg.point.z
        if not self.data_received: # 데이터 첫 수신 시 로그 기록
             self.log_once_and_set_flag()

    def angle_callback(self, msg):
        """'/aruco/rotation_angle' 토픽 콜백 함수. 마커의 회전 각도를 수신하여 멤버 변수에 저장합니다."""
        self.aruco_rotation_angle = msg.data
        if not self.data_received: # 데이터 첫 수신 시 로그 기록
             self.log_once_and_set_flag()

    def log_once_and_set_flag(self):
        """ ArUco 좌표와 각도 데이터가 모두 수신되면, 한 번만 로그를 남기고 데이터 수신 완료 플래그를 설정합니다. """
        if self.aruco_center_x != 0 and self.aruco_rotation_angle != 0:
            rospy.loginfo("Initial ArUco pose received successfully.")
            rospy.loginfo(f"  - Center: (x={self.aruco_center_x:.1f}, y={self.aruco_center_y:.1f})")
            rospy.loginfo(f"  - Angle: {self.aruco_rotation_angle:.2f} degrees")
            self.data_received = True

    def move_jointx(self, pos, vel=100, acc=100, time=5.0, sol=2):
        """DSR 로봇의 MoveJointx 서비스를 호출하는 래퍼(wrapper) 함수."""
        try:
            req = MoveJointxRequest(pos=pos, vel=vel, acc=acc, time=time, radius=0, mode=0, blendType=0, syncType=0, sol=sol)
            self.move_jointx_srv(req)
            rospy.loginfo(f"move_jointx to {pos} success")
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to call service move_jointx: {e}")
            rospy.signal_shutdown("Service call failed")

    def move_line(self, pos, vel=MOVE_VEL, acc=MOVE_ACC, time=3.0):
        """DSR 로봇의 MoveLine 서비스를 호출하는 래퍼(wrapper) 함수."""
        try:
            req = MoveLineRequest(pos=pos, vel=vel, acc=acc, time=time, radius=0, mode=0, blendType=0, syncType=0)
            self.move_line_srv(req)
            rospy.loginfo(f"move_line to {pos} success")
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to call service move_line: {e}")
            rospy.signal_shutdown("Service call failed")

    def grip(self, width, speed=20, force=10):
        """XML-RPC를 통해 그리퍼를 제어하는 래퍼(wrapper) 함수."""
        if self.gripper_proxy:
            try:
                self.gripper_proxy.twofg_grip_external(0, float(width), int(speed), int(force))
                rospy.loginfo(f"Gripper command sent: width={width}")
            except Exception as e:
                rospy.logerr(f"Gripper control failed: {e}")
        else:
            rospy.logwarn("Gripper not connected, skipping grip command.")

    def run(self):
        """메인 실행 로직. ArUco 데이터 수신 후 로봇 동작 시퀀스를 수행합니다."""
        rospy.loginfo("Waiting for ArUco marker data...")
        
        # ArUco 데이터가 수신될 때까지 최대 10초간 대기
        wait_rate = rospy.Rate(10)
        for _ in range(100): # 10Hz * 10초 = 100회 체크
            if self.data_received:
                break
            wait_rate.sleep()
        
        # 타임아웃 처리: 시간 내에 데이터를 받지 못하면 노드 종료
        if not self.data_received:
            rospy.logerr("Timeout: Did not receive ArUco data. Shutting down.")
            return

        # --- 로봇 동작 시퀀스 시작 ---
        
        # 1. 마커 위치 기반으로 로봇의 목표 위치 계산
        target_x = BASE_X_OFFSET + self.aruco_center_x
        target_y = BASE_Y_OFFSET + self.aruco_center_y + APPROACH_Y_OFFSET
        
        # 수신된 각도를 로봇의 RZ(Yaw)축 회전에 적용 (기본 자세 180도에 더함)
        target_rz = DEFAULT_RZ + self.aruco_rotation_angle
        
        # 동작에 사용할 주요 위치들을 사전 정의
        approach_pos = [target_x, target_y, APPROACH_Z, DEFAULT_RX, DEFAULT_RY, target_rz] # 물체 위 접근 위치
        pick_pos =     [target_x, target_y, PICK_Z,     DEFAULT_RX, DEFAULT_RY, target_rz] # 물체 잡는 위치
        initial_pos =  [BASE_X_OFFSET, BASE_Y_OFFSET, APPROACH_Z, DEFAULT_RX, DEFAULT_RY, DEFAULT_RZ] # 초기/대기 위치

        rospy.loginfo("Starting robot motion sequence...")

        # 2. 초기 위치(initial_pos)로 이동
        self.move_jointx(initial_pos, time=5)
        time.sleep(1)

        # 3. 그리퍼 열기 (물건을 잡을 준비)
        self.grip(width=70.0) # Open
        time.sleep(1)

        # 4. 마커 위 접근 위치(approach_pos)로 이동 (회전각 적용됨)
        self.move_jointx(approach_pos, time=5)
        time.sleep(1)

        # 5. 마커를 잡기 위해 직선으로 하강(pick_pos)
        self.move_line(pick_pos, time=3)
        time.sleep(1)
        
        # 6. 그리퍼 닫기 (물건 잡기)
        self.grip(width=16.0) # Close
        time.sleep(1)

        # 7. 물건을 들어올리기 (다시 접근 위치로)
        self.move_jointx(approach_pos, time=5)
        time.sleep(1)

        # 8. 초기 위치(initial_pos)로 복귀
        self.move_jointx(initial_pos, time=5)
        time.sleep(1)

        # 9. 물건을 내려놓기 위해 초기 위치에서 직선으로 하강
        self.move_line([initial_pos[0], initial_pos[1], PICK_Z, DEFAULT_RX, DEFAULT_RY, DEFAULT_RZ], time=3)
        time.sleep(1)

        # 10. 그리퍼 열기 (물건 놓기)
        self.grip(width=70.0) # Open
        time.sleep(1)
        
        # 11. 다시 초기 위치로 복귀 (Z축 상승)
        self.move_line(initial_pos, time=3)
        time.sleep(1)

        rospy.loginfo("Motion sequence completed successfully.")


if __name__ == '__main__':
    """
    스크립트의 메인 실행 블록.
    TrayController 객체를 생성하고, 예외 처리를 포함하여 실행합니다.
    """
    try:
        controller = TrayController()
        controller.run()
    except rospy.ROSInterruptException:
        # 사용자가 Ctrl+C 등으로 ROS 노드를 종료한 경우
        pass
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred in main: {e}")