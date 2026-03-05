#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cup_placement_controller.py

ArUco/컵 중심을 추적하여 컵을 집어 트레이로 옮기는 로직과
'/speed_multiplier' 토픽을 이용한 실시간 속도 조절 로직을 병합한 노드.
"""

import xmlrpc.client
import rospy
import time
import math
import numpy as np

from dsr_msgs.srv import (
    MoveLine, MoveJointx, MoveJoint,
    MoveJointRequest, MoveJointxRequest, MoveLineRequest,
    GetCurrentPose, GetCurrentPoseRequest
)
from std_msgs.msg import Float32MultiArray, Int32, Float32
from dsr_msgs.srv import SetCtrlBoxDigitalOutput, SetCtrlBoxDigitalOutputRequest
from geometry_msgs.msg import PointStamped, PolygonStamped

class CupPlacementController:

    def __init__(self):
        # 1. ROS 노드 초기화
        rospy.init_node('cup_placement_controller', anonymous=True)
        rospy.loginfo("컵 플레이스먼트 실시간 속도 조절 노드를 시작합니다.")

        # 2. ✅ [MERGED] 실시간 속도 조절 변수 초기화
        self.speed_multiplier = 1.0
        self.vel = [0.0, 0.0]  # DSR 시간 기반 제어를 위해 vel/acc는 0으로 고정
        self.acc = [0.0, 0.0]

        # 3. ArUco 및 컵 관련 변수 초기화 (전역 변수 -> 클래스 멤버)
        self.Xg, self.Yg, self.Zg = 0.0, 0.0, 0.0
        self.ArUco_stamp = None 
        self.Xc, self.Yc, self.Zc = 0.0, 0.0, 0.0
        self.cup_stamp = None
        
        self.Xc0 = self.Yc0 = self.Zc0 = None # 컵 고정 값
        self.cup_locked = False
        self.Xg0 = self.Yg0 = self.Zg0 = None # ArUco 고정 값
        self.ArUco_locked = False

        self.received_angle = None
        self.corners_tool = []

        # 4. XML-RPC 그리퍼 클라이언트 설정
        try:
            self.server_url = "http://192.168.137.101:41414/"
            self.proxy = xmlrpc.client.ServerProxy(self.server_url)
            rospy.loginfo(f"XML-RPC 서버 연결 성공: {self.server_url}")
        except Exception as e:
            rospy.logerr(f"XML-RPC 서버 연결 실패: {e}")
            rospy.signal_shutdown("그리퍼 서버 연결 실패")

        # 5. DSR 모션 서비스 클라이언트 준비 (딱 한 번만 생성)
        try:
            rospy.loginfo("DSR 모션 서비스 연결 대기 중...")
            rospy.wait_for_service('/dsr01m1013/motion/move_joint', timeout=5.0)
            rospy.wait_for_service('/dsr01m1013/motion/move_line', timeout=5.0)
            rospy.wait_for_service('/dsr01m1013/motion/move_jointx', timeout=5.0)
            
            self.move_joint_srv = rospy.ServiceProxy('/dsr01m1013/motion/move_joint', MoveJoint)
            self.move_line_srv = rospy.ServiceProxy('/dsr01m1013/motion/move_line', MoveLine)
            self.move_jointx_srv = rospy.ServiceProxy('/dsr01m1013/motion/move_jointx', MoveJointx)
            
            rospy.loginfo("모든 DSR 모션 서비스에 성공적으로 연결되었습니다.")
            
        except Exception as e:
            rospy.logerr(f"DSR 서비스 연결 실패: {e}")
            rospy.signal_shutdown("DSR 서비스 연결 실패. 드라이버를 확인하세요.")

        # 6. ROS Subscribers 설정
        # ✅ [MERGED] 속도 배율 토픽 구독 추가
        rospy.Subscriber('/speed_multiplier', Float32, self.speed_callback)
        rospy.loginfo("'/speed_multiplier' 토픽 구독 중...")
        
        # ArUco 및 컵 관련 토픽 구독
        rospy.Subscriber("/aruco/center_tool",  PointStamped,  self.center_callback)    
        rospy.Subscriber("/aruco/corners_tool", PolygonStamped, self.corners_callback)  
        rospy.Subscriber("/cup/center_tool",    PointStamped,  self.cup_center_callback)
        rospy.Subscriber("/aruco/yaw_deg",      Float32,       self.angle_callback)     
        rospy.loginfo("ArUco 및 컵 토픽 구독 중...")

    # --- [MERGED] 실시간 속도 조절 로직 ---

    def speed_callback(self, msg):
        """
        /speed_multiplier 토픽 수신 시 호출 (별도 스레드).
        수신된 배율을 self.speed_multiplier에 저장합니다.
        """
        received_value = msg.data
        
        if received_value <= 0.01:
            if self.speed_multiplier != 0.01:
                rospy.logwarn(f"수신된 배율({received_value:.2f})이 너무 낮습니다. 0.01 (100배 느림)로 강제 적용.")
            self.speed_multiplier = 0.01
        else:
            self.speed_multiplier = received_value

    def get_scaled_time(self, base_time):
        """
        현재 self.speed_multiplier를 기준으로 스케일링된 시간을 즉시 계산하여 반환합니다.
        로직: 배율 2.0 -> 시간 1/2 (2배 빠름)
        """
        try:
            # 배율 2.0일 때 1/2 시간이 되도록 (속도 2배)
            scaled_time = base_time * self.speed_multiplier 
            return scaled_time
        except ZeroDivisionError:
            rospy.logerr_throttle(5, "Speed multiplier가 0입니다. 기본 시간으로 복구.")
            return base_time
        except Exception as e:
            rospy.logerr_throttle(5, f"시간 계산 오류: {e}. 기본 시간({base_time})으로 복구.")
            return base_time

    # --- ArUco 및 컵 콜백 함수들 ---

    def center_callback(self, msg: PointStamped):
        """(옵션) /aruco/center_tool → 필요시 사용"""
        self.Xg = msg.point.x * 1000.0
        self.Yg = msg.point.y * 1000.0
        self.Zg = msg.point.z * 1000.0
        self.ArUco_stamp = msg.header.stamp
        rospy.loginfo_throttle(1.0, f"[aruco/center_tool] Xg={self.Xg:.1f}, Yg={self.Yg:.1f}, Zg={self.Zg:.1f} (mm)")

    def angle_callback(self, msg: Float32):
        self.received_angle = float(msg.data)
        rospy.loginfo_throttle(1.0, f"[aruco/yaw_deg] {self.received_angle:.2f} deg")

    def cup_center_callback(self, msg: PointStamped):
        """컵 중심 좌표 최신값 갱신 (mm)"""
        self.Xc = msg.point.x * 1000.0
        self.Yc = - msg.point.y * 1000.0   # ← 현장 좌표계 보정 그대로 유지
        self.Zc = msg.point.z * 1000.0
        self.cup_stamp = msg.header.stamp
        rospy.loginfo_throttle(1.0, f"[cup/center_tool] latest: Xc={self.Xc:.1f}, Yc={self.Yc:.1f}, Zc={self.Zc:.1f} (mm)")

    def corners_callback(self, msg: PolygonStamped):
        self.corners_tool = [(p.x, p.y, p.z) for p in msg.polygon.points]
        rospy.loginfo_throttle(1.0, f"[aruco/corners_tool] {len(self.corners_tool)} corners")

    # --- 유틸리티 함수들 ---

    def wait_and_lock_cup_center(self, timeout=5.0, stable_samples=5, tol_mm=3.0, freshness=0.3):
        """신선한 컵 중심을 기다렸다가 평균값으로 Xc0,Yc0,Zc0를 잠근다."""
        t0 = rospy.Time.now().to_sec()
        samples = []
        rate = rospy.Rate(50)

        while not rospy.is_shutdown():
            if self.cup_stamp is not None:
                age = (rospy.Time.now() - self.cup_stamp).to_sec()
                if age < freshness and np.isfinite([self.Xc, self.Yc, self.Zc]).all():
                    samples.append((self.Xc, self.Yc, self.Zc))
                    if len(samples) >= stable_samples:
                        arr = np.array(samples[-stable_samples:])
                        spread = arr.max(axis=0) - arr.min(axis=0)
                        if np.all(spread <= tol_mm):
                            self.Xc0, self.Yc0, self.Zc0 = np.mean(arr, axis=0).tolist()
                            self.cup_locked = True
                            rospy.loginfo(f"[LOCK] Cup center: ({self.Xc0:.1f}, {self.Yc0:.1f}, {self.Zc0:.1f}) mm "
                                          f"(spread={spread[0]:.2f},{spread[1]:.2f},{spread[2]:.2f} mm)")
                            return True

            if rospy.Time.now().to_sec() - t0 > timeout:
                rospy.logwarn("Timed out waiting for stable cup center")
                return False
            rate.sleep()

    def wait_and_lock_ArUco_center(self, timeout=5.0, stable_samples=5, tol_mm=6.0, freshness=2):
        """신선한 아르코 중심을 기다렸다가 평균값으로 Xg0,Yg0,Zg0를 잠근다."""
        t0 = rospy.Time.now().to_sec()
        samples = []
        rate = rospy.Rate(50)

        while not rospy.is_shutdown():
            if self.ArUco_stamp is not None:
                age = (rospy.Time.now() - self.ArUco_stamp).to_sec()
                if age < freshness and np.isfinite([self.Xg, self.Yg, self.Zg]).all():
                    samples.append((self.Xg, self.Yg, self.Zg))
                    if len(samples) >= stable_samples:
                        arr = np.array(samples[-stable_samples:])
                        spread = arr.max(axis=0) - arr.min(axis=0)
                        if np.all(spread <= tol_mm):
                            self.Xg0, self.Yg0, self.Zg0 = np.mean(arr, axis=0).tolist()
                            self.ArUco_locked = True
                            rospy.loginfo(f"[LOCK] ArUco center: ({self.Xg0:.1f}, {self.Yg0:.1f}, {self.Zg0:.1f}) mm "
                                          f"(spread={spread[0]:.2f},{spread[1]:.2f},{spread[2]:.2f} mm)")
                            return True

            if rospy.Time.now().to_sec() - t0 > timeout:
                rospy.logwarn("Timed out waiting for stable ArUco center")
                return False
            rate.sleep()

    @staticmethod
    def rotate_passive_deg(x: float, y: float, theta_deg: float):
        """좌표축을 반시계로 theta_deg(도)만큼 회전 (패시브 회전)."""
        if theta_deg is None:
            rospy.logwarn("Rotation angle is None, using 0.0 deg.")
            theta_deg = 0.0
        theta = math.radians(theta_deg)
        ct, st = math.cos(theta), math.sin(theta)
        x_pos = x * ct + y * st
        y_pos = -x * st + y * ct
        return x_pos, y_pos

    # --- DSR 모션 서비스 호출 함수 (최적화) ---

    def movejoint(self, posx, vel_, acc_, time_):
        try:
            req = MoveJointRequest()
            req.pos = posx; req.vel = vel_; req.acc = acc_; req.time = time_
            req.radius = 0; req.mode = 0; req.blendType = 0; req.syncType = 0
            self.move_joint_srv(req) # ✅ __init__에서 생성한 프록시 사용
        except rospy.ServiceException as e:
            rospy.logerr_throttle(5, f"move_joint 서비스 호출 실패: {e}")

    def moveline(self, posx, vel_, acc_, time_):
        try:
            req = MoveLineRequest()
            req.pos = posx; req.vel = vel_; req.acc = acc_; req.time = time_
            req.radius = 0; req.mode = 0; req.blendType = 0; req.syncType = 0
            self.move_line_srv(req) # ✅ __init__에서 생성한 프록시 사용
        except rospy.ServiceException as e:
            rospy.logerr_throttle(5, f"moveline 서비스 호출 실패: {e}")

    def movejointx(self, posx, vel_, acc_, time_, sol):
        try:
            req = MoveJointxRequest()
            req.pos = posx; req.vel = vel_; req.acc = acc_; req.time = time_
            req.sol = sol
            req.radius = 0; req.mode = 0; req.blendType = 0; req.syncType = 0
            self.move_jointx_srv(req) # ✅ __init__에서 생성한 프록시 사용
        except rospy.ServiceException as e:
            rospy.logerr(f"move_jointx 서비스 호출 실패: {e}")

    # --- 메인 실행 로직 ---

    def run_main_logic(self):
        # 구독자 연결을 위한 초기 대기 (스케일링 X)
        rospy.sleep(0.5)
        rate = rospy.Rate(10)

        # 고정 파라미터
        pos_basic   = [-200, 600, 500, 0, 180, 180]
        search_tray = [-536, 660, 500, 0, 180, 90]
        search_cup  = [-260, 700, 310, 90 ,90, -90]

        # --- [MERGED] 모든 시간 값에 get_scaled_time() 적용 ---

        # 1) 기본에서 → 트레이 검색 포즈로 이동
        self.proxy.twofg_grip_external(0, 33.0, 20, 10)
        time.sleep(1)
        
        self.movejointx(pos_basic,  0, 0, self.get_scaled_time(3.0), 2) # ✅
        time.sleep(1)
        
        self.movejointx(search_tray, 0, 0, self.get_scaled_time(3.0), 2) # ✅
        time.sleep(1)

        # 2) 여기서 트레이 중심을 안정적으로 락
        ok = self.wait_and_lock_ArUco_center(timeout=5.0, stable_samples=5, tol_mm=3.0, freshness=0.3)
        if not ok:
            rospy.logerr("Could not capture ArUco center. Abort.")
            return
        
        rospy.loginfo(f"Using locked ArUco center: Xg0={self.Xg0:.1f}, Yg0={self.Yg0:.1f}, Zg0={self.Zg0:.1f} (mm)")

        # ===== 컵 도착 지점 표시 =====
        # (self.received_angle이 None일 경우 대비하여 rotate_passive_deg 내부 수정)
        x_pos, y_pos = self.rotate_passive_deg(-120, 130, self.received_angle)
        time.sleep(1)
        rospy.loginfo(f"Pos: X_pos={x_pos:.1f}, Y_pos={y_pos:.1f} (mm)")
        
        # 3) 트레이 검색 포즈에서 → 컵 검색 포즈로 이동
        self.movejointx(search_cup, 0, 0, self.get_scaled_time(3.0), 2) # ✅
        time.sleep(1)

        # 4) 여기서 컵 중심을 '처음 한 번' 안정적으로 락
        ok = self.wait_and_lock_cup_center(timeout=5.0, stable_samples=5, tol_mm=3.0, freshness=0.3)
        if not ok:
            rospy.logerr("Could not capture cup center at search pose. Abort.")
            return

        rospy.loginfo(f"Using locked cup center: Xc0={self.Xc0:.1f}, Yc0={self.Yc0:.1f}, Zc0={self.Zc0:.1f} (mm)")

        # 목표 포즈들 (고정된 Xc0, Yc0, Xg0, Yg0 값 사용)
        pos_grip           = [-260 - self.Yc0 + 74,             700 , 310, 90 ,90, -90]
        pos_grip_start     = [-260 - self.Yc0 + 74,  700 + self.Xc0 - 87, 310, 90 ,90, -90]
        pos_grip_start_up  = [-260 - self.Yc0 + 74,  700 + self.Xc0 - 87, 450, 90 ,90, -90] 
        movetotray         = [-341 + self.Yg0 - y_pos, 660 + self.Xg0 + x_pos, 450, 180 ,90, -90]
        movetotray_down    = [-341 + self.Yg0 - y_pos, 660 + self.Xg0 + x_pos, 120, 180 ,90, -90]
        basic              = [-301 + self.Yg0 - y_pos, 660 + self.Xg0 + x_pos, 450, 180 ,90, -90]
        basic2             = [-301 + self.Yg0 - y_pos, 660 + self.Xg0 + x_pos, 500, 0 ,180, 180]

        # ===== 동작 시퀀스 =====
        try:
            rospy.loginfo("--- 동작 시퀀스 시작 (실시간 속도 적용 모드) ---")
            
            # (디버깅용 로그) 현재 적용될 배율을 5초에 한 번씩 출력
            rospy.loginfo_throttle(5.0, f"현재 배율 {self.speed_multiplier:.2f} (다음 동작에 즉시 적용)")

            # 접근 경로
            self.moveline(pos_grip,    self.vel, self.acc, self.get_scaled_time(3.0)) # ✅
            time.sleep(1)

            # 그리퍼 오픈 → 접근 → 클로즈 → 상승
            self.proxy.twofg_grip_external(0, 71.0, 20, 10)
            time.sleep(1)

            self.moveline(pos_grip_start,     self.vel, self.acc, self.get_scaled_time(3.0)) # ✅
            time.sleep(1)
            
            self.proxy.twofg_grip_external(0, 42.0, 20, 10)
            time.sleep(1)
            
            self.moveline(pos_grip_start_up,  self.vel, self.acc, self.get_scaled_time(3.0)) # ✅
            time.sleep(1)

            # 트레이로 이동, 내려놓기
            self.moveline(movetotray, self.vel, self.acc, self.get_scaled_time(6.0)) # ✅
            time.sleep(1)
            
            self.moveline(movetotray_down, self.vel, self.acc, self.get_scaled_time(3.0)) # ✅
            time.sleep(1)
            
            self.proxy.twofg_grip_external(0, 71.0, 20, 10)
            time.sleep(1)
            
            self.moveline(movetotray, self.vel, self.acc, self.get_scaled_time(3.0)) # ✅
            time.sleep(1)
            
            self.proxy.twofg_grip_external(0, 33.0, 20, 10)
            time.sleep(1)

            # 복귀
            self.movejointx(basic, 0, 0, self.get_scaled_time(5.0), 2) # ✅
            self.movejointx(basic2, 0, 0, self.get_scaled_time(5.0), 2) # ✅
            self.movejointx(pos_basic, 0, 0, self.get_scaled_time(5.0), 2) # ✅

            rospy.loginfo("--- 모든 시퀀스 완료 ---")

        except Exception as e:
            rospy.logerr(f"시퀀스 실행 중 예외 발생: {e}")

if __name__ == '__main__':
    try:
        # 1. 컨트롤러 객체 생성 (이때 __init__이 실행됨)
        controller = CupPlacementController()
        
        # 2. 메인 로직 실행 (전체 시퀀스가 완료될 때까지 블로킹)
        controller.run_main_logic()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS 노드가 중단되었습니다.")
    except Exception as e:
        rospy.logerr(f"예기치 않은 오류 발생: {e}")