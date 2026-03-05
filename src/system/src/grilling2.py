import xmlrpc.client
import rospy
import time
import math
import numpy as np
from nav_msgs.msg import Path
from std_msgs.msg import Float64MultiArray
from dsr_msgs.srv import (
    MoveLine, MoveJointx, MoveJoint,
    MoveJointRequest, MoveJointxRequest, MoveLineRequest,
    MoveCircle , MoveCircleRequest,
    GetCurrentPose, GetCurrentPoseRequest
)
from std_msgs.msg import Float32MultiArray, Int32, Float32
from dsr_msgs.srv import SetCtrlBoxDigitalOutput, SetCtrlBoxDigitalOutputRequest
from geometry_msgs.msg import PointStamped, PolygonStamped

class CupPlacementController:

    def __init__(self):
        # 1. ROS 노드 초기화
        rospy.init_node('cup_placement_controller', anonymous=True)
        rospy.loginfo("튀김 플레이스먼트 실시간 속도 조절 노드를 시작합니다.")

        # 2. ✅ [MERGED] 실시간 속도 조절 변수 초기화
        self.speed_multiplier = 1.0
        self.received_path = []

        self.vel = [0.0, 0.0]  # DSR 시간 기반 제어를 위해 vel/acc는 0으로 고정
        self.acc = [0.0, 0.0]

        # 3. ArUco 및 컵 관련 변수 초기화 (전역 변수 -> 클래스 멤버)
        self.Xg, self.Yg, self.Zg = 0.0, 0.0, 0.0
        self.ArUco_stamp = None 
        
        self.Xc0 = self.Yc0 = self.Zc0 = None # 컵 고정 값
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
        rospy.Subscriber("/aruco/yaw_deg",      Float32,       self.angle_callback) 
        rospy.Subscriber("/path_cook", Path, self.path_callback)    
        rospy.loginfo("ArUco 토픽 구독 중...")

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
    
    def path_callback(self, msg: Path):
        self.received_path = []  # 기존 데이터 초기화

        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            z = pose_stamped.pose.position.z
            self.received_path.append((x, y, z))  # 클래스 내부 변수에 저장

        rospy.loginfo_throttle(1.0, f"[path_cook] {len(self.received_path)} waypoints stored.")


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

    def corners_callback(self, msg: PolygonStamped):
        self.corners_tool = [(p.x, p.y, p.z) for p in msg.polygon.points]
        rospy.loginfo_throttle(1.0, f"[aruco/corners_tool] {len(self.corners_tool)} corners")

    # --- 유틸리티 함수들 ---

    def execute_path(self):
        vel = [0, 0]
        acc = [0, 0]

        # ✅ [MERGED] 속도 로직 적용
        if not self.received_path:
            rospy.logwarn("No path data received yet.")
            return
        
        rospy.loginfo(f"[Executor] Executing {len(self.received_path)} waypoints..")

        for (x, y, z) in self.received_path:
            posx = [x, y, z, 0, 180, 180]
            # ✅ [MERGED] 고정 시간 2초 대신 get_scaled_time 사용
            self.moveline(posx, vel, acc, self.get_scaled_time(5.0))
            # ✅ [MERGED] 고정 sleep 0.5초 대신 get_scaled_time 사용
            time.sleep(0.5)

    def execute_path2(self):
        vel = [0, 0]
        acc = [0, 0]

        # ✅ [MERGED] 속도 로직 적용
        if not self.received_path:
            rospy.logwarn("No path data received yet.")
            return
        
        rospy.loginfo(f"[Executor] Executing {len(self.received_path)} waypoints in reverse order..")

        for (x, y, z) in reversed(self.received_path):
            posx = [x, y, z, 0, 180, 180]
            # ✅ [MERGED] 고정 시간 2초 대신 get_scaled_time 사용
            self.moveline(posx, vel, acc, self.get_scaled_time(5.0))
            # ✅ [MERGED] 고정 sleep 0.5초 대신 get_scaled_time 사용
            time.sleep(0.5)

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

    def list_to_float64multiarray_list(self, pos_list):
        arr_list = []
        for p in pos_list:
            m = Float64MultiArray()
            m.data = p
            arr_list.append(m)
        return arr_list

    def movecircle(self, pos_list, vel_, acc_, time_):
        rospy.wait_for_service('/dsr01m1013/motion/move_circle')
        try:
            move_circle_srv = rospy.ServiceProxy('/dsr01m1013/motion/move_circle', MoveCircle)

            req = MoveCircleRequest()
            req.pos      = self.list_to_float64multiarray_list(pos_list)
            req.vel      = vel_
            req.acc      = acc_
            req.time     = time_
            req.radius   = 0
            req.ref      = 0
            req.mode     = 0
            req.angle1   = 0
            req.angle2   = 0
            req.blendType = 0
            req.syncType  = 0

            resp = move_circle_srv(req)
            if resp.success:
                rospy.loginfo("move_circle Success")
            else:
                rospy.logwarn("move_circle called but success == False")

        except rospy.ServiceException as e:
            rospy.logerr("Failed to call service move_circle: %s", e)
            rospy.signal_shutdown("Service call failed")


    # --- 메인 실행 로직 ---

    def run_main_logic(self):
        # 구독자 연결을 위한 초기 대기 (스케일링 X)
        rospy.sleep(0.5)
        rate = rospy.Rate(10)

        # 고정 파라미터
        pos_basic   = [   -200,      600,     500, 0, 180, 180]   #정위치
        search_tray = [   -536,      660,     500, 0, 180,  90]   #트레이 위치 확인하러 이동
        move2ATC    = [-41.080, -745.700, 421.640, 0, 180, 180]   #석쇠 뒤집는 용도 ATC 체결하러 이동
        move2ATC2   = [-772.100, -343.850, 452.350, 0, 180, 90]   #음식 뜨는 용도 ATC 체결하러 이동

        # --- [MERGED] 모든 시간 값에 get_scaled_time() 적용 ---

        # 1) 기본에서 → 트레이 검색 포즈로 이동
        self.movejointx(pos_basic,  0, 0, self.get_scaled_time(3.0), 2)
        time.sleep(1)
        
        self.movejointx(search_tray, 0, 0, self.get_scaled_time(3.0), 2)
        time.sleep(1)

        # 2) 여기서 트레이 중심을 안정적으로 락
        ok = self.wait_and_lock_ArUco_center(timeout=5.0, stable_samples=5, tol_mm=3.0, freshness=0.3)
        if not ok:
            rospy.logerr("Could not capture ArUco center. Abort.")
            return
        
        rospy.loginfo(f"Using locked ArUco center: Xg0={self.Xg0:.1f}, Yg0={self.Yg0:.1f}, Zg0={self.Zg0:.1f} (mm)")

        # ===== 튀김 도착 지점 표시 =====
        # (self.received_angle이 None일 경우 대비하여 rotate_passive_deg 내부 수정)
        x_pos, y_pos = self.rotate_passive_deg(80, 110, self.received_angle)
        time.sleep(1)
        rospy.loginfo(f"Pos: X_pos={x_pos:.1f}, Y_pos={y_pos:.1f} (mm)")
        
        # 3) 트레이 검색 포즈에서 → ATC 체결하러 이동
        self.movejointx(move2ATC, 0, 0, self.get_scaled_time(8.0), 2)
        time.sleep(1)

        # 목표 포즈들 (고정된 Xg0, Yg0 값 사용)
       
        atc1        = [-41.080, -745.700, 321.640, 0, 180,  180]    #ATC 체결 하강
        atc2        = [-41.080, -695.700, 321.640, 0, 180,  180]   #ATC 체결 후 스테이션과 분리
        atc3        = [-41.080, -695.700, 621.640, 0, 180,  180]    #스테이션과 분리 후 상승  

        ready1      = [-200.200, -653.910, 566.550, 180, 90, -90]
        ready2      = [-200.200, -653.910, 266.550, 180, 90, -90]  #굽기 준비 ########################3333333333 좌표 따야함
        grilling1   = [-370.200, -653.910, 266.550, 180, 90, -90]  #굽기 하러 체결       
        grilling2   = [-370.200, -653.910, 566.550, 180, 90, -90]  #체결 후 상승
        grilling3   = [-370.200, -653.910, 566.550, 180, 90, 90]  #상승 후 회전
        grilling4   = [-370.200, -653.910, 266.550, 180, 90, 90]  #하강
        grilling5   = [-200.200, -653.910, 266.550, 180, 90, 90]  #하강 후 분리


        atc4        = [-772.100, -343.850, 352.350, 0, 180, 90]    #ATC 체결 하강
        atc5        = [-672.100, -343.850, 352.350, 0, 180, 90]   #ATC 체결 후 스테이션과 분리
        atc6        = [-672.100, -343.850, 552.350, 0, 180, 90]    #스테이션과 분리 후 상승 
        move2cook   = [  -200 ,  -654.95,  550.880, 0, 180,  270]    #조리대로 이동 
        move2cook2  = [  -200 , -654.95, 354.44, 0, 180, 270]    #조리대로 이동

        #조리 모션 수행
        step1       = [  -330 , -654.95, 354.44, 0, 180, 270]     #석쇠 열기 위해 출발
        pos_list    = [
                        [-435.800 , -654.790, 581.64, 0, 180, 270],  # 석쇠를 열기 위한 원형 궤적
                        [    -657 , -650.790, 647.84, 0, 180, 270]   
                      ]
        step2      = [  -630 , -654.95, 554.44, 0, 180, 270]         #석쇠와 분리
        step3      = [  -330 , -654.95, 554.44, 0, 180, 270]    #석쇠와 분리 후 음식 뜰 준비
        step4      = [  -330 , -654.95, 354.44, 0, 180, 270]   
        step5      = [  -620 , -654.95, 354.44, 0, 180, 270]    #음식 뜨러 출발
        up         = [  -620 , -654.95, 454.44, 0, 180, 270]     #음식 뜨고 상승
        move2tray  = [  -620 ,  654.95, 454.44, 0, 180, 270]     #음식 들고 트레이 쪽으로 이동
    
        move2plate = [-235.470 + self.Yg0 - y_pos, 649.750 + self.Xg0 + x_pos, 215.270,      0,     180,   270]  #음식 놓을 위치로 이동
        put        = [-357.320 + self.Yg0 - y_pos, 649.750 + self.Xg0 + x_pos, 275.210, 173.16, -144.26, 88.36] # 음식 놓기
        step6      = [-300, 550.280, 402.960, 0, 180, 270]


        # ===== 동작 시퀀스 =====
        try:
            rospy.loginfo("--- 동작 시퀀스 시작 (실시간 속도 적용 모드) ---")
            
            # (디버깅용 로그) 현재 적용될 배율을 5초에 한 번씩 출력
            rospy.loginfo_throttle(5.0, f"현재 배율 {self.speed_multiplier:.2f} (다음 동작에 즉시 적용)")

            # 석쇠 뒤집는 용 ATC 체결 과정
            self.moveline(atc1,    self.vel, self.acc, self.get_scaled_time(2.0))
            time.sleep(0.5)
            self.moveline(atc2,   self.vel, self.acc, self.get_scaled_time(2.0))
            time.sleep(0.5)
            self.moveline(atc3,     self.vel, self.acc, self.get_scaled_time(2.0))
            time.sleep(0.5)
            
            # 석쇠 뒤집는 과정
            self.movejointx(ready1, 0, 0, self.get_scaled_time(4.0), 2)
            time.sleep(0.5)
            self.moveline(ready2,  self.vel, self.acc, self.get_scaled_time(3.0))
            time.sleep(0.5)
            self.moveline(grilling1,  self.vel, self.acc, self.get_scaled_time(3.0))
            time.sleep(0.5)
            self.moveline(grilling2,  self.vel, self.acc, self.get_scaled_time(2.0))
            time.sleep(0.5)
            self.moveline(grilling3,  self.vel, self.acc, self.get_scaled_time(2.0))
            time.sleep(0.5)
            self.moveline(grilling4,  self.vel, self.acc, self.get_scaled_time(2.0))
            time.sleep(0.5)
            self.moveline(grilling5,  self.vel, self.acc, self.get_scaled_time(3.0))
            time.sleep(0.5)
            self.moveline(ready1,  self.vel, self.acc, self.get_scaled_time(3.0))
            time.sleep(0.5)


            # 석쇠 뒤집는 용 ATC 분리 과정
            self.moveline(atc3,    self.vel, self.acc, self.get_scaled_time(3.0))
            time.sleep(0.5)
            self.moveline(atc2,   self.vel, self.acc, self.get_scaled_time(2.0))
            time.sleep(0.5)
            self.moveline(atc1,     self.vel, self.acc, self.get_scaled_time(2.0))
            time.sleep(0.5)
            self.moveline(move2ATC,  self.vel, self.acc, self.get_scaled_time(2.0))
            time.sleep(0.5)

            # 음식 뜨는 ATC 체결 과정
            
            self.moveline(move2ATC2,  self.vel, self.acc, self.get_scaled_time(6.0))
            time.sleep(0.5)
            self.moveline(atc4,    self.vel, self.acc, self.get_scaled_time(2.0))
            time.sleep(0.5)
            self.moveline(atc5,   self.vel, self.acc, self.get_scaled_time(2.0))
            time.sleep(0.5)
            self.moveline(atc6,     self.vel, self.acc, self.get_scaled_time(2.0))
            time.sleep(0.5)

            # 석쇠 여는 과정
            self.moveline(move2cook,    self.vel, self.acc, self.get_scaled_time(4.0))
            time.sleep(0.5)
            self.moveline(move2cook2,    self.vel, self.acc, self.get_scaled_time(4.0))
            time.sleep(0.5)
            self.moveline(step1,  self.vel, self.acc, self.get_scaled_time(2.0))
            time.sleep(0.5)
            self.movecircle(pos_list, self.vel, self.acc, 4)
            time.sleep(2)
            self.moveline(step2,    self.vel, self.acc, self.get_scaled_time(2.0))
            time.sleep(0.5)
            self.moveline(step3,   self.vel, self.acc, self.get_scaled_time(2.0))
            time.sleep(0.5)
            self.moveline(step4,     self.vel, self.acc, self.get_scaled_time(2.0))
            time.sleep(0.5)


            # 음식 뜨는 과정
            self.moveline(step5,    self.vel, self.acc, self.get_scaled_time(4.0))
            time.sleep(0.5)
            self.moveline(up,   self.vel, self.acc, self.get_scaled_time(2.0))
            time.sleep(0.5)
            self.moveline(move2tray,     self.vel, self.acc, self.get_scaled_time(8.0))
            time.sleep(0.5)

            # 음식 트레이 위로 올리는 과정
            self.movejointx(move2plate, 0, 0, self.get_scaled_time(3.0), 2)
            time.sleep(0.5)
            self.movejointx(put, 0, 0, self.get_scaled_time(2.0), 2)
            time.sleep(0.5)
            self.movejointx(step6, 0, 0, self.get_scaled_time(4.0), 2)
            time.sleep(0.5)
            
            # 음식 뜨는 ATC 분리 과정
            self.movejointx(atc6, 0, 0, self.get_scaled_time(6.0), 2)
            time.sleep(0.5)
            self.moveline(atc5,    self.vel, self.acc, self.get_scaled_time(2.0))
            time.sleep(0.5)
            self.moveline(atc4,   self.vel, self.acc, self.get_scaled_time(2.0))
            time.sleep(0.5)
            self.moveline(move2ATC2,     self.vel, self.acc, self.get_scaled_time(2.0))
            time.sleep(0.5)
            self.movejointx(pos_basic,  0, 0, self.get_scaled_time(6.0), 2)
            time.sleep(1)

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