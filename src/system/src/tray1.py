import xmlrpc.client
import rospy
import time
import math
import numpy as np
from nav_msgs.msg import Path
from dsr_msgs.srv import (
    MoveLine, MoveJointx, MoveJoint,
    MoveJointRequest, MoveJointxRequest, MoveLineRequest,
    GetCurrentPose, GetCurrentPoseRequest,
    SetCtrlBoxDigitalOutput, SetCtrlBoxDigitalOutputRequest
)
from std_msgs.msg import Float32MultiArray, Int32, Float32
from geometry_msgs.msg import PointStamped, PolygonStamped

class ArucoMotionController:
    def __init__(self):
        # 1. ROS 노드 초기화
        rospy.init_node('aruco_motion_controller', anonymous=True)
        rospy.loginfo("ArUco 기반 실시간 속도 조절 노드를 시작합니다.")

        # 2. ✅ [MERGED] 실시간 속도 조절 변수 초기화
        self.speed_multiplier = 1.0
        self.vel = [0.0, 0.0]  # DSR 시간 기반 제어를 위해 vel/acc는 0으로 고정
        self.acc = [0.0, 0.0]

        # 3. ArUco 및 경로 관련 변수 초기화 (전역 변수 -> 클래스 멤버로)
        self.Xg, self.Yg, self.Zg = 0, 0, 0
        self.ArUco_stamp = None
        self.received_angle = 0.0
        self.corners_tool = []
        self.received_path = []
        
        self.Xg0, self.Yg0, self.Zg0 = 0, 0, 0 # 잠금(Lock)될 ArUco 좌표
        self.ArUco_locked = False

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
        
        # ArUco 및 경로 관련 토픽 구독
        rospy.Subscriber("/aruco/center_tool", PointStamped, self.center_callback)
        rospy.Subscriber("/aruco/corners_tool", PolygonStamped, self.corners_callback)
        rospy.Subscriber("/cup/center_tool", PointStamped, self.center_callback) # 원본 코드와 동일하게 유지
        rospy.Subscriber("/aruco/yaw_deg", Float32, self.angle_callback)
        rospy.Subscriber("/path_move", Path, self.path_callback)
        rospy.loginfo("ArUco 및 경로 토픽 구독 중...")
        

    # --- [MERGED] 실시간 속도 조절 로직 ---

    def speed_callback(self, msg):
        """
        /speed_multiplier 토픽 수신 시 호출 (별도 스레드).
        수신된 배율을 self.speed_multiplier에 저장합니다.
        """
        received_value = msg.data
        
        # 0 또는 음수로 인한 ZeroDivisionError 방지
        if received_value <= 0.01:
            if self.speed_multiplier != 0.01:
                rospy.logwarn(f"수신된 배율({received_value:.2f})이 너무 낮습니다. 0.01 (100배 느림)로 강제 적용.")
            self.speed_multiplier = 0.01
        else:
            self.speed_multiplier = received_value
            # rospy.loginfo_throttle(5.0, f"새 배율 수신: {self.speed_multiplier:.2f}")

    def get_scaled_time(self, base_time):
        """
        현재 self.speed_multiplier를 기준으로 스케일링된 시간을 즉시 계산하여 반환합니다.
        (참고: 배율 2.0 -> 시간 1/2 (2배 빠름)로 로직 수정)
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

    def get_scaled_vel(self, base_vel):
        """
        base_vel: [linear_vel, angular_vel] 형식의 리스트/튜플
        self.speed_multiplier: 1, 2, 5 중 하나
        1 -> 1배
        2 -> 1/2배
        5 -> 1/5배
        """
        try:
            factor = 1.0 / float(self.speed_multiplier)   # 1 → 1.0 / 1 = 1.0
                                                        # 2 → 1.0 / 2 = 0.5
                                                        # 5 → 1.0 / 5 = 0.2
            return [v * factor for v in base_vel]

        except ZeroDivisionError:
            rospy.logerr_throttle(5, "speed_multiplier가 0입니다. 배율 1.0으로 복구.")
            return list(base_vel)
        except Exception as e:
            rospy.logerr_throttle(5, f"속도 스케일링 오류: {e}. 기본 값({base_vel}) 사용.")
            return list(base_vel)


    # --- ArUco 및 경로 콜백 함수들 ---

    def center_callback(self, msg):
        # 전역 변수 대신 self 사용
        self.Xg = msg.point.x*1000
        self.Yg = msg.point.y*1000
        self.Zg = msg.point.z*1000
        self.ArUco_stamp = msg.header.stamp
        # rospy.loginfo(f"Received ArUco center: Xg={self.Xg}, Yg={self.Yg}, Zg={self.Zg}")

    def angle_callback(self, msg):
        self.received_angle = msg.data
        # rospy.loginfo(f"Received rotation angle: {self.received_angle} degrees")

    def corners_callback(self, msg):
        self.corners_tool = [(point.x, point.y, point.z) for point in msg.polygon.points]
        # rospy.loginfo(f"Received corners in tool frame: {self.corners_tool}")

    def path_callback(self, msg):
        self.received_path = []
        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            z = pose_stamped.pose.position.z
            self.received_path.append((x, y, z))
        rospy.loginfo(f"[Path Callback] {len(self.received_path)} points received")

    # --- DSR 모션 서비스 호출 함수 ---
    # (효율을 위해 __init__에서 생성된 프록시를 사용하도록 수정)

    def movejoint(self, posx, vel_, acc_, time_):
        try:
            req = MoveJointRequest()
            req.pos = posx; req.vel = vel_; req.acc = acc_; req.time = time_
            req.radius = 0; req.mode = 0; req.blendType = 0; req.syncType = 0
            self.move_joint_srv(req) # ✅ self.move_joint_srv 사용
        except rospy.ServiceException as e:
            rospy.logerr_throttle(5, f"move_joint 서비스 호출 실패: {e}")

    def moveline(self, posx, vel_, acc_, time_):
        try:
            req = MoveLineRequest()
            req.pos = posx; req.vel = vel_; req.acc = acc_; req.time = time_
            req.radius = 0; req.mode = 0; req.blendType = 0; req.syncType = 0
            self.move_line_srv(req) # ✅ self.move_line_srv 사용
        except rospy.ServiceException as e:
            rospy.logerr_throttle(5, f"moveline 서비스 호출 실패: {e}")

    def movejointx(self, posx, vel_, acc_, time_, sol):
        try:
            req = MoveJointxRequest()
            req.pos = posx; req.vel = vel_; req.acc = acc_; req.time = time_
            req.sol = sol
            req.radius = 0; req.mode = 0; req.blendType = 0; req.syncType = 0
            self.move_jointx_srv(req) # ✅ self.move_jointx_srv 사용
        except rospy.ServiceException as e:
            rospy.logerr(f"move_jointx 서비스 호출 실패: {e}")

    # --- 경로 실행 헬퍼 함수 ---

    def execute_path(self):
        base_vel = [200.0, 30.0]
        acc      = [300.0, 60.0]

        # ✅ [MERGED] 속도 로직 적용
        if not self.received_path:
            rospy.logwarn("No path data received yet.")
            return
        
        rospy.loginfo(f"[Executor] Executing {len(self.received_path)} waypoints..")

        for (x, y, z) in self.received_path:
            posx = [x, y, z, 0, 180, 180]
            # ✅ [MERGED] 고정 시간 2초 대신 get_scaled_time 사용
            self.moveline(posx, self.get_scaled_vel(base_vel), acc, 0)
            # ✅ [MERGED] 고정 sleep 0.5초 대신 get_scaled_time 사용
            time.sleep(0.5)

    def execute_path2(self):
        base_vel = [200.0, 30.0]
        acc      = [300.0, 60.0]

        # ✅ [MERGED] 속도 로직 적용
        if not self.received_path:
            rospy.logwarn("No path data received yet.")
            return
        
        rospy.loginfo(f"[Executor] Executing {len(self.received_path)} waypoints in reverse order..")

        for (x, y, z) in reversed(self.received_path):
            posx = [x, y, z, 0, 180, 180]
            # ✅ [MERGED] 고정 시간 2초 대신 get_scaled_time 사용
            self.moveline(posx, self.get_scaled_vel(base_vel), acc, 0)
            # ✅ [MERGED] 고정 sleep 0.5초 대신 get_scaled_time 사용
            time.sleep(0.5)

    # --- ArUco 락 헬퍼 함수 ---

    def wait_and_lock_ArUco_center(self, timeout=5.0, stable_samples=5, tol_mm=6.0, freshness=2):
        # 전역 변수 대신 self 사용
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

    # --- 메인 실행 로직 (main() -> 클래스 메서드로) ---

    def run_main_logic(self):
        rospy.sleep(1) # 구독자 연결 대기
        rate = rospy.Rate(10)

        # 기본 위치 설정
        search = [-206, 660, 500, 0, 180, 90]
        
        # ✅ [MERGED] 고정 시간 5초 대신 get_scaled_time 사용
        self.movejointx(search, 0, 0, self.get_scaled_time(5.0), 2)
        time.sleep(1.0)

        # 2) 트레이 중심 락
        ok = self.wait_and_lock_ArUco_center(timeout=5.0, stable_samples=5, tol_mm=3.0, freshness=0.3)
        if not ok:
            rospy.logerr("Could not capture ArUco center. Abort.")
            return
        
        rospy.loginfo(f"Using locked ArUco center: Xg0={self.Xg0:.1f}, Yg0={self.Yg0:.1f}, Zg0={self.Zg0:.1f} (mm)")

        # 잠금된 좌표(self.Xg0) 기준으로 목표 위치 계산
        pos       = [-206 + self.Yg0 + 4, 660 + self.Xg0 + 1, 500, 0, 180,  90]
        pos2      = [-206 + self.Yg0 + 4, 660 + self.Xg0 + 1, 500, 0, 180, 180]
        pos_basic = [               -206,                660, 500, 0, 180,  90]

        
        # 메인 루프 (원본 코드는 break로 한 번만 실행됨)
        while not rospy.is_shutdown():
            rate.sleep()

            # (디버깅용 로그) 현재 적용될 배율을 5초에 한 번씩 출력
            rospy.loginfo_throttle(5.0, f"현재 배율 {self.speed_multiplier:.2f} (다음 동작에 즉시 적용)")

            angle_with_offset = 87 + self.received_angle

            # pos_with_rotation을 회전 각도를 반영하여 설정
            pos_with_rotation = [-206 + self.Yg0 + 4, 660 + self.Xg0 + 1, 500,   0, 180, angle_with_offset]
            pos_down          = [-206 + self.Yg0 + 4, 660 + self.Xg0 + 1, 271,   0, 180, angle_with_offset]
            pos_up            = [-206 + self.Yg0 + 4, 660 + self.Xg0 + 1, 500,   0, 180, angle_with_offset]
            finish            = [               1000,                350, 273,   0, 180,               270]
            finish_up         = [               1000,                350, 500,   0, 180,               270]

            # --- 모든 모션에 get_scaled_time() 적용 ---
            
            # 트레이 그립 과정
            self.movejointx(pos, 0, 0, self.get_scaled_time(3.0), 2)
            time.sleep(0.5)           
            self.proxy.twofg_grip_external(0, 70.0, 20, 10)
            time.sleep(0.5)
            self.movejointx(pos_with_rotation, 0, 0, self.get_scaled_time(3.0), 2)
            time.sleep(0.5)
            self.moveline(pos_down, self.vel, self.acc, self.get_scaled_time(2.0)) 
            time.sleep(0.5)
            self.proxy.twofg_grip_external(0, 16.0, 20, 10)
            time.sleep(1.5)
            self.movejointx(pos_up, 0, 0, self.get_scaled_time(3.0), 2) 
            time.sleep(0.5)
            self.movejointx(pos2, 0, 0, self.get_scaled_time(3.0), 2)
            time.sleep(0.5)
            
            # 트레이 이송
            self.execute_path()
            
            # 트레이 이송 후 고객에게 전달
            self.moveline(finish_up, self.vel, self.acc, self.get_scaled_time(2.0)) 
            time.sleep(0.5)
            self.moveline(finish, self.vel, self.acc, self.get_scaled_time(2.0)) 
            self.proxy.twofg_grip_external(0, 70.0, 20, 10)
            time.sleep(0.5)
            self.moveline(finish_up, self.vel, self.acc, self.get_scaled_time(2.0)) 
            time.sleep(0.5)

            # 고객에게 전달 후 복귀
            self.execute_path2()
            time.sleep(0.5)
            self.movejointx(pos_basic, 0, 0, self.get_scaled_time(4.0), 2)
            time.sleep(0.5)
            
            break

if __name__ == '__main__':
    try:
        # 1. 컨트롤러 객체 생성 (이때 __init__이 실행됨)
        controller = ArucoMotionController()
        
        # 2. 메인 로직 실행
        controller.run_main_logic()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS 노드가 중단되었습니다.")
    except Exception as e:
        rospy.logerr(f"예기치 않은 오류 발생: {e}")