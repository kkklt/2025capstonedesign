#!/usr/bin/env python3

import time
import math
import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs

# === ROS ===
import rospy
from std_msgs.msg import Int32, Float32, Header
from geometry_msgs.msg import PointStamped, PolygonStamped, Point32, Point
from dsr_msgs.srv import GetCurrentPose, GetCurrentPoseRequest

# ========================
# 🔧 사용자 설정 (상수)
# ========================
TEXT_COLOR = (255, 255, 255)
TEXT_BG = True
TEXT_SCALE = 0.55
TEXT_THICK = 1
DEPTH_ROI_RADIUS = 3     # 깊이 추정용 ROI 반경(픽셀)
OUT_UNIT_MM = True       # 좌표를 mm로 퍼블리시/표시
SCALE_OUT = 1000.0 if OUT_UNIT_MM else 1.0
UNIT_LABEL = "mm" if OUT_UNIT_MM else "m"

# ========================
# 🔁 카메라→그리퍼 변환 (캘리브레이션 결과)
#   - R_c2g, t_c2g는 '미터(m)' 단위로 유지
# ========================
R_c2g = np.array([
    [0.9998396867445545, -0.007328346659852211, -0.01633695644188792],
    [0.007276998328315886, 0.9999684011488463, -0.003200312351867875],
    [0.0163598932111673, 0.003080915294658917, 0.9998614213255086]
], dtype=np.float64)
t_c2g_mm = np.array([
    [-25.46297744346341],
    [-121.2978522193079],
    [5.415990194692839]
], dtype=np.float64)
t_c2g = t_c2g_mm / 1000.0  # (3,1) meters


def euler_deg_to_R(rx_deg, ry_deg, rz_deg, order='zyx'):
    """
    rx,ry,rz(deg) → 회전행렬
    order: 'zyx' = Rz @ Ry @ Rx (기본값), 'xyz' = Rx @ Ry @ Rz
    ※ 컨트롤러의 오일러 정의/순서와 반드시 맞추세요.
    """
    rx, ry, rz = map(math.radians, (rx_deg, ry_deg, rz_deg))
    Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
    Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
    Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
    if order == 'zyx':
        return Rz @ Ry @ Rx
    elif order == 'xyz':
        return Rx @ Ry @ Rz
    else:
        raise ValueError("order must be 'zyx' or 'xyz'")

def draw_label(img, text, x, y,
               text_color=TEXT_COLOR, bg=TEXT_BG,
               scale=TEXT_SCALE, thick=TEXT_THICK):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    if bg:
        cv2.rectangle(img, (x, y - th - baseline), (x + tw, y + baseline),
                      (0, 0, 0), thickness=-1)
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thick, cv2.LINE_AA)

class ArucoDetector:
    """
    RealSense 카메라와 ArUco 마커를 사용하여 마커의 3D 위치를 탐지하고,
    로봇의 툴(tool) 및 베이스(base) 좌표계로 변환하여 ROS 토픽으로 발행하는 클래스.
    """
    def __init__(self):
        """ArucoDetector 클래스의 생성자. ROS 노드, 리소스, 설정을 초기화합니다."""
        rospy.init_node("aruco_pub", anonymous=True)

        # === 좌표 변환 행렬 초기화 ===
        # 카메라(c) -> 그리퍼(g, tool) 변환 행렬 (사전 캘리브레이션 값)
        self.R_c2g = R_c2g
        self.t_c2g = t_c2g
        # 그리퍼(g, tool) -> 로봇 베이스(b) 변환 행렬 (로봇에서 동적으로 수신)
        self.R_g2b = np.eye(3, dtype=np.float64)
        self.t_g2b = np.zeros((3,1), dtype=np.float64)
        
        # 깊이 값 유효 범위 (미터 단위)
        self.DEPTH_VALID_MIN = 0.10
        self.DEPTH_VALID_MAX = 6.0

        # --- 초기화 메서드 호출 ---
        self._setup_ros()
        self._setup_realsense()
        self._setup_aruco()
        
        # --- OpenCV 시각화 창 설정 ---
        self.WIN_NAME = "ArUco + Depth (ROS publish, mm)"
        cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN_NAME, 960, 720)

    def _setup_ros(self):
        """ROS 퍼블리셔와 서비스 프록시를 설정합니다."""
        # --- 마커 정보 퍼블리셔 ---
        self.pub_id = rospy.Publisher("/aruco/id", Int32, queue_size=10)
        self.pub_center_tool = rospy.Publisher("/aruco/center_tool", PointStamped, queue_size=10)
        self.pub_corners_tool = rospy.Publisher("/aruco/corners_tool", PolygonStamped, queue_size=10)
        self.pub_center_base = rospy.Publisher("/aruco/center_base", PointStamped, queue_size=10)
        self.pub_corners_base = rospy.Publisher("/aruco/corners_base", PolygonStamped, queue_size=10)
        self.pub_angle = rospy.Publisher("/aruco/rotation_angle", Float32, queue_size=10)
        
        # --- 두산 로봇 현재 TCP 포즈 서비스 클라이언트 ---
        try:
            rospy.wait_for_service('/dsr01m1013/system/get_current_pose', timeout=2.0)
            self._get_pose_proxy = rospy.ServiceProxy('/dsr01m1013/system/get_current_pose', GetCurrentPose)
        except (rospy.ServiceException, rospy.ROSException, rospy.exceptions.ROSInitException) as e:
            rospy.logerr(f"Failed to connect to DSR service: {e}")
            self._get_pose_proxy = None

    def _setup_realsense(self):
        """RealSense 카메라 파이프라인과 설정을 초기화합니다."""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        profile = self.pipeline.start(config)
        dev = profile.get_device()
        name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else "Unknown"

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale() # z16 depth map을 미터 단위로 변환하는 스케일 값
        
        rospy.loginfo(f"[INFO] Device: {name}")
        rospy.loginfo(f"[INFO] Depth Scale (z16→m): {self.depth_scale}")
        
        # D435 모델의 경우 최소 인식 거리가 더 길어서 파라미터 조정
        if "D435" in name:
            self.DEPTH_VALID_MIN = 0.12
        rospy.loginfo(f"[INFO] Depth valid range set to: {self.DEPTH_VALID_MIN:.2f} m ~ {self.DEPTH_VALID_MAX:.1f} m")

        # 컬러와 깊이 프레임의 좌표계를 일치시키는 정렬(align) 객체 생성
        self.align = rs.align(rs.stream.color)
        
    def _setup_aruco(self):
        """ArUco 마커 탐지기(detector)를 설정합니다."""
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        try: # OpenCV 버전에 따라 파라미터 생성 함수 이름이 다름
            parameters = aruco.DetectorParameters()
        except TypeError:
            parameters = aruco.DetectorParameters_create()
        
        # --- 파라미터 미세 조정 ---
        # AprilTag의 코너 검출 방식을 사용하여 정확도 향상 (노이즈/경계에 더 강인함)
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
        # 다각형 근사 정확도를 높여 더 정확한 사각형 검출 유도 (기본값 0.03 -> 0.04로 약간 완화)
        parameters.polygonalApproxAccuracyRate = 0.04
        # adaptiveThresholding의 상수 값을 조정하여 조명 변화에 좀 더 강인하게 만듭니다 (기본값 7)
        parameters.adaptiveThreshConstant = 10

        try: # 최신 OpenCV 버전에서는 ArucoDetector 클래스를 사용
            self._detector = aruco.ArucoDetector(self.aruco_dict, parameters)
        except AttributeError: # 구버전 호환용
            self._detector = None
            self.parameters = parameters # detectMarkers에 전달하기 위해 저장
    
    def detect_markers(self, img_bgr):
        """입력 이미지에서 ArUco 마커를 탐지합니다."""
        if self._detector is not None:
            # 최신 OpenCV 방식
            return self._detector.detectMarkers(img_bgr)
        else:
            # 구버전 OpenCV 방식
            return aruco.detectMarkers(img_bgr, self.aruco_dict, parameters=self.parameters)

    def calculate_rotation_angle(self, corner_pixels):
        """마커의 코너 픽셀 좌표를 이용해 2D 이미지 평면에서의 회전 각도를 계산합니다."""
        points = corner_pixels.reshape((4, 2))
        # ArUco 코너 순서: 0: 좌상단, 1: 우상단, 2: 우하단, 3: 좌하단
        point1 = points[1]  # 우상단(Top-Right)
        point2 = points[2]  # 우하단(Bottom-Right)
        
        # 두 점 사이의 벡터를 이용해 각도 계산 (x축 양의 방향 기준)
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        angle = math.degrees(math.atan2(dy, dx))
        return angle

    def publish_rotation_angle(self, marker_id, angle):
        """계산된 회전 각도를 ROS 토픽으로 발행합니다."""
        self.pub_angle.publish(Float32(angle))
        # 디버깅 시 아래 주석을 해제하여 발행되는 각도 값을 확인할 수 있습니다.
        # rospy.loginfo(f"Published rotation angle for marker {marker_id}: {angle:.2f} degrees")

    def update_tool_to_base_from_robot(self, order='zyx'):
        """
        두산 로봇의 'get_current_pose' 서비스를 호출하여,
        현재 로봇의 TCP(툴) 좌표를 기준으로 '툴→베이스' 변환 행렬을 갱신합니다.
        """
        if self._get_pose_proxy is None:
            rospy.logwarn_throttle(5.0, "DSR service proxy not available.")
            return

        # 서비스 요청: space_type=1은 TCP 기준 좌표(task space)를 의미
        req = GetCurrentPoseRequest()
        req.space_type = 1
        resp = self._get_pose_proxy(req)

        pos = resp.pos
        if len(pos) < 6:
            raise RuntimeError(f"get_current_pose returned invalid pos: {pos}")

        # 로봇 컨트롤러에서 받은 TCP 포즈 (x,y,z는 mm, rx,ry,rz는 degree)
        x_mm, y_mm, z_mm, rx, ry, rz = pos
        # 베이스 -> 툴(그리퍼) 변환
        R_b2g = euler_deg_to_R(rx, ry, rz, order=order)
        t_b2g = np.array([[x_mm],[y_mm],[z_mm]], dtype=np.float64) / 1000.0  # mm to meters

        # 역변환을 통해 툴(그리퍼) -> 베이스 변환 행렬 계산
        self.R_g2b = R_b2g.T
        self.t_g2b = -self.R_g2b @ t_b2g

    def cam_to_gripper(self, p_c_m: np.ndarray) -> np.ndarray:
        """3D 점을 카메라 좌표계에서 그리퍼(툴) 좌표계로 변환합니다."""
        return (self.R_c2g @ p_c_m) + self.t_c2g

    def gripper_to_base(self, p_g_m: np.ndarray) -> np.ndarray:
        """3D 점을 그리퍼(툴) 좌표계에서 로봇 베이스 좌표계로 변환합니다."""
        return (self.R_g2b @ p_g_m) + self.t_g2b

    def get_distance_median(self, depth_frame, cx, cy, r):
        """
        주어진 픽셀 (cx, cy) 주변의 반경(r) 내 ROI(관심 영역)에서 
        깊이 값들의 중간값(median)을 계산하여 노이즈에 강한 깊이 추정을 합니다.
        """
        h, w = depth_frame.get_height(), depth_frame.get_width()
        xs = range(max(cx - r, 0), min(cx + r, w - 1) + 1)
        ys = range(max(cy - r, 0), min(cy + r, h - 1) + 1)
        vals = [depth_frame.get_distance(x, y) for y in ys for x in xs]
        # 유효한 깊이 값만 필터링
        valid_vals = [d for d in vals if self.DEPTH_VALID_MIN < d < self.DEPTH_VALID_MAX and np.isfinite(d)]
        return float(np.median(valid_vals)) if valid_vals else None

    def publish_center_and_corners(self, marker_id: int,
                                   center_g_m: np.ndarray, corners_g_m_list: list,
                                   center_b_m: np.ndarray, corners_b_m_list: list):
        """마커의 ID, 중심점, 코너 좌표를 툴/베이스 좌표계로 ROS 토픽에 발행합니다."""
        header = Header()
        header.stamp = rospy.Time.now()
        self.pub_id.publish(Int32(marker_id))

        # --- 툴(Tool) 좌표계 기준 데이터 발행 (단위: mm) ---
        cx, cy, cz = (center_g_m.ravel() * SCALE_OUT).tolist()
        cy = -cy # y 좌표 부호 반전 (필요에 따라 조정)
        header.frame_id = "tool"
        msg_c = PointStamped(header=header, point=Point(x=cx, y=cy, z=cz))
        self.pub_center_tool.publish(msg_c)
        
        msg_poly = PolygonStamped(header=header)
        for p in corners_g_m_list:
            x, y, z = (p.ravel() * SCALE_OUT).tolist()
            y = -y
            msg_poly.polygon.points.append(Point32(x=x, y=y, z=z))
        self.pub_corners_tool.publish(msg_poly)

        # --- 베이스(Base) 좌표계 기준 데이터 발행 (단위: mm) ---
        if center_b_m is not None and corners_b_m_list is not None:
            header.frame_id = "base"
            cbx, cby, cbz = (center_b_m.ravel() * SCALE_OUT).tolist()
            cby = -cby
            msg_cb = PointStamped(header=header, point=Point(x=cbx, y=cby, z=cbz))
            self.pub_center_base.publish(msg_cb)
            
            msg_poly_b = PolygonStamped(header=header)
            for p in corners_b_m_list:
                x, y, z = (p.ravel() * SCALE_OUT).tolist()
                y = -y
                msg_poly_b.polygon.points.append(Point32(x=x, y=y, z=z))
            self.pub_corners_base.publish(msg_poly_b)

    def run(self):
        """메인 루프. 카메라로부터 프레임을 받아 마커 탐지 및 정보 발행을 반복합니다."""
        # FPS 계산용 변수
        fps_t0 = time.time()
        fps_cnt = 0
        
        # 로봇 포즈 갱신 주기 제어용 변수
        last_pose_update_t = 0.0
        POSE_UPDATE_PERIOD = 0.02  # 50 Hz, 로봇 서비스가 느리면 0.05 등으로 조정

        while not rospy.is_shutdown():
            # (1) 로봇 TCP 포즈 갱신 (주기적으로)
            now_ros = rospy.Time.now().to_sec()
            if now_ros - last_pose_update_t >= POSE_UPDATE_PERIOD:
                try:
                    self.update_tool_to_base_from_robot(order='zyx')
                except Exception as e:
                    rospy.logwarn_throttle(1.0, f"update_tool_to_base failed: {e}")
                last_pose_update_t = now_ros

            # (2) RealSense 카메라 프레임 획득 및 정렬
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            vis = color_image.copy() # 시각화용 이미지 복사

            # (3) ArUco 마커 탐지
            corners, ids, _ = self.detect_markers(color_image)
            
            if ids is not None: # 마커가 하나 이상 탐지된 경우
                aruco.drawDetectedMarkers(vis, corners, ids)
                # 디버깅: rospy.logdebug(f"Detected markers: {ids.ravel()}")
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

                for i, corner in enumerate(corners):
                    marker_id = int(ids[i][0])
                    pts = corner.reshape((4, 2)).astype(int)

                    # 각 코너에 번호 표시
                    for j, point in enumerate(pts):
                        px, py = point[0], point[1]
                        draw_label(vis, str(j), px, py - 10, text_color=(0, 255, 0), bg=True, scale=0.5)

                    # 마커 중심점의 2D 픽셀 좌표 및 깊이 값 계산
                    cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                    d_center = self.get_distance_median(depth_frame, cx, cy, r=DEPTH_ROI_RADIUS)
                    # 디버깅: rospy.logdebug(f"Center depth: {d_center}")

                    # (4) 3D 좌표 변환 (카메라 → 툴 → 베이스)
                    center_g_m, center_b_m = None, None
                    if d_center is not None:
                        try:
                            # 2D 픽셀 좌표와 깊이값을 이용해 3D 카메라 좌표(p_c_m) 계산
                            p_c_m = np.array(rs.rs2_deproject_pixel_to_point(
                                depth_intrin, [float(cx), float(cy)], float(d_center)
                            ), dtype=np.float64).reshape(3,1)
                            
                            # 카메라 -> 툴 -> 베이스 순서로 좌표 변환
                            center_g_m = self.cam_to_gripper(p_c_m)
                            center_b_m = self.gripper_to_base(center_g_m)

                            # 디버깅용 출력 (콘솔)
                            Xg, Yg, Zg = (center_g_m.ravel() * SCALE_OUT).tolist()
                            rospy.loginfo(f"[TOOL][ID {marker_id}] Center ≈ ({Xg:.1f}, {-Yg:.1f}, {Zg:.1f}) {UNIT_LABEL}")

                            # 마커 회전 각도 계산 및 발행
                            rotation_angle = self.calculate_rotation_angle(corner)
                            rospy.loginfo(f"[TOOL][ID {marker_id}] Rotation Angle: {rotation_angle:.2f} degrees")
                            self.publish_rotation_angle(marker_id, rotation_angle)

                        except Exception as e:
                            rospy.logerr(f"[ERR] center transform: {type(e).__name__}: {e}")

                    # 마커의 4개 코너에 대해서도 동일하게 3D 좌표 변환 수행
                    corners_g_m, corners_b_m = [], []
                    all_corners_ok = True
                    for j, (px, py) in enumerate(pts):
                        d_corner = self.get_distance_median(depth_frame, int(px), int(py), r=DEPTH_ROI_RADIUS)
                        # 코너 깊이 측정이 실패하면 중심 깊이 값으로 대체
                        if d_corner is None: d_corner = d_center
                        if d_corner is None:
                            rospy.logwarn(f"[TOOL][ID {marker_id}] V{j}: depth N/A → skip")
                            all_corners_ok = False
                            break
                        
                        try:
                            p_c_m = np.array(rs.rs2_deproject_pixel_to_point(
                                depth_intrin, [float(px), float(py)], float(d_corner)
                            ), dtype=np.float64).reshape(3,1)

                            p_g_m = self.cam_to_gripper(p_c_m)
                            corners_g_m.append(p_g_m)
                            corners_b_m.append(self.gripper_to_base(p_g_m))
                            
                            # 디버깅용 출력 (콘솔)
                            Xg, Yg, Zg = (p_g_m.ravel() * SCALE_OUT).tolist()
                            rospy.loginfo(f"[TOOL][ID {marker_id}] Corner {j} ≈ ({Xg:.1f}, {-Yg:.1f}, {Zg:.1f}) {UNIT_LABEL}")

                        except Exception as e:
                            rospy.logerr(f"[ERR] corner V{j} transform: {type(e).__name__}: {e}")
                            all_corners_ok = False
                            break
                    
                    # (5) ROS 토픽 발행
                    if center_g_m is not None and all_corners_ok and len(corners_g_m) == 4:
                        self.publish_center_and_corners(
                            marker_id, center_g_m, corners_g_m, center_b_m, corners_b_m
                        )

            # ----- FPS 계산 및 표시 -----
            fps_cnt += 1
            now = time.time()
            if now - fps_t0 >= 1.0: # 1초마다 FPS 갱신
                fps = fps_cnt / (now - fps_t0)
                fps_cnt = 0
                fps_t0 = now
                draw_label(vis, f"FPS: {fps:.1f}", 10, 25, bg=True)

            # ----- 결과 영상 출력 -----
            cv2.imshow(self.WIN_NAME, vis)
            if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' 키를 누르면 종료
                break

    def shutdown(self):
        """노드 종료 시 호출될 클린업 함수. 사용된 리소스를 해제합니다."""
        self.pipeline.stop()
        cv2.destroyAllWindows()
        rospy.loginfo("Aruco detector node shut down.")

if __name__ == '__main__':
    """
    스크립트의 메인 실행 블록.
    ArucoDetector 객체를 생성하고, 예외 처리를 포함하여 실행합니다.
    """
    try:
        detector = ArucoDetector()
        detector.run()
    except KeyboardInterrupt:
        # 사용자가 Ctrl+C로 종료한 경우
        pass
    except Exception as e:
        rospy.logfatal(f"Unhandled exception: {e}")
    finally:
        # 프로그램 종료 시 항상 shutdown 메서드를 호출하여 리소스를 정리
        if 'detector' in locals() and detector:
            detector.shutdown()