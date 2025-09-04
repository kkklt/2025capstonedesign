import time
import math
import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs

# === ROS ===
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PointStamped, PolygonStamped, Point32
from dsr_msgs.srv import GetCurrentPose, GetCurrentPoseRequest
from std_msgs.msg import Float32

# ========================
# 🔧 사용자 설정
# ========================
TEXT_COLOR = (255, 255, 255)
TEXT_BG = True
TEXT_SCALE = 0.55
TEXT_THICK = 1

# 깊이 사용 범위(기종에 따라 start 후 동적으로 조정)
DEPTH_VALID_MIN = 0.10   # m (D435i면 코드에서 0.12로 올려줌)
DEPTH_VALID_MAX = 6.0    # m
DEPTH_ROI_RADIUS = 3     # 깊이 추정용 ROI 반경(픽셀)

SHOW_CORNER_TOOL_LABELS = False

# ===== 출력 단위 설정 =====
OUT_UNIT_MM = True                # 좌표를 mm로 퍼블리시/표시
SCALE_OUT   = 1000.0 if OUT_UNIT_MM else 1.0
UNIT_LABEL  = "mm" if OUT_UNIT_MM else "m"

# ========================
# 🔁 카메라→그리퍼 변환 (캘리브레이션 결과 반영)
#   - R_c2g, t_c2g는 '미터(m)' 단위로 유지하세요
# ========================
R_c2g = np.array([
    [0.9998396867445545, -0.007328346659852211, -0.01633695644188792],
    [0.007276998328315886, 0.9999684011488463, -0.003200312351867875],
    [0.0163598932111673,   0.003080915294658917, 0.9998614213255086]
], dtype=np.float64)

t_c2g_mm = np.array([
    [-25.46297744346341],
    [-121.2978522193079],
    [  5.415990194692839]
], dtype=np.float64)

t_c2g = t_c2g_mm / 1000.0  # (3,1) meters

def cam_to_gripper(p_c_m: np.ndarray) -> np.ndarray:
    """카메라 좌표계 3D점(m)을 그리퍼(툴) 좌표계로 변환"""
    return (R_c2g @ p_c_m) + t_c2g  # (3,1) m

# ========================
# 🧭 툴→베이스 변환 (Doosan에서 TCP 포즈로 동적 갱신)
# ========================
# 기본값: 항등행렬/제로 (초기화용; 아래 update 함수로 매 프레임 갱신)
R_g2b = np.eye(3, dtype=np.float64)
t_g2b = np.zeros((3,1), dtype=np.float64)

def euler_deg_to_R(rx_deg, ry_deg, rz_deg, order='zyx'):
    """
    rx,ry,rz(deg) → 회전행렬
    order:
      'zyx' = Rz @ Ry @ Rx (yaw→pitch→roll)  ← 기본값
      'xyz' = Rx @ Ry @ Rz (roll→pitch→yaw)
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

# 퍼블리셔 추가
pub_angle = rospy.Publisher("/aruco/rotation_angle", Float32, queue_size=10)

# 회전 각도를 퍼블리시하는 부분
def publish_rotation_angle(marker_id, angle):
    """회전 각도를 퍼블리시"""
    msg = Float32()
    msg.data = angle
    pub_angle.publish(msg)
    rospy.loginfo(f"Published rotation angle for marker {marker_id}: {angle:.2f} degrees")

# 기존 코드에서 'calculate_rotation_angle' 함수 추가
def calculate_rotation_angle(corners):
    # 코너 좌표 4개를 사용하여 회전 각도를 계산
    points = np.array(corners, dtype=np.float32)
    
    point1 = corners_g_m[0]  # 첫 번째 코너 좌표
    point2 = corners_g_m[3]  # 두 번째 코너 좌표

    # 각도를 계산하기 위해 x, y 좌표를 추출
    dx = point2[0] - point1[0]  # x 방향 차이
    dy = point2[1] - point1[1]  # y 방향 차이
    
    # 아크탄젠트를 사용하여 각도 계산
    angle = math.degrees(math.atan2(dy, dx))

    return angle


# Doosan 현재 포즈 서비스(proxy는 아래에서 초기화)
_get_pose_proxy = None

def update_tool_to_base_from_robot(order='zyx'):
    """
    /dsr01m1013/system/get_current_pose (space_type=1) 호출로
    base→tool 포즈를 받아, tool→base 변환(R_g2b, t_g2b)을 갱신.
    - 위치 단위: mm → 내부 m로 변환
    - 각도 단위: deg
    """
    global _get_pose_proxy, R_g2b, t_g2b
    if _get_pose_proxy is None:
        rospy.wait_for_service('/dsr01m1013/system/get_current_pose')
        _get_pose_proxy = rospy.ServiceProxy('/dsr01m1013/system/get_current_pose', GetCurrentPose, persistent=True)

    req = GetCurrentPoseRequest()
    req.space_type = 1  # 0: joint, 1: task(space)
    resp = _get_pose_proxy(req)

    pos = resp.pos  # [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg] (일반적으로)
    if len(pos) < 6:
        raise RuntimeError(f"get_current_pose returned invalid pos: {pos}")

    x_mm, y_mm, z_mm, rx, ry, rz = pos
    R_b2g = euler_deg_to_R(rx, ry, rz, order=order)
    t_b2g = np.array([[x_mm],[y_mm],[z_mm]], dtype=np.float64) / 1000.0  # mm → m

    # inverse(base->tool) → tool->base
    R_g2b = R_b2g.T
    t_g2b = - R_g2b @ t_b2g

def gripper_to_base(p_g_m: np.ndarray) -> np.ndarray:
    """그리퍼 좌표계(m) → 베이스 좌표계(m) (동적으로 갱신된 R_g2b, t_g2b 사용)"""
    return (R_g2b @ p_g_m) + t_g2b  # (3,1) m

# ========================
# 유틸 함수
# ========================
def draw_label(img, text, x, y,
               text_color=TEXT_COLOR, bg=TEXT_BG,
               scale=TEXT_SCALE, thick=TEXT_THICK):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    if bg:
        cv2.rectangle(img, (x, y - th - baseline), (x + tw, y + baseline),
                      (0, 0, 0), thickness=-1)
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thick, cv2.LINE_AA)

def get_distance_median(depth_frame, cx, cy, r, dmin, dmax):
    """RealSense API(get_distance)로 ROI 메디안 깊이(m) 계산"""
    h = depth_frame.get_height()
    w = depth_frame.get_width()
    xs = range(max(cx - r, 0), min(cx + r, w - 1) + 1)
    ys = range(max(cy - r, 0), min(cy + r, h - 1) + 1)
    vals = []
    for y in ys:
        for x in xs:
            d = depth_frame.get_distance(x, y)
            if dmin < d < dmax and np.isfinite(d):
                vals.append(d)
    if len(vals) == 0:
        return None
    return float(np.median(vals))

# ========================
# ROS 노드 & 퍼블리셔
# ========================
rospy.init_node("aruco_pub", anonymous=True)
pub_id = rospy.Publisher("/aruco/id", Int32, queue_size=10)
pub_center_tool = rospy.Publisher("/aruco/center_tool", PointStamped, queue_size=10)
pub_corners_tool = rospy.Publisher("/aruco/corners_tool", PolygonStamped, queue_size=10)
pub_center_base = rospy.Publisher("/aruco/center_base", PointStamped, queue_size=10)
pub_corners_base = rospy.Publisher("/aruco/corners_base", PolygonStamped, queue_size=10)

# ========================
# RealSense 파이프라인
# ========================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

# ----- 장치/센서 정보 출력 & 기종별 파라미터 보정 -----
dev = profile.get_device()
name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else "Unknown"
serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else "N/A"
fw = dev.get_info(rs.camera_info.firmware_version) if dev.supports(rs.camera_info.firmware_version) else "N/A"

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

print(f"[INFO] Device: {name} (S/N: {serial}) | FW: {fw}")
print(f"[INFO] Depth Scale (z16→m): {depth_scale}")

if "D435" in name:
    DEPTH_VALID_MIN = 0.12
    DEPTH_VALID_MAX = 6.0
else:
    DEPTH_VALID_MIN = max(DEPTH_VALID_MIN, 0.10)
    DEPTH_VALID_MAX = max(DEPTH_VALID_MAX, 6.0)
print(f"[INFO] Depth valid range set to: {DEPTH_VALID_MIN:.2f} m ~ {DEPTH_VALID_MAX:.1f} m")

try:
    if depth_sensor.supports(rs.option.emitter_enabled):
        cur = depth_sensor.get_option(rs.option.emitter_enabled)
        if cur == 0:
            depth_sensor.set_option(rs.option.emitter_enabled, 1)
            print("[INFO] Enabled emitter to improve depth on low-texture surfaces.")
except Exception as e:
    print(f"[WARN] Could not set emitter: {e}")

align = rs.align(rs.stream.color)

# ========================
# ArUco 설정 (OpenCV 버전 호환)
# ========================
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
try:
    parameters = aruco.DetectorParameters()
except TypeError:
    parameters = aruco.DetectorParameters_create()

try:
    _detector = aruco.ArucoDetector(aruco_dict, parameters)
except AttributeError:
    _detector = None

def detect_markers(img_bgr):
    if _detector is not None:
        return _detector.detectMarkers(img_bgr)
    else:
        return aruco.detectMarkers(img_bgr, aruco_dict, parameters=parameters)

WIN = "ArUco + Depth (ROS publish, mm)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 960, 720)

# ========================
# 퍼블리시 헬퍼
# ========================
# 퍼블리시 헬퍼
def publish_center_and_corners(marker_id: int,
                               center_g_m: np.ndarray,
                               corners_g_m_list: list,
                               center_b_m: np.ndarray = None,
                               corners_b_m_list: list = None):
    """툴/베이스 좌표계 center와 v0~v3를 퍼블리시 (출력은 mm)"""
    stamp = rospy.Time.now()
    pub_id.publish(Int32(marker_id))

    # Tool frame (mm)
    cx, cy, cz = (center_g_m.ravel() * SCALE_OUT).tolist()
    # y 좌표의 부호를 반전시킴
    cy = -cy  # y 좌표 부호 반전
    msg_c = PointStamped()
    msg_c.header.stamp = stamp
    msg_c.header.frame_id = "tool"
    msg_c.point.x, msg_c.point.y, msg_c.point.z = float(cx), float(cy), float(cz)
    pub_center_tool.publish(msg_c)

    msg_poly = PolygonStamped()
    msg_poly.header.stamp = stamp
    msg_poly.header.frame_id = "tool"
    for p in corners_g_m_list:  # 코너 좌표 퍼블리시
        x, y, z = (p.ravel() * SCALE_OUT).tolist()
        y = -y  # y 좌표 부호 반전
        msg_poly.polygon.points.append(Point32(x=float(x), y=float(y), z=float(z)))
    pub_corners_tool.publish(msg_poly)

    # Base frame (mm)
    if (center_b_m is not None) and (corners_b_m_list is not None):
        cbx, cby, cbz = (center_b_m.ravel() * SCALE_OUT).tolist()
        # y 좌표의 부호를 반전시킴
        cby = -cby  # y 좌표 부호 반전
        msg_cb = PointStamped()
        msg_cb.header.stamp = stamp
        msg_cb.header.frame_id = "base"
        msg_cb.point.x, msg_cb.point.y, msg_cb.point.z = float(cbx), float(cby), float(cbz)
        pub_center_base.publish(msg_cb)

        msg_poly_b = PolygonStamped()
        msg_poly_b.header.stamp = stamp
        msg_poly_b.header.frame_id = "base"
        for p in corners_b_m_list:  # 코너 좌표 퍼블리시
            x, y, z = (p.ravel() * SCALE_OUT).tolist()
            y = -y  # y 좌표 부호 반전
            msg_poly_b.polygon.points.append(Point32(x=float(x), y=float(y), z=float(z)))
        pub_corners_base.publish(msg_poly_b)


def detect_markers(img_bgr):
    if _detector is not None:
        return _detector.detectMarkers(img_bgr)
    else:
        return aruco.detectMarkers(img_bgr, aruco_dict, parameters=parameters)

# 메인 루프
fps_t0 = time.time()
fps_cnt = 0
last_pose_update_t = 0.0     # 툴→베이스 갱신 주기 제어
POSE_UPDATE_PERIOD = 0.02    # 50 Hz로 갱신(서비스가 감당되면), 느리면 0.05로

try:
    while not rospy.is_shutdown():
        # --- (1) 현재 로봇 TCP 포즈로 툴→베이스 변환 갱신 ---
        now_ros = rospy.Time.now().to_sec()
        if now_ros - last_pose_update_t >= POSE_UPDATE_PERIOD:
            try:
                # Doosan 컨트롤러의 오일러 순서에 맞춰 'zyx' 또는 'xyz'로 선택
                update_tool_to_base_from_robot(order='zyx')
            except Exception as e:
                rospy.logwarn_throttle(1.0, f"update_tool_to_base failed: {e}")
            last_pose_update_t = now_ros

        # --- (2) 카메라 프레임 획득 / 정렬 ---
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            print("No frames yet...")
            continue

        color_image = np.asanyarray(color_frame.get_data())
        vis = color_image.copy()

        # --- (3) ArUco 탐지 ---
        corners, ids, _ = detect_markers(color_image)
        if ids is not None:
            print(f"Detected markers: {ids}")  # 디버깅 출력

            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            for i, corner in enumerate(corners):
                marker_id = int(ids[i][0])
                pts = corner.reshape((4, 2)).astype(int)

                # 중심 픽셀/깊이(ROI 메디안)
                cx = int(np.mean(pts[:, 0])); cy = int(np.mean(pts[:, 1]))
                d_center = get_distance_median(depth_frame, cx, cy, r=DEPTH_ROI_RADIUS,
                                               dmin=DEPTH_VALID_MIN, dmax=DEPTH_VALID_MAX)
                print(f"Center depth: {d_center}")  # 깊이 정보 디버깅 출력

                # --- (4) 중심점/꼭짓점 카메라→툴→베이스 변환 ---
                center_g_m = None; center_b_m = None
                if d_center is not None:
                    try:
                        Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(
                            depth_intrin, [float(cx), float(cy)], float(d_center)
                        )
                        p_c_m = np.array([[Xc], [Yc], [Zc]], dtype=np.float64)  # m
                        p_g_m = cam_to_gripper(p_c_m)                           # (3,1) m
                        center_g_m = p_g_m
                        center_b_m = gripper_to_base(p_g_m)

                        # 콘솔(mm)
                        Xg, Yg, Zg = (p_g_m.ravel() * SCALE_OUT).tolist()
                        print(f"[TOOL][ID {marker_id}] Center ≈ ({Xg:.1f}, {-Yg:.1f}, {Zg:.1f}) {UNIT_LABEL}")

                        # --- 0번과 1번 코너 좌표로 회전 각도 계산 ---
                        rotation_angle = calculate_rotation_angle(corner)
                        print(f"[TOOL][ID {marker_id}] Rotation Angle (0, 1 corners): {rotation_angle:.2f} degrees")

                        # 회전 각도 퍼블리시
                        publish_rotation_angle(marker_id, rotation_angle)  # 회전 각도 퍼블리시

                    except Exception as e:
                        print(f"[ERR] center transform: {type(e).__name__}: {e}")

                corners_g_m = []
                corners_b_m = []
                all_ok = True
                for j, (px, py) in enumerate(pts):
                    d_corner = get_distance_median(depth_frame, int(px), int(py), r=DEPTH_ROI_RADIUS,
                                                   dmin=DEPTH_VALID_MIN, dmax=DEPTH_VALID_MAX)
                    if d_corner is None:
                        d_corner = d_center
                    if d_corner is None:
                        print(f"[TOOL][ID {marker_id}] V{j}: depth N/A → skip")
                        all_ok = False
                        break

                    try:
                        Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(
                            depth_intrin, [float(px), float(py)], float(d_corner)
                        )
                        p_c_m = np.array([[Xc], [Yc], [Zc]], dtype=np.float64)
                        p_g_m = cam_to_gripper(p_c_m)
                        corners_g_m.append(p_g_m)

                        # 콘솔에 각 코너의 (x, y, z) 좌표 출력
                        Xg, Yg, Zg = (p_g_m.ravel() * SCALE_OUT).tolist()
                        print(f"[TOOL][ID {marker_id}] Corner {j} ≈ ({Xg:.1f}, {-Yg:.1f}, {Zg:.1f}) {UNIT_LABEL}")
                        corners_b_m.append(gripper_to_base(p_g_m))
                    except Exception as e:
                        print(f"[ERR] corner V{j} transform: {type(e).__name__}: {e}")
                        all_ok = False
                        break

                # 검출 마커 표시
                try:
                    aruco.drawDetectedMarkers(vis, corners, ids)
                except Exception:
                    pass

                # --- (5) 퍼블리시 ---
                if (center_g_m is not None) and all_ok and (len(corners_g_m) == 4):
                    publish_center_and_corners(
                        marker_id,
                        center_g_m,
                        corners_g_m,
                        center_b_m,
                        corners_b_m
                    )

        # ----- FPS -----
        fps_cnt += 1
        now = time.time()
        if now - fps_t0 >= 0.5:
            fps = fps_cnt / (now - fps_t0)
            fps_cnt = 0
            fps_t0 = now
            draw_label(vis, f"FPS: {fps:.1f}", 10, 25, bg=True)

        # ----- 결과 출력 -----
        cv2.imshow(WIN, vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
