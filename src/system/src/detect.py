import pyrealsense2 as rs
import cv2
import cv2.aruco as aruco
import numpy as np
import torch
import time
import math

# ==== ROS ====
import rospy
from geometry_msgs.msg import PointStamped, PolygonStamped, Point32
from std_msgs.msg import Float32, Int32

# ========================
# 🔧 사용자 설정
# ========================
CONF_THRES = 0.60
YOLO_IMG_SIZE = 640
BOX_COLOR = (0, 0, 225)
TEXT_COLOR = (255, 255, 255)
TEXT_BG = True
TEXT_SCALE = 0.55
TEXT_THICK = 1
BOX_THICK = 3

# 기종에 따라 아래 값은 start 후 동적으로 조정됨(기본값은 안전한 범위로)
DEPTH_VALID_MIN = 0.10   # m (D435i면 코드에서 0.12로 올려줌)
DEPTH_VALID_MAX = 6.0    # m
DEPTH_ROI_RADIUS = 3     # 깊이 추정용 ROI 반경(픽셀)
ALLOW_CLASSES = None     # 예: [0, 1] 특정 클래스만 표시, None이면 전부

# 꼭짓점 TOOL 좌표를 화면에도 띄우고 싶으면 True
SHOW_CORNER_TOOL_LABELS = False

# YOLO 설정
YOLO_DIR = '/home/hyunsoo/capstone2025/src/yolov5'
YOLO_WEIGHTS = '/home/hyunsoo/capstone2025/src/yolov5/train/0909cup/weights/best.pt'
CUP_CLASS_ID = 0         # 컵 클래스 ID

# 프레임 아이디(필요시 변경)
FRAME_ID_TOOL = 'tool0'
FRAME_ID_BASE = 'base'

# ========================
# 🔁 카메라→그리퍼 변환 (캘리브레이션 결과 반영)
# ========================
R_c2g = np.array([
    [0.9987670687919746, -0.04441860801095171, -0.02216595495475445],
    [0.04450782768642642, 0.9990027357184897, 0.003547858735748418],
    [0.02198625869315107, -0.00453004297362275, 0.999748009820142]
], dtype=np.float64)

t_c2g_mm = np.array([
    [-25.22925474699036],
    [-125.9568164136453],
    [17.75056071168541]
], dtype=np.float64)

t_c2g = t_c2g_mm / 1000.0  # (3,1) meters

def cam_to_gripper(p_c_m: np.ndarray) -> np.ndarray:
    """카메라 좌표계 3D점(m)을 그리퍼(툴) 좌표계로 변환"""
    return (R_c2g @ p_c_m) + t_c2g  # (3,1)

# (선택) 툴→베이스 변환이 있으면 여기에 입력
USE_BASE_FRAME = False
R_g2b = np.eye(3, dtype=np.float64)
t_g2b = np.zeros((3,1), dtype=np.float64)

def gripper_to_base(p_g_m: np.ndarray) -> np.ndarray:
    return (R_g2b @ p_g_m) + t_g2b  # (3,1)

# ========================
# 유틸 함수
# ========================
def draw_label(img, text, x, y,
               text_color=TEXT_COLOR, bg=TEXT_BG,
               scale=TEXT_SCALE, thick=TEXT_THICK):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    if bg:
        cv2.rectangle(img, (int(x), int(y - th - baseline)), (int(x + tw), int(y + baseline)),
                      (0, 0, 0), thickness=-1)
    cv2.putText(img, text, (int(x), int(y)),
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

# 🔸 회전각 계산(0→3 에지 기준, XY평면; 단위 무관)
def calculate_rotation_angle(corners_xy):
    """
    corners_xy: 길이 4의 리스트/배열, 각 원소는 (x, y)
                예: [p0, p1, p2, p3], ArUco 코너 순서 그대로
    반환: 0번→3번 벡터가 +X축과 이루는 각도(도), 범위 [-180, 180)
    """
    p1 = np.array(corners_xy[0], dtype=np.float64)  # corner 0
    p2 = np.array(corners_xy[3], dtype=np.float64)  # corner 3
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    ang = math.degrees(math.atan2(dy, dx))
    
    return ang


# ========================
# ROS 노드 & 퍼블리셔
# ========================
rospy.init_node('yolo_aruco_depth_node', anonymous=True)

pub_aruco_marker_id   = rospy.Publisher('/aruco/marker_id', Int32, queue_size=10)
pub_aruco_center_tool = rospy.Publisher('/aruco/center_tool', PointStamped, queue_size=10)
pub_aruco_corners_tool= rospy.Publisher('/aruco/corners_tool', PolygonStamped, queue_size=10)
pub_aruco_yaw_deg     = rospy.Publisher('/aruco/yaw_deg', Float32, queue_size=10)
pub_cup_center_tool   = rospy.Publisher('/cup/center_tool', PointStamped, queue_size=10)

# (옵션) 베이스 프레임 퍼블리셔
if USE_BASE_FRAME:
    pub_aruco_center_base  = rospy.Publisher('/aruco/center_base', PointStamped, queue_size=10)
    pub_aruco_corners_base = rospy.Publisher('/aruco/corners_base', PolygonStamped, queue_size=10)
    pub_cup_center_base    = rospy.Publisher('/cup/center_base', PointStamped, queue_size=10)

# ========================
# RealSense 파이프라인
# ========================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_device("102422076836")

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
# YOLOv5 로드
# ========================
model = torch.hub.load(YOLO_DIR, 'custom',
                       path=YOLO_WEIGHTS,
                       source='local')
model.conf = CONF_THRES
model.iou = 0.45

# ========================
# ArUco 설정
# ========================
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
try:
    parameters = aruco.DetectorParameters()
except TypeError:
    parameters = aruco.DetectorParameters_create()

detector = aruco.ArucoDetector(aruco_dict, parameters)

WIN = "Yolo+ArUco+Depth"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 960, 720)

# ========================
# 메인 루프
# ========================
fps_t0 = time.time()
fps_cnt = 0
try:
    with torch.no_grad():
        while not rospy.is_shutdown():
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                print("No frames yet...")
                continue

            color_image = np.asanyarray(color_frame.get_data())
            yolo_frame = color_image.copy()
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            # ----- YOLO 추론 및 3D 변환 (컵 중심점 툴 좌표 퍼블리시) -----
            results = model(color_image, size=YOLO_IMG_SIZE)
            try:
                det_df = results.pandas().xyxy[0]
            except Exception:
                det_df = None

            best_conf = -1.0
            best_p_g_cup = None
            best_bbox = None

            if det_df is not None and len(det_df) > 0:
                for _, row in det_df.iterrows():
                    conf = float(row['confidence'])
                    cls = int(row['class'])

                    # 컵 클래스만 처리
                    if cls == CUP_CLASS_ID and conf >= CONF_THRES:
                        x1, y1 = int(row['xmin']), int(row['ymin'])
                        x2, y2 = int(row['xmax']), int(row['ymax'])
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        # 바운딩 박스 중심의 깊이 추정
                        d_cup_center = get_distance_median(
                            depth_frame, cx, cy, r=DEPTH_ROI_RADIUS,
                            dmin=DEPTH_VALID_MIN, dmax=DEPTH_VALID_MAX
                        )

                        # 시각화: 바운딩 박스, 중심점
                        cv2.rectangle(yolo_frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICK)
                        cv2.circle(yolo_frame, (cx, cy), 4, (0, 255, 0), -1)

                        label_text = f"Cup {conf:.2f}"
                        if d_cup_center is not None:
                            try:
                                # 카메라 좌표계 3D 변환
                                Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(
                                    depth_intrin, [float(cx), float(cy)], float(d_cup_center)
                                )
                                p_c_cup = np.array([[Zc], [Xc], [Yc]], dtype=np.float64)

                                # ⭐ 그리퍼 좌표계로 변환 ⭐
                                p_g_cup = cam_to_gripper(p_c_cup)  # meters
                                Xg_mm, Yg_mm, Zg_mm = (p_g_cup * 1000).ravel()
                                label_text += f" -> ({Xg_mm:.2f}, {Yg_mm:.2f}, {Zg_mm:.2f}) mm"
                                print(f"[TOOL][Cup] Center ≈ ({Xg_mm:.3f}, {Yg_mm:.3f}, {Zg_mm:.3f}) mm")

                                # 최고 신뢰도 컵 저장(퍼블리시용)
                                if conf > best_conf:
                                    best_conf = conf
                                    best_p_g_cup = p_g_cup.copy()
                                    best_bbox = (cx, cy)
                            except Exception as e:
                                print(f"[ERR] Cup transform: {type(e).__name__}: {e}")
                                label_text += " -> N/A"
                        else:
                            label_text += " -> Z: N/A"

                        draw_label(yolo_frame, label_text, x1, max(0, y1 - 5))

            # 컵 퍼블리시 (최고 신뢰도 한 개)
            if best_p_g_cup is not None:
                stamp = rospy.Time.now()
                cup_msg = PointStamped()
                cup_msg.header.stamp = stamp
                cup_msg.header.frame_id = FRAME_ID_TOOL
                cup_msg.point.x = float(best_p_g_cup[0, 0])
                cup_msg.point.y = float(best_p_g_cup[1, 0])
                cup_msg.point.z = float(best_p_g_cup[2, 0])
                pub_cup_center_tool.publish(cup_msg)

                if USE_BASE_FRAME:
                    p_b_cup = gripper_to_base(best_p_g_cup)
                    cup_b = PointStamped()
                    cup_b.header.stamp = stamp
                    cup_b.header.frame_id = FRAME_ID_BASE
                    cup_b.point.x = float(p_b_cup[0, 0])
                    cup_b.point.y = float(p_b_cup[1, 0])
                    cup_b.point.z = float(p_b_cup[2, 0])
                    pub_cup_center_base.publish(cup_b)

            # ----- ArUco 탐지 -----
            corners, ids, _ = detector.detectMarkers(color_image)
            if ids is not None:
                for i, corner in enumerate(corners):
                    pts = corner.reshape((4, 2)).astype(int)

                    # 중심 픽셀/깊이(ROI 메디안)
                    cx_aruco = int(np.mean(pts[:, 0]))
                    cy_aruco = int(np.mean(pts[:, 1]))
                    d_center = get_distance_median(
                        depth_frame, cx_aruco, cy_aruco, r=DEPTH_ROI_RADIUS,
                        dmin=DEPTH_VALID_MIN, dmax=DEPTH_VALID_MAX
                    )

                    # 시각화: 중심/꼭짓점
                    cv2.circle(yolo_frame, (cx_aruco, cy_aruco), 5, (0, 255, 0), -1)
                    for j, (px, py) in enumerate(pts):
                        cv2.circle(yolo_frame, (px, py), 4, (255, 0, 0), -1)
                        draw_label(yolo_frame, f"{j}", px + 6, py - 6, bg=False)

                    # 중심 깊이 라벨
                    if d_center is not None:
                        draw_label(yolo_frame, f"ID {ids[i][0]}  Z≈{d_center*1000:.3f}mm", cx_aruco + 8, cy_aruco - 8)
                    else:
                        draw_label(yolo_frame, f"ID {ids[i][0]}  Z: N/A", cx_aruco + 8, cy_aruco - 8)

                    # ===== 1) 중심점 → 카메라3D → TOOL 변환 =====
                    p_g_center = None
                    if d_center is not None:
                        try:
                            Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(
                                depth_intrin, [float(cx_aruco), float(cy_aruco)], float(d_center)
                            )
                            p_c = np.array([[Xc], [Yc], [Zc]], dtype=np.float64)
                            p_g = cam_to_gripper(p_c)  # meters
                            p_g_center = p_g.copy()
                            Xg_mm, Yg_mm, Zg_mm = (p_g * 1000).ravel()
                            Yg_mm = - Yg_mm
                            print(f"[TOOL][ID {ids[i][0]}] Center ≈ ({Xg_mm:.3f}, {Yg_mm:.3f}, {Zg_mm:.3f}) mm")
                        except Exception as e:
                            print(f"[ERR] center transform: {type(e).__name__}: {e}")

                    # 🔹 회전각/코너 퍼블리시를 위한 컨테이너 (미터 단위)
                    corners_tool_xy_m = [None] * 4     # (x,y) in m
                    corners_tool_xyz_m = [None] * 4    # (x,y,z) in m
                    if USE_BASE_FRAME:
                        corners_base_xyz_m = [None] * 4

                    # ===== 2) 각 꼭짓점 → 카메라3D → TOOL 변환 =====
                    for j, (px, py) in enumerate(pts):
                        d_corner = get_distance_median(
                            depth_frame, int(px), int(py), r=DEPTH_ROI_RADIUS,
                            dmin=DEPTH_VALID_MIN, dmax=DEPTH_VALID_MAX
                        )
                        if d_corner is None:
                            d_corner = d_center

                        if d_corner is None:
                            print(f"[TOOL][ID {ids[i][0]}] V{j}: depth N/A → skip")
                            continue

                        try:
                            Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(
                                depth_intrin, [float(px), float(py)], float(d_corner)
                            )
                            p_c = np.array([[Xc], [Yc], [Zc]], dtype=np.float64)
                            p_g = cam_to_gripper(p_c)  # meters
                            Xg_m, Yg_m, Zg_m = p_g.ravel()
                            Xg_mm, Yg_mm, Zg_mm = (p_g * 1000).ravel()
                            Yg_mm = - Yg_mm
                            print(f"[TOOL][ID {ids[i][0]}] V{j} ≈ ({Xg_mm:.3f}, {Yg_mm:.3f}, {Zg_mm:.3f}) mm")

                            # 저장(회전각/코너 퍼블리시용)
                            corners_tool_xy_m[j]  = (float(Xg_m), float(Yg_m))
                            corners_tool_xyz_m[j] = (float(Xg_m), float(Yg_m), float(Zg_m))

                            if SHOW_CORNER_TOOL_LABELS:
                                draw_label(yolo_frame, f"({Xg_mm:.2f},{Yg_mm:.2f},{Zg_mm:.2f})mm",
                                           px + 8, py + 14, bg=True)

                            if USE_BASE_FRAME:
                                p_b = gripper_to_base(p_g)
                                Xb_m, Yb_m, Zb_m = p_b.ravel()
                                corners_base_xyz_m[j] = (float(Xb_m), float(Yb_m), float(Zb_m))
                                print(f"[BASE][ID {ids[i][0]}] V{j} ≈ ({Xb_m*1000:.3f}, {Yb_m*1000:.3f}, {Zb_m*1000:.3f}) mm")
                        except Exception as e:
                            print(f"[ERR] corner V{j} transform: {type(e).__name__}: {e}")
 
                    # 🔻 퍼블리시 (ID, 중심, 코너, 회전각)
                    stamp = rospy.Time.now()
                    # 1) 마커 ID
                    pub_aruco_marker_id.publish(Int32(data=int(ids[i][0])))

                    # 2) 중심 좌표 (tool/base)
                    if p_g_center is not None:
                        center_msg = PointStamped()
                        center_msg.header.stamp = stamp
                        center_msg.header.frame_id = FRAME_ID_TOOL
                        center_msg.point.x = float(p_g_center[0, 0])
                        center_msg.point.y = float(p_g_center[1, 0])
                        center_msg.point.z = float(p_g_center[2, 0])
                        pub_aruco_center_tool.publish(center_msg)

                        if USE_BASE_FRAME:
                            p_b = gripper_to_base(p_g_center)
                            center_b = PointStamped()
                            center_b.header.stamp = stamp
                            center_b.header.frame_id = FRAME_ID_BASE
                            center_b.point.x = float(p_b[0, 0])
                            center_b.point.y = float(p_b[1, 0])
                            center_b.point.z = float(p_b[2, 0])
                            pub_aruco_center_base.publish(center_b)

                    # 3) 코너 좌표 폴리곤 (tool/base)
                    if all(c is not None for c in corners_tool_xyz_m):
                        poly = PolygonStamped()
                        poly.header.stamp = stamp
                        poly.header.frame_id = FRAME_ID_TOOL
                        for (x, y, z) in corners_tool_xyz_m:
                            poly.polygon.points.append(Point32(x=x, y=y, z=z))
                        pub_aruco_corners_tool.publish(poly)

                        if USE_BASE_FRAME and all(c is not None for c in corners_base_xyz_m):
                            poly_b = PolygonStamped()
                            poly_b.header.stamp = stamp
                            poly_b.header.frame_id = FRAME_ID_BASE
                            for (x, y, z) in corners_base_xyz_m:
                                poly_b.polygon.points.append(Point32(x=x, y=y, z=z))
                            pub_aruco_corners_base.publish(poly_b)

                    # 4) 회전각(Yaw) 퍼블리시 (tool XY)
                    if all(c is not None for c in corners_tool_xy_m):
                        theta_tool = calculate_rotation_angle(corners_tool_xy_m)
                        pub_aruco_yaw_deg.publish(Float32(data=float(theta_tool)))
                        print(f"[TOOL][ID {ids[i][0]}] Yaw(0→3) ≈ {theta_tool:.2f}°")

                aruco.drawDetectedMarkers(yolo_frame, corners, ids)

            # ----- FPS -----
            fps_cnt += 1
            now = time.time()
            if now - fps_t0 >= 0.5:
                fps = fps_cnt / (now - fps_t0)
                fps_cnt = 0
                fps_t0 = now
                draw_label(yolo_frame, f"FPS: {fps:.1f}", 10, 25, bg=True)

            # ----- 결과 출력 -----
            cv2.imshow(WIN, yolo_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
