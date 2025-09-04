import xmlrpc.client
import rospy
import time
import math
import numpy as np
from dsr_msgs.srv import MoveLine, MoveJointx, MoveJoint, MoveJointRequest, MoveJointxRequest, MoveLineRequest, GetCurrentPose, GetCurrentPoseRequest
from std_msgs.msg import Float32MultiArray, Int32
from dsr_msgs.srv import SetCtrlBoxDigitalOutput, SetCtrlBoxDigitalOutputRequest
from geometry_msgs.msg import PointStamped, PolygonStamped
from std_msgs.msg import Float32

# Xg, Yg, Zg를 저장할 변수
Xg, Yg, Zg = 0, 0, 0

def center_callback(msg):
    global Xg, Yg, Zg
    # 수신한 중심 좌표를 Xg, Yg, Zg로 저장
    Xg = msg.point.x
    Yg = msg.point.y
    Zg = msg.point.z
    rospy.loginfo(f"Received ArUco center coordinates: Xg={Xg}, Yg={Yg}, Zg={Zg}")

received_angle = 0.0

# 회전 각도를 수신하는 콜백 함수
def angle_callback(msg):
    global received_angle
    received_angle = msg.data  # 퍼블리시된 회전 각도 값을 저장
    rospy.loginfo(f"Received rotation angle: {received_angle} degrees")

# 코너 좌표를 받는 콜백 함수
def corners_callback(msg):
    global corners_tool, corners_base
    # 툴 좌표계에서 코너 좌표를 저장
    corners_tool = [(point.x, point.y, point.z) for point in msg.polygon.points]
    rospy.loginfo(f"Received corners in tool frame: {corners_tool}")

# 회전 각도를 계산하는 함수 (코너 좌표에서)
def calculate_rotation_angle(corners):
    # 코너 좌표 4개를 사용하여 회전 각도를 계산
    points = np.array(corners_tool, dtype=np.float32)
    
    point1 = corners_tool[0]  # 첫 번째 코너 좌표
    point2 = corners_tool[3]  # 두 번째 코너 좌표

    # 각도를 계산하기 위해 x, y 좌표를 추출
    dx = point2[0] - point1[0]  # x 방향 차이
    dy = point2[1] - point1[1]  # y 방향 차이
    
    # 아크탄젠트를 사용하여 각도 계산
    angle = math.degrees(math.atan2(dy, dx))

    angle = - angle

    return angle

def movejoint(posx, vel_, acc_, time_):
    rospy.wait_for_service('/dsr01m1013/motion/move_joint')
    try:
        move_joint_srv = rospy.ServiceProxy('/dsr01m1013/motion/move_joint', MoveJoint)
        req = MoveJointRequest()
        req.pos = posx  # posx 리스트에 6개의 값이 있어야 함
        req.vel = vel_
        req.acc = acc_
        req.time = time_
        req.radius = 0
        req.mode = 0
        req.blendType = 0
        req.syncType = 0
        resp = move_joint_srv(req)
        rospy.loginfo("move_joint Success")
    except rospy.ServiceException as e:
        rospy.logerr("Failed to call service move_joint: %s", e)
        rospy.signal_shutdown("Service call failed")

def moveline(posx, vel_, acc_, time_):
    rospy.wait_for_service('/dsr01m1013/motion/move_line')
    try:
        move_line_srv = rospy.ServiceProxy('/dsr01m1013/motion/move_line', MoveLine)
        req = MoveLineRequest()
        req.pos = posx  # posx 리스트에 6개의 값이 있어야 함
        req.vel = vel_  # vel_ 리스트에 2개의 값
        req.acc = acc_  # acc_ 리스트에 2개의 값
        req.time = time_
        req.radius = 0
        req.mode = 0
        req.blendType = 0
        req.syncType = 0
        resp = move_line_srv(req)
        rospy.loginfo("move_line Success")
    except rospy.ServiceException as e:
        rospy.logerr("Failed to call service move_line: %s", e)
        rospy.signal_shutdown("Service call failed")

def movejointx(posx, vel_, acc_, time_, sol):
    rospy.wait_for_service('/dsr01m1013/motion/move_jointx')
    try:
        move_jointx_srv = rospy.ServiceProxy('/dsr01m1013/motion/move_jointx', MoveJointx)
        req = MoveJointxRequest()
        req.pos = posx  # posx 리스트에 6개의 값이 있어야 함
        req.vel = vel_
        req.acc = acc_
        req.time = time_
        req.radius = 0
        req.mode = 0
        req.blendType = 0
        req.syncType = 0
        req.sol = sol
        resp = move_jointx_srv(req)
        rospy.loginfo("move_jointx Success")
    except rospy.ServiceException as e:
        rospy.logerr("Failed to call service move_jointx: %s", e)
        rospy.signal_shutdown("Service call failed")

server_url = "http://192.168.137.101:41414/"
proxy = xmlrpc.client.ServerProxy(server_url)

def get_current_pose(space_type=1): # 0 : joint 1: space
    rospy.wait_for_service('/dsr01m1013/system/get_current_pose')
    try:
        get_current_pose = rospy.ServiceProxy('/dsr01m1013/system/get_current_pose', GetCurrentPose)
        req = GetCurrentPoseRequest()
        req.space_type = space_type
        response = get_current_pose(req)
        current_pose = response.pos
        rospy.loginfo(f"현재 : {current_pose}")
        return current_pose  # 위치 값 리스트를 반환
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return None

def main():
    rospy.init_node('square_center_node', anonymous=True)

    # center_tool 토픽을 구독하여 ArUco 마커의 중심 좌표를 받아옵니다.
    rospy.Subscriber("/aruco/center_tool", PointStamped, center_callback)
    # corners_tool 토픽을 구독하여 ArUco 마커의 코너 좌표를 받아옵니다.
    rospy.Subscriber("/aruco/corners_tool", PolygonStamped, corners_callback)
    rospy.Subscriber("/aruco/corners_base", PolygonStamped, corners_callback)
    rospy.Subscriber("/aruco/rotation_angle", Float32, angle_callback)

    # 좌표를 받기 전에 약간의 대기시간을 줍니다
    rospy.sleep(1)

    rate = rospy.Rate(10)
    current_position = get_current_pose(space_type=1)
    x_coord = current_position[0]
    y_coord = current_position[1]
    z_coord = current_position[2]

    # 기본 위치 설정
    pos = [400+Xg, 400+Yg-5, 400, 0, 180, 180]
    pos_down2 = [400+Xg, 400+Yg-5, 254, 0, 180, 180]
    pos_basic = [400, 400, 400, 0, 180, 180]
    vel = [0, 0]   # 속도 설정 (두 값)
    acc = [0, 0]   # 가속도 설정 (두 값)

    while not rospy.is_shutdown():
        rate.sleep()

        # corners_tool의 값이 설정되었을 때, 회전 각도를 계산합니다
        if len(corners_tool) >= 4:
            # 코너 좌표에서 회전 각도를 계산하고 angle 변수에 할당
            angle = calculate_rotation_angle(corners_tool)
            rospy.loginfo(f"Calculated rotation angle: {angle} degrees")
            print(f"Calculated rotation angle: {angle} degrees")  # 콘솔에 출력
            angle_with_offset = 180 + received_angle

            # pos_with_rotation을 회전 각도를 반영하여 설정
            pos_with_rotation = [400+Xg, 400+Yg-5, 400, 0, 180, angle_with_offset]
            pos_down = [400+Xg, 400+Yg-5, 254, 0, 180, angle_with_offset]
            pos_up = [400+Xg, 400+Yg-5, 400, 0, 180, angle_with_offset]

        # 기본 위치로 move_jointx 실행
        movejointx(pos, 0, 0, 5, 2)
        time.sleep(1)   
        proxy.twofg_grip_external(0, 70.0, 20, 10)
        time.sleep(1)
        movejointx(pos_with_rotation, 0, 0, 5, 2)
        time.sleep(1)
        moveline(pos_down, vel, acc, 3)
        time.sleep(1) 
        proxy.twofg_grip_external(0, 16.0, 20, 10)
        time.sleep(1)
        movejointx(pos_up, 0, 0, 5, 2)
        time.sleep(1)
        movejointx(pos, 0, 0, 5, 2)
        time.sleep(1)
        moveline(pos_down2, vel, acc, 3)
        time.sleep(1)
        proxy.twofg_grip_external(0, 70.0, 20, 10)
        time.sleep(1) 
        moveline(pos, vel, acc, 3)
        time.sleep(1)
        movejointx(pos_basic, 0, 0, 5, 2)

        break  # 한 번 실행 후 종료

if __name__ == '__main__':
    main()
