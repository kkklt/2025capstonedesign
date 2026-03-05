import numpy as np
import math
import rospy
import time
from capstone.srv import CheckCollision, CheckCollisionRequest
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float64


occupied_points = []
# ---- 고정 장애물 정의 (월·고정 설비 등) ----
# type: 'box'인 경우, min = [xmin, ymin, zmin], max = [xmax, ymax, zmax]
STATIC_OBSTACLES = [
    {
        "type": "box",
        "min": np.array([-0.8, -0.8, 0]),  # 예시: 튀김기/벽 같은 고정 구조물
        "max": np.array([0,  0, 0.3]),
    },
    {
        "type": "box",
        "min": np.array([0, -0.8, 0]),  # 예시: 튀김기/벽 같은 고정 구조물
        "max": np.array([0.8,  -0.6, 0.3]),
    },
    {
        "type": "box",
        "min": np.array([-0.35, -0.35, 0]),  # 예시: 튀김기/벽 같은 고정 구조물
        "max": np.array([0.35,  0.35, 0.5]),
    }
    
]


cm2m = 0.001
m2cm = 1000.0

# PRM 파라미터 설정
N_KNN = 20
MAX_EDGE_LEN = 0.28
minDist = 0.05
maxDist = MAX_EDGE_LEN


# 시작점과 목표점
startpoint_cm = [-206, 660, 500]
goalpoint_cm = [567.330, -749.240, 619.800]
start_m = [v * cm2m for v in startpoint_cm]
goal_m  = [v * cm2m for v in goalpoint_cm]


collision_checker_service = None

z_up_margin = 0.3
z_down_margin = 0.1
radius_margin = 0.28

# ---- 전역 가중치(지금은 0 → 기존 동작 그대로) ----
W_NODE = 0.0    # 노드 패널티 가중치
W_EDGE = 0.0    # 엣지 패널티 가중치

def node_cost(x, y, z):
    return 0.0  # TODO: 장애물 근접 패널티 등 나중에 정의(≥0 권장)

def edge_penalty(a, b):
    return 0.0  # TODO: 통로폭/기울기 등 엣지 기반 패널티(≥0 권장)

def octomap_callback(msg):
    global occupied_points
    points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    occupied_points = np.array(points)

class Node:
    def __init__(self, x, y, z, cost, parent_index):
        self.x = x
        self.y = y
        self.z = z
        self.cost = cost
        self.parent_index = parent_index
        self.neighbors = []
        self.id = 0
        self.gscore = math.inf
        self.fscore = math.inf
        self.parent = None

    def __eq__(self, other):
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def addNeighbors(self, neighbor):
        self.neighbors.append(neighbor)

    def setParent(self, parent):
        self.parent = parent

def c2g(current, goal):
    return np.linalg.norm(np.array([current.x, current.y, current.z]) - np.array([goal.x, goal.y, goal.z]))

def direction_reward(current, neighbor, goal):
    v1 = np.array([neighbor.x - current.x, neighbor.y - current.y, neighbor.z - current.z])
    v2 = np.array([goal.x - current.x, goal.y - current.y, goal.z - current.z])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-9)
    return -0.05 * (1 - cos_theta)

def astar(start, goal, nodes, max_waypoints=None):
    open_set = {start: start}
    closed_set = {}

    start.gscore = 0.0
    start.fscore = c2g(start, goal)

    while open_set:
        current = min(open_set, key=lambda o: open_set[o].fscore)
        if current == goal:
            path = []
            while current:
                path.append([current.x, current.y, current.z])
                current = current.parent
            path = path[::-1]

            return path

        del open_set[current]
        closed_set[current] = current

        for neighbor in current.neighbors:
            if neighbor in closed_set:
                continue

            edge_len = c2g(current, neighbor)
            tentative_gscore = (current.gscore
                                + edge_len
                                + W_NODE * neighbor.cost
                                + W_EDGE * edge_penalty(current, neighbor)
                                + direction_reward(current, neighbor, goal))

            if neighbor not in open_set or tentative_gscore < neighbor.gscore:
                neighbor.gscore = tentative_gscore
                neighbor.fscore = tentative_gscore + 1.5 * c2g(neighbor, goal)
                neighbor.setParent(current)
                open_set[neighbor] = neighbor

    return None


def create_prm(nodes):
    coords = np.array([[n.x, n.y, n.z] for n in nodes])
    for i, node in enumerate(nodes):
        d = np.linalg.norm(coords - coords[i], axis=1)
        mask = (d >= minDist) & (d <= maxDist)
        cand = np.where(mask)[0]
        # 가까운 순 정렬 후 상위 K개
        cand = cand[np.argsort(d[cand])][:N_KNN]

        for j in cand:
            if j <= i:
                continue  # 중복 간선 방지(한 번만 검사)
            neighbor = nodes[j]
            if is_edge_collision_free([node.x, node.y, node.z],
                                      [neighbor.x, neighbor.y, neighbor.z],
                                      step=EDGE_STEP):
                node.addNeighbors(neighbor)
                neighbor.addNeighbors(node)
    return nodes

def is_in_static_obstacle(point):
    """
    point: [x, y, z] (m)
    STATIC_OBSTACLES 안에 정의된 고정 장애물 안에 포함되면 True
    """
    p = np.array(point)
    for obs in STATIC_OBSTACLES:
        if obs["type"] == "box":
            mn = obs["min"]
            mx = obs["max"]
            # AABB 내부에 있는지 확인
            if np.all(p >= mn) and np.all(p <= mx):
                return True
    return False

def is_collision(point):
    global collision_checker_service

    if is_in_static_obstacle(point):
        return True
    
    try:
        request = CheckCollisionRequest()
        request.point.x = point[0]
        request.point.y = point[1]
        request.point.z = point[2]

        response = collision_checker_service(request)
        return response.is_occupied
    
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
        return True
    

def is_edge_collision_free(p_from, p_to, step=None):
    # step≈ 옥토맵 해상도(res)의 0.5배 권장(예: res=0.05m → step=0.025m)
    if step is None:
        step = EDGE_STEP

    v = np.array(p_to) - np.array(p_from)
    L = np.linalg.norm(v)
    if L == 0.0:
        return True
    
    n = max(1, int(math.ceil(L / step)))

    # 간단한 팽창: z 한 단계, 각도 4개만
    dz_list = [0.0, z_up_margin, -z_down_margin ]   # 필요하면 [0.0, 0.05] 정도
    theta_list = [0, math.pi/2, math.pi, 3*math.pi/2]

    for i in range(1, n):
        center = (np.array(p_from) + v * (i / float(n)))
        for dz in dz_list:
            for theta in theta_list:
                dx = radius_margin * math.cos(theta)
                dy = radius_margin * math.sin(theta)
                pt = [center[0] + dx, center[1] + dy, center[2] + dz]
                if is_collision(pt):
                    return False

    return True

    
def is_connected(nodes, start_node, goal_node):
    visited = set()

    stack = [start_node]

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        for neighbor in current.neighbors:
            if neighbor == goal_node:
                return True
            stack.append(neighbor)
    return False

def calculate_path_length(path):
    length = 0.0
    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i+1])
        length += np.linalg.norm(p1 - p2)
    return length

# ---------- 옥토맵 기반 시각화 함수 ----------
def draw_box(ax, mn, mx, **plot_kwargs):
    """
    mn: np.array([xmin, ymin, zmin])
    mx: np.array([xmax, ymax, zmax])
    """
    # 8개 꼭짓점
    x0, y0, z0 = mn
    x1, y1, z1 = mx
    corners = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ])

    # 12개 엣지(꼭짓점 인덱스 쌍)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 아래 사각형
        (4, 5), (5, 6), (6, 7), (7, 4),  # 위 사각형
        (0, 4), (1, 5), (2, 6), (3, 7),  # 세로 엣지
    ]

    for i, j in edges:
        xs = [corners[i, 0], corners[j, 0]]
        ys = [corners[i, 1], corners[j, 1]]
        zs = [corners[i, 2], corners[j, 2]]
        ax.plot(xs, ys, zs, **plot_kwargs)

def visualize_prm_octomap(nodes, path, occupied_points=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # ① 옥토맵 점 표시
    if occupied_points is not None and len(occupied_points) > 0:
        occ = np.array(occupied_points)
        ax.scatter(occ[:,0], occ[:,1], occ[:,2],
                   c='red', s=5, alpha=0.4, label='Occupied (Octomap)')
        
     # ② 고정 장애물 박스 표시
    static_labeled = False
    for obs in STATIC_OBSTACLES:
        if obs["type"] == "box":
            mn = obs["min"]
            mx = obs["max"]
            # 첫 번째 박스만 legend label 달기
            if not static_labeled:
                draw_box(ax, mn, mx,
                         color='purple', linewidth=1.5, alpha=0.8, label='Static obstacle')
                static_labeled = True
            else:
                draw_box(ax, mn, mx,
                         color='purple', linewidth=1.5, alpha=0.8)

    # ② PRM 그래프 간선
    for node in nodes:
        for nb in node.neighbors:
            ax.plot([node.x, nb.x], [node.y, nb.y], [node.z, nb.z],
                    color='gray', linewidth=0.5, alpha=0.4)

    # ③ A* 경로
    if path:
        path = np.array(path)
        ax.plot(path[:,0], path[:,1], path[:,2],
                color='blue', linewidth=2.5, label='A* Path')

    # ④ 시작/목표
    ax.scatter(start_m[0], start_m[1], start_m[2],
               color='green', s=80, label='Start')
    ax.scatter(goal_m[0], goal_m[1], goal_m[2],
               color='orange', s=80, label='Goal')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.legend()
    ax.view_init(elev=25, azim=-60)
    plt.title("PRM + A* Path + Octomap Occupied Voxels")
    plt.tight_layout()
    plt.show()

def main():
    global collision_checker_service, EDGE_STEP
    rospy.init_node('prm_astar_planner', anonymous=True)

    res = rospy.get_param('/octomap_server/resolution', 0.05)  # m
    EDGE_STEP = 1.5 * float(res)
    rospy.loginfo(f'OctoMap resolution={res:.3f} m → EDGE_STEP={EDGE_STEP:.3f} m')

    rospy.Subscriber("/octomap_point_cloud_centers", PointCloud2, octomap_callback)

    path_publisher = rospy.Publisher('/path_cook', Path, queue_size=10, latch=True)
    planning_time_pub = rospy.Publisher('/path_planning_time', Float64, queue_size=10, latch=True)

    rospy.loginfo("Waiting for collision check service ...")
    rospy.wait_for_service('check_collision_at_point')
    try: 
        collision_checker_service = rospy.ServiceProxy('check_collision_at_point', CheckCollision)
        rospy.loginfo("Collision check service available.")
    except rospy.ServiceException as e:
        rospy.logerr("Service proxy creation failed: %s" %e)
        return
    

    start_node = Node(start_m[0], start_m[1], start_m[2], node_cost(*start_m), -1)
    goal_node = Node(goal_m[0], goal_m[1], goal_m[2], node_cost(*goal_m), -1)

        # 노드 리스트 생성 (랜덤 노드)
    nodes = [start_node, goal_node]
    for _ in range(400):
        x, y, z = np.random.uniform(-0.8, 0.7), np.random.uniform(-0.8, 0.8), np.random.uniform(0.10, 0.80)
        if is_collision([x,y,z]):
            continue

        nodes.append(Node(x, y, z, node_cost(x,y,z), -1))

    rospy.loginfo("Creating PRM graph..")
    prm_start_time = time.time()
    nodes = create_prm(nodes)
    prm_duration = time.time() - prm_start_time
    rospy.loginfo(f"PRM graph created in {prm_duration: .4f} seconds.")

    

    if not is_connected(nodes, start_node, goal_node):
        rospy.logwarn("Path is not connected. PRM failed to find a connection.")
        print(f"RESULTS,FAIL,{prm_duration: .4f} seconds.")
        return
    
    print("시작점과 목표점이 연결되어 있습니다.")

    rospy.loginfo("Searching for a path using A*...")
    astar_start_time = time.time()
    path = astar(start_node, goal_node, nodes, max_waypoints=10)
    astar_duration = time.time() - astar_start_time
    rospy.loginfo (f"A* search completed in {astar_duration: .4f} seconds.")

    total_planning_time = prm_duration + astar_duration
    rospy.loginfo(f"Total planning time = {total_planning_time:.4f} s")


    visualize_prm_octomap(nodes, path, occupied_points)
    
    if path:
        total_planning_time = prm_duration + astar_duration

        # 1) 총 planning time publish
        planning_msg = Float64()
        planning_msg.data = total_planning_time
        planning_time_pub.publish(planning_msg)
        rospy.loginfo(f"Published planning time: {total_planning_time:.4f} s")
        
        path_len = calculate_path_length(path)
        num_waypoints = len(path)
        rospy.loginfo(f"Path found! Length: {path_len:.2f}, Waypoints: {num_waypoints}")
        print(f"RESULTS,SUCCESS,{prm_duration:.4f}, {astar_duration:.4f}, {path_len:.2f}, {num_waypoints}")
        
        rospy.loginfo("Publishing the planned path ...")

        path_msg_m = Path()
        path_msg_m.header.stamp = rospy.Time.now()
        path_msg_m.header.frame_id = "world"

        path_msg_cm = Path()
        path_msg_cm.header.stamp = rospy.Time.now()
        path_msg_cm.header.frame_id = "world"

        for point in path:
            pose_m = PoseStamped()
            pose_m.header.stamp = rospy.Time.now()
            pose_m.header.frame_id = "world"
            pose_m.pose.position.x = point[0]
            pose_m.pose.position.y = point[1]
            pose_m.pose.position.z = point[2]
            pose_m.pose.orientation.w = 1.0
            path_msg_m.poses.append(pose_m)

            pose_cm = PoseStamped()
            pose_cm.header.stamp = rospy.Time.now()
            pose_cm.header.frame_id = "world"
            pose_cm.pose.position.x = point[0] * m2cm
            pose_cm.pose.position.y = point[1] * m2cm
            pose_cm.pose.position.z = point[2] * m2cm
            pose_cm.pose.orientation.w = 1.0
            path_msg_cm.poses.append(pose_cm)

        rospy.sleep(1.0) 
        path_publisher.publish(path_msg_cm)
        rospy.loginfo("Path published to /path_cook topic.")

        rospy.spin()
        
    else:
        rospy.logwarn("Path not found by A*.")
        print(f"RESULTS,FAIL,{prm_duration:.4f}, {astar_duration:.4f},N/A, N/A")

    

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass