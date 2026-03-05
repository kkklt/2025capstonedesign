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
from scipy.stats import qmc
from scipy.spatial import cKDTree 


occupied_points = []
collision_checker_service = None
point_kdtree = None

startpoint_cm = [-274, 650, 450]
goalpoint_cm = [1000, 350, 400]

cm2m = 0.001
m2cm = 1000.0


start_m = [v * cm2m for v in startpoint_cm]
goal_m  = [v * cm2m for v in goalpoint_cm]


z_up_margin = 0.25
z_down_margin = 0.3
radius_margin = 0.28

EDGE_STEP = 0.05

# 유틸 함수
def node_cost(x, y, z):
    return 0.0  

def update_kdtree():
    global point_kdtree, occupied_points
    if len(occupied_points) > 0:
        point_kdtree = cKDTree(occupied_points)

def octomap_callback(msg):
    global occupied_points
    points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    occupied_points = np.array(points)
    update_kdtree()

# 노드클래스 
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


# PRM 관련 함수
def c2g(current, goal):
    return np.linalg.norm(np.array([current.x, current.y, current.z]) - np.array([goal.x, goal.y, goal.z]))

def direction_reward(current, neighbor, goal):
    v1 = np.array([neighbor.x - current.x, neighbor.y - current.y, neighbor.z - current.z])
    v2 = np.array([goal.x - current.x, goal.y - current.y, goal.z - current.z])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-9)
    return -0.05 * (1 - cos_theta)

def generate_nodes_collision_free(total_halton=600):
    
    nodes=[]

    sampler = qmc.Halton(d=3, scramble=True)
    halton_points = sampler.random(total_halton)

    #알아서 크기에 맞게 조정
    x_range = (-1.0,1.1)
    y_range = (0.15, 0.95)
    z_range = (0.35, 0.95)

    def scale(v, lo, hi):
        return lo + v * (hi - lo)

    for h in halton_points:
        x = scale(h[0], *x_range)
        y = scale(h[1], *y_range)
        z = scale(h[2], *z_range)
        
        if not is_collision([x, y, z]):
            nodes.append(Node(x, y, z, node_cost(x, y, z), -1))

    rospy.loginfo(f"[Halton] {len(nodes)}")
    return nodes

def connect_knn(nodes, k=12):
    coords = np.array([[n.x, n.y, n.z] for n in nodes])
    tree = cKDTree(coords)

    for idx, node in enumerate(nodes):
        dists, idxs = tree.query(coords[idx], k=k+1)

        for j in idxs[1:]:
            node.addNeighbors(nodes[j])

    rospy.loginfo(f"k-NN connected (k={k})")
    return nodes 

# A* 알고리즘
def astar(start, goal, nodes,max_waypoints=None ):

    open_set = {start}
    closed_set = set()

    start.gscore = 0.0
    start.fscore = c2g(start,goal)

    while open_set:
        current = min(open_set, key=lambda n: n.fscore)

        if current == goal:
            path = []
            while current:
                path.append([current.x, current.y, current.z])
                current = current.parent
            return path[::-1]
        
        open_set.remove(current)
        closed_set.add(current)

        for neighbor in current.neighbors:
            if neighbor not in closed_set:
                if not hasattr(current, 'validated_edges'):
                    current.validated_edges = {}

                if neighbor not in current.validated_edges:

                    if is_edge_collision_free([current.x, current.y, current.z],
                                            [neighbor.x, neighbor.y, neighbor.z]):
                        current.validated_edges[neighbor] = True
                    else:
                        current.validated_edges[neighbor] = False
                        continue

                else:

                    if current.validated_edges[neighbor] is False:
                        continue
        
            tentative_g = current.gscore + c2g(current, neighbor)

            if tentative_g < neighbor.gscore:
                neighbor.parent = current
                neighbor.gscore = tentative_g
                neighbor.fscore = tentative_g + 1.6 * c2g(neighbor, goal)
                open_set.add(neighbor)

    return None

# 스무딩, 경로
def is_line_collision_free(p1, p2, step=None):
    return is_edge_collision_free(p1, p2, step)

def smooth_path(path, iterations = 200):
    if len(path) <= 2:
        return path
    
    path = [np.array(p) for p in path]

    for _ in range(iterations):

        if len(path) <= 2:
            break
        max_i = len(path) - 2
        if max_i <= 0:
            break

        i = np.random.randint(0, len(path) - 2)
        j = np.random.randint(i + 2, len(path))

        p_i = path[i]
        p_j = path[j]

        if is_line_collision_free(p_i.tolist(), p_j.tolist(), step=EDGE_STEP):
            path = path[:i + 1] + path[j:]

    return [p.tolist() for p in path]  

def densify_path(path, min_points=4):
    """
    path: [[x,y,z], ...]
    min_points: 최소 waypoint 개수
    """
    pts = [np.array(p) for p in path]

    # 이미 충분하면 그대로 리턴
    if len(pts) >= min_points:
        return [p.tolist() for p in pts]

    # 가장 긴 segment를 계속 반으로 쪼개면서 개수 늘리기
    while len(pts) < min_points:
        max_len = -1.0
        max_idx = None
        for i in range(len(pts) - 1):
            seg_len = np.linalg.norm(pts[i+1] - pts[i])
            if seg_len > max_len:
                max_len = seg_len
                max_idx = i

        # 모든 구간 길이가 0이면 더이상 쪼갤 수 없음
        if max_len <= 0.0 or max_idx is None:
            break

        mid = (pts[max_idx] + pts[max_idx + 1]) / 2.0
        pts.insert(max_idx + 1, mid)

    return [p.tolist() for p in pts] 

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

def subdivide_by_distance(path, max_seg=0.2):
    """
    path: [[x,y,z], ...]
    max_seg: segment length threshold (meters)
    Return: subdivided path with all segments <= max_seg
    """
    new_path = []

    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i+1])

        # 현재 segment 길이
        dist = np.linalg.norm(p2 - p1)

        # 몇 개로 쪼갤지 결정
        if dist > max_seg:
            n = int(math.ceil(dist / max_seg))
            for t in range(n):
                new_pt = p1 + (p2 - p1) * (t / float(n))
                new_path.append(new_pt.tolist())
        else:
            new_path.append(p1.tolist())

    # 마지막 점 추가
    new_path.append(path[-1])
    return new_path


#충돌검사
def is_collision_distance(pt, threshold=0.04):
    global point_kdtree
    if point_kdtree is None:
        return False
    
    dist, idx = point_kdtree.query(pt)
    return dist < threshold

def is_collision(point):
    global collision_checker_service
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
                
                if is_collision_distance(pt, threshold=0.03):
                    return False
                if is_collision(pt):
                    return False

    return True



# ---------- 옥토맵 기반 시각화 함수 ----------
def visualize_prm_octomap(nodes, path, occupied_points=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # ① 옥토맵 점 표시
    if occupied_points is not None and len(occupied_points) > 0:
        occ = np.array(occupied_points)
        ax.scatter(occ[:,0], occ[:,1], occ[:,2],
                   c='red', s=5, alpha=0.4, label='Occupied (Octomap)')

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

# main
def main():
    global collision_checker_service, EDGE_STEP

    rospy.init_node('prm_astar_planner2', anonymous=True)

    res = rospy.get_param('/octomap_server/resolution', 0.05)  # m
    EDGE_STEP = 1.0 * float(res)
    rospy.loginfo(f'OctoMap resolution={res:.3f} m → EDGE_STEP={EDGE_STEP:.3f} m')

    rospy.Subscriber("/octomap_point_cloud_centers", PointCloud2, octomap_callback)

    path_publisher = rospy.Publisher('/path_move', Path, queue_size=10, latch=True)
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

    #prm, k-nn연결
    prm_start_time = time.time()
    nodes = [start_node, goal_node]
    sampled_nodes = generate_nodes_collision_free()
    nodes.extend(sampled_nodes)
    nodes = connect_knn(nodes, k=8)
    prm_duration = time.time() - prm_start_time
    rospy.loginfo(f"PRM graph created in {prm_duration: .4f} seconds.")

    if not is_connected(nodes, start_node, goal_node):
        rospy.logwarn("Path is not connected. PRM failed to find a connection.")
        print(f"RESULTS,FAIL,{prm_duration: .4f} seconds.")
        return
    
    print("시작점과 목표점이 연결되어 있습니다.")

    #A*
    rospy.loginfo("Searching for a path using A*...")
    astar_start_time = time.time()
    path = astar(start_node, goal_node, nodes)
    
    # 스무딩
    if path and len(path) > 2:
        smoothed = smooth_path(path, iterations = 20)
        path = smoothed
    if path:
        path = densify_path(path, min_points=4)
    if path:
        path = subdivide_by_distance(path, max_seg=0.18)

    astar_duration = time.time() - astar_start_time
    rospy.loginfo (f"A* search completed in {astar_duration: .4f} seconds.")
    total_planning_time = prm_duration + astar_duration
    rospy.loginfo(f"Total planning time = {total_planning_time:.4f} s")

    #시각화
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

        path_msg_cm = Path()
        path_msg_cm.header.stamp = rospy.Time.now()
        path_msg_cm.header.frame_id = "world"

        for point in path:
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
        rospy.loginfo("Path published to /path_move topic.")

        rospy.spin()
        
    else:
        rospy.logwarn("Path not found by A*.")
        print(f"RESULTS,FAIL,{prm_duration:.4f}, {astar_duration:.4f},N/A, N/A")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass