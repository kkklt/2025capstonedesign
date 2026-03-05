# obstacle_detector/msg/ObstacleInfo.msg
std_msgs/Header header       # 메시지 헤더 (시간 스탬프, 프레임 ID)
geometry_msgs/Point min_point # 바운딩 박스 최소점 (로봇 base_link 기준)
geometry_msgs/Point max_point # 바운딩 박스 최대점 (로봇 base_link 기준)
float32 width                # 바운딩 박스 너비 (Y축 방향)
float32 height               # 바운딩 박스 높이 (Z축 방향)
float32 depth                # 바운딩 박스 깊이 (X축 방향)
int32 cluster_id             # 장애물 클러스터 ID
float32 camera_depth_z       # 카메라 기준 z(depth) 길이
