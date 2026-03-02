#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h> // pcl::getMinMax3D 사용
#include <visualization_msgs/MarkerArray.h> // MarkerArray 메시지 사용
#include <cstdlib> // rand() 사용
#include <algorithm> // std::min, std::max 사용을 위해 추가
#include <limits> // std::numeric_limits 사용

// OpenCV 관련 헤더 (cv::Matx44f 사용)
#include <opencv2/core/matx.hpp>
#include <opencv2/core/eigen.hpp> // cv::Matx를 Eigen::Matrix로 변환하기 위해

// Eigen 라이브러리 (3D 변환을 위해 PCL과 함께 많이 사용됨)
#include <Eigen/Geometry>

// 커스텀 메시지 헤더와 표준 헤더 추가
#include <realsense_obstacle_detector/ObstacleInfo.h> // <-- 이 줄 추가
#include <std_msgs/Header.h>                         // <-- 이 줄 추가 (ObstacleInfo 메시지 헤더에 필요)

// Point Cloud 타입을 정의 (RGB 정보 포함)
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;

class ObstacleDetector
{
public:
    ObstacleDetector(ros::NodeHandle& nh) : nh_(nh) // <-- 생성자 시그니처 변경 (nh_ 멤버 초기화)
    {
        // ROS Subscriber
        sub_ = nh_.subscribe("/camera/depth/color/points", 1, &ObstacleDetector::pointCloudCallback, this);

        // ROS Publisher (처리된 Point Cloud)
        pub_processed_cloud_ = nh_.advertise<PointCloud>("processed_point_cloud", 1);

        // ROS Publisher (장애물 Bounding Box MarkerArray)
        pub_markers_ = nh_.advertise<visualization_msgs::MarkerArray>("detected_obstacle_boxes", 1);

        // ROS Publisher (커스텀 장애물 정보 메시지)
        obstacle_info_pub_ = nh_.advertise<realsense_obstacle_detector::ObstacleInfo>("detected_obstacle_info", 10); // <-- 이 줄 추가

        // 파라미터 로드
        nh_.param("z_filter_min", z_filter_min_, 0.02); // Z축 최소 필터 (m)
        nh_.param("z_filter_max", z_filter_max_, 0.7); // Z축 최대 필터 (m)
        nh_.param("voxel_leaf_size", voxel_leaf_size_, 0.04); // Voxel Grid 필터 크기 (m)
        nh_.param("plane_distance_threshold", plane_distance_threshold_, 0.03); // 평면 분리 임계값 (m)
        nh_.param("cluster_tolerance", cluster_tolerance_, 0.05); // 클러스터링 거리 허용 오차 (m)
        nh_.param("min_cluster_size", min_cluster_size_, 30); // 최소 클러스터 크기 (점의 개수)
        nh_.param("max_cluster_size", max_cluster_size_, 300000); // 최대 클러스터 크기 (점의 개수)

        ROS_INFO("Obstacle Detector Node Initialized.");
        ROS_INFO("Subscribing to: /camera/depth/color/points");
        ROS_INFO("Publishing to: detected_obstacle_info"); // <-- 추가된 퍼블리셔 정보
        ROS_INFO("Z-filter limits: min=%.2fm, max=%.2fm", z_filter_min_, z_filter_max_);
        ROS_INFO("Voxel leaf size: %.2fm", voxel_leaf_size_);
        ROS_INFO("Plane distance threshold: %.2fm", plane_distance_threshold_);
        ROS_INFO("Cluster tolerance: %.2fm", cluster_tolerance_);
        ROS_INFO("Min/Max cluster size: %d/%d", min_cluster_size_, max_cluster_size_);

        // *************** 캘리브레이션 결과 행렬 정의 ***************
        // campus_pj.cpp에서 얻은 c2g 행렬 값을 여기에 직접 입력합니다.
        // **주의**: 이 행렬의 이동(translation) 부분이 mm 단위라면,
        //        아래에서 m 단위로 변환하기 위해 1000.0으로 나누어주어야 합니다.
        //        여러분의 실제 캘리브레이션 결과에 따라 이 값을 변경해야 합니다.
        cv::Matx44f c2g_cv = {0.9998396867445545, -0.007328346659852211, -0.01633695644188792, -25.46297744346341,
                              0.007276998328315886, 0.9999684011488463, -0.003200312351867875, -121.2978522193079,
                              0.0163598932111673, 0.003080915294658917, 0.9998614213255086, 5.415990194692839,
                               0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00};

        // cv::Matx44f를 Eigen::Matrix4f로 변환
        cv::cv2eigen(c2g_cv, c2g_eigen_mat_);

        // 만약 translation 값이 mm 단위라면, m 단위로 변환 (필요에 따라 주석 처리 또는 수정)
        c2g_eigen_mat_(0, 3) /= 1000.0; // X
        c2g_eigen_mat_(1, 3) /= 1000.0; // Y
        c2g_eigen_mat_(2, 3) /= 1000.0; // Z
        
        ROS_INFO_STREAM("Loaded c2g_eigen_mat_ (m unit):\n" << c2g_eigen_mat_);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher pub_processed_cloud_;
    ros::Publisher pub_markers_; // MarkerArray 발행을 위한 Publisher
    ros::Publisher obstacle_info_pub_; // <-- 이 줄 추가 (커스텀 메시지 퍼블리셔)

    // 캘리브레이션 결과 행렬 (Eigen::Matrix4f 형태로 저장)
    Eigen::Matrix4f c2g_eigen_mat_; 

    // 파라미터 변수
    double z_filter_min_;
    double z_filter_max_;
    double voxel_leaf_size_;
    double plane_distance_threshold_;
    double cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;

    void pointCloudCallback(const PointCloud::ConstPtr& cloud_msg)
    {
        if (cloud_msg->empty())
        {
            ROS_WARN("Received empty point cloud. Skipping processing.");
            return;
        }

        // 1. Z축 필터링 (PassThrough Filter)
        pcl::PassThrough<pcl::PointXYZRGB> pass;
        pass.setInputCloud(cloud_msg);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(z_filter_min_, z_filter_max_);
        PointCloud::Ptr cloud_filtered_z(new PointCloud);
        pass.filter(*cloud_filtered_z);

        if (cloud_filtered_z->empty())
        {
            ROS_WARN("Point cloud empty after Z-filtering. Skipping processing.");
            return;
        }

        // 2. Voxel Grid 필터링 (Downsampling)
        pcl::VoxelGrid<pcl::PointXYZRGB> vg;
        vg.setInputCloud(cloud_filtered_z);
        vg.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
        PointCloud::Ptr cloud_filtered(new PointCloud);
        vg.filter(*cloud_filtered);

        if (cloud_filtered->empty())
        {
            ROS_WARN("Point cloud empty after Voxel Grid filtering. Skipping processing.");
            return;
        }

        // 3. 평면 제거 (Plane Segmentation - RANSAC)
        /*
        pcl::SACSegmentation<pcl::PointXYZRGB> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        PointCloud::Ptr cloud_objects(new PointCloud); 

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE); 
        seg.setMethodType(pcl::SAC_RANSAC);   
        seg.setMaxIterations(1000);           
        seg.setDistanceThreshold(plane_distance_threshold_); 

        seg.setInputCloud(cloud_filtered);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.size() == 0)
        {
            *cloud_objects = *cloud_filtered;
            ROS_WARN("Could not estimate a planar model for the given dataset. Considering all points as objects.");
        }
        else
        {
            pcl::ExtractIndices<pcl::PointXYZRGB> extract;
            extract.setInputCloud(cloud_filtered);
            extract.setIndices(inliers);
            extract.setNegative(true); 
            extract.filter(*cloud_objects);
        }
        
        if (cloud_objects->empty())
        {
            ROS_WARN("Point cloud empty after plane segmentation. Skipping processing.");
            return;
        }
        */

        // 변경: 평면 제거를 건너뛰고 바로 클러스터링 단계로 넘어갑니다.
        // 클러스터링을 위해 기존 필터링된 클라우드를 그대로 사용
        PointCloud::Ptr cloud_objects = cloud_filtered;


        // 4. 클러스터링 (Euclidean Clustering)
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
        tree->setInputCloud(cloud_objects);

        std::vector<pcl::PointIndices> cluster_indices; 
        pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
        ec.setClusterTolerance(cluster_tolerance_); // 클러스터링 거리 허용 오차
        ec.setMinClusterSize(min_cluster_size_);    // 최소 클러스터 크기
        ec.setMaxClusterSize(max_cluster_size_);    // 최대 클러스터 크기
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_objects);
        ec.extract(cluster_indices);

        // 5. 처리된 Point Cloud 발행 (옵션: 평면 제거 후의 객체 클라우드)
        pub_processed_cloud_.publish(*cloud_objects);


        // 6. 클러스터별 Bounding Box 계산 및 가장 가까운 장애물 선택
        visualization_msgs::MarkerArray marker_array;
        marker_array.markers.clear(); 

        std::string robot_frame_id = "base_link"; 
        
        // 가장 가까운 장애물을 찾기 위한 변수 초기화
        bool found_closest_obstacle = false;
        double closest_x = std::numeric_limits<double>::max(); // 가장 작은 X 좌표를 찾기 위해 최대값으로 초기화

        Eigen::Vector4f closest_min_pt_robot_final, closest_max_pt_robot_final; // 로봇 프레임
        int closest_cluster_id = -1; // 가장 가까운 클러스터의 ID

        int current_cluster_id = 0; // 현재 처리 중인 클러스터의 ID (marker.id 용도)

        // ⚠ closest_cluster_cloud 변수를 루프 외부에 선언하여 전체 함수에서 접근 가능하게 함
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr closest_cluster_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        for (const auto& cluster : cluster_indices)
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            for (const auto& idx : cluster.indices)
            {
                cluster_cloud->points.push_back(cloud_objects->points[idx]);
            }
            cluster_cloud->width = cluster_cloud->points.size();
            cluster_cloud->height = 1;
            cluster_cloud->is_dense = true; 

            pcl::PointXYZRGB min_pt_camera, max_pt_camera;
            pcl::getMinMax3D(*cluster_cloud, min_pt_camera, max_pt_camera);

            double box_size_x = max_pt_camera.x - min_pt_camera.x;
            double box_size_y = max_pt_camera.y - min_pt_camera.y;
            double box_size_z = max_pt_camera.z - min_pt_camera.z;

            // 너무 작거나 비정상적으로 큰 마커는 제외 (필요에 따라 조절)
            if (box_size_x < 0.05 || box_size_y < 0.05 || box_size_z < 0.05 ||
                box_size_x > 2.0 || box_size_y > 2.0 || box_size_z > 2.0)
            {
                current_cluster_id++; // 건너뛰는 클러스터도 ID는 증가
                continue; 
            }

            // --- 직접 행렬 변환: 카메라 프레임 -> 로봇 베이스 프레임 ---
            Eigen::Vector4f min_pt_camera_homo(min_pt_camera.x, min_pt_camera.y, min_pt_camera.z, 1.0f);
            Eigen::Vector4f max_pt_camera_homo(max_pt_camera.x, max_pt_camera.y, max_pt_camera.z, 1.0f);

            Eigen::Vector4f min_pt_robot_homo_raw = c2g_eigen_mat_ * min_pt_camera_homo;
            Eigen::Vector4f max_pt_robot_homo_raw = c2g_eigen_mat_ * max_pt_camera_homo;
            
            Eigen::Vector4f current_min_pt_robot_final;
            Eigen::Vector4f current_max_pt_robot_final;

            current_min_pt_robot_final.x() = std::min(min_pt_robot_homo_raw.x(), max_pt_robot_homo_raw.x());
            current_max_pt_robot_final.x() = std::max(min_pt_robot_homo_raw.x(), max_pt_robot_homo_raw.x());

            current_min_pt_robot_final.y() = std::min(min_pt_robot_homo_raw.y(), max_pt_robot_homo_raw.y());
            current_max_pt_robot_final.y() = std::max(min_pt_robot_homo_raw.y(), max_pt_robot_homo_raw.y());

            current_min_pt_robot_final.z() = std::min(min_pt_robot_homo_raw.z(), max_pt_robot_homo_raw.z());
            current_max_pt_robot_final.z() = std::max(min_pt_robot_homo_raw.z(), max_pt_robot_homo_raw.z());

            current_min_pt_robot_final.w() = 1.0f;
            current_max_pt_robot_final.w() = 1.0f;
            
            // ************ 가장 가까운 장애물 선택 로직 ************
            // 로봇의 정면(X축)에 가장 가까운 장애물을 선택 (X 좌표가 가장 작은 것)
            if (current_min_pt_robot_final.x() < closest_x)
            {
                closest_x = current_min_pt_robot_final.x();
                closest_min_pt_robot_final = current_min_pt_robot_final;
                closest_max_pt_robot_final = current_max_pt_robot_final;
                closest_cluster_id = current_cluster_id;
                found_closest_obstacle = true;
                // ⚠ 가장 가까운 클러스터의 데이터를 저장합니다.
                *closest_cluster_cloud = *cluster_cloud; 
            }
            
            current_cluster_id++; // 다음 클러스터를 위해 ID 증가
        }

        // 모든 클러스터를 확인한 후, 가장 가까운 장애물만 출력하고 마커를 발행합니다.
        if (found_closest_obstacle)
        {
            // ⚠ 이곳에서 min_pt_camera를 다시 계산하면 closest_cluster_cloud가 유효함
            pcl::PointXYZRGB min_pt_camera, max_pt_camera;
            pcl::getMinMax3D(*closest_cluster_cloud, min_pt_camera, max_pt_camera);
            
            ROS_INFO("--- Closest Obstacle Bounding Box (Robot Frame - %s) ---", robot_frame_id.c_str());
            ROS_INFO("  Cluster ID: %d", closest_cluster_id);
            ROS_INFO("  Min Point: (X=%.4f, Y=%.4f, Z=%.4f)", closest_min_pt_robot_final.x(), closest_min_pt_robot_final.y(), closest_min_pt_robot_final.z());
            ROS_INFO("  Max Point: (X=%.4f, Y=%.4f, Z=%.4f)", closest_max_pt_robot_final.x(), closest_max_pt_robot_final.y(), closest_max_pt_robot_final.z());
            
            // ************ ObstacleInfo 메시지 생성 및 발행 ************
            realsense_obstacle_detector::ObstacleInfo obstacle_msg; // <-- 커스텀 메시지 객체 생성

            // 1. 헤더 정보 채우기
            obstacle_msg.header.stamp = ros::Time::now(); // 현재 시간 스탬프
            obstacle_msg.header.frame_id = "base_link";   // 이 정보는 'base_link' 프레임 기준임을 명시

            // 2. min_point, max_point 좌표 채우기
            // geometry_msgs::Point 타입이므로 x,y,z 멤버 사용
            obstacle_msg.min_point.x = closest_min_pt_robot_final.x();
            obstacle_msg.min_point.y = closest_min_pt_robot_final.y();
            obstacle_msg.min_point.z = closest_min_pt_robot_final.z();

            obstacle_msg.max_point.x = closest_max_pt_robot_final.x();
            obstacle_msg.max_point.y = closest_max_pt_robot_final.y();
            obstacle_msg.max_point.z = closest_max_pt_robot_final.z();

            // 3. 바운딩 박스 크기 (width, height, depth) 계산 및 채우기
            // 로봇의 base_link 프레임 기준:
            // X축: 로봇의 앞/뒤 (depth)
            // Y축: 로봇의 좌/우 (width)
            // Z축: 로봇의 상/하 (height)
            obstacle_msg.width = closest_max_pt_robot_final.y() - closest_min_pt_robot_final.y();
            obstacle_msg.height = closest_max_pt_robot_final.z() - closest_min_pt_robot_final.z();
            obstacle_msg.depth = closest_max_pt_robot_final.x() - closest_min_pt_robot_final.x();

            // 4. 클러스터 ID 채우기
            obstacle_msg.cluster_id = closest_cluster_id;
            
            // 5. 카메라 기준의 깊이 값 채우기 (새로 추가)
            obstacle_msg.camera_depth_z = min_pt_camera.z; 

            // 6. 메시지 발행 
            obstacle_info_pub_.publish(obstacle_msg); // <-- 커스텀 메시지 발행

            // PRM 알고리즘에 전달할 데이터: closest_min_pt_robot_final, closest_max_pt_robot_final
            // (이미 obstacle_msg로 발행하고 있으므로, PRM은 이 메시지를 구독하면 됩니다.)

            // Marker 메시지 생성 (가장 가까운 Bounding Box)
            visualization_msgs::Marker marker;
            marker.header.frame_id = robot_frame_id;
            marker.header.stamp = ros::Time::now();
            marker.ns = "closest_obstacle"; // 이름 공간을 'closest_obstacle'로 변경하여 다른 마커와 구분
            marker.id = closest_cluster_id;
            marker.type = visualization_msgs::Marker::CUBE;
            marker.action = visualization_msgs::Marker::ADD;

            marker.pose.position.x = (closest_min_pt_robot_final.x() + closest_max_pt_robot_final.x()) / 2.0;
            marker.pose.position.y = (closest_min_pt_robot_final.y() + closest_max_pt_robot_final.y()) / 2.0;
            marker.pose.position.z = (closest_min_pt_robot_final.z() + closest_max_pt_robot_final.z()) / 2.0;

            marker.scale.x = closest_max_pt_robot_final.x() - closest_min_pt_robot_final.x();
            marker.scale.y = closest_max_pt_robot_final.y() - closest_min_pt_robot_final.y();
            marker.scale.z = closest_max_pt_robot_final.z() - closest_min_pt_robot_final.z();

            marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0; marker.color.a = 0.8; // 가장 가까운 장애물은 눈에 띄게 빨간색으로
            marker.lifetime = ros::Duration(0.5);
            marker_array.markers.push_back(marker);

            pub_markers_.publish(marker_array);
        }
        else
        {
            // 가장 가까운 장애물이 없으면 모든 마커를 지웁니다.
            visualization_msgs::Marker marker;
            marker.header.frame_id = robot_frame_id;
            marker.header.stamp = ros::Time::now();
            marker.ns = "closest_obstacle";
            marker.id = 0; // 특정 ID만 삭제하거나, 모든 ID를 삭제
            marker.action = visualization_msgs::Marker::DELETEALL; // 모든 마커 삭제
            marker_array.markers.push_back(marker);
            pub_markers_.publish(marker_array);
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "obstacle_detector_node");
    ros::NodeHandle nh; // <-- NodeHandle 인스턴스 생성
    ObstacleDetector od(nh); // <-- 생성자에 NodeHandle 인스턴스 전달
    ros::spin();
    return 0;
}
