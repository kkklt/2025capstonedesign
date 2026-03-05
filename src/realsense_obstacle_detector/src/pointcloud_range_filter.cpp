#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h> // PassThrough 필터를 사용합니다.

// PCL 포인트 클라우드 타입을 PointCloudXYZ로 정의합니다. (RGB 정보는 필터링에 필요 없으므로 제외)
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;

class PointCloudRangeFilter
{
public:
    PointCloudRangeFilter()
    {
        // ROS 노드 핸들을 생성합니다.
        ros::NodeHandle nh;

        // 필터링된 포인트 클라우드를 발행할 Publisher를 설정합니다.
        // 토픽 이름은 "/cloud_filtered_by_range" 입니다.
        pub_ = nh.advertise<PointCloudXYZ>("/cloud_filtered_by_range", 1);

        // Realsense 카메라의 원본 포인트 클라우드를 구독할 Subscriber를 설정합니다.
        // 원본 토픽 이름은 "/camera/depth/color/points" 입니다.
        sub_ = nh.subscribe("/camera/depth/color/points", 1, &PointCloudRangeFilter::cloudCallback, this);

        ROS_INFO("Point Cloud Range Filter node started. Listening for topics...");
    }

    void cloudCallback(const PointCloudXYZ::ConstPtr& input_cloud)
    {
        // 1. Z축 필터링 (카메라 정면 방향으로 50cm)
        pcl::PassThrough<pcl::PointXYZ> pass_z;
        pass_z.setInputCloud(input_cloud);
        pass_z.setFilterFieldName("z");       // Z축 (카메라 앞뒤 방향)
        pass_z.setFilterLimits(0.0, 0.5);     // 0m ~ 0.5m 사이의 점만 남김
        
        PointCloudXYZ::Ptr cloud_filtered_z(new PointCloudXYZ());
        pass_z.filter(*cloud_filtered_z);

        // 2. X축 필터링 (카메라 좌우 방향으로 50cm)
        pcl::PassThrough<pcl::PointXYZ> pass_x;
        pass_x.setInputCloud(cloud_filtered_z);
        pass_x.setFilterFieldName("x");       // X축 (카메라 좌우 방향)
        pass_x.setFilterLimits(-0.25, 0.25);  // -25cm ~ +25cm 사이의 점만 남김

        PointCloudXYZ::Ptr cloud_filtered_zx(new PointCloudXYZ());
        pass_x.filter(*cloud_filtered_zx);

        // 3. Y축 필터링 (카메라 상하 방향으로 50cm)
        pcl::PassThrough<pcl::PointXYZ> pass_y;
        pass_y.setInputCloud(cloud_filtered_zx);
        pass_y.setFilterFieldName("y");       // Y축 (카메라 상하 방향)
        pass_y.setFilterLimits(-0.25, 0.25);  // -25cm ~ +25cm 사이의 점만 남김

        PointCloudXYZ::Ptr final_cloud(new PointCloudXYZ());
        pass_y.filter(*final_cloud);

        // 최종 필터링된 포인트 클라우드의 헤더 정보를 채우고 발행합니다.
        // 타임스탬프와 프레임 ID는 원본 데이터의 것을 그대로 사용합니다.
        final_cloud->header = input_cloud->header;
        pub_.publish(final_cloud);
    }

private:
    ros::Publisher pub_;
    ros::Subscriber sub_;
};

int main(int argc, char** argv)
{
    // ROS를 초기화하고 노드를 생성합니다.
    ros::init(argc, argv, "pointcloud_range_filter");
    PointCloudRangeFilter filter;
    ros::spin(); // 콜백 함수가 계속 호출되도록 대기합니다.
    return 0;
}
