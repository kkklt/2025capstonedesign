cmake_minimum_required(VERSION 3.0.2)
project(realsense_obstacle_detector)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  pcl_ros
  tf
  geometry_msgs
  visualization_msgs
  message_generation
)

find_package(OpenCV REQUIRED)

add_message_files(
  FILES
  ObstacleInfo.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES realsense_obstacle_detector
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs pcl_ros tf geometry_msgs visualization_msgs message_runtime opencv
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  /usr/include/eigen3
)


#add_executable(obstacle_detector_node src/obstacle_detector_node.cpp)

#add_dependencies(obstacle_detector_node realsense_obstacle_detector_generate_messages_cpp)

#target_link_libraries(obstacle_detector_node
#  ${catkin_LIBRARIES}
#  ${PCL_LIBRARIES}
#  ${OpenCV_LIBS} #
#)

##09/18추가
add_executable(pointcloud_range_filter src/pointcloud_range_filter.cpp)
target_link_libraries(pointcloud_range_filter
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)
