import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

def publish_saved_path():
    rospy.init_node('path_republisher', anonymous=True)
    pub = rospy.Publisher('/path_cook', Path, queue_size=10)
    rate = rospy.Rate(1)  # 1Hz

    # ---- 저장된 path 값 ----
    path = Path()
    path.header.frame_id = "world"

    coords = [
        (-206, 660, 500),
        (200, 400.0, 500.0),
        (567.330, -749.240, 619.800)
    ]

    for x, y, z in coords:
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "world"
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = z
        pose_stamped.pose.orientation.w = 1.0
        path.poses.append(pose_stamped)

    rospy.loginfo("Publishing saved /path_cook continuously...")
    while not rospy.is_shutdown():
        path.header.stamp = rospy.Time.now()
        pub.publish(path)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_saved_path()
    except rospy.ROSInterruptException:
        pass
