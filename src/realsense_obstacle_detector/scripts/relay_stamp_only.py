#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2

def cb(msg):
    msg.header.stamp = rospy.Time.now()
    pub.publish(msg)

if __name__ == "__main__":
    rospy.init_node("relay_stamp_only")
    pub = rospy.Publisher("/pc_in_cam", PointCloud2, queue_size=1)
    rospy.Subscriber("/camera/depth/color/points", PointCloud2, cb, queue_size=1)
    rospy.spin()
