#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs as tf2_sm

TARGET = "base_link"   # OctoMap frame과 동일

def cb(msg):
    try:
        trans = tfbuf.lookup_transform(TARGET, msg.header.frame_id,
                                       rospy.Time(0), rospy.Duration(1.0))
        out = tf2_sm.do_transform_cloud(msg, trans)
        out.header.frame_id = TARGET
        out.header.stamp = rospy.Time.now()   # <-- 추가: 현재 시간으로 스탬프!
        pub.publish(out)
    except Exception as e:
        rospy.logwarn_throttle(1.0, "relay_to_base: %s" % str(e))

if __name__ == "__main__":
    rospy.init_node("relay_to_base")
    tfbuf = tf2_ros.Buffer()
    tflist = tf2_ros.TransformListener(tfbuf)
    pub = rospy.Publisher("/pc_in_base", PointCloud2, queue_size=1)
    
    #rospy.Subscriber("/pc_in_cam", PointCloud2, cb, queue_size=1) 
    #09/18 수정
    rospy.Subscriber("/cloud_filtered_by_range", PointCloud2, cb, queue_size=1)
    rospy.spin()

