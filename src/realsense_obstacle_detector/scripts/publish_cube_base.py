#!/usr/bin/env python3
import rospy, numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

def make_cube(frame="base_link"):
    xs = np.linspace(0.5, 1.5, 60)
    ys = np.linspace(-0.4, 0.4, 60)
    zs = np.linspace(0.0, 1.0, 60)
    pts = np.array(np.meshgrid(xs, ys, zs)).reshape(3,-1).T
    header = Header(stamp=rospy.Time.now(), frame_id=frame)
    fields = [PointField('x',0,PointField.FLOAT32,1),
              PointField('y',4,PointField.FLOAT32,1),
              PointField('z',8,PointField.FLOAT32,1)]
    return pc2.create_cloud(header, fields, pts.astype(np.float32))

if __name__=="__main__":
    rospy.init_node("publish_cube_base")
    pub = rospy.Publisher("/pc_in_base", PointCloud2, queue_size=1)
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        pc = make_cube("base_link")
        pub.publish(pc)
        rate.sleep()
