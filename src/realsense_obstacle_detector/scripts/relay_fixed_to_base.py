#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
import tf2_sensor_msgs
from sensor_msgs.msg import PointCloud2

# 이 노드의 역할:
# 1. 고정된 천장 카메라의 포인트 클라우드 토픽('/fixed_camera/pointcloud')을 구독(subscribe)합니다.
# 2. TF(Transform) 정보를 기다립니다. ('base_link'와 'fixed_camera_link' 사이의 관계)
# 3. 수신한 포인트 클라우드 데이터를 'base_link' 좌표계 기준으로 변환합니다.
# 4. 변환된 데이터를 OctoMap 서버가 구독하는 '/pc_in_base' 토픽으로 발행(publish)합니다.

def pointcloud_callback(msg):
    global pub
    if tf_buffer.can_transform('base_link', msg.header.frame_id, msg.header.stamp, timeout=rospy.Duration(1.0)):
        try:
            # TF 변환을 실행합니다.
            transformed_cloud = tf_buffer.transform(msg, 'base_link')
            # 변환된 포인트 클라우드를 발행합니다.
            pub.publish(transformed_cloud)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("TF transform error: {}".format(e))
    else:
        rospy.logwarn("Cannot transform from {} to base_link".format(msg.header.frame_id))

if __name__ == '__main__':
    rospy.init_node('relay_fixed_to_base')

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    # ### 여기가 바로 핵심 변경 부분입니다! ###
    # 기존 '/pc_in_cam' 대신, 우리가 새로 만든 '/fixed_camera/pointcloud' 토픽을 구독합니다.
    sub = rospy.Subscriber('/fixed_camera/pointcloud', PointCloud2, pointcloud_callback, queue_size=1)
    
    # 발행하는 토픽은 기존과 동일하게 '/pc_in_base' 입니다.
    # 이렇게 해야 OctoMap 서버가 두 카메라의 데이터를 모두 받을 수 있습니다.
    pub = rospy.Publisher('/pc_in_base', PointCloud2, queue_size=1)

    rospy.spin()
