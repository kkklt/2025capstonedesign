#!/usr/bin/env python3
import rospy
import tf_conversions
import tf2_ros
import geometry_msgs.msg
import math

if __name__ == '__main__':
    rospy.init_node('fake_robot_tf_publisher')

    # tf 정보를 발행할 broadcaster 생성
    broadcaster = tf2_ros.StaticTransformBroadcaster()

    # 1. tool0 -> camera_link (Hand-Eye) 관계는 '고정'이므로 static으로 한 번만 발행
    # 이 값은 사용자가 제공한 값을 그대로 사용합니다.
    static_transformStamped = geometry_msgs.msg.TransformStamped()
    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = "tool0"
    static_transformStamped.child_frame_id = "camera_link"
    static_transformStamped.transform.translation.x = 0.02735643
    static_transformStamped.transform.translation.y = 0.10486931
    static_transformStamped.transform.translation.z = -0.00648899
    
    q = tf_conversions.transformations.quaternion_from_euler(-0.02368569, 0.00894470, 3.12296149)
    static_transformStamped.transform.rotation.x = q[0]
    static_transformStamped.transform.rotation.y = q[1]
    static_transformStamped.transform.rotation.z = q[2]
    static_transformStamped.transform.rotation.w = q[3]

    broadcaster.sendTransform(static_transformStamped)

    # 2. base -> tool0 관계는 '움직임'을 표현해야 하므로, 루프를 돌며 계속 발행
    dynamic_broadcaster = tf2_ros.TransformBroadcaster()
    rate = rospy.Rate(50) # 50Hz로 tf 정보 발행

    angle = 0.0
    while not rospy.is_shutdown():
        # 시간이 지남에 따라 angle 값을 변화시켜 회전 운동 생성
        angle += 0.01

        # 회전 변환 생성
        q_dynamic = tf_conversions.transformations.quaternion_from_euler(0, 0, angle)

        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "base"
        t.child_frame_id = "tool0"

        # 로봇 베이스로부터 0.5m 떨어진 위치에서 Z축을 중심으로 회전
        t.transform.translation.x = 0.5 * math.cos(angle)
        t.transform.translation.y = 0.5 * math.sin(angle)
        t.transform.translation.z = 0.3 # 약간 위로

        t.transform.rotation.x = q_dynamic[0]
        t.transform.rotation.y = q_dynamic[1]
        t.transform.rotation.z = q_dynamic[2]
        t.transform.rotation.w = q_dynamic[3]

        dynamic_broadcaster.sendTransform(t)
        rate.sleep()
