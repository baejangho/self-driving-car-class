#!/usr/bin/env python
# -*- coding:utf-8 -*-

import rospy
from std_msgs.msg import Int32
import math
from geometry_msgs.msg import Twist

class move_limo:
    def __init__(self):
        rospy.init_node('control')
        self.drive_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        # 메시지 발행 주기를 설정합니다. 이 예제에서는 10Hz로 설정합니다.
        self.rate = rospy.Rate(10)  # ROS 2-1단계(옵션): 발행 주기 설정

    def drive_control(self, event):

            try:
                #rospy.loginfo("off_center, lateral_gain = {}, {}".format(self.off_center, self.LATERAL_GAIN))
                self.speed = 0.1
                self.angle = 0.1
                drive = Twist()
                drive.linear.x = self.speed
                drive.angular.z = self.angle
                self.drive_pub.publish(drive)
                
                # 지정한 발행 주기에 따라 슬립합니다. 이것은 메시지를 일정한 주기로 발행하기 위해 사용됩니다.
                self.rate.sleep()  # ROS 3-1단계(옵션): 퍼블리셔 - 주기 실행
            except Exception as e:
                print("error:",e)
                                 

if __name__ == '__main__':
    MoveCar = move_limo()
    try:
        while not rospy.is_shutdown():
            MoveCar.drive_control()
    except KeyboardInterrupt:
        print("program down")
