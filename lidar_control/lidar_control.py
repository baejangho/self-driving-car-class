#!/usr/bin/env python
# -*- coding:utf-8 -*-

import rospy
from std_msgs.msg import String
import math
from geometry_msgs.msg import Twist

class move_limo:
    def __init__(self):
        rospy.init_node('mission_control')

        self.BASE_SPEED = 0.2

        self.front_wall = rospy.Subscriber("/front_wall", String, self.front_wall_sub)
        self.drive_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)

        self.rate = rospy.Rate(10)  # ROS 2-1단계(옵션): 발행 주기 설정

    def front_wall_sub(self, data):
        self.front_wall_state = data.data
        print('front wall :', self.front_wall_state)
        
    def mission_control(self):

            try:
                drive = Twist()
                if self.obstacle_detect:
                    drive.linear.x = 0.0
                    drive.angular.z = 0.0
                    #print('stop')
                else:
                    drive.linear.x = 0.0
                    drive.angular.z = 0.0
                    #print('go')
                self.drive_pub.publish(drive)
                
                self.rate.sleep()
            except:
                pass 
                

    def detect_obstacle(self, data):
        if data.data == "obstacle":
            self.obstacle_detect = True
        else:
            self.obstacle_detect = False
        print(self.obstacle_detect)
    
    def obstacle_list(self, data):
        self.obs_list = list(data.data)
        print(self.obs_list)
        
            



if __name__ == '__main__':
    MoveCar = move_limo()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("program down")
