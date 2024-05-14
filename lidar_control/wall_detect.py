#! /usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

import math

class LidarExample():
    def __init__(self):
        # ROS part
        
        rospy.init_node("obstacle_detect")
        rospy.Subscriber("scan",LaserScan,self.scan_callback)
        self.pub_obstacle = rospy.Publisher("/lidar_obstacle",String,queue_size=1)
        self.pub_left_wall = rospy.Publisher('/left_wall', String, queue_size=1)
        self.pub_front_wall = rospy.Publisher('/front_wall', String, queue_size=1)
        self.pub_right_wall = rospy.Publisher('/right_wall', String, queue_size=1)
        self.MAX_ANGLE_DEG = 180
        self.MIN_ANGLE_DEG = -180
        #self.MAX_OBST_ANGLE_DEG = 10
        #self.MIN_OBST_ANGLE_DEG = -10

# ==================================================
#                 Callback Functions
# ==================================================
    def scan_callback(self, data):
        left_wall = []
        front_wall = []
        right_wall = []
        
        for i,n in enumerate(data.ranges):
            angle = data.angle_min + data.angle_increment * i
            angle_deg = angle * 180 / math.pi
            
            x = n * math.cos(angle)
            y = n * math.sin(angle)
            
            # if angle_deg < self.MAX_ANGLE_DEG and angle_deg > self.MIN_ANGLE_DEG and not n == 0:
            #     print("lidar angle and range : ({},{})".format(angle, n))
            #     print("lidar x and y : ({},{})".format(x, y))
            if y > -0.15 and y < 0.15 and x < 0.5 and x > 0:
                right_wall.append((x,y))
        if len(right_wall) > 5:
            self.pub_front_wall.publish("on")
        else:
            self.pub_front_wall.publish("off")


def run():
    new_class = LidarExample()
    rospy.loginfo_once("ROS Node Initialized")
    rospy.spin()
            
if __name__=="__main__":
    run()
