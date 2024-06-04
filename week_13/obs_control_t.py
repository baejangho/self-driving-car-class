#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 위의 코드는 ROS(로봇 운영 체제)에서 동작하는 Python 클래스 예제입니다.
# 이 예제를 통해 학생들은 LIDAR(Laser Range Finder) 데이터를 활용하여 로봇을 제어하는 방법을 배울 수 있습니다.

# rospy 라이브러리와 필요한 메시지 유형(sensor_msgs.msg,geometry_msgs.msg) 및 수학 라이브러리를 가져옵니다.
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
from std_msgs.msg import Bool
from math import *

# Class_Name 클래스 정의: ROS 노드를 클래스로 정의하여 코드를 구조화합니다.
class OBS_Control:  # 1단계: 클래스 이름 정의
    def __init__(self):  # 2단계: 클래스 초기화 및 초기 설정
        # ROS 노드를 초기화합니다. 노드 이름은 "wego_sub_node"로 지정됩니다.
        rospy.init_node("control_node")  # ROS 1단계(필수): 노드 이름 정의

        # ROS 서브스크라이버(Subscriber)를 설정합니다.
        # "/scan" 토픽에서 LaserScan 메시지를 구독하고, 콜백 함수(lidar_CB)를 호출합니다.
        rospy.Subscriber("/dist90R", Int32, self.dist90R_CB)  # ROS 2단계: 노드 역할 - 서브스크라이버 설정
        rospy.Subscriber("/dist90L", Int32, self.dist90L_CB)  # ROS 2단계: 노드 역할 - 서브스크라이버 설정
        rospy.Subscriber("/dist45R", Int32, self.dist45R_CB)  # ROS 2단계: 노드 역할 - 서브스크라이버 설정
        rospy.Subscriber("/dist45L", Int32, self.dist45L_CB)  # ROS 2단계: 노드 역할 - 서브스크라이버 설정
        rospy.Subscriber("/obsR", Bool, self.obsR_CB)  # ROS 2단계: 노드 역할 - 서브스크라이버 설정
        rospy.Subscriber("/obsL", Bool, self.obsL_CB)  # ROS 2단계: 노드 역할 - 서브스크라이버 설정
        rospy.Subscriber("/obsC", Bool, self.obsC_CB)  # ROS 2단계: 노드 역할 - 서브스크라이버 설정

        # ROS 퍼블리셔(Publisher)를 설정합니다.
        # "/cmd_vel" 토픽에 Twist 메시지를 발행합니다.
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)  # ROS 2단계: 노드 역할 - 퍼블리셔 설정

        # 메시지 타입 설정 및 초기화
        self.cmd_msg = Twist()
        self.obsC = False
        self.obsR = False
        self.obsL = False
        self.dist90R = 0.5
        self.dist90L = 0.5
        self.dist45R = 0.5
        self.dist45L = 0.5
        self.wall_dist = 0.2
        self.kp = 10

        # ROS 퍼블리셔 주기 설정
        self.rate = rospy.Rate(10)  # ROS 2-1단계(옵션): 퍼블리셔 - 주기 설정

    def dist90R_CB(self, data):
        self.dist90R = data  # 메세지를 self.dist90R에 저장
    def dist90L_CB(self, data):
        self.dist90L = data  # 메세지를 self.dist90L에 저장
    def dist45R_CB(self, data):
        self.dist45R = data  # 메세지를 self.dist45R에 저장
    def dist45L_CB(self, data):
        self.dist45L = data  # 메세지를 self.dist45L에 저장
    def obsR_CB(self, data):
        self.obsR = data  # 메세지를 self.obsR에 저장
    def obsL_CB(self, data):        
        self.obsL = data  # 메세지를 self.obsL에 저장
    def obsC_CB(self, data):    
        self.obsC = data  # 메세지를 self.obsC에 저장

    def control(self):  # 제어 함수
        if self.obsC and self.obsR and self.obsL: # Stop
            speed = 0
            steer = 0
        elif not self.obsC and self.obsR: # following the right wall
            speed = 0.2
            theta = acos(self.dist45R / self.dist90R)*180/pi
            steer = (45-theta) + self.kp*(self.wall_dist - self.dist90R)
        elif self.obsC and self.obsR: # turn left
            speed = 0.1
            steer = 20
        else:
            print('조건 없음')
            speed = 0
            steer = 0
        
        self.cmd_msg.linear.x = speed
        self.cmd_msg.angular.z = steer
        self.pub.publish(self.cmd_msg)

# 메인 함수 정의: ROS 노드를 실행하기 위한 메인 함수입니다.
def main():  # 4단계: 메인 함수 정의
    OBS_driving = OBS_Control()  # 클래스(Class_Name)를 변수(class_name)에 저장
    while not rospy.is_shutdown():
        OBS_driving.control()


# 직접 실행 코드: 스크립트가 직접 실행될 때 main() 함수를 호출합니다.
if __name__ == "__main__":  # 5단계: 직접 실행 구문 정의
    main()
