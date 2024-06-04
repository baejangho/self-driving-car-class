#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 위의 코드는 ROS(로봇 운영 체제)에서 동작하는 Python 클래스 예제입니다.
# 이 예제를 통해 학생들은 LIDAR(Laser Range Finder) 데이터를 활용하여 로봇을 제어하는 방법을 배울 수 있습니다.

# rospy 라이브러리와 필요한 메시지 유형(sensor_msgs.msg,geometry_msgs.msg) 및 수학 라이브러리를 가져옵니다.
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from std_msgs.msg import Bool
import math

# Class_Name 클래스 정의: ROS 노드를 클래스로 정의하여 코드를 구조화합니다.
class OBS_Control:  # 1단계: 클래스 이름 정의
    def __init__(self):  # 2단계: 클래스 초기화 및 초기 설정
        # ROS 노드를 초기화합니다. 노드 이름은 "wego_sub_node"로 지정됩니다.
        rospy.init_node("control_node")  # ROS 1단계(필수): 노드 이름 정의

        # ROS 서브스크라이버(Subscriber)를 설정합니다.
        # "/scan" 토픽에서 LaserScan 메시지를 구독하고, 콜백 함수(lidar_CB)를 호출합니다.
        rospy.Subscriber("/dist90R", Float32, self.dist90R_CB)  # ROS 2단계: 노드 역할 - 서브스크라이버 설정
        rospy.Subscriber("/dist75R", Float32, self.dist75R_CB)  # ROS 2단계: 노드 역할 - 서브스크라이버 설정
        rospy.Subscriber("/dist90L", Float32, self.dist90L_CB)  # ROS 2단계: 노드 역할 - 서브스크라이버 설정
        rospy.Subscriber("/dist75L", Float32, self.dist75L_CB)  # ROS 2단계: 노드 역할 - 서브스크라이버 설정
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
        self.dist75R = 0.5
        self.dist75L = 0.5
        self.wall_dist = 0.2
        self.kp = 0.01
        self.pre_steer = 0

        # ROS 퍼블리셔 주기 설정
        self.rate = rospy.Rate(2)  # ROS 2-1단계(옵션): 퍼블리셔 - 주기 설정

    def dist90R_CB(self, data):
        self.dist90R = data.data  # 메세지를 self.dist90R에 저장
        # print('dist90R:',self.dist90R)
    def dist75R_CB(self, data):
        self.dist75R = data.data  # 메세지를 self.dist75R에 저장
    def dist90L_CB(self, data):
        self.dist90L = data.data  # 메세지를 self.dist90L에 저장
    def dist75L_CB(self, data):
        self.dist75L = data.data  # 메세지를 self.dist75L에 저장
    def obsR_CB(self, data):
        self.obsR = data.data  # 메세지를 self.obsR에 저장
        # print('obsR:',self.obsR)
    def obsL_CB(self, data):        
        self.obsL = data.data  # 메세지를 self.obsL에 저장
        # print('obsL:',self.obsL)
    def obsC_CB(self, data):    
        self.obsC = data.data  # 메세지를 self.obsC에 저장
        # print('obsC:',self.obsC)

    def control(self):  # 제어 함수
        if not self.obsC and self.obsR: # following the right wall
            speed = 0.2
            if self.dist90R == 0:
                speed = 0.0
                steer = self.pre_steer
                print('dist90R = 0')
                pass
            else:
                value = self.dist90R /self.dist75R
                # print('acos value:',value)
                if value > 1:
                    value = 1
                elif value < -1:
                    value = -1

                theta = math.acos(value)*180/math.pi
                print('theta:',theta)
                steer = (15-theta)*math.pi/180 + self.kp*(self.wall_dist - self.dist90R)
                print('steer:',steer)
            print('wall')
        elif self.obsC and self.obsR: # turn left
            speed = 0.1
            steer = 0.1
            print('turn left')
        elif self.obsC and self.obsR and self.obsL: # Stop
            speed = 0
            steer = 0
            print('stop')
        else:
            print('조건 없음')
            speed = 0
            steer = 0
        
        self.cmd_msg.linear.x = speed
        self.cmd_msg.angular.z = steer
        self.pub.publish(self.cmd_msg)
        self.pre_steer = steer

# 메인 함수 정의: ROS 노드를 실행하기 위한 메인 함수입니다.
def main():  # 4단계: 메인 함수 정의
    OBS_driving = OBS_Control()  # 클래스(Class_Name)를 변수(class_name)에 저장
    while not rospy.is_shutdown():
        OBS_driving.control()
        OBS_driving.rate.sleep()


# 직접 실행 코드: 스크립트가 직접 실행될 때 main() 함수를 호출합니다.
if __name__ == "__main__":  # 5단계: 직접 실행 구문 정의
    main()
