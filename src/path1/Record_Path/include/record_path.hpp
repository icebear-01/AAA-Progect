/*
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2024-06-25 10:39:52
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2024-12-14 19:54:25
 * @FilePath: /src/planning/src/path1/Record_Path/include/record_path.hpp
 * @Description: 
 * 
 * Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
 */
#ifndef __RECORD_PATH_HPP
#define __RECORD_PATH_HPP
#include <iostream>

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "yaml-cpp/yaml.h"
#include <pthread.h>
#include <cfloat>
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <sensor_msgs/PointCloud2.h>
#include <vector>
#include <cmath>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
// #include <geometry_msgs/Pose2D.h>
#include <planning_msgs/car_info.h>
#include <fstream>
#include <sstream>

#include <sys/types.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <thread>
#include <ros/package.h>
//using namespace std;
const std::string record_path = ros::package::getPath("record_path");
// const std::string yaml_path1 =ugv_path+"/config.yaml";
// const YAML::Node config_path = YAML::LoadFile(yaml_path1);
const std::string FILE_NAME = record_path+"/path/elevator_path.txt"; // 文件名称

// fstream file(FILE_NAME.c_str());
static float wheel_speedometer_speed;
static float Steering_wheel_turn_angle = 22;
static float Using_wheel_speedometer_speed;
static float Using_wheel_turn_angle;

static float get_car_speed;
static float get_car_turn;
static float set_car_speed;
static float set_car_turn;

int init_can0();
int can0_get_status(int can0_fd);
static float L1=1.15;
class TrajectoryRecorder {
public:
    std::fstream& file_;
    ros::Publisher  line_pub;
    ros::Subscriber sub_pos;
    void callBack_Pos(const planning_msgs::car_info::ConstPtr &car_pose);
    TrajectoryRecorder(std::fstream& file,ros::NodeHandle &nh);
    ~TrajectoryRecorder();
};

#endif