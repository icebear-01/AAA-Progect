/*
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2024-07-23 20:09:29
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2024-07-24 09:01:20
 * @FilePath: /src/planning/src/path1/path_optimization/include/path_optimization.hpp
 * @Description: 
 * 
 * Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
 */

#ifndef __OPTIMIZATION_PATH_HPP
#define __OPTIMIZATION_PATH_HPP
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "yaml-cpp/yaml.h"
#include <pthread.h>
#include <limits.h>
#include <cfloat>
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <message_filters/subscriber.h>
#include <sensor_msgs/PointCloud2.h>
#include <vector>
#include <cmath>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <geometry_msgs/Pose2D.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <ros/package.h>
#include "OsqpEigen/OsqpEigen.h"
//using namespace std;
const std::string record_path = ros::package::getPath("record_path");
const std::string play_path = ros::package::getPath("record_path");
// const std::string yaml_path1 =ugv_path+"/config.yaml";
// const YAML::Node config_path = YAML::LoadFile(yaml_path1);
const std::string raw_FILE_NAME = record_path + "/path/elevator_path.txt"; // 文件名称
// const std::string raw_FILE_NAME = "/home/wmd/P2P_fast_try718/P2P_fast/src/planning/src/path1/Record_Path/path/raw_trajectory.txt"; // 文件名称
const std::string optimized_FILE_NAME = play_path+"/path/elevator_path_op.txt"; // 文件名称
pcl::PointCloud<pcl::PointXYZI>::Ptr opt_line_record(new pcl::PointCloud<pcl::PointXYZI>);
int opt_total_points_num; //原始数据中共多少点

class TrajectoryOptimization {
public:
    
    std::fstream& file_;
    ros::Publisher  data_pub;
    TrajectoryOptimization(std::fstream& file,ros::NodeHandle &nh);
    Eigen::VectorXd Smooth_Reference_Line(pcl::PointCloud<pcl::PointXYZI>::Ptr opt_line_record);
    ~TrajectoryOptimization();
};


#endif