/*
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2024-12-26 21:37:36
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2024-12-27 16:25:10
 * @FilePath: /src/planning/src/planner/include/planner.h
 * @Description: 
 * 
 * Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
 */
#ifndef __PLANNER_H  
#define __PLANNER_H
#include <ros/ros.h>
#include <iostream>
#include <string>
#include "include/emplanner.hpp"
#include "hybrid_a_star/hybrid_a_star_flow.h"
#include "include/rtk_planner.h"
#include <planning_msgs/car_scene.h>

class Planner {
private:
    planning_msgs::car_scene scene;
    int target_floor_number=1;
    ros::Publisher car_scene_pub;
    ros::Subscriber scene_chang_task_info_sub;
    ros::Subscriber scene_chang_floor_info_sub;
    void change_scene_task_callBack(const planning_msgs::car_scene::ConstPtr &scene_task_change);
    void change_scene_floor_callBack(const planning_msgs::car_scene::ConstPtr &scene_task_change);

public:
    Planner(ros::NodeHandle& nh);
    void plan(ros::NodeHandle& nh);
    int last_task_type; // 记录上一次的任务类型
    int start_floor;  // 车辆起始点
    int start_task_type;  // 车辆起始点
};

#endif