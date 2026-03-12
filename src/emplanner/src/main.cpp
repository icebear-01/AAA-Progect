/*
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2024-12-26 21:24:20
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2024-12-27 10:44:51
 * @FilePath: /src/planning/src/emplanner/src/main.cpp
 * @Description: 
 * 
 * Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
 */
#include "include/emplanner.hpp"

using namespace std;

int main(int argc, char* argv[])  {
    ros::init(argc, argv, "play_recordPath");
    ros::NodeHandle nh;
    planning_msgs::car_scene car_scene;
    EMPlanner EMPlanner(nh,car_scene);
    ros::Rate rate(10);
    while (ros::ok()) {
        EMPlanner.Plan(car_scene);
        ros::spinOnce();
        rate.sleep();
    }
    ros::shutdown();
    
    return 0;
}