#ifndef __SIMDATA_H  
#define __SIMDATA_H
#include <ros/ros.h>
#include <iostream>
#include <string>

#include <planning_msgs/car_scene.h>

class Simulated_data {
private:
    planning_msgs::car_scene scene;
    ros::Publisher car_scene_pub;
    ros::Publisher scene_chang_task_info_pub;
    
    ros::Subscriber  scene_chang_task_info_sub ; 
    ros::Subscriber  scene_chang_floor_info_sub ; 

    void change_scene_task_callBack(const planning_msgs::car_scene::ConstPtr &scene_task_change);
    void change_scene_floor_callBack(const planning_msgs::car_scene::ConstPtr &scene_task_change);
    
public:
    Simulated_data(ros::NodeHandle& nh);
    void publish(ros::NodeHandle& nh);
    int last_task_type; // 记录上一次的任务类型
    int start_floor;  // 车辆起始点
    int start_task_type;  // 车辆起始点
};

#endif