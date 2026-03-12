#ifndef __RTKPLANNER_H
#define __RTKPLANNER_H

#include <ros/ros.h>
#include <iostream>
#include <fstream>  // 用于读取文件
#include <vector>
#include <sstream>
#include <string>
#include <ros/package.h>
#include <planning_msgs/hybrid_astar_path_point.h>
#include <planning_msgs/hybrid_astar_path.h>
#include <planning_msgs/hybrid_astar_paths.h>
#include <planning_msgs/car_path.h>
#include <planning_msgs/path_point.h>
#include <planning_msgs/hybrid_astar_paths.h>
#include <planning_msgs/car_scene.h>
#include <planning_msgs/car_info.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Pose.h>  // 用于表示单个点
#include <geometry_msgs/Pose2D.h>

class RTK_planner
{
private:
    ros::Publisher RawRoute_pub;  // 未优化路线发布器
    ros::Publisher PatrolRoute_pub;  // 优化rtk路线发布器
    ros::Publisher RawRoute_pub_watch;  // 未优化路线发布器

    ros::Subscriber sub_location;

    
    std::shared_ptr<planning_msgs::car_scene> car_scene_; 
    pcl::PointCloud<pcl::PointXYZI>::Ptr rtk_path_watch; // qp——path rviz观测值
    sensor_msgs::PointCloud2 ss_local_path_watch;
    int last_task_type;
    planning_msgs::hybrid_astar_paths route;  // 注意这里使用的是 'planning_msgs::hybrid_astar_paths'
    bool SwitchScene(planning_msgs::car_scene &car_scene);
    bool loadRoute(planning_msgs::car_scene car_scene);  // 加载路线数据的函数，floor 用于选择文件夹
    bool loadElevatorPath(planning_msgs::car_scene car_scene);
    void publishRoute();  // 发布路线
    void callBack_location(const planning_msgs::car_info::ConstPtr &car_pose);
    
    
public:

    RTK_planner(ros::NodeHandle& nh,const planning_msgs::car_scene &car_scene);
    geometry_msgs::Pose2D::Ptr Car_Pose; // 车辆位姿信息(车前轴的坐标)
    
    void Plan(planning_msgs::car_scene &car_scene);
    
    std::string base_path = ros::package::getPath("planner") + "/text/F";  // 基础路径
    ~RTK_planner();
};

#endif
