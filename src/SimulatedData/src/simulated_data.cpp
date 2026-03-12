#include "include/simulated_data.h"

Simulated_data::Simulated_data(ros::NodeHandle& nh)
{
    // TODO: 测试用
    start_floor=3;
    start_task_type=10;
    // scene_chang_task_info_sub = nh.subscribe<planning_msgs::car_scene>("/scene_change_task_info", 100, &Simulated_data::change_scene_task_callBack, this);
    // scene_chang_floor_info_sub = nh.subscribe<planning_msgs::car_scene>("/scene_change_floor_info", 100, &Simulated_data::change_scene_floor_callBack, this); 
    car_scene_pub = nh.advertise<planning_msgs::car_scene>("/car_scene_sim", 100);
    scene_chang_task_info_pub = nh.advertise<planning_msgs::car_scene>("/scene_change_task_info", 100);
}

void Simulated_data::change_scene_task_callBack(const planning_msgs::car_scene::ConstPtr &scene_task_change)
{
    scene.task_type=scene_task_change->task_type;
}

void Simulated_data::change_scene_floor_callBack(const planning_msgs::car_scene::ConstPtr &scene_task_change)
{
    scene.floor=scene_task_change->floor;
}

void Simulated_data::publish(ros::NodeHandle& nh)
{
    ros::Rate rate(10);
    // planning_msgs::car_scene scene;
    scene.floor=1;
    scene.task_type=10;
    int num=0;

    while (ros::ok()) {
        if (scene.task_type==10)
        {
            num++;
        }
        
        if (num<10)
        {
            planning_msgs::car_scene scene_task_info_change;
            scene_task_info_change.task_type=10;
            scene_task_info_change.floor=scene.floor;
            scene_chang_task_info_pub.publish(scene_task_info_change);
            std::cout<<"task_type:"<<10<<std::endl;
        }
        
        else if(num>=10&&num<11)
        {
            planning_msgs::car_scene scene_task_info_change;
            scene_task_info_change.task_type=11;
            scene_task_info_change.floor=scene.floor;
            scene_chang_task_info_pub.publish(scene_task_info_change);
            std::cout<<"task_type:"<<11<<std::endl;
        }
        else 
        {
            planning_msgs::car_scene scene_task_info_change;
            scene_task_info_change.task_type=10;
            scene_task_info_change.floor=scene.floor;
            scene_chang_task_info_pub.publish(scene_task_info_change);
            std::cout<<"task_type:"<<10<<std::endl;
        }
        
        
        car_scene_pub.publish(scene);

        ros::spinOnce();
        rate.sleep();
    }
}

