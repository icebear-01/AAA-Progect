#include"include/rtk_planner.h"

using namespace std;

int main(int argc, char* argv[]) 
{
    ros::init(argc, argv, "rtk_planner");
    ros::NodeHandle nh;
    planning_msgs::car_scene scene;  //测试用
    
    RTK_planner rtk_planner(nh,scene);
    rtk_planner.Plan(scene);
    
    ros::shutdown();
    return 0;
}