#include "include/planner.h"

using namespace std;

int main(int argc, char* argv[]) 
{
    ros::init(argc, argv, "planner");
    ros::NodeHandle nh;
    Planner planner(nh);
    planner.plan(nh);

    ros::shutdown();
    return 0;
}