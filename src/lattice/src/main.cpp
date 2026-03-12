#include<lattice/lattice_planner.h>

int main(int argc, char *argv[])
{
    ros::init(argc,argv,"lattice_planner");
    ros::NodeHandle nh;

    lattice_planner Lattice_plan_pub(nh);

    return 0;
}
