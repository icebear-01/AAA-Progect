#include "include/simulated_data.h"

using namespace std;

int main(int argc, char* argv[]) 
{
    ros::init(argc, argv, "SimulatedData");
    ros::NodeHandle nh;
    Simulated_data simulated_data(nh);
    simulated_data.publish(nh);

    return 0;
}