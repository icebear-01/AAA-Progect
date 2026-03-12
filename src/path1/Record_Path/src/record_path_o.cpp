/**
 * @FilePath     : /Record_Path/src/record_path.cpp
 * @Description  :  
 * @Author       : WMD
 * @Version      : 0.0.1
 * @LastEditors  : WMD email:732485622@qq.com
 * @LastEditTime : 2023-12-14 16:12:13
 * @Copyright    : G AUTOMOBILE RESEARCH INSTITUTE CO.,LTD Copyright (c) 2023.
**/
#include "record_path.hpp"
#include <fstream>
using namespace std;

//初始化can0
int init_can0()
{
    // // 防止出现RTNETLINK answers: Device or resource busy 先关闭can
    system("sudo ip link set down can0");
    // //加载can设备驱动
    // system("sudo modprobe can");
    // system("sudo modprobe can_raw");
    // system("sudo modprobe mttcan");
    system("sudo modprobe gs_usb");
    // 设置波特率 500Kbs
    system("sudo ip link set can0 type can bitrate 500000");
    // // 使能can
    system("sudo ip link set up can0");
    struct sockaddr_can addr;
    struct ifreq ifr;
    struct can_frame frame[1] = {{0}};
    int can_fd = socket(PF_CAN, SOCK_RAW, CAN_RAW); // 创建套接字
    strcpy(ifr.ifr_name, "can0");
    ioctl(can_fd, SIOCGIFINDEX, &ifr);
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    bind(can_fd, (struct sockaddr *)&addr, sizeof(addr));
    /********************* 过滤规则设置 *********************/
    // CAN_EFF_MASK 0x1FFFFFFFU
    // 此处设置三组过滤规则，只接收 ID 为 1、2、3 的三种数据帧
    // struct can_filter rfilter[3];
    // rfilter[0].can_id = 1;
    // rfilter[0].can_mask = CAN_EFF_MASK; // 扩展帧 (EFF: extend frame format)

    // rfilter[1].can_id = 2;
    // rfilter[1].can_mask = CAN_EFF_MASK; // 扩展帧

    // rfilter[2].can_id = 3;
    // rfilter[2].can_mask = CAN_EFF_MASK; // 扩展帧
    // setsockopt(can_fd, SOL_CAN_RAW, CAN_RAW_FILTER, &rfilter, sizeof(rfilter));
    return can_fd;
}

////////////////////////////得到域控制器的状态//////////////////////////////////////
int can0_get_status(int can0_fd)
{
    struct can_frame frame;
    double num = 0;
    while (1)
    {
        int nbytes = read(can0_fd, &frame, sizeof(frame)); // 接收报文
       
        if (nbytes > 0 && frame.can_id == 0x221)
        {
            
            int16_t get_car_speed_ = (frame.data[0]<<8)+frame.data[1];
            get_car_speed = (float)get_car_speed_/1000;
            int16_t get_car_turn_ = (frame.data[6]<<8)+frame.data[7];
            get_car_turn = float(get_car_turn_)/1000;

        }

    }
}
//////////////////////////////////////////////////////////////////////////////////

TrajectoryRecorder::TrajectoryRecorder(std::fstream& file,ros::NodeHandle &nh) : file_(file)
{
    TrajectoryRecorder::sub_pos = nh.subscribe<planning_msgs::car_info>("/car_pos", 100, &TrajectoryRecorder::callBack_Pos, this); 
    // TrajectoryRecorder::line_pub = nh.publisher<geometry_msgs::Pose2D>("/car_pos", 100, &pureNode::callBack2, this); // 订阅车位置姿态
    ros::spin();
}

void TrajectoryRecorder::callBack_Pos(const planning_msgs::car_info::ConstPtr &car_pose)
{
    //Steering_wheel_turn_angle = 1;
    //Using_wheel_speedometer_speed = wheel_speedometer_speed;
    //cout<<" path:"<<FILE_NAME<<endl;
    //cout<<"Using_Steering_wheel_turn_angle = "<<Using_wheel_turn_angle<<endl;
    this->file_<<((car_pose->x) - L1 * cos(car_pose->yaw))<<" "<<((car_pose->y) - L1 * sin(car_pose->yaw))<<" "<<car_pose->yaw<<" "<<get_car_turn<<std::endl;
    cout<<"x,y,  yaw , angle:"<<car_pose->x<<" "<<car_pose->y<<" "<<car_pose->yaw<<" "<<get_car_turn<<std::endl;
}

TrajectoryRecorder::~TrajectoryRecorder()
{
    if (file_.is_open()) {
        file_.close();
    }
    cout<<"路径记录完成，请使用 ‘rosrun optimizetion_path optimize_path’，对路径优化后使用!"<<endl;
}
