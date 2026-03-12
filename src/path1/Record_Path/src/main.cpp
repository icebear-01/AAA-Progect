/**
 * @FilePath     : /Record_Path/src/main.cpp
 * @Description  :  
 * @Author       : WMD
 * @Version      : 0.0.1
 * @LastEditors  : WMD email:732485622@qq.com
 * @LastEditTime : 2023-11-30 18:59:38
 * @Copyright    : G AUTOMOBILE RESEARCH INSTITUTE CO.,LTD Copyright (c) 2023.
**/
#include "record_path.hpp"

using namespace std;

int main(int argc, char* argv[]) 
{
    ros::init(argc, argv, "record_Path");
    ros::NodeHandle nh;


    int can0_fd = init_can0();

    //////////////////////////////////////////can读//////////////////////////////
    // 读取整车控制信息
    thread can0read_thread1(can0_get_status, can0_fd);
    can0read_thread1.detach();

    fstream file(FILE_NAME.c_str());
    TrajectoryRecorder TrajectoryRecorder(file,nh);

    return 0;
}