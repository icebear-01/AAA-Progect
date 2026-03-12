/*
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2024-07-23 20:17:33
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2024-07-23 22:01:44
 * @FilePath: /src/planning/src/path1/path_optimization/src/main.cpp
 * @Description: 
 * 
 * Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
 */
/**
 * @FilePath     : /Record_Path/src/main.cpp
 * @Description  :  
 * @Author       : WMD
 * @Version      : 0.0.1
 * @LastEditors  : WMD email:732485622@qq.com
 * @LastEditTime : 2023-11-30 18:59:38
 * @Copyright    : G AUTOMOBILE RESEARCH INSTITUTE CO.,LTD Copyright (c) 2023.
**/
#include "path_optimization.cpp"

using namespace std;

int main(int argc, char* argv[]) 
{
    ros::init(argc, argv, "path_optimization");
    ros::NodeHandle nh;

    fstream file(raw_FILE_NAME.c_str());
    TrajectoryOptimization TrajectoryOptimization(file,nh);

    return 0;
}