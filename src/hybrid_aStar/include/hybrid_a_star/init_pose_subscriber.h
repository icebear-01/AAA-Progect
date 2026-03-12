/*
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2024-10-15 10:24:45
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2024-10-16 09:45:53
 * @FilePath: /src/planning/src/hybrid_aStar/Hybrid_A_Star-main/include/hybrid_a_star/init_pose_subscriber.h
 * @Description: 
 * 
 * Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
 */
/*******************************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2022 Zhang Zhimeng
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 * WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

#ifndef HYBRID_A_STAR_INIT_POSE_SUBSCRIBER_H
#define HYBRID_A_STAR_INIT_POSE_SUBSCRIBER_H

#include <ros/ros.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Pose2D.h>
#include <planning_msgs/car_info.h>

#include <deque>
#include <mutex>

struct VehiclePosition
{
    float x=0;
    float y=0;
    float yaw=0;
    bool is_initialized=false;
};

class InitPoseSubscriber2D {
public:
    InitPoseSubscriber2D(ros::NodeHandle &nh, const std::string &topic_name,
                         size_t buff_size);

    void ParseData(std::deque<geometry_msgs::PoseWithCovarianceStampedPtr> &pose_data_buff);
    void ParseData(geometry_msgs::Pose2D::Ptr& car_pose);
    std::unique_ptr<VehiclePosition> car_position_ptr = std::make_unique<VehiclePosition>();

private:
    void MessageCallBack(const geometry_msgs::PoseWithCovarianceStampedPtr &init_pose_ptr);
    void MessageCallBack(const planning_msgs::car_info::ConstPtr& car_pose);

private:
    ros::Subscriber subscriber_;
    std::deque<geometry_msgs::PoseWithCovarianceStampedPtr> init_poses_;
    // VehiclePosition car_position_;  //存储车辆位置信息

    std::mutex buff_mutex_;
};

#endif //HYBRID_A_STAR_INIT_POSE_SUBSCRIBER_H
