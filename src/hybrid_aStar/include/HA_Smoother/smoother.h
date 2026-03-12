/*
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2024-11-20 17:19:23
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2024-11-26 11:18:20
 * @FilePath: /src/planning/src/hybrid_aStar/Hybrid_A_Star-main/include/HA_Smoother/smoother.h
 * @Description: 
 * 
 * Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
 */
#pragma once
#include <vector>
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <algorithm>
#include <sensor_msgs/PointCloud2.h>
#include <boost/make_shared.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <geometry_msgs/Pose2D.h>
#include "hybrid_a_star/type.h"
#include "OsqpEigen/OsqpEigen.h"

class HA_Smoother
{
public:
    HA_Smoother(std::vector<VectorVec4d> &RawPath);
    // ~HA_Smoother();
    // int Smooth(std::vector<VectorVec4d> &RawPath_);
    std::vector<VectorVec4d>  Smooth(std::vector<VectorVec4d> &RawPath_);
    std::vector<std::vector<double>> CalculateLinearizedCurvatureParams(const VectorVec4d &RawPath) ;

    VectorVec4d smoothed_path_points_;
    const double curvature_constraint_ = 0.5; //最大曲率限制 
    
private:
    int total_PathPointsNum;
    double PathLength;
};



