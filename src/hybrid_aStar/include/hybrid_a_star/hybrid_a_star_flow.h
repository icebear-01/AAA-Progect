/*
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2024-10-08 11:16:16
 * 
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2024-12-20 21:23:38
 * @FilePath: /src/planning/src/hybrid_aStar/Hybrid_A_Star-main/include/hybrid_a_star/hybrid_a_star_flow.h
 * @Description: 
 * 
 * Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
 */
#ifndef HYBRID_A_STAR_HYBRID_A_STAR_FLOW_H
#define HYBRID_A_STAR_HYBRID_A_STAR_FLOW_H

#include <fstream>
#include <sstream> 
#include <ros/ros.h>
#include <ros/package.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>

#include "hybrid_a_star.h"
#include "costmap_subscriber.h"
#include "init_pose_subscriber.h"
#include "goal_pose_subscriber.h"
#include "guided_frontend_onnx.h"
#include "yaml-cpp/yaml.h"

#include <planning_msgs/path_point.h>
#include <planning_msgs/point.h>
#include <planning_msgs/car_path.h>
#include <planning_msgs/hybrid_astar_paths.h>
#include <planning_msgs/hybrid_astar_path.h>
#include <planning_msgs/hybrid_astar_path_point.h>
#include <planning_msgs/car_scene.h>
// #include "HA_Smoother/smoother.h"

class HybridAStarFlow {
public:
    HybridAStarFlow() = default;  //显式声明构造函数，但是是空，这样可以默认构造

    explicit HybridAStarFlow(ros::NodeHandle &nh);// 初始化为 [0.0, 1.0) 的均匀分布;  //explicit 禁止隐式调用，只能显式调用

    void Run(ros::NodeHandle &nh,planning_msgs::car_scene &car_scene);

private:
    void InitPoseData();

    void ReadData();

    bool HasStartPose();

    bool HasGoalPose();

    void PublishPathToControl(const VectorVec4d &path);

    void PublishPath(const VectorVec4d &path);

    void PublishCurvatureProfiles(const VectorVec4d &raw_path, const VectorVec4d &smoothed_path);

    void PublishSearchedTree(const VectorVec4d &searched_tree);

    void PublishVehiclePath(const VectorVec4d &path, double width,
                            double length, unsigned int vehicle_interval);

    bool NextPlan(const VectorVec4d &path);

    void InitStaticMap();  //初始化静态障碍物地图

    void updateOccupancyHybridAStar(ros::NodeHandle &nh);

    void ComputeCurvatureProfile(const VectorVec4d &path, std::vector<double> &s_out,
                                 std::vector<double> &kappa_out);

    bool PlanWithCurrentFrontend(const Vec4d &start_state, const Vec4d &goal_state,
                                 VectorVec4d &path_original, VectorVec4d &path_smoothed);

    bool PlanWithTransformerGuidedAstar(const Vec4d &start_state, const Vec4d &goal_state,
                                        VectorVec4d &path_original, VectorVec4d &path_smoothed);

    bool PlanWithTransformerGuidedAstarPython(const Vec4d &start_state, const Vec4d &goal_state,
                                              VectorVec4d &path_original, VectorVec4d &path_smoothed);

    bool PlanWithTransformerGuidedAstarOnnx(const Vec4d &start_state, const Vec4d &goal_state,
                                            VectorVec4d &path_original, VectorVec4d &path_smoothed);

    bool ExportGuidedFrontendRequest(const Vec4d &start_state, const Vec4d &goal_state,
                                     const std::string &request_json_path) const;

    bool LoadGuidedFrontendPath(const std::string &path_csv, VectorVec4d &path_original) const;

    void PublishPlanningOutputs(const VectorVec4d &path_original, const VectorVec4d &path_smoothed);

    bool InitStaticMapFromOccupancyGrid();
    void InvalidateGuidedFrontendOccupancyCache();
    const std::vector<int>& GetGuidedFrontendOccupancyRowMajor();

private:
    std::shared_ptr<HybridAStar> kinodynamic_astar_searcher_ptr_;
    std::shared_ptr<CostMapSubscriber> costmap_sub_ptr_;
    std::shared_ptr<InitPoseSubscriber2D> init_pose_sub_ptr_;
    std::shared_ptr<GoalPoseSubscriber2D> goal_pose_sub_ptr_;
    std::shared_ptr<InitPoseSubscriber2D> init_pose_car_position_sub_ptr_;

    ros::Publisher path_pub_;
    ros::Publisher path_pub_planning_path;  //自定义消息发布类型
    ros::Publisher searched_tree_pub_;
    ros::Publisher vehicle_path_pub_;
    ros::Publisher line_pub_local_path;  
    ros::Publisher line_pub_smoothed_path;  //平滑路线  
    ros::Publisher curvature_marker_pub_;   //曲率对比曲线

    ros::Subscriber matching_points_sub_;
    ros::Subscriber scene_chang_task_info_sub;
    planning_msgs::car_scene car_scene_;  //车辆场景

    std::deque<geometry_msgs::PoseWithCovarianceStampedPtr> init_pose_deque_;
    std::deque<geometry_msgs::PoseStampedPtr> goal_pose_deque_;
    std::deque<nav_msgs::OccupancyGridPtr> costmap_deque_;
    std::vector<std::vector<int>> grid_static_occupancy;  // 二维动态+静态总地图网格占用状态
    std::vector<std::vector<int>> grid_static_occupancy_map;  // 二维静态地图网格占用状态
    std::vector<std::vector<int>> grid_point_count;

    Vec4d goal_state_last;

    geometry_msgs::PoseWithCovarianceStampedPtr current_init_pose_ptr_;  //初始化起始点
    geometry_msgs::PoseStampedPtr current_goal_pose_ptr_;       //起始化终点
    nav_msgs::OccupancyGridPtr current_costmap_ptr_;        //代价地图
    double map_max_x_;
    double map_max_y_;
    double map_min_x_;
    double map_min_y_;
    double grid_obstacle_resolution_;  //地图障碍物栅格分辨率
    double grid_state_resolution_;     //状态物栅格分辨率
    int grid_obstacle_width_;  //
    int grid_obstacle_height_;

    geometry_msgs::Pose2D::Ptr car_pose_flow_ptr;  //车辆位置信息

    VectorVec4d pub_path;

    double path_end_x=0.0;
    double path_end_y=0.0;
    float plan_vel=0.4;

    bool use_transformer_guided_frontend_{false};
    bool fallback_to_hybrid_astar_{true};
    bool last_plan_used_transformer_frontend_{false};
    std::string guided_frontend_python_{"python3"};
    std::string guided_frontend_backend_{"python"};
    std::string guided_frontend_script_path_;
    std::string guided_frontend_ckpt_path_;
    std::string guided_frontend_onnx_path_;
    std::string guided_frontend_device_{"cpu"};
    double guided_frontend_lambda_{1.0};
    double guided_frontend_heuristic_weight_{1.0};
    std::string guided_frontend_heuristic_mode_{"octile"};
    std::string guided_frontend_integration_mode_{"g_cost"};
    double guided_frontend_bonus_threshold_{0.5};
    double guided_frontend_clearance_weight_{0.0};
    double guided_frontend_clearance_safe_distance_{0.0};
    double guided_frontend_clearance_power_{2.0};
    std::string guided_frontend_clearance_integration_mode_{"g_cost"};
    bool guided_frontend_allow_corner_cut_{false};
    bool guided_frontend_invert_guidance_cost_{false};
    int guided_frontend_onnx_intra_threads_{1};
    int guided_frontend_onnx_inter_threads_{1};
    std::string static_map_source_{"pcd"};
    std::string static_map_topic_{"/guided_frontend_random_map"};
    double static_map_timeout_{3.0};
    bool static_map_fallback_to_pcd_{true};

    std::unique_ptr<guided_frontend::GuidanceCostMapOnnx> guided_frontend_onnx_;
    std::vector<int> guided_frontend_occupancy_row_major_;
    bool guided_frontend_occupancy_row_major_valid_{false};

    ros::Time timestamp_;

    bool has_map_{};
    bool next_plan=true;
    int path_type=0; //规划的第几个路线编号
};

#endif //HYBRID_A_STAR_HYBRID_A_STAR_FLOW_H
