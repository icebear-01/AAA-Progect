/**
 * @FilePath     : /Play_RecordPath/include/play_path.cpp
 * @Description  : 版本0.01：此版本为稳定运行路径规划的画图版本，添加了matplotlib cpp画图库，稳定运行路径规划，实现SL动态图
 *                 版本0.02：添加了起始点的优化，现在的起点及车辆路径更加稳定
 *                 版本0.03：修复了dp_path和qp_path的终点不一致的Bug，优化了部分程序
 *                 版本0.04：添加了速度规划的DP，QP速度规划算法，及matplotlib cpp 的ST动态图
 *                 版本0.05：修复了速度规划中DP规划的重大bug，DP_speed正常运行，但是，QP_speed存在问题
 *                 版本0.06：修复了QP_speed中躲避障碍物错误问题，较好的运行路径规划的流程
 *                 版本0.07：修复了ST图中起点无解问题，优化了碰瓷问题的解决
 *                 版本0.08：稳定运行版本，粗略完成决策与规划功能
 *                 版本0.09：针对小车的版本，修改针对小车的参数，考虑障碍物的尺寸与道路边界限制
 * @Author       : WMD
 * @Version      : 0.0.9
 * @LastEditors  : 
 * @LastEditTime : 2024-6-28 22:53:20
 * 
 * @Copyright    : G AUTOMOBILE RESEARCH INSTITUTE CO.,LTD Copyright (c) 2023.
 **/
#include <include/emplanner.hpp>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include "RL_DP/rl_dp.h"
using namespace std;
// namespace plt = matplotlibcpp;
#define SHAN "\033[5;30;41m"
#define RED "\033[0;30;41m"
#define GREEN "\033[0;30;42m"
#define YELLOW "\033[0;30;43m"
#define BULE1 "\033[0;30;44m"
#define PURPLE "\033[0;30;45m"
#define BULE2 "\033[0;30;46m"
#define WHITE "\033[0;30;47m"
#define ENDL "\033[0m\n"

#define LIGHT_BLACK "\033[0;90m"
#define LIGHT_RED "\033[0;91m"
#define LIGHT_GREEN "\033[0;92m"
#define LIGHT_YELLOW "\033[0;93m"
#define LIGHT_BLUE "\033[0;94m"
#define LIGHT_PURPLE "\033[0;95m"
#define LIGHT_CYAN "\033[0;96m"
#define LIGHT_WHITE "\033[0;97m"

namespace {

struct SlPoint
{
    double s;
    double l;
};

std::vector<SlPoint> BuildObstacleSlCorners(const planning_msgs::Obstacle& obstacle)
{
    std::vector<SlPoint> corners;
    corners.reserve(4);
    double raw_min_s = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 4; ++i) {
        raw_min_s = std::min(raw_min_s, static_cast<double>(obstacle.bounding_boxs_SL[i].x));
    }
    const double s_offset = static_cast<double>(obstacle.min_s) - raw_min_s;
    for (int i = 0; i < 4; ++i) {
        corners.push_back(
            {
                static_cast<double>(obstacle.bounding_boxs_SL[i].x) + s_offset,
                static_cast<double>(obstacle.bounding_boxs_SL[i].y),
            });
    }
    return corners;
}

SlPoint InterpolateAtL(const SlPoint& a, const SlPoint& b, double target_l)
{
    const double dl = b.l - a.l;
    if (std::abs(dl) < 1e-9) {
        return {0.5 * (a.s + b.s), target_l};
    }
    const double ratio = (target_l - a.l) / dl;
    return {a.s + ratio * (b.s - a.s), target_l};
}

std::vector<SlPoint> ClipPolygonByLowerL(const std::vector<SlPoint>& polygon, double lower_l)
{
    std::vector<SlPoint> output;
    if (polygon.empty()) {
        return output;
    }
    SlPoint prev = polygon.back();
    bool prev_inside = prev.l >= lower_l - 1e-9;
    for (const auto& curr : polygon) {
        const bool curr_inside = curr.l >= lower_l - 1e-9;
        if (curr_inside != prev_inside) {
            output.push_back(InterpolateAtL(prev, curr, lower_l));
        }
        if (curr_inside) {
            output.push_back(curr);
        }
        prev = curr;
        prev_inside = curr_inside;
    }
    return output;
}

std::vector<SlPoint> ClipPolygonByUpperL(const std::vector<SlPoint>& polygon, double upper_l)
{
    std::vector<SlPoint> output;
    if (polygon.empty()) {
        return output;
    }
    SlPoint prev = polygon.back();
    bool prev_inside = prev.l <= upper_l + 1e-9;
    for (const auto& curr : polygon) {
        const bool curr_inside = curr.l <= upper_l + 1e-9;
        if (curr_inside != prev_inside) {
            output.push_back(InterpolateAtL(prev, curr, upper_l));
        }
        if (curr_inside) {
            output.push_back(curr);
        }
        prev = curr;
        prev_inside = curr_inside;
    }
    return output;
}

std::vector<SlPoint> ClipObstacleToLateralBand(
    const planning_msgs::Obstacle& obstacle,
    double time_s,
    double lower_l,
    double upper_l)
{
    auto polygon = BuildObstacleSlCorners(obstacle);
    for (auto& point : polygon) {
        point.s += static_cast<double>(obstacle.s_vel) * time_s;
        point.l += static_cast<double>(obstacle.l_vel) * time_s;
    }
    polygon = ClipPolygonByLowerL(polygon, lower_l);
    polygon = ClipPolygonByUpperL(polygon, upper_l);
    return polygon;
}

bool ComputeObstacleStripSRange(
    const planning_msgs::Obstacle& obstacle,
    double time_s,
    double lower_l,
    double upper_l,
    double* min_s,
    double* max_s)
{
    const auto polygon = ClipObstacleToLateralBand(obstacle, time_s, lower_l, upper_l);
    if (polygon.empty()) {
        return false;
    }
    double local_min_s = std::numeric_limits<double>::infinity();
    double local_max_s = -std::numeric_limits<double>::infinity();
    for (const auto& point : polygon) {
        local_min_s = std::min(local_min_s, point.s);
        local_max_s = std::max(local_max_s, point.s);
    }
    if (min_s != nullptr) {
        *min_s = local_min_s;
    }
    if (max_s != nullptr) {
        *max_s = local_max_s;
    }
    return true;
}

std::string ResolveEmplannerPackagePath()
{
    const char* env_pkg_path = std::getenv("EMPLANNER_PKG_PATH");
    if (env_pkg_path != nullptr && env_pkg_path[0] != '\0') {
        return std::string(env_pkg_path);
    }

    const std::string source_path = __FILE__;
    const std::string suffix = "/src/emplanner.cpp";
    const size_t suffix_pos = source_path.rfind(suffix);
    if (suffix_pos != std::string::npos) {
        return source_path.substr(0, suffix_pos);
    }

    const std::string pkg_path = ros::package::getPath("emplanner");
    if (!pkg_path.empty()) {
        return pkg_path;
    }

    return std::string();
}

}  // namespace

EMPlanner::EMPlanner(ros::NodeHandle &nh,planning_msgs::car_scene car_scene) : car_scene_(car_scene)
{
    EMPlanner::line_pub = nh.advertise<sensor_msgs::PointCloud2>("line_Path_tra", 10);

    EMPlanner::line_pub_watch = nh.advertise<sensor_msgs::PointCloud2>("watch_line_Path_tra", 10);
    EMPlanner::line_pub_local_path = nh.advertise<sensor_msgs::PointCloud2>("local_Path", 10);
    EMPlanner::line_pub_local_qp_path = nh.advertise<planning_msgs::car_path>("em_Path", 10);
    EMPlanner::line_pub_path = nh.advertise<planning_msgs::car_path>("line_Path", 10);
    EMPlanner::obs_watch_pub = nh.advertise<sensor_msgs::PointCloud2>("obs_watch", 10);
    EMPlanner::line_pub_local_qp_path_watch = nh.advertise<sensor_msgs::PointCloud2>("local_qp_Path", 10);
    
    EMPlanner::sub_obstacle_list_lidar = nh.subscribe<planning_msgs::ObstacleList>("/obstacleList_lidar", 10, &EMPlanner::callBack_obstacleList_lidar, this); // 订阅车位置姿态
    EMPlanner::sub_obstacle_list_vision = nh.subscribe<obj_msgs::ObstacleList>("/object_detection_local", 10, &EMPlanner::callBack_obstacle_vision_List, this); // 订阅车位置姿态
    EMPlanner::sub_location = nh.subscribe<planning_msgs::car_info>("/car_pos", 10, &EMPlanner::callBack_location, this);        // 订阅车位置姿态
    EMPlanner::sub_car_info = nh.subscribe<planning_msgs::car_info>("/car_info", 10, &EMPlanner::callBack_carInfo, this);
    EMPlanner::sub_car_stop = nh.subscribe<std_msgs::Float32>("/car_stop", 10, &EMPlanner::callBack_car_stop, this);

    Car_Pose = boost::make_shared<geometry_msgs::Pose2D>();
  
    Car_Pose_middle = boost::make_shared<geometry_msgs::Pose2D>();
    Car_Pose_back = boost::make_shared<geometry_msgs::Pose2D>();
    line_record = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    line_record_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    line_record_opt = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    line_qp_Interpolation = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    line_record_watch = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    local_path = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    local_qp_path = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    obs_watch = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

    obstacle_list = boost::make_shared<planning_msgs::ObstacleList>();
    obstacle_list_qp_path_sl = boost::make_shared<planning_msgs::ObstacleList>();
    obstacle_list_swap = boost::make_shared<planning_msgs::ObstacleList>();
    obstacle_list_vision = boost::make_shared<planning_msgs::ObstacleList>();
    obstacle_list_lidar = boost::make_shared<planning_msgs::ObstacleList>();
    QP_path = boost::make_shared<planning_msgs::car_path>();

    ros::NodeHandle pnh("~");
    pnh.param("sample_s", sample_s, sample_s);
    pnh.param("sample_l", sample_l, sample_l);
    pnh.param("sample_s_num", sample_s_num, sample_s_num);
    pnh.param("sample_s_per_meters", sample_s_per_meters, sample_s_per_meters);
    pnh.param("col_node_num", col_node_num, col_node_num);
    pnh.param("row_node_num", row_node_num, row_node_num);
    pnh.param("w_qp_l", w_qp_l, w_qp_l);
    pnh.param("w_qp_dl", w_qp_dl, w_qp_dl);
    pnh.param("w_qp_ddl", w_qp_ddl, w_qp_ddl);
    pnh.param("w_qp_ref_dp", w_qp_ref_dp, w_qp_ref_dp);
    pnh.param("plan_time", plan_time, plan_time);
    pnh.param("speed_plan_t_dt", speed_plan_t_dt, speed_plan_t_dt);
    pnh.param("st_s_min_step", ST_s_min_step, ST_s_min_step);
    pnh.param("speed_plan_distance", speed_plan_distance, speed_plan_distance);
    pnh.param("st_lateral_limit", st_lateral_limit, st_lateral_limit);
    pnh.param("dynamic_vel_threshold", dynamic_vel_threshold, dynamic_vel_threshold);
    pnh.param("speed_reference", speed_reference, speed_reference);
    pnh.param("speed_dp_ref_vel_weight", w_SpeedDpPlan_ref_vel, w_SpeedDpPlan_ref_vel);
    pnh.param("speed_dp_hard_distance", SpeedDpPlanMinObstacleDistance, SpeedDpPlanMinObstacleDistance);
    pnh.param("dp_vel_max", dp_vel_max, dp_vel_max);
    pnh.param("dp_a_max", dp_a_max, dp_a_max);
    pnh.param("speed_dp_soft_distance", SpeedDpPlanMaxObstacleDistance, SpeedDpPlanMaxObstacleDistance);
    pnh.param("speed_qp_ref_s_weight", w_SpeedQpPlan_ref_s, w_SpeedQpPlan_ref_s);
    pnh.param("speed_qp_safe_distance", SpeedQpPlan_SafeDistance, SpeedQpPlan_SafeDistance);
    pnh.param("speed_qp_v_max", SpeedQP_v_max, SpeedQP_v_max);
    pnh.param("speed_qp_a_max", SpeedQP_a_max, SpeedQP_a_max);
    if (sample_s <= 0.0f) {
        ROS_WARN_STREAM("sample_s <= 0, reset to 0.8");
        sample_s = 0.8f;
    }
    if (sample_l <= 0.0f) {
        ROS_WARN_STREAM("sample_l <= 0, reset to 0.35");
        sample_l = 0.35f;
    }
    if (sample_s_num <= 0.0f) {
        ROS_WARN_STREAM("sample_s_num <= 0, reset to 10");
        sample_s_num = 10.0f;
    }
    if (sample_s_per_meters <= 0.0f) {
        ROS_WARN_STREAM("sample_s_per_meters <= 0, reset to 20");
        sample_s_per_meters = 20.0f;
    }
    if (col_node_num < 2) {
        ROS_WARN_STREAM("col_node_num < 2, reset to 6");
        col_node_num = 6;
    }
    if (row_node_num < 3) {
        ROS_WARN_STREAM("row_node_num < 3, reset to 11");
        row_node_num = 11;
    }
    if (plan_time <= 0.0f) {
        ROS_WARN_STREAM("plan_time <= 0, reset to 4.0");
        plan_time = 4.0f;
    }
    if (speed_plan_t_dt <= 0.0f) {
        ROS_WARN_STREAM("speed_plan_t_dt <= 0, reset to 0.5");
        speed_plan_t_dt = 0.5f;
    }
    if (ST_s_min_step <= 0.0f) {
        ROS_WARN_STREAM("st_s_min_step <= 0, reset to 0.2");
        ST_s_min_step = 0.2f;
    }
    if (speed_plan_distance <= 0.0f) {
        ROS_WARN_STREAM("speed_plan_distance <= 0, reset to 5.0");
        speed_plan_distance = 5.0f;
    }
    if (st_lateral_limit <= 0.0f) {
        ROS_WARN_STREAM("st_lateral_limit <= 0, reset to 1.0");
        st_lateral_limit = 1.0f;
    }
    if (dynamic_vel_threshold < 0.0f) {
        ROS_WARN_STREAM("dynamic_vel_threshold < 0, reset to 0.0");
        dynamic_vel_threshold = 0.0f;
    }
    if (speed_reference <= 0.0f) {
        ROS_WARN_STREAM("speed_reference <= 0, reset to 0.5");
        speed_reference = 0.5f;
    }
    if (w_SpeedDpPlan_ref_vel < 0.0f) {
        ROS_WARN_STREAM("speed_dp_ref_vel_weight < 0, reset to 500.0");
        w_SpeedDpPlan_ref_vel = 500.0f;
    }
    if (SpeedDpPlanMinObstacleDistance < 0.0f) {
        ROS_WARN_STREAM("speed_dp_hard_distance < 0, reset to 0.4");
        SpeedDpPlanMinObstacleDistance = 0.4f;
    }
    if (dp_vel_max <= 0.0f) {
        ROS_WARN_STREAM("dp_vel_max <= 0, reset to 1.2");
        dp_vel_max = 1.2f;
    }
    if (dp_a_max <= 0.0f) {
        ROS_WARN_STREAM("dp_a_max <= 0, reset to 0.6");
        dp_a_max = 0.6f;
    }
    if (w_SpeedQpPlan_ref_s < 0.0f) {
        ROS_WARN_STREAM("speed_qp_ref_s_weight < 0, reset to 0.0");
        w_SpeedQpPlan_ref_s = 0.0f;
    }
    if (SpeedQP_v_max <= 0.0f) {
        ROS_WARN_STREAM("speed_qp_v_max <= 0, reset to 1.2");
        SpeedQP_v_max = 1.2f;
    }
    if (SpeedQP_a_max <= 0.0f) {
        ROS_WARN_STREAM("speed_qp_a_max <= 0, reset to 0.5");
        SpeedQP_a_max = 0.5f;
    }
    if (SpeedQpPlan_SafeDistance < 0.0f) {
        ROS_WARN_STREAM("speed_qp_safe_distance < 0, reset to 0.01");
        SpeedQpPlan_SafeDistance = 0.01f;
    }
    if (SpeedDpPlanMaxObstacleDistance < SpeedDpPlanMinObstacleDistance + 0.05f) {
        ROS_WARN_STREAM("speed_dp_soft_distance too small, reset to dp_hard_distance + 0.05");
        SpeedDpPlanMaxObstacleDistance = SpeedDpPlanMinObstacleDistance + 0.05f;
    }
    if (w_qp_l <= 0.0f) {
        ROS_WARN_STREAM("w_qp_l <= 0, reset to 800");
        w_qp_l = 800.0f;
    }
    if (w_qp_dl <= 0.0f) {
        ROS_WARN_STREAM("w_qp_dl <= 0, reset to 200");
        w_qp_dl = 200.0f;
    }
    if (w_qp_ddl <= 0.0f) {
        ROS_WARN_STREAM("w_qp_ddl <= 0, reset to 600");
        w_qp_ddl = 600.0f;
    }
    if (w_qp_ref_dp <= 0.0f) {
        ROS_WARN_STREAM("w_qp_ref_dp <= 0, reset to 50");
        w_qp_ref_dp = 50.0f;
    }
    pnh.param("use_rl_dp", use_rl_dp_, true);
    pnh.param("timing_print_every", timing_print_every_, 1);
    if (timing_print_every_ < 1) {
        timing_print_every_ = 1;
    }
    pnh.param("rl_dp_model_path", rl_dp_model_path_,
              std::string("/home/wmd/rl_dp/main_DP/main/checkpoints/ppo_policy_20251225_101829_update_12450.onnx"));
    if (!use_rl_dp_) {
        rl_dp_disable_reason_ = "use_rl_dp param false";
    }
    if (use_rl_dp_) {
        pnh.param("rl_dp_vehicle_scale", rl_dp_vehicle_scale_, 0.9f);
        float rl_dp_coarse_inflation = 0.0f;
        float rl_dp_fine_inflation = 0.0f;
        pnh.param("rl_dp_coarse_inflation", rl_dp_coarse_inflation, 0.0f);
        pnh.param("rl_dp_fine_inflation", rl_dp_fine_inflation, 0.0f);
        if (rl_dp_coarse_inflation < 0.0f) {
            ROS_WARN_STREAM("rl_dp_coarse_inflation < 0, clamp to 0");
            rl_dp_coarse_inflation = 0.0f;
        }
        if (rl_dp_fine_inflation < 0.0f) {
            ROS_WARN_STREAM("rl_dp_fine_inflation < 0, clamp to 0");
            rl_dp_fine_inflation = 0.0f;
        }
        if (!(rl_dp_vehicle_scale_ > 0.1f && rl_dp_vehicle_scale_ <= 1.0f)) {
            ROS_WARN_STREAM("rl_dp_vehicle_scale out of range: " << rl_dp_vehicle_scale_
                            << ", clamp to 1.0");
            rl_dp_vehicle_scale_ = 1.0f;
        }
        pnh.param("rl_dp_s_samples", rl_dp_s_samples_, 9);
        pnh.param("rl_dp_l_samples", rl_dp_l_samples_, 19);
        pnh.param("rl_dp_s_min", rl_dp_s_min_, 0.0f);
        pnh.param("rl_dp_s_max", rl_dp_s_max_, 8.0f);
        pnh.param("rl_dp_l_min", rl_dp_l_min_, -4.0f);
        pnh.param("rl_dp_l_max", rl_dp_l_max_, 4.0f);
        if (rl_dp_s_samples_ < 2) {
            ROS_WARN_STREAM("rl_dp_s_samples < 2, reset to 9");
            rl_dp_s_samples_ = 9;
        }
        if (rl_dp_l_samples_ < 2) {
            ROS_WARN_STREAM("rl_dp_l_samples < 2, reset to 19");
            rl_dp_l_samples_ = 19;
        }
        if (rl_dp_s_max_ <= rl_dp_s_min_) {
            ROS_WARN_STREAM("rl_dp_s_max <= rl_dp_s_min, reset to horizon");
            rl_dp_s_min_ = 0.0f;
            rl_dp_s_max_ = sample_s * static_cast<float>(col_node_num);
        }
        if (rl_dp_l_max_ <= rl_dp_l_min_) {
            ROS_WARN_STREAM("rl_dp_l_max <= rl_dp_l_min, reset to +/-4");
            rl_dp_l_min_ = -4.0f;
            rl_dp_l_max_ = 4.0f;
        }
        float rl_dp_s_step = (rl_dp_s_max_ - rl_dp_s_min_) / std::max(1, rl_dp_s_samples_ - 1);
        float rl_dp_l_step = (rl_dp_l_max_ - rl_dp_l_min_) / std::max(1, rl_dp_l_samples_ - 1);
        int lateral_move_limit = std::max(
            1, static_cast<int>(std::floor(1.2f * rl_dp_s_step / rl_dp_l_step)));
        std::ifstream model_file(rl_dp_model_path_);
        if (!model_file.good()) {
            ROS_WARN_STREAM("RL_DP model not found: " << rl_dp_model_path_ << ". Fallback to classic DP.");
            rl_dp_disable_reason_ = "model not found: " + rl_dp_model_path_;
            use_rl_dp_ = false;
        } else {
            try {
                rl_dp_.reset(new RL_DP(rl_dp_model_path_,
                                       rl_dp_s_samples_,
                                       rl_dp_l_samples_,
                                       rl_dp_s_min_,
                                       rl_dp_s_max_,
                                       rl_dp_l_min_,
                                       rl_dp_l_max_,
                                       lateral_move_limit,
                                       3,
                                       rl_dp_coarse_inflation,
                                       rl_dp_fine_inflation,
                                       carLength * rl_dp_vehicle_scale_,
                                       car_width * rl_dp_vehicle_scale_));
                rl_dp_disable_reason_.clear();
            } catch (const std::exception& e) {
                ROS_WARN_STREAM("RL_DP init failed: " << e.what() << ". Fallback to classic DP.");
                rl_dp_disable_reason_ = std::string("init failed: ") + e.what();
                use_rl_dp_ = false;
                rl_dp_.reset();
            }
        }
    }
    
    First_run();
}


EMPlanner::~EMPlanner()
{
}

void EMPlanner::InjectSimCarState(const planning_msgs::car_info &car_pose)
{
    planning_msgs::car_info::ConstPtr car_pose_ptr =
        boost::make_shared<const planning_msgs::car_info>(car_pose);
    callBack_location(car_pose_ptr);
    callBack_carInfo(car_pose_ptr);
}

void EMPlanner::InjectSimObstacleList(const planning_msgs::ObstacleList &obstacle_list_msg)
{
    planning_msgs::ObstacleList::ConstPtr obstacle_list_ptr =
        boost::make_shared<const planning_msgs::ObstacleList>(obstacle_list_msg);
    callBack_obstacleList_lidar(obstacle_list_ptr);
}

void EMPlanner::InjectSimCarStop(float flag_car_stop)
{
    std_msgs::Float32 msg;
    msg.data = flag_car_stop;
    std_msgs::Float32::ConstPtr stop_ptr = boost::make_shared<const std_msgs::Float32>(msg);
    callBack_car_stop(stop_ptr);
}

void EMPlanner::ReinitializeForCurrentInputs()
{
    First_run();
}

bool EMPlanner::HasLatestPlanResult() const
{
    return has_latest_plan_result_;
}

bool EMPlanner::GetLatestQpRunningNormally() const
{
    return latest_qp_running_normally_;
}

const std::string& EMPlanner::GetLatestDpSource() const
{
    return latest_dp_source_;
}

const Frenet_path_points& EMPlanner::GetLatestDpPathSL() const
{
    return latest_dp_path_sl_;
}

const Frenet_path_points& EMPlanner::GetLatestQpPathSL() const
{
    return latest_qp_path_sl_;
}

const std::vector<Eigen::Vector2d>& EMPlanner::GetLatestDpPathXY() const
{
    return latest_dp_path_xy_;
}

const std::vector<Eigen::Vector2d>& EMPlanner::GetLatestQpPathXY() const
{
    return latest_qp_path_xy_;
}

const planning_msgs::car_path& EMPlanner::GetLatestQpPathMsg() const
{
    return latest_qp_path_msg_;
}

double EMPlanner::GetLatestPlannerCycleMs() const
{
    return latest_planner_cycle_ms_;
}

double EMPlanner::GetLatestDpSamplingMs() const
{
    return latest_dp_sampling_ms_;
}

double EMPlanner::GetLatestQpOptimizationMs() const
{
    return latest_qp_optimization_ms_;
}

double EMPlanner::GetLatestSpeedPlanningMs() const
{
    return latest_speed_planning_ms_;
}

bool EMPlanner::HasLatestSpeedPlanResult() const
{
    return latest_speed_plan_available_;
}

const Speed_plan_points& EMPlanner::GetLatestSpeedQpPoints() const
{
    return latest_speed_qp_points_;
}

const std::vector<double>& EMPlanner::GetLatestSpeedDpPathS() const
{
    return latest_speed_dp_path_s_;
}

int EMPlanner::GetLatestSpeedDpLastFeasibleIndex() const
{
    return latest_speed_dp_last_feasible_index_;
}

const Speed_Plan_DP_ST_nodes& EMPlanner::GetLatestSpeedDpStNodes() const
{
    return latest_speed_dp_st_nodes_;
}

const planning_msgs::ObstacleList& EMPlanner::GetLatestSpeedObstacleListSL() const
{
    return latest_speed_obstacle_list_qp_path_sl_;
}

void EMPlanner::callBack_car_stop(const std_msgs::Float32::ConstPtr& flag_car_stop)
{
    flagCarStop=flag_car_stop->data;
}

void EMPlanner::callBack_obstacleList_lidar(const planning_msgs::ObstacleList::ConstPtr &Obstacle_list_lidar) //车速赋值回调
{
    obstacle_list_lidar.reset(new planning_msgs::ObstacleList);
    *obstacle_list_lidar=*Obstacle_list_lidar;
}


void EMPlanner::callBack_carInfo(const planning_msgs::car_info::ConstPtr &car_info) //车速赋值回调
{
    real_vehicle_speed=car_info->speed;
}
void EMPlanner::callBack_location(const planning_msgs::car_info::ConstPtr &car_pose)//NDT定位回调
{
    Car_Pose->x = ((car_pose->x) + L * cos(car_pose->yaw)); // 纠正为后车身XY坐标
    Car_Pose->y = ((car_pose->y) + L * sin(car_pose->yaw)); // 纠正为后车身XY坐标
    Car_Pose->theta = car_pose->yaw;

    Car_Pose_middle->x = car_pose->x;
    Car_Pose_middle->y = car_pose->y;
    Car_Pose_middle->theta = car_pose->yaw;
    Car_Pose_back->x = ((car_pose->x) - L * cos(car_pose->yaw)); // 纠正为后车身XY坐标
    Car_Pose_back->y = ((car_pose->y) - L * sin(car_pose->yaw)); // 纠正为后车身XY坐标
    Car_Pose_back->theta = car_pose->yaw;

    obstacle_list_qp_path_sl.reset(new planning_msgs::ObstacleList);
    obstacle_list.reset(new planning_msgs::ObstacleList);
    
    car_pos_is_arrive = true;
}

void EMPlanner::callBack_obstacle_vision_List(const obj_msgs::ObstacleList::ConstPtr &Obstacle_list_vision) //视觉回调
{
    obstacle_list_vision.reset(new planning_msgs::ObstacleList);
    for (int i = 0; i < Obstacle_list_vision->obstacles.size(); i++)
    {
        planning_msgs::Obstacle obs_vision;
        for (int j = 0; j < obstacle_list_vision->obstacles[i].bounding_boxs.size(); j++)
        {
            obs_vision.bounding_boxs[j].x =Obstacle_list_vision->obstacles[i].bounding_boxs[j].x;
            obs_vision.bounding_boxs[j].y =Obstacle_list_vision->obstacles[i].bounding_boxs[j].y;
        }
        
        obs_vision.x_vel =Obstacle_list_vision->obstacles[i].vel.x;

        obs_vision.y_vel =Obstacle_list_vision->obstacles[i].vel.y;
        obstacle_list_vision->obstacles.push_back(obs_vision);
    }
}


/**
 * @description: 障碍物参数计算
 * @param {Ptr} &Obstacle_list
 * @return {*}
 */
void EMPlanner::Obstacle_list_Initialization(planning_msgs::ObstacleList::Ptr &Obstacle_list) 
{
    for (int i = 0; i < Obstacle_list->obstacles.size(); i++)
    {
        const double obstacle_speed =
            std::sqrt(std::pow(Obstacle_list->obstacles[i].x_vel, 2) +
                      std::pow(Obstacle_list->obstacles[i].y_vel, 2));
        Obstacle_list->obstacles[i].is_dynamic_obs =
            Obstacle_list->obstacles[i].is_dynamic_obs || obstacle_speed > dynamic_vel_threshold;
        float max_s;
        float min_s;
        float max_l;
        float min_l;

        vector<float> s_temp;
        vector<float> l_temp;

        // Eigen::Vector2d obstacle_point_center(Obstacle_list->obstacles[i].x, Obstacle_list->obstacles[i].y);
        Eigen::Vector2d obstacle_point_point1(Obstacle_list->obstacles[i].max_x, Obstacle_list->obstacles[i].max_y);
        Eigen::Vector2d obstacle_point_point2(Obstacle_list->obstacles[i].max_x, Obstacle_list->obstacles[i].min_y);
        Eigen::Vector2d obstacle_point_point3(Obstacle_list->obstacles[i].min_x, Obstacle_list->obstacles[i].max_y);
        Eigen::Vector2d obstacle_point_point4(Obstacle_list->obstacles[i].min_x, Obstacle_list->obstacles[i].min_y);

        Eigen::Vector2d obstacle_point_point1_sl = calc_obstacle_sl(obstacle_point_point1); // 上下左右障碍物四个角点计算
        Eigen::Vector2d obstacle_point_point2_sl = calc_obstacle_sl(obstacle_point_point2);
        // Eigen::Vector2d obstacle_point_center_sl=calc_obstacle_sl(obstacle_point_center);//障碍物中心点计算
        Eigen::Vector2d obstacle_point_point3_sl = calc_obstacle_sl(obstacle_point_point3);
        Eigen::Vector2d obstacle_point_point4_sl = calc_obstacle_sl(obstacle_point_point4);

        s_temp.push_back(obstacle_point_point1_sl(0));
        s_temp.push_back(obstacle_point_point2_sl(0));
        s_temp.push_back(obstacle_point_point3_sl(0));
        s_temp.push_back(obstacle_point_point4_sl(0));

        l_temp.push_back(obstacle_point_point1_sl(1));
        l_temp.push_back(obstacle_point_point2_sl(1));
        l_temp.push_back(obstacle_point_point3_sl(1));
        l_temp.push_back(obstacle_point_point4_sl(1));

        auto s_point_min = std::min_element(s_temp.begin(), s_temp.end());
        auto s_point_max = std::max_element(s_temp.begin(), s_temp.end());
        auto l_point_min = std::min_element(l_temp.begin(), l_temp.end());
        auto l_point_max = std::max_element(l_temp.begin(), l_temp.end());

        Obstacle_list->obstacles[i].min_s = *s_point_min;
        Obstacle_list->obstacles[i].max_s = *s_point_max;
        Obstacle_list->obstacles[i].min_l = *l_point_min;
        Obstacle_list->obstacles[i].max_l = *l_point_max;
        Obstacle_list->obstacles[i].s = (*s_point_min + *s_point_max) / 2;
        Obstacle_list->obstacles[i].l = (*l_point_min + *l_point_max) / 2;
        // if (abs(Obstacle_list->obstacles[i].l) < 1.5)
        // {
        //     cout << i << " 个障碍物 l<1.5; l:" << Obstacle_list->obstacles[i].l << "!!!!!!!!!!" << endl;
        // }

        // Obstacle_list->obstacles[i].max_length = sqrt(pow(Obstacle_list->obstacles[i].max_x - Obstacle_list->obstacles[i].min_x, 2) + pow(Obstacle_list->obstacles[i].max_y - Obstacle_list->obstacles[i].min_y, 2));
    }
}


/**
 * @description: 障碍物参数计算
 * @param {Ptr} &Obstacle_list
 * @return {*}
 */
void EMPlanner::Obstacle_list_Initialization_vision(planning_msgs::ObstacleList::Ptr &Obstacle_list)
{
    for (int i = 0; i < Obstacle_list->obstacles.size(); i++)
    {
        Obstacle_list->obstacles[i].is_consider=true;
        if(sqrt(pow(Obstacle_list->obstacles[i].x_vel,2)+pow(Obstacle_list->obstacles[i].y_vel,2))>dynamic_vel_threshold)
            Obstacle_list->obstacles[i].is_dynamic_obs=true;
        else
            Obstacle_list->obstacles[i].is_dynamic_obs=false;
        float max_s;
        float min_s;
        float max_l;
        float min_l;
        float middle_x,middle_y;

        vector<float> s_temp;
        vector<float> l_temp;

        // Eigen::Vector2d obstacle_point_center(Obstacle_list->obstacles[i].x, Obstacle_list->obstacles[i].y);
        Eigen::Vector2d obstacle_point_point1(Obstacle_list->obstacles[i].bounding_boxs[0].x, Obstacle_list->obstacles[i].bounding_boxs[0].y);
        Eigen::Vector2d obstacle_point_point2(Obstacle_list->obstacles[i].bounding_boxs[1].x, Obstacle_list->obstacles[i].bounding_boxs[1].y);
        Eigen::Vector2d obstacle_point_point3(Obstacle_list->obstacles[i].bounding_boxs[2].x, Obstacle_list->obstacles[i].bounding_boxs[2].y);
        Eigen::Vector2d obstacle_point_point4(Obstacle_list->obstacles[i].bounding_boxs[3].x, Obstacle_list->obstacles[i].bounding_boxs[3].y);
        middle_x=(Obstacle_list->obstacles[i].bounding_boxs[0].x+Obstacle_list->obstacles[i].bounding_boxs[1].x+Obstacle_list->obstacles[i].bounding_boxs[2].x+Obstacle_list->obstacles[i].bounding_boxs[3].x)/4;
        middle_y=(Obstacle_list->obstacles[i].bounding_boxs[0].y+Obstacle_list->obstacles[i].bounding_boxs[1].y+Obstacle_list->obstacles[i].bounding_boxs[2].y+Obstacle_list->obstacles[i].bounding_boxs[3].y)/4;
        
        Eigen::Vector2d obstacle_point_point1_sl = calc_obstacle_sl(obstacle_point_point1); // 上下左右障碍物四个角点计算
        Eigen::Vector2d obstacle_point_point2_sl = calc_obstacle_sl(obstacle_point_point2);
        // Eigen::Vector2d obstacle_point_center_sl=calc_obstacle_sl(obstacle_point_center);//障碍物中心点计算
        Eigen::Vector2d obstacle_point_point3_sl = calc_obstacle_sl(obstacle_point_point3);
        Eigen::Vector2d obstacle_point_point4_sl = calc_obstacle_sl(obstacle_point_point4);
        Obstacle_list->obstacles[i].bounding_boxs_SL[0].x=obstacle_point_point1_sl(0); Obstacle_list->obstacles[i].bounding_boxs_SL[0].y=obstacle_point_point1_sl(1);
        Obstacle_list->obstacles[i].bounding_boxs_SL[1].x=obstacle_point_point2_sl(0); Obstacle_list->obstacles[i].bounding_boxs_SL[1].y=obstacle_point_point2_sl(1);
        Obstacle_list->obstacles[i].bounding_boxs_SL[2].x=obstacle_point_point3_sl(0); Obstacle_list->obstacles[i].bounding_boxs_SL[2].y=obstacle_point_point3_sl(1);
        Obstacle_list->obstacles[i].bounding_boxs_SL[3].x=obstacle_point_point4_sl(0); Obstacle_list->obstacles[i].bounding_boxs_SL[3].y=obstacle_point_point4_sl(1);
        
        s_temp.push_back(obstacle_point_point1_sl(0));
        s_temp.push_back(obstacle_point_point2_sl(0));
        s_temp.push_back(obstacle_point_point3_sl(0));
        s_temp.push_back(obstacle_point_point4_sl(0));

        l_temp.push_back(obstacle_point_point1_sl(1));
        l_temp.push_back(obstacle_point_point2_sl(1));
        l_temp.push_back(obstacle_point_point3_sl(1));
        l_temp.push_back(obstacle_point_point4_sl(1));

        auto s_point_min = std::min_element(s_temp.begin(), s_temp.end());
        auto s_point_max = std::max_element(s_temp.begin(), s_temp.end());
        auto l_point_min = std::min_element(l_temp.begin(), l_temp.end());
        auto l_point_max = std::max_element(l_temp.begin(), l_temp.end());

        Obstacle_list->obstacles[i].x = middle_x;
        Obstacle_list->obstacles[i].y = middle_y;
        Obstacle_list->obstacles[i].min_s = *s_point_min;
        Obstacle_list->obstacles[i].max_s = *s_point_max;
        Obstacle_list->obstacles[i].min_l = *l_point_min;
        Obstacle_list->obstacles[i].max_l = *l_point_max;
        Obstacle_list->obstacles[i].s = (*s_point_min + *s_point_max) / 2;
        Obstacle_list->obstacles[i].l = (*l_point_min + *l_point_max) / 2;

        Obstacle_list->obstacles[i].absolute_s=Obstacle_list->obstacles[i].s;
        Obstacle_list->obstacles[i].absolute_s_min=Obstacle_list->obstacles[i].min_s ;
        Obstacle_list->obstacles[i].absolute_s_max=Obstacle_list->obstacles[i].max_s;
        // cout<<"sl min max:"<<Obstacle_list->obstacles[i].min_s<<","<<Obstacle_list->obstacles[i].max_s<<","<<Obstacle_list->obstacles[i].min_l<<","<<Obstacle_list->obstacles[i].max_l<<endl;
        
        
        Obstacle_list->obstacles[i].max_length=abs(Obstacle_list->obstacles[i].max_l-Obstacle_list->obstacles[i].min_l);
        // cout<<"obs "<<i<<":"<<Obstacle_list->obstacles[i].s<<","<<Obstacle_list->obstacles[i].l<<","<<Obstacle_list->obstacles[i].x_vel<<endl;

        // Obstacle_list->obstacles[i].x_vel=Obstacle_list->obstacles[i].x_vel+real_vehicle_speed;
        Obstacle_list->obstacles[i].x_vel=Obstacle_list->obstacles[i].x_vel;
    }
}

/**
 * @description: 针对qp_path_路线计算各个障碍物的SL
 * @param {Ptr} &Obstacle_list
 * @return {*}
 */
void EMPlanner::Obstacle_list_Initialization_qp_path(planning_msgs::ObstacleList::Ptr &Obstacle_list,Eigen::Vector2d planning_frist_star)
{
    for (int i = 0; i < Obstacle_list->obstacles.size(); i++)
    {
        float max_s,min_s,max_l,min_l;
        float middle_x,middle_y;

        vector<float> s_temp;
        vector<float> l_temp;

        // Eigen::Vector2d obstacle_point_center(Obstacle_list->obstacles[i].x, Obstacle_list->obstacles[i].y);
        Eigen::Vector2d obstacle_point_point1(Obstacle_list->obstacles[i].bounding_boxs[0].x, Obstacle_list->obstacles[i].bounding_boxs[0].y);
        Eigen::Vector2d obstacle_point_point2(Obstacle_list->obstacles[i].bounding_boxs[1].x, Obstacle_list->obstacles[i].bounding_boxs[1].y);
        Eigen::Vector2d obstacle_point_point3(Obstacle_list->obstacles[i].bounding_boxs[2].x, Obstacle_list->obstacles[i].bounding_boxs[2].y);
        Eigen::Vector2d obstacle_point_point4(Obstacle_list->obstacles[i].bounding_boxs[3].x, Obstacle_list->obstacles[i].bounding_boxs[3].y);
        middle_x=(Obstacle_list->obstacles[i].bounding_boxs[0].x+Obstacle_list->obstacles[i].bounding_boxs[1].x+Obstacle_list->obstacles[i].bounding_boxs[2].x+Obstacle_list->obstacles[i].bounding_boxs[3].x)/4;
        middle_y=(Obstacle_list->obstacles[i].bounding_boxs[0].y+Obstacle_list->obstacles[i].bounding_boxs[1].y+Obstacle_list->obstacles[i].bounding_boxs[2].y+Obstacle_list->obstacles[i].bounding_boxs[3].y)/4;

        Eigen::Vector2d obstacle_point_point1_sl = calc_obstacle_sl(obstacle_point_point1,QP_path,planning_frist_star); // 上下左右障碍物四个角点计算
        Eigen::Vector2d obstacle_point_point2_sl = calc_obstacle_sl(obstacle_point_point2,QP_path,planning_frist_star);  //sl为qp坐标系的sl
        // Eigen::Vector2d obstacle_point_center_sl=calc_obstacle_sl(obstacle_point_center);//障碍物中心点计算
        Eigen::Vector2d obstacle_point_point3_sl = calc_obstacle_sl(obstacle_point_point3,QP_path,planning_frist_star);
        Eigen::Vector2d obstacle_point_point4_sl = calc_obstacle_sl(obstacle_point_point4,QP_path,planning_frist_star);
        Obstacle_list->obstacles[i].bounding_boxs_SL[0].x=obstacle_point_point1_sl(0); Obstacle_list->obstacles[i].bounding_boxs_SL[0].y=obstacle_point_point1_sl(1);
        Obstacle_list->obstacles[i].bounding_boxs_SL[1].x=obstacle_point_point2_sl(0); Obstacle_list->obstacles[i].bounding_boxs_SL[1].y=obstacle_point_point2_sl(1);
        Obstacle_list->obstacles[i].bounding_boxs_SL[2].x=obstacle_point_point3_sl(0); Obstacle_list->obstacles[i].bounding_boxs_SL[2].y=obstacle_point_point3_sl(1);
        Obstacle_list->obstacles[i].bounding_boxs_SL[3].x=obstacle_point_point4_sl(0); Obstacle_list->obstacles[i].bounding_boxs_SL[3].y=obstacle_point_point4_sl(1);
        // cout<<"sl1:"<<obstacle_point_point1_sl(0)<<","<<obstacle_point_point1_sl(1)<<endl;

        int index_match_middle_points = get_closest_point(middle_x, middle_y, *QP_path);
        Eigen::Vector2d d(QP_path->points[index_match_middle_points].tor.x,QP_path->points[index_match_middle_points].tor.y);  //路径方向
        Eigen::Vector2d v(Obstacle_list->obstacles[i].x_vel,Obstacle_list->obstacles[i].y_vel); //速度
        double s_dot = d.dot(v);  //s_vel

        Eigen::Vector2d v_tor=s_dot*d;
        Eigen::Vector2d v_nor=v-v_tor;
        double l_dot=v_nor.norm();  //l_vel
        double cross_sign = d.x() * v.y() - d.y() * v.x(); // 这是二维向量叉积的计算方式
        if (cross_sign < 0)
        {
            l_dot=-l_dot;
        }

        Obstacle_list->obstacles[i].s_vel=s_dot;  //障碍物sl的速度赋值
        Obstacle_list->obstacles[i].l_vel=l_dot;  

        s_temp.push_back(obstacle_point_point1_sl(0));
        s_temp.push_back(obstacle_point_point2_sl(0));
        s_temp.push_back(obstacle_point_point3_sl(0));
        s_temp.push_back(obstacle_point_point4_sl(0));

        l_temp.push_back(obstacle_point_point1_sl(1));
        l_temp.push_back(obstacle_point_point2_sl(1));
        l_temp.push_back(obstacle_point_point3_sl(1));
        l_temp.push_back(obstacle_point_point4_sl(1));

        auto s_point_min = std::min_element(s_temp.begin(), s_temp.end());
        auto s_point_max = std::max_element(s_temp.begin(), s_temp.end());
        auto l_point_min = std::min_element(l_temp.begin(), l_temp.end());
        auto l_point_max = std::max_element(l_temp.begin(), l_temp.end());

        Obstacle_list->obstacles[i].min_s = *s_point_min-lidarToCarHead;
        Obstacle_list->obstacles[i].max_s = *s_point_max-lidarToCarHead;
        // Lateral offset should stay in the local path Frenet frame; subtracting lidarToCarHead
        // here would incorrectly pull every obstacle 0.55 m toward the center line.
        Obstacle_list->obstacles[i].min_l = *l_point_min;
        Obstacle_list->obstacles[i].max_l = *l_point_max;
        Obstacle_list->obstacles[i].s = (*s_point_min + *s_point_max) / 2-lidarToCarHead;
        Obstacle_list->obstacles[i].l = (*l_point_min + *l_point_max) / 2;
        // cout<<"!!!qp_path_sl:"<<Obstacle_list->obstacles[i].s<<","<<Obstacle_list->obstacles[i].l<<endl;
    }
    if(!QpPathRunningNormally&&car_direct==1)  //无解，QP替换为上一时刻的QP_path
    {
        // car_wait=1;
        float stop_s=10;
        for (int i = 0; i < Obstacle_list->obstacles.size(); i++)
        {
            if((Obstacle_list->obstacles[i].min_l< (-car_width/2 -safe_distance) &&Obstacle_list->obstacles[i].max_l< (-car_width/2 -safe_distance))
            || (Obstacle_list->obstacles[i].min_l> (car_width/2+safe_distance) &&Obstacle_list->obstacles[i].max_l> (car_width/2+safe_distance))) //无影响障碍物           
            {
                continue;
            }
            else //有影响障碍物           
            {
                if(Obstacle_list->obstacles[i].min_s<stop_s) //找到最近的影响无解的障碍物s ：stop_s
                {
                    stop_s=Obstacle_list->obstacles[i].min_s;
                }
            }
        }
        if(stop_s<10&&stop_s>2)
        {
            car_wait=1;
        }
        // cout<<SHAN<<"stop_s:"<<stop_s<<ENDL;
        
    }

}

Eigen::Vector2d EMPlanner::calc_obstacle_sl(Eigen::Vector2d obstacle_point)
{
    Eigen::Vector2d obstacle_sl;
    int index_match_points = get_closest_point(obstacle_point(0), obstacle_point(1), path, index_closest_Car);

    planning_msgs::path_point host_projected_point1 = find_projected_point_Frenet(path, index_match_points, obstacle_point);
    Eigen::Vector2d AB(host_projected_point1.tor.x, host_projected_point1.tor.y);
    Eigen::Vector2d AC(obstacle_point(0) - host_projected_point1.x, obstacle_point(1) - host_projected_point1.y);
    obstacle_sl(0) = host_projected_point1.absolute_s;
    float l = AB.x() * AC.y() - AB.y() * AC.x();

    // float l=sqrt(pow(obstacle_point(0) - host_projected_point1.x, 2) + pow(obstacle_point(1) - host_projected_point1.y, 2));

    obstacle_sl(1) = l;
    return obstacle_sl;
}

Eigen::Vector2d EMPlanner::calc_obstacle_sl(Eigen::Vector2d obstacle_point,planning_msgs::car_path::Ptr &path,Eigen::Vector2d planning_frist_star)
{
    Eigen::Vector2d obstacle_sl;
    int index_match_points = get_closest_point(obstacle_point(0), obstacle_point(1), *path);
    planning_msgs::path_point host_projected_point1 = find_projected_point_Frenet(*path, index_match_points, obstacle_point);
    Eigen::Vector2d AB(host_projected_point1.tor.x, host_projected_point1.tor.y);
    Eigen::Vector2d AC(obstacle_point(0) - host_projected_point1.x, obstacle_point(1) - host_projected_point1.y);
    // obstacle_sl(0) = host_projected_point1.absolute_s+planning_frist_star(0);
    obstacle_sl(0) = host_projected_point1.absolute_s;
    float l = AB.x() * AC.y() - AB.y() * AC.x();

    obstacle_sl(1) = l;
    return obstacle_sl;
}

void EMPlanner::First_run()
{
    string line;
    // 读取txt文件
    int lineCount = 0;
    std::string line_c;
    const std::string emplanner_pkg_path = ResolveEmplannerPackagePath();
    const std::string default_trajectory_file =
        emplanner_pkg_path.empty() ? std::string() : (emplanner_pkg_path + "/text/trajectory.txt");
    ros::NodeHandle pnh("~");
    std::string trajectory_file = default_trajectory_file;
    pnh.param("trajectory_file", trajectory_file, default_trajectory_file);
    fstream file_(trajectory_file.c_str());
    if (!file_.is_open()) {
        ROS_ERROR_STREAM("Failed to open trajectory file: " << trajectory_file);
        return;
    }

    path.points.clear();
    line_record->points.clear();
    line_record_->points.clear();
    line_record_opt->points.clear();
    line_record_watch->points.clear();
    line_qp_Interpolation->points.clear();
    local_path->points.clear();
    local_qp_path->points.clear();
    obs_watch->points.clear();

    // 逐行读取文件，直到读到文件末尾
    while (getline(file_, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream iss(line);
        double x = 0.0;
        double y = 0.0;
        if (!(iss >> x >> y)) {
            continue;
        }
        ++lineCount;
    }
    if (lineCount < 21) {
        ROS_ERROR_STREAM("Trajectory file has too few valid points: " << lineCount
                         << ". Need at least 21 points. file=" << trajectory_file);
        return;
    }

    // 重新定位文件流到文件开头
    file_.clear();  // 清除文件流的状态
    file_.seekg(0, std::ios::beg);  // 将文件流定位到文件开头
    int i_num=0; 
    Eigen::VectorXd QPSolution(lineCount*2);
    while (getline(file_, line))
    {
        std::istringstream iss(line);
        double x = 0.0;
        double y = 0.0;
        if (!(iss >> x >> y)) {
            continue;
        }
        QPSolution(i_num) = x;
        QPSolution(i_num+1) = y;
        i_num=i_num+2;
        // cout<<"lineCount:"<<lineCount<<"   i_num:"<<i_num<<endl;
    }
    total_points_num = lineCount;
    // 路径优化
    // Eigen::VectorXd QPSolution = Smooth_Reference_Line(line_record_);

    // 计算路线航向角(斜率）
    vector<float> temp_phi(total_points_num);
    // cout<<"QPSolution.size:"<<QPSolution.size()<<endl;
    Calculate_OptimizedPath_Heading(QPSolution, temp_phi);

    // 进行均值插值
    Mean_Interpolation(line_record_opt, line_record);
    // 路径参数计算
    Path_Parameter_Calculate(path);

    // 写入优化后路线点    filename << base_path <<  std::to_string(car_scene_.floor) << ;
    // std::ofstream outfile_watch(base_path + "/inspection_path.txt");
    // 检查文件是否成功打开
    // if (!outfile_watch.is_open())
    // {
    //     std::cerr << "Error: Unable to open the file for writing." << std::endl;
    // }

    // for (int i = 0; i < path.points.size(); i++)
    // {
    //     outfile_watch << path.points[i].number << " ";
    //     outfile_watch << path.points[i].x << " ";
    //     outfile_watch << path.points[i].y << " ";
    //     outfile_watch << path.points[i].yaw << " ";
    //     outfile_watch << path.points[i].ds << " ";
    //     outfile_watch << path.points[i].theta << " ";
    //     outfile_watch << path.points[i].kappa << endl;
    // }
    // // 关闭文件流
    // outfile_watch.close();

    // rviz观测数据
    for (int i = 0; i < path.points.size(); i++)
    {
        pcl::PointXYZI points_watch;
        points_watch.x = path.points[i].x;
        points_watch.y = path.points[i].y;
        points_watch.z = 0;
        points_watch.intensity = path.points[i].vel;
        line_record_watch->push_back(points_watch);
        // cout << "number,int,vel,right,left: " << path.number[i - 1] << " " << path.theta[i - 1] << " " << path.vel[i - 1] << " " << path.flag_right_turn[i - 1] << " " << path.flag_left_turn[i - 1] << endl;
    }
    cout<<"line_record_watch_size："<<line_record_watch->points.size();

    cout << "Initialization successful!!! ";
    cout << " Path_total_num:" << path.points.size() << endl;

    index_closest_Car = get_closest_point(Car_Pose->x, Car_Pose->y, path);
    cout<<GREEN<<" Car_Pose-:"<<Car_Pose->x<<","<<Car_Pose->y<<ENDL;

    float x_middle = Car_Pose_middle->x ; // 车轴中点坐标
    float y_middle = Car_Pose_middle->y ;
    float x_back = Car_Pose_back->x; // 车轴中点坐标
    float y_back = Car_Pose_back->y;

    index_closest_Car = get_closest_point(Car_Pose->x, Car_Pose->y, path);
    if (index_closest_Car == 0)
    {
        index_middle_Car = 0;
        index_back_Car = 0;
    }
    else
    {
        index_middle_Car = get_closest_point(x_middle, y_middle, path, 1, index_closest_Car);
        if (index_middle_Car == 0)
        {
            index_back_Car = 0;
        }
        else
        {
            index_back_Car = get_closest_point(x_back, y_back, path, 1, index_middle_Car);
        }
    }

    Eigen::Vector2d host_point(x_middle, y_middle); // 此处的host点是车身中间点
    planning_msgs::path_point host_projected_point = find_projected_point_Frenet(path, index_middle_Car, host_point);
    Eigen::Vector2d planning_frist_star = Find_start_sl_FirstRun(host_point, host_projected_point, index_middle_Car);//车的sl
    float calc_start_dl=tan(Car_Pose->theta-path.points[index_middle_Car].yaw);

    const auto planner_cycle_start = std::chrono::steady_clock::now();
    double latest_dp_cycle_ms = 0.0;
    double latest_qp_cycle_ms = 0.0;

    obstacle_list->obstacles.clear();
    obstacle_list_swap->obstacles.clear();
    obstacle_list_qp_path_sl->obstacles.clear();
    Obstacle_list_Initialization_vision(obstacle_list_lidar);
    for (auto& obstacle : obstacle_list_lidar->obstacles)
    {
        obstacle.isLidarObs = true;
        obstacle_list->obstacles.push_back(obstacle);
    }
    Obstacle_list_Initialization_vision(obstacle_list_vision);
    for (auto& obstacle : obstacle_list_vision->obstacles)
    {
        obstacle.isLidarObs = false;
        obstacle_list->obstacles.push_back(obstacle);
    }
    for (const auto& obstacle : obstacle_list->obstacles)
    {
        pcl::PointXYZI obs;
        obs.x = obstacle.x;
        obs.y = obstacle.y;
        obs_watch->points.push_back(obs);
    }

    ros::WallTime dp_start = ros::WallTime::now();
    const auto dp_cycle_start = std::chrono::steady_clock::now();
    Min_path_nodes min_cost_path_frist;
    bool rl_dp_ok = BuildRlDpMinPath(planning_frist_star(0), planning_frist_star(1), min_cost_path_frist);
    bool rl_dp_used = rl_dp_ok || rl_dp_soft_fail_;
    const bool rl_dp_disabled_by_param =
        (!use_rl_dp_ && rl_dp_disable_reason_ == "use_rl_dp param false");
    if (!rl_dp_ok) {
        if (!rl_dp_disabled_by_param) {
            ++rl_dp_fail_count_;
        }
        if (!rl_dp_soft_fail_) {
            min_cost_path_frist = CalcNodeMinCost(planning_frist_star(0), planning_frist_star(1), calc_start_dl); // 最小路径错误，只有两组点
        }
    }
    
    DP_path_sl = InterpolatePoints(min_cost_path_frist, planning_frist_star(0));
    latest_dp_cycle_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - dp_cycle_start).count();
    ros::WallDuration dp_cost = ros::WallTime::now() - dp_start;
    ROS_INFO_STREAM("DP source=" << (rl_dp_used ? "RL_DP" : "classic")
                    << ", time=" << dp_cost.toSec() * 1000.0 << " ms (first_run)"
                    << ", rl_dp_fail_count=" << rl_dp_fail_count_);
    if (!rl_dp_ok && !rl_dp_disabled_by_param) {
        const std::string reason = rl_dp_last_fail_reason_.empty() ? "unknown" : rl_dp_last_fail_reason_;
        ROS_WARN_STREAM_THROTTLE(1.0, "RL_DP " << (rl_dp_soft_fail_ ? "soft" : "hard")
                                   << " fail count=" << rl_dp_fail_count_
                                   << ", reason: " << reason);
    }

    const auto qp_cycle_start = std::chrono::steady_clock::now();
    QP_path_sl_global = cacl_qp_path(planning_frist_star(0), planning_frist_star(1), 0, 0);
    latest_qp_cycle_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - qp_cycle_start).count();

    vector<Eigen::Vector2d> planning_dp_path_xy = FrenetToXY(DP_path_sl, path);
    vector<Eigen::Vector2d> planning_qp_path_xy = FrenetToXY(QP_path_sl_global, path);
    // 计算路线航向角(斜率）
    vector<float> temp_phi_qp(planning_qp_path_xy.size());
    pcl::PointCloud<pcl::PointXYZI>::Ptr line_qp_path(new pcl::PointCloud<pcl::PointXYZI>); // 存储计算osqp求解出的解
    Calculate_OptimizedPath_Heading(planning_qp_path_xy, temp_phi_qp, line_qp_path);

    // 进行均值插值
    Mean_Interpolation(line_qp_path, line_qp_Interpolation);
    planning_msgs::car_path::Ptr QP_path(new planning_msgs::car_path);
    // QP_xy转化为pcl发布，用来观测
    QP_Path_Publish(QP_path, line_qp_Interpolation);
    Calc_QP_path_param(QP_path);

    for (int i = 0; i < planning_dp_path_xy.size(); i++)
    {
        pcl::PointXYZI local_dp_path_point;
        local_dp_path_point.x = planning_dp_path_xy[i](0);
        local_dp_path_point.y = planning_dp_path_xy[i](1);
        local_path->points.push_back(local_dp_path_point);
    }
    for (int i = 0; i < planning_qp_path_xy.size(); i++)
    {
        pcl::PointXYZI local_qp_path_point;
        local_qp_path_point.x = planning_qp_path_xy[i](0);
        local_qp_path_point.y = planning_qp_path_xy[i](1);
        local_qp_path->points.push_back(local_qp_path_point);
    }

    // dp_path发布
    sensor_msgs::PointCloud2 ss_local_path;
    pcl::toROSMsg(*local_path, ss_local_path);
    ss_local_path.header.frame_id = "velodyne";
    line_pub_local_path.publish(ss_local_path);
    local_path.reset(new pcl::PointCloud<pcl::PointXYZI>);
    // qp_path观测发布
    sensor_msgs::PointCloud2 ss_local_qp_path;
    pcl::toROSMsg(*local_qp_path, ss_local_qp_path);
    ss_local_qp_path.header.frame_id = "velodyne";
    line_pub_local_qp_path_watch.publish(ss_local_qp_path);
    local_qp_path.reset(new pcl::PointCloud<pcl::PointXYZI>);

    QP_path->RunningNormally = QpPathRunningNormally;
    latest_dp_path_sl_ = DP_path_sl;
    latest_qp_path_sl_ = QP_path_sl_global;
    latest_dp_path_xy_ = planning_dp_path_xy;
    latest_qp_path_xy_ = planning_qp_path_xy;
    latest_qp_path_msg_ = *QP_path;
    latest_qp_running_normally_ = QpPathRunningNormally;
    latest_dp_source_ = rl_dp_used ? "RL_DP" : "classic";
    latest_dp_sampling_ms_ = latest_dp_cycle_ms;
    latest_qp_optimization_ms_ = latest_qp_cycle_ms;
    latest_speed_planning_ms_ = 0.0;
    latest_speed_plan_available_ = false;
    latest_speed_qp_points_ = Speed_plan_points();
    latest_speed_dp_path_s_.clear();
    latest_speed_dp_st_nodes_ = Speed_Plan_DP_ST_nodes();
    latest_speed_obstacle_list_qp_path_sl_ = planning_msgs::ObstacleList();
    latest_planner_cycle_ms_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - planner_cycle_start).count();
    has_latest_plan_result_ =
        !latest_dp_path_xy_.empty() && !latest_qp_path_msg_.points.empty();
    line_pub_local_qp_path.publish(*QP_path);

    line_pub_path.publish(path);

}

void EMPlanner::Plan(planning_msgs::car_scene &car_scene)
{
    const auto planner_cycle_start = std::chrono::steady_clock::now();
    double dp_cycle_ms = 0.0;
    double qp_cycle_ms = 0.0;
    double speed_planning_ms = 0.0;

    if (path.points.size() < 5) {
        ROS_ERROR_THROTTLE(1.0, "EMPlanner path is not initialized, skip planning.");
        return;
    }

    // cout<<SHAN<<"Car_Pose_H:"<<index_closest_Car<<" ,"<<Car_Pose->x<<" ,"<<Car_Pose->y<<ENDL;
    // clock_t time_end2 = clock();
    // std::cout<<"plan!!!"<<std::endl;

    // if (flag_is_first_run)
    // {


    // } // 首次循环结束

    // else
    // {
        clock_t time_start = clock();

        planning_msgs::ObstacleList::Ptr obstacle_list_temp(new planning_msgs::ObstacleList);
        obstacle_list->obstacles.clear();
        obstacle_list_swap->obstacles.clear();
        obstacle_list_qp_path_sl->obstacles.clear();
        obs_watch->points.clear();
         
        // *obstacle_list = *Obstacle_list;
        // Obstacle_list_Initialization(obstacle_list);

        Obstacle_list_Initialization_vision(obstacle_list_lidar);//视觉感知障碍物初始化
        for (auto& obstacle: obstacle_list_lidar->obstacles) //视觉与雷达感知合并为一个变量
        {   
            obstacle.isLidarObs=true;
            obstacle_list->obstacles.push_back(obstacle);
        }

        Obstacle_list_Initialization_vision(obstacle_list_vision);//视觉感知障碍物初始化
        for (auto& obstacle: obstacle_list_vision->obstacles) //视觉与雷达感知合并为一个变量
        {
            obstacle.isLidarObs=false;
            obstacle_list->obstacles.push_back(obstacle);
        }

        for (int i = 0; i < obstacle_list->obstacles.size(); i++)
        {
            pcl::PointXYZI obs;
            obs.x=obstacle_list->obstacles[i].x;
            obs.y=obstacle_list->obstacles[i].y;
            obs_watch->points.push_back(obs);
        }
        // clock_t time_end01 = clock();
        // double time_diff01 = static_cast<double>(time_end01 - time_start) / CLOCKS_PER_SEC;
        // std::cout << "obs初始化: " << time_diff01 << " 秒" << std::endl;
        time_start = clock();
        // for (auto& obstacle: obstacle_list->obstacles) //视觉与雷达感知合并为一个变量
        // {
        //     cout<<"obs:s:"<<obstacle.s-host_middle_sl(0)<<","<<obstacle.max_s-host_middle_sl(0)<<","<<obstacle.min_s-host_middle_sl(0)<<" l:"<<obstacle.max_l<<","<<obstacle.min_l<<" vel:"<< obstacle.x_vel<<","<<obstacle.y_vel<<endl;
        // }
        // for (int i = 0; i < obstacle_list->obstacles.size(); i++)
        // {
        //     if (abs(obstacle_list->obstacles[i].l) < 2)
        //     {
        //         obstacle_list_swap->obstacles.push_back(obstacle_list->obstacles[i]);
        //     }
        // }
        // *obstacle_list = *obstacle_list_swap;


        // obstacle_list_swap.reset(new planning_msgs::ObstacleList);

        float x_middle = Car_Pose_middle->x ; // 车轴中点坐标
        float y_middle = Car_Pose_middle->y ;
        float x_back = Car_Pose_back->x; // 车轴中点坐标
        float y_back = Car_Pose_back->y;
        index_closest_Car = get_closest_point(Car_Pose->x, Car_Pose->y, path);

        if (index_closest_Car == 0)
        {
            index_middle_Car = 0;
            index_back_Car = 0;
        }
        else
        {
            index_middle_Car = get_closest_point(x_middle, y_middle, path, 1, index_closest_Car);
            if (index_middle_Car == 0)
            {
                index_back_Car = 0;
            }
            else
            {
                index_back_Car = get_closest_point(x_back, y_back, path, 1, index_middle_Car);
            }
        }
        float control_error=sqrt(pow(path.points[index_closest_Car].x - Car_Pose->x, 2) +pow(path.points[index_closest_Car].y - Car_Pose->y, 2));
        if(control_error>err_max)
        {err_max=control_error;}
        cout<<LIGHT_RED<<" 跟踪误差："<<control_error<<" max_error:"<<err_max<<ENDL;
                // 写入优化后路线点
        // std::ofstream outfile_error(play_path_path + "/text/data_error.txt");
        // // 检查文件是否成功打开
        // if (!outfile_error.is_open())
        // {
        //     std::cerr << "Error: Unable to open the file for writing." << std::endl;
        // }

        // for (int i = 0; i < path.points.size(); i++)
        // {
        //     outfile_error << control_error << " ";
        //     outfile_error << Car_Pose->x << " ";
        //     outfile_error << Car_Pose->y << " ";
        //     outfile_error << path.points[index_closest_Car].x << " ";
        //     outfile_error << path.points[index_closest_Car].y << endl;
        // }
        // // 关闭文件流
        // outfile_error.close();
        // clock_t time_end00 = clock();
        // double time_diff00 = static_cast<double>(time_end00 - time_start) / CLOCKS_PER_SEC;
        // std::cout << "文件读写: " << time_diff00 << " 秒" << std::endl;
        // time_start = clock();
        // cout << " index_closest_Car:" << index_closest_Car << " index_middle_Car:" << index_middle_Car << " index_back_Car:" << index_back_Car << endl;

        Eigen::Vector2d host_point(x_middle, y_middle); // 此处的host点是车身中间点
        planning_msgs::path_point host_projected_middle_point = find_projected_point_Frenet(path, index_middle_Car, host_point);
        // Eigen::Vector2d planning_frist_star = Find_start_sl(host_point, host_projected_middle_point,index_middle_Car);
        host_middle_sl = Find_start_sl(host_point, host_projected_middle_point, index_middle_Car);  //全局路径下
        Eigen::Vector2d host_forward_point(Car_Pose->x, Car_Pose->y); // 此处的host点是车前的点的投影
        planning_msgs::path_point host_projected_forward_point = find_projected_point_Frenet(path, index_closest_Car, host_forward_point);
        host_forward_sl = Find_start_sl(host_forward_point, host_projected_forward_point, index_closest_Car); //全局路径下

        Eigen::Vector2d host_back_point(x_back, y_back); // 此处的host点是车前的点的投影
        planning_msgs::path_point host_projected_back_point = find_projected_point_Frenet(path, index_back_Car, host_back_point);
        host_back_sl = Find_start_sl(host_back_point, host_projected_back_point, index_back_Car); //全局路径下
   
        // int index_car_forward_qp_path = index2s(QP_path_sl_global, host_forward_sl(0));
        // int index_car_middle_qp_path = index2s(QP_path_sl_global, host_middle_sl(0));
        // //起始点为上一规划路径的车前点
        // Eigen::Vector2d planning_frist_star(QP_path_sl_global.s(index_car_forward_qp_path), QP_path_sl_global.l(index_car_forward_qp_path));
        Eigen::Vector2d planning_frist_star=host_middle_sl;
        // cout<<"host_middle_sl"<<host_middle_sl(0)<<","<<host_middle_sl(1)<<endl;
        //保留上一个时刻的车中点到车前点的规划路线的sl no.(1)
        // Eigen::VectorXd swap_qp_path_s = QP_path_sl_global.s.segment(index_car_middle_qp_path, index_car_forward_qp_path - index_car_middle_qp_path + 1);
        // Eigen::VectorXd swap_qp_path_l = QP_path_sl_global.l.segment(index_car_middle_qp_path, index_car_forward_qp_path - index_car_middle_qp_path + 1);
        //模式判断
        // if(!QpPathRunningNormally&&real_vehicle_speed<=0.01)
        // {
        //     car_direct=-1; //倒车
        // }
        // else{
        //     car_direct=1; //前进
        // }

        //模式判断
        // real_vehicle_speed=0.5;
        // clock_t time_end1 = clock();
        // double time_diff1 = static_cast<double>(time_end1 - time_start) / CLOCKS_PER_SEC;
        // std::cout << "模式识别: " << time_diff1 << " 秒" << std::endl;
        time_start = clock();

        // cout<<LIGHT_CYAN<<"reverseDistance:"<<reverseDistance<<ENDL;
        if(!QpPathRunningNormally ||flagCarStop==1)  //无解且停车
        {   sum_forward_reverse_num++;
            if(sum_forward_reverse_num>20)
            {
                if(flag_is_first_reverse)
                {
                    firstReversePosition(0)=x_middle;
                    firstReversePosition(1)=y_middle;
                }
                flag_is_first_reverse=false;
                flagCalcBackDistance=true;
                car_direct=-1; //倒车
                cout<<RED<<" 无解且停车,开始倒车 "<<ENDL;
            }
            if(!QpPathRunningNormally&&reverseDistance>=maxReverseDistance) //无解且倒车到一定距离，停止倒车并停车等待
            {
                car_wait=1;
            }
            else
            {
                car_wait=0;
            }
        }
        else if(QpPathRunningNormally&&(reverseDistance>=minReverseDistance)  ) //有解且倒车到一定距离了,只要进一次足够
        {
            flagCalcBackDistance=false;
            flag_is_first_reverse=true;
            car_direct=1; //前进
            car_wait=0;
            sum_forward_reverse_num=0;
            sum_back_reverse_num=0;
            firstReversePosition(0)=x_middle;
            firstReversePosition(1)=y_middle;
            cout<<GREEN<<" 有解,且倒车到一定距离,开始前进 "<<ENDL;
        }
        else if(QpPathRunningNormally&&(flagCarStop==0||flagCarStop==2)&&abs(real_vehicle_speed)<0.02&&car_direct==-1)  //为了解决前方无障碍物但是倒车的情况
        {
            sum_flag_cancel_reverse_num++;
            if(sum_flag_cancel_reverse_num>=5)
            {
                flag_is_first_reverse=true;
                car_direct=1; //前进
                car_wait=0;
                sum_forward_reverse_num=0;
                sum_back_reverse_num=0;
                cout<<GREEN<<" 取消倒车,开始前进 "<<ENDL;
            }
        }
        else if(QpPathRunningNormally&&(flagCarStop==0||flagCarStop==2)&&car_direct==1)  //正常行驶状态,参数初始化
        {
            sum_forward_reverse_num=0;
            sum_back_reverse_num=0;
            car_wait=0;
            flag_is_first_reverse=true;
            sum_flag_cancel_reverse_num=0;
            firstReversePosition(0)=x_middle;
            firstReversePosition(1)=y_middle;
            cout<<GREEN<<" 正常行驶状态"<<ENDL;
        }

        if(flagCarStop==2&&car_direct==-1)  //倒车时有后方障碍物
        {
            sum_back_reverse_num++;
            if(sum_back_reverse_num>2)
            {
                flag_is_first_reverse=true;
                car_direct=1; //前进
                car_wait=0;
                sum_forward_reverse_num=0;
                cout<<YELLOW<<" 倒车时有后方障碍物,开始前进 "<<ENDL;
            }
        }
        // cout<<LIGHT_BLUE<<"car_wait = "<<car_wait<<" flagCarStop:"<<flagCarStop<<" real_vehicle_speed:"<<real_vehicle_speed<<ENDL;
        clock_t time_end2 = clock();
        double time_diff2 = static_cast<double>(time_end2 - time_start) / CLOCKS_PER_SEC;
        std::cout << "模式识别: " << time_diff2 << " 秒" << std::endl;
        time_start = clock();
        // //过近状态改变
        // if(flagCarStop==1)
        // {
        //     sum_flag_stop_forward_num++;
        //     if(sum_flag_stop_forward_num>21)
        //     {
        //         car_direct=-1; //倒车
        //         sum_flag_stop_back_num=0;
        //     }
        // }
        // else if(flagCarStop==2)
        // {
        //     sum_flag_stop_back_num++;
        //     if(sum_flag_stop_forward_num>21)
        //     {
        //         car_direct=1; //前进
        //         sum_flag_stop_forward_num=0;
        //     }
        // }
        if(flagCalcBackDistance)
        {
            reverseDistance=sqrt(pow(x_middle-firstReversePosition(0),2)+pow(y_middle-firstReversePosition(1),2));
        }
        else
        {
            reverseDistance=0;
        }
        float calc_start_dl=tan(Car_Pose->theta-path.points[index_middle_Car].yaw);

        //倒车模式
        
        //计算DP最小节点
        const auto dp_cycle_start = std::chrono::steady_clock::now();
        ros::WallTime dp_start = ros::WallTime::now();
        Min_path_nodes min_cost_path_frist;
        bool rl_dp_attempted = (car_direct == 1);
        bool rl_dp_ok = false;
        const bool rl_dp_disabled_by_param =
            (!use_rl_dp_ && rl_dp_disable_reason_ == "use_rl_dp param false");
        if (rl_dp_attempted) {
            rl_dp_ok = BuildRlDpMinPath(planning_frist_star(0), planning_frist_star(1), min_cost_path_frist);
        }
        bool rl_dp_used = rl_dp_ok || (rl_dp_attempted && rl_dp_soft_fail_);
        if (rl_dp_attempted && !rl_dp_ok && !rl_dp_disabled_by_param) {
            ++rl_dp_fail_count_;
        }
        clock_t time_start_dp = clock();
        
        if (!rl_dp_used) {
            min_cost_path_frist = CalcNodeMinCost(planning_frist_star(0), planning_frist_star(1), calc_start_dl); //局部的DP_path
        }
        // clock_t time_end3 = clock();
        // double time_diff3 = static_cast<double>(time_end3 - time_start) / CLOCKS_PER_SEC;
        // std::cout << "DPpath: " << time_diff3 << " 秒" << std::endl;
        time_start = clock();
        
        DP_path_sl = InterpolatePoints(min_cost_path_frist, planning_frist_star(0));  //全局S的DP_path
        const auto dp_cycle_end = std::chrono::steady_clock::now();
        dp_cycle_ms += std::chrono::duration<double, std::milli>(dp_cycle_end - dp_cycle_start).count();
        clock_t time_end1 = clock();
        double time_diff1 = static_cast<double>(time_end1 - time_start_dp) / CLOCKS_PER_SEC;
        std::cout << "CalcNodeMinCost程序运行时间: " << time_diff1 << " 秒 " << std::endl;
        
        if (car_direct == 1) {
            ros::WallDuration dp_cost = ros::WallTime::now() - dp_start;
            ROS_INFO_STREAM("DP source=" << (rl_dp_used ? "RL_DP" : "classic")
                            << ", time=" << dp_cost.toSec() * 1000.0 << " ms"
                            << ", rl_dp_fail_count=" << rl_dp_fail_count_);
            if (!rl_dp_ok && !rl_dp_disabled_by_param) {
                const std::string reason = rl_dp_last_fail_reason_.empty() ? "unknown" : rl_dp_last_fail_reason_;
                ROS_WARN_STREAM_THROTTLE(1.0, "RL_DP " << (rl_dp_soft_fail_ ? "soft" : "hard")
                                           << " fail count=" << rl_dp_fail_count_
                                           << ", reason: " << reason);
            }
        }
        // for (int i = 0; i < DP_path_sl.s.size(); i++)
        // {
        //     cout<<"in_DP_PATH_S:"<<DP_path_sl.s(i)<<endl;
        // }
        
        const auto qp_cycle_start = std::chrono::steady_clock::now();
        QP_path_sl = cacl_qp_path(planning_frist_star(0), planning_frist_star(1), calc_start_dl, 0);
        qp_cycle_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - qp_cycle_start).count();
        // clock_t time_end4 = clock();
        // double time_diff4 = static_cast<double>(time_end4 - time_start) / CLOCKS_PER_SEC;
        // std::cout << "QPpath: " << time_diff4 << " 秒" << std::endl;
        time_start = clock();

        if(car_direct==-1) //倒车
        {
            const auto reverse_dp_cycle_start = std::chrono::steady_clock::now();
            ros::WallTime dp_reverse_start = ros::WallTime::now();
            Min_path_nodes min_cost_path_frist = CalcDpPathNodeMinCost_reverse(planning_frist_star(0), planning_frist_star(1));
            DP_path_sl_r = InterpolateDpPathPoints_reverse(min_cost_path_frist,planning_frist_star(0));
            const auto reverse_dp_cycle_end = std::chrono::steady_clock::now();
            dp_cycle_ms += std::chrono::duration<double, std::milli>(reverse_dp_cycle_end - reverse_dp_cycle_start).count();
            const auto reverse_qp_cycle_start = std::chrono::steady_clock::now();
            QP_path_sl_r = calcQpPath_reverse(planning_frist_star(0), planning_frist_star(1), calc_start_dl, 0);
            qp_cycle_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - reverse_qp_cycle_start).count();
            DP_path_sl=DP_path_sl_r;
            QP_path_sl=QP_path_sl_r;
            planning_dp_path_xy = FrenetToXY_r(DP_path_sl, path); //dp由frenet转XY坐标
            planning_qp_path_xy = FrenetToXY_r(QP_path_sl, path); //qp由frenet转XY坐标
            ros::WallDuration dp_reverse_cost = ros::WallTime::now() - dp_reverse_start;
            ROS_INFO_STREAM("DP source=reverse_classic, time="
                            << dp_reverse_cost.toSec() * 1000.0 << " ms");
            // clock_t time_end5 = clock();
            // double time_diff5 = static_cast<double>(time_end5 - time_start) / CLOCKS_PER_SEC;
            // std::cout << "倒车DP: " << time_diff5 << " 秒" << std::endl;
        }
        else
        {
            planning_dp_path_xy = FrenetToXY(DP_path_sl, path); //dp由frenet转XY坐标
            planning_qp_path_xy = FrenetToXY(QP_path_sl, path); //qp由frenet转XY坐标
        }
        const std::string dp_source_label =
            (car_direct == -1) ? "reverse_classic" : (rl_dp_used ? "RL_DP" : "classic");
        
        // for (int i = 0; i < QP_path_sl.s.size(); i++)
        // {
        //     cout<<"QP DP SL:"<<QP_path_sl.s(i)<<","<<QP_path_sl.dl(i)<<";"<<DP_path_sl.s(i)<<","<<DP_path_sl.l(i)<<endl;
        // }
        


        //保留上一个时刻的车中点到车前点的规划路线的sl no.(2)
        // int concatenation_size = QP_path_sl.s.size() + index_car_forward_qp_path - index_car_middle_qp_path + 1;
        // Eigen::VectorXd QP_path_sl_new_s(concatenation_size);
        // Eigen::VectorXd QP_path_sl_new_l(concatenation_size);
        // QP_path_sl_new_s << swap_qp_path_s, QP_path_sl.s;
        // QP_path_sl_new_l << swap_qp_path_l, QP_path_sl.l;
        // if (RunningNormally)
        // {
        //     QP_path_sl.s = QP_path_sl_new_s;velodyne
        //     QP_path_sl.l = QP_path_sl_new_l;
        //     QP_path_sl_global.s = QP_path_sl_new_s;
        //     QP_path_sl_global.l = QP_path_sl_new_l;
        // }

        
        // cout<<"planning_dp_path_xy_size:"<<planning_dp_path_xy.size()<<endl;
        // for (int i = 0; i < planning_dp_path_xy.size(); i++)
        // {
        //     cout<<"planning_dp_path_xy xy:"<<planning_dp_path_xy[i][0]<<","<<planning_dp_path_xy[i][1]<<endl;
        // }
       
        // Plot_SL_Graph(host_forward_sl, DP_patvelodyneh_sl, QP_path_sl); //SL图绘画
      
        vector<float> temp_phi_qp(planning_qp_path_xy.size());
        pcl::PointCloud<pcl::PointXYZI>::Ptr line_qp_path(new pcl::PointCloud<pcl::PointXYZI>);
        if(car_direct==1) //前进
        {
            Calculate_OptimizedPath_Heading(planning_qp_path_xy, temp_phi_qp, line_qp_path);
        }
        else  //倒车
        {
            Calculate_OptimizedPath_Heading_reverse(planning_qp_path_xy, temp_phi_qp, line_qp_path);
        }
        // else{
        //     cout<<"Car_State Error 02!"<<endlvelodyne;
        // }
      
        // 进行均值插值
        Mean_Interpolation(line_qp_path, line_qp_Interpolation);

        // QP_xy转化为pcl发布，用来观测
        QP_Path_Publish(QP_path, line_qp_Interpolation);
        Calc_QP_path_param(QP_path);
        //记录最近一次的路径规划可行解
        if(QpPathRunningNormally)
        {
            QP_path_last_success_run.reset(new planning_msgs::car_path);
            planning_qp_path_xy_last_success_run=planning_qp_path_xy;
            *QP_path_last_success_run=*QP_path;
        }
        else if(car_direct==1) //无解且车为前进状态
        {
            planning_qp_path_xy=planning_qp_path_xy_last_success_run;
            if (QP_path_last_success_run) {  // 确保 path_ptr 不为空
                *QP_path=*QP_path_last_success_run;
            } else {
               
                ROS_ERROR("Error! No successful_path!");
                //  return ;
            }
            
            cout<<LIGHT_WHITE<<"当前无解，显示为最后一次可行解！！！"<<ENDL;
        }
          
        time_start = clock();
        bool speed_plan_available = false;
        Speed_plan_points latest_speed_qp_points_local;
        std::vector<double> latest_speed_dp_path_s_local;
        int latest_speed_dp_last_feasible_index_local = -1;
        Speed_Plan_DP_ST_nodes latest_speed_dp_st_nodes_local;
        planning_msgs::ObstacleList latest_speed_obstacle_list_qp_path_sl_local;
        //Speed Plan
        if(car_direct==1)
        {
            const auto speed_plan_start = std::chrono::steady_clock::now();
            *obstacle_list_qp_path_sl=*obstacle_list;   //实际测试要注释开***
            
            Obstacle_list_Initialization_qp_path(obstacle_list_qp_path_sl,host_forward_sl); //将障碍物投影到QPpath下

            Speed_plan_calc_obs_ST(obstacle_list_qp_path_sl);   
            // Speed_Plan_DP_ST_nodes ST_Nodes= CalcSpeedPlanDp_StNodes(QP_path_sl_global.s(index_car_forward_qp_path));

            Speed_Plan_DP_ST_nodes ST_Nodes= CalcSpeedPlanDp_StNodes(host_middle_sl(0));  //生成DP坐标节点

            vector<double> Speed_DP_path_s=CalcSpeedDpCost(obstacle_list_qp_path_sl, ST_Nodes);

            Speed_plan_points speed_qp_points=CalcSpeedPlan_QpPath(obstacle_list_qp_path_sl, ST_Nodes, Speed_DP_path_s,host_middle_sl(0));//速度qp规划的结果
            speed_planning_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - speed_plan_start).count();
            latest_speed_qp_points_local = speed_qp_points;
            latest_speed_dp_path_s_local = Speed_DP_path_s;
            latest_speed_dp_last_feasible_index_local = speed_dp_last_feasible_index_;
            latest_speed_dp_st_nodes_local = ST_Nodes;
            latest_speed_obstacle_list_qp_path_sl_local = *obstacle_list_qp_path_sl;
            speed_plan_available = !speed_qp_points.s_time.empty();

        }

        // plt::ion(); // 动态画图
        // Plot_ST_Graph(speed_qp_points,Speed_DP_path_s,obstacle_list_qp_path_sl); //ST图绘画
        //dp_path转化为pcl格式方便rviz观察
        for (int i = 0; i < planning_dp_path_xy.size(); i++)
        {
            pcl::PointXYZI local_dp_path_point;
            local_dp_path_point.x = planning_dp_path_xy[i](0);
            local_dp_path_point.y = planning_dp_path_xy[i](1);
            local_path->points.push_back(local_dp_path_point);
        }
        //qp_path转化为pcl格式方便rviz观察
        
        local_qp_path->points.reserve(planning_qp_path_xy.size());
        for (int i = 0; i < planning_qp_path_xy.size(); i++)
        {
            pcl::PointXYZI local_qp_path_point;
            local_qp_path_point.x = planning_qp_path_xy[i](0);
            local_qp_path_point.y = planning_qp_path_xy[i](1);
            local_qp_path->points.push_back(local_qp_path_point);
        }
        
        // dp_path发布
        sensor_msgs::PointCloud2 ss_local_path;
        pcl::toROSMsg(*local_path, ss_local_path);
        ss_local_path.header.frame_id = "velodyne";
        line_pub_local_path.publish(ss_local_path);
        local_path.reset(new pcl::PointCloud<pcl::PointXYZI>);
        // qp_path观测发布
        sensor_msgs::PointCloud2 ss_local_qp_path;
        pcl::toROSMsg(*local_qp_path, ss_local_qp_path);
        ss_local_qp_path.header.frame_id = "velodyne";
        line_pub_local_qp_path_watch.publish(ss_local_qp_path);
        local_qp_path.reset(new pcl::PointCloud<pcl::PointXYZI>);
        // obs观测发布
        sensor_msgs::PointCloud2 ss_local_obs;
        pcl::toROSMsg(*obs_watch, ss_local_obs);
        ss_local_obs.header.frame_id = "velodyne";
        obs_watch_pub.publish(ss_local_obs);
        obs_watch.reset(new pcl::PointCloud<pcl::PointXYZI>);
        
        // if(!QpPathRunningNormally&&car_direct==1)
        // {
        //     QP_path_last_success_run->car_direct=car_direct;
        //     QP_path_last_success_rund->V_straight=calc_car_speed;
        //     line_pub_local_qp_path.publish(*QP_path_last_success_run);
        // }
        // else
        // {
            reach_end=false;
            double dis=sqrt(pow(path.points[path.points.size() - 1].x - Car_Pose->x, 2) +pow(path.points[path.points.size() - 1].y - Car_Pose->y, 2));
            if (dis < END_Stop_Distance || index_middle_car==path.points.size() - 1)
            {
                QP_path->ReachTarget=true;
                if(Car_Pose->x!=0.0&&Car_Pose->y!=0.0)//为防止坐标未初始化
                    reach_end=true;
            }

            QP_path->car_direct=car_direct;
            QP_path->path_type=1;
            if(QP_path->car_direct==-1)
            {
                 QP_path->V_straight=-0.5;
            }
            else
            {
                QP_path->V_straight=calc_car_speed;
            }
            QP_path->car_wait=car_wait;
            QP_path->RunningNormally = QpPathRunningNormally;
            latest_dp_path_sl_ = DP_path_sl;
            latest_qp_path_sl_ = QP_path_sl;
            latest_dp_path_xy_ = planning_dp_path_xy;
            latest_qp_path_xy_ = planning_qp_path_xy;
            latest_qp_path_msg_ = *QP_path;
            latest_qp_running_normally_ = QpPathRunningNormally;
            latest_dp_source_ = dp_source_label;
            latest_dp_sampling_ms_ = dp_cycle_ms;
            latest_qp_optimization_ms_ = qp_cycle_ms;
            latest_speed_planning_ms_ = speed_planning_ms;
            latest_speed_plan_available_ = speed_plan_available;
            latest_speed_qp_points_ = latest_speed_qp_points_local;
            latest_speed_dp_path_s_ = latest_speed_dp_path_s_local;
            latest_speed_dp_last_feasible_index_ = latest_speed_dp_last_feasible_index_local;
            latest_speed_dp_st_nodes_ = latest_speed_dp_st_nodes_local;
            latest_speed_obstacle_list_qp_path_sl_ = latest_speed_obstacle_list_qp_path_sl_local;
            has_latest_plan_result_ =
                !latest_dp_path_xy_.empty() && !latest_qp_path_msg_.points.empty();
            line_pub_local_qp_path.publish(*QP_path);
        // }


        // QP_path.header.frame_id = "velodyne";
        
        QP_path.reset(new planning_msgs::car_path);

        // line_pub_local_qp_path.publish(QP_path);
        // line_qp_Interpolation

        // path.header.frame_id = "velodyne";
        path.car_direct=car_direct;
        path.car_wait=car_wait;
        path.V_straight=calc_car_speed;
        line_pub_path.publish(path);

        // clock_t time_end = clock();
        // double time_diff = static_cast<double>(time_end - time_start) / CLOCKS_PER_SEC;
        // std::cout << "程序运行时间: " << time_diff << " 秒" << std::endl;
    // }

    // 发布全局观测路线
    sensor_msgs::PointCloud2 line_record_pub_watch;
    // cout<<"!!!!!!!!!line_record_watch_size!!!!!!!!!!!!!!!!!!!："<<line_record_watch->points.size();
    pcl::toROSMsg(*line_record_watch, line_record_pub_watch);
    line_record_pub_watch.header.frame_id = "velodyne";
    line_pub_watch.publish(line_record_pub_watch);
    // path.header.frame_id = "velodyne";
    // line_pub_path.publish(path);

    line_qp_Interpolation.reset(new pcl::PointCloud<pcl::PointXYZI>);

    line_qp_Interpolation.reset(new pcl::PointCloud<pcl::PointXYZI>);
    const auto planner_cycle_end = std::chrono::steady_clock::now();
    const double planner_cycle_ms =
        std::chrono::duration<double, std::milli>(planner_cycle_end - planner_cycle_start).count();
    latest_planner_cycle_ms_ = planner_cycle_ms;
    dp_time_total_ms_ += dp_cycle_ms;
    planner_time_total_ms_ += planner_cycle_ms;
    dp_time_max_ms_ = std::max(dp_time_max_ms_, dp_cycle_ms);
    planner_time_max_ms_ = std::max(planner_time_max_ms_, planner_cycle_ms);
    ++timing_sample_count_;
    if (timing_sample_count_ % timing_print_every_ == 0) {
        const double samples = static_cast<double>(timing_sample_count_);
        const double dp_avg_ms = dp_time_total_ms_ / samples;
        const double planner_avg_ms = planner_time_total_ms_ / samples;
        ROS_INFO_STREAM("[Timing] sample=" << timing_sample_count_
                        << ", dp_cycle_ms=" << dp_cycle_ms
                        << ", dp_total_ms=" << dp_time_total_ms_
                        << ", dp_avg_ms=" << dp_avg_ms
                        << ", dp_max_ms=" << dp_time_max_ms_
                        << ", planner_cycle_ms=" << planner_cycle_ms
                        << ", planner_total_ms=" << planner_time_total_ms_
                        << ", planner_avg_ms=" << planner_avg_ms
                        << ", planner_max_ms=" << planner_time_max_ms_);
    }
    // cout << "Path_total_num:" << line_record->points.size();
    // cout << "   Successfully load path!!!" << endl;
}

Eigen::Vector2d EMPlanner::ObstacleSL2XY(planning_msgs::Obstacle obs)
{
    int index_obs2=index2s(path,obs.s);
    planning_msgs::path_point proj_point_ = find_projected_point_Frenet(path, index_obs2, obs.s, obs.l);
    Eigen::Vector2d match_point_xy(proj_point_.x, proj_point_.y);
    Eigen::Vector2d point_nor(proj_point_.nor.x, proj_point_.nor.y);
    Eigen::Vector2d point_temp_xy = match_point_xy + obs.l * point_nor;
    return point_temp_xy;
}

void EMPlanner::Calc_QP_path_param(planning_msgs::car_path::Ptr &QP_path)
{
    QP_path->points[0].absolute_s=0;
    QP_path->points[0].ds=0;
    QP_path->points[0].tor.x=cos(QP_path->points[0].yaw);
    QP_path->points[0].tor.y=sin(QP_path->points[0].yaw);
    QP_path->points[0].nor.x=-sin(QP_path->points[0].yaw);
    QP_path->points[0].nor.y=cos(QP_path->points[0].yaw);
    double absolute_s=0;
    for (int i = 1; i < QP_path->points.size(); i++)
    {
        QP_path->points[i].ds=sqrt(pow(QP_path->points[i].x-QP_path->points[i-1].x,2)+pow(QP_path->points[i].y-QP_path->points[i-1].y,2)); 
        absolute_s+=QP_path->points[i].ds;
        QP_path->points[i].absolute_s=absolute_s;
        QP_path->points[i].tor.x=cos(QP_path->points[i].yaw);
        QP_path->points[i].tor.y=sin(QP_path->points[i].yaw);
        QP_path->points[i].nor.x=-sin(QP_path->points[i].yaw);
        QP_path->points[i].nor.y=cos(QP_path->points[i].yaw);
    }
}

// void EMPlanner::Plot_ST_Graph(Speed_plan_points speed_qp_points,vector<double> Speed_DP_path_s,planning_msgs::ObstacleList::Ptr &obstacle_list_qp_path_sl)
// {
//     plt::ion();
//     plt::figure(2); // 动态画图
//     plt::clf(); // 清空画布
//     plt::named_plot("QP_ST", speed_qp_points.s_time, speed_qp_points.s, "--");
//     plt::named_plot("DP_ST", speed_qp_points.s_time, Speed_DP_path_s, "green");
//     // for (int i = 0; i < obstacle_list_qp_path_sl->obstacles.size(); i++)
//     // {
//     //     vector<double> obs_x,obs_y;
//     //     obs_x.push_back(obstacle_list_qp_path_sl->obstacles[i].speed_plan_s_in);
//     //     obs_x.push_back(obstacle_list_qp_path_sl->obstacles[i].speed_plan_s_out);
//     //     obs_y.push_back(obstacle_list_qp_path_sl->obstacles[i].min_s);
//     //     obs_y.push_back(obstacle_list_qp_path_sl->obstacles[i].max_s);
//     //     plt::named_plot("DP_ST", speed_qp_points.s_time, Speed_DP_path_s, "red");
//     // }
//     for (int i = 0; i < obstacle_list_qp_path_sl->obstacles.size(); i++)
//     {
//         if(obstacle_list_qp_path_sl->obstacles[i].is_consider)
//         {
//             std::vector<double> obs_t = {obstacle_list_qp_path_sl->obstacles[i].speed_plan_t_in,obstacle_list_qp_path_sl->obstacles[i].speed_plan_t_out};
//             std::vector<double> obs_s = {obstacle_list_qp_path_sl->obstacles[i].speed_plan_s_in, obstacle_list_qp_path_sl->obstacles[i].speed_plan_s_out};
//             // cout<<"obs_t_in/out:"<<obstacle_list_qp_path_sl->obstacles[i].speed_plan_t_in<<","<<obstacle_list_qp_path_sl->obstacles[i].speed_plan_t_out<<endl;
//             plt::plot(obs_t, obs_s,"red");
//         }
//     }
    
//     // plt::named_plot("Obstacle", speed_qp_points.s_time, Speed_DP_path_s, "green");
//     int max_y=Speed_DP_path_s.back()>speed_qp_points.s.back()?Speed_DP_path_s.back():speed_qp_points.s.back();
//     // plt::ylim(int(speed_qp_points.s[0]), max_y+1);
//     plt::xlim(0, 7);
//     plt::legend();
//     plt::title("ST");
//     plt::pause(0.0000001); // 等待0.001秒，让GUI有机会响应用户交互
//     plt::show();
// }

//这里的T_in,T_out,应该考虑的更细致一些，应该考虑障碍物的外形（待优化）
void EMPlanner::Speed_plan_calc_obs_ST(planning_msgs::ObstacleList::Ptr &obstacle_list_qp_path_sl)
{
    const float st_left_bound = st_lateral_limit;
    const float st_right_bound = -st_lateral_limit;
    for (int i = 0; i < obstacle_list_qp_path_sl->obstacles.size(); i++)
    {
        if(QpPathRunningNormally)
        {
            if (obstacle_list_qp_path_sl->obstacles[i].is_dynamic_obs==false ||
                obstacle_list_qp_path_sl->obstacles[i].absolute_s_max<=host_back_sl(0))
            {
                obstacle_list_qp_path_sl->obstacles[i].is_consider = false;
                continue;
            }
        }
        else
        {
            if (obstacle_list_qp_path_sl->obstacles[i].absolute_s_max<=host_back_sl(0))
            {
                obstacle_list_qp_path_sl->obstacles[i].is_consider = false;
                continue;
            }
        }
        // 当无解时会把静态障碍物也纳入速度规划；这时给它一个很小的朝中心线速度，
        // 只用于构造 ST 占据区，不能覆盖真实动态障碍物的 SL 速度。
        if(!obstacle_list_qp_path_sl->obstacles[i].is_dynamic_obs)
        {
            obstacle_list_qp_path_sl->obstacles[i].l_vel =
                obstacle_list_qp_path_sl->obstacles[i].l >= 0.0 ? -0.01 : 0.01;
            obstacle_list_qp_path_sl->obstacles[i].s_vel = 0.01;
        }
        if (std::abs(obstacle_list_qp_path_sl->obstacles[i].l_vel) < 1e-6)
        {
            if (obstacle_list_qp_path_sl->obstacles[i].max_l < st_right_bound ||
                obstacle_list_qp_path_sl->obstacles[i].min_l > st_left_bound)
            {
                obstacle_list_qp_path_sl->obstacles[i].is_consider = false;
                continue;
            }
            obstacle_list_qp_path_sl->obstacles[i].is_consider = true;
            obstacle_list_qp_path_sl->obstacles[i].speed_plan_t_in = 0.0;
            obstacle_list_qp_path_sl->obstacles[i].speed_plan_t_out = plan_time;
            double s_min_at_start = obstacle_list_qp_path_sl->obstacles[i].min_s;
            double s_min_at_end = obstacle_list_qp_path_sl->obstacles[i].min_s +
                                  obstacle_list_qp_path_sl->obstacles[i].s_vel * plan_time;
            double ignored_max_s = obstacle_list_qp_path_sl->obstacles[i].max_s;
            ComputeObstacleStripSRange(
                obstacle_list_qp_path_sl->obstacles[i],
                0.0,
                st_right_bound,
                st_left_bound,
                &s_min_at_start,
                &ignored_max_s);
            ComputeObstacleStripSRange(
                obstacle_list_qp_path_sl->obstacles[i],
                plan_time,
                st_right_bound,
                st_left_bound,
                &s_min_at_end,
                &ignored_max_s);
            obstacle_list_qp_path_sl->obstacles[i].speed_plan_s_in =
                s_min_at_start > sMinDistance ? s_min_at_start : sMinDistance;
            obstacle_list_qp_path_sl->obstacles[i].speed_plan_s_out =
                s_min_at_end > sMinDistance ? s_min_at_end : sMinDistance;
            continue;
        }
        const float t_cross_upper =
            (st_left_bound - obstacle_list_qp_path_sl->obstacles[i].min_l) /
            obstacle_list_qp_path_sl->obstacles[i].l_vel;
        const float t_cross_lower =
            (st_right_bound - obstacle_list_qp_path_sl->obstacles[i].max_l) /
            obstacle_list_qp_path_sl->obstacles[i].l_vel;
        float t_in = std::min(t_cross_upper, t_cross_lower);
        float t_out = std::max(t_cross_upper, t_cross_lower);
        double s_in = obstacle_list_qp_path_sl->obstacles[i].min_s +
                      obstacle_list_qp_path_sl->obstacles[i].s_vel * t_in;
        double s_out = obstacle_list_qp_path_sl->obstacles[i].min_s +
                       obstacle_list_qp_path_sl->obstacles[i].s_vel * t_out;
        if (t_in > plan_time || t_out < 0) //在plan_time秒内无影响则舍弃
        {
            obstacle_list_qp_path_sl->obstacles[i].is_consider = false;
            continue;
        }
        t_in=t_in>0?t_in:0;
        t_out=t_out>plan_time?plan_time:t_out;
        obstacle_list_qp_path_sl->obstacles[i].is_consider = true;
        double ignored_max_s = obstacle_list_qp_path_sl->obstacles[i].max_s;
        ComputeObstacleStripSRange(
            obstacle_list_qp_path_sl->obstacles[i],
            t_in,
            st_right_bound,
            st_left_bound,
            &s_in,
            &ignored_max_s);
        ComputeObstacleStripSRange(
            obstacle_list_qp_path_sl->obstacles[i],
            t_out,
            st_right_bound,
            st_left_bound,
            &s_out,
            &ignored_max_s);

        obstacle_list_qp_path_sl->obstacles[i].speed_plan_s_in = s_in>sMinDistance?s_in:sMinDistance;
        obstacle_list_qp_path_sl->obstacles[i].speed_plan_s_out = s_out>sMinDistance?s_out:sMinDistance;
        obstacle_list_qp_path_sl->obstacles[i].speed_plan_t_in = t_in;
        obstacle_list_qp_path_sl->obstacles[i].speed_plan_t_out = t_out;
        // cout<<"obs_ST "<<i<<": s_in:"<<obstacle_list_qp_path_sl->obstacles[i].speed_plan_s_in<<",s_out: "<<obstacle_list_qp_path_sl->obstacles[i].speed_plan_s_out<<",t_in:"<<obstacle_list_qp_path_sl->obstacles[i].speed_plan_t_in<<",t_out:"<<obstacle_list_qp_path_sl->obstacles[i].speed_plan_t_out<<endl;

    }

}

Speed_Plan_DP_ST_nodes EMPlanner::CalcSpeedPlanDp_StNodes(float plan_start_s)
{
    const int t_node_num = std::max(1, static_cast<int>(std::lround(plan_time / speed_plan_t_dt)));
    Speed_Plan_DP_ST_nodes DP_ST_nodes;
    vector<float> s_node;
    s_node.reserve(speed_plan_distance/ST_s_min_step+1);

    for (float i = 0.0; i <= speed_plan_distance; i += ST_s_min_step) // 0-7m 分辨率 0.1m 
    {
        s_node.push_back(i);
    }

    const size_t s_node_num = s_node.size();

    Speed_DP_col_nodes speed_DP_col_start_nodes;    
    Speed_DP_Single_node speed_single_start_node;
    speed_single_start_node.node_s = s_node[0];
    speed_single_start_node.node_t = 0;
    speed_single_start_node.node_s_dot = real_vehicle_speed; // 待修改：这里如果接入车速应该设置为当前车速！！！
    speed_DP_col_start_nodes.rol_nodes.push_back(speed_single_start_node);

    DP_ST_nodes.col_nodes.push_back(speed_DP_col_start_nodes);

    for (size_t j = 1; j <= t_node_num; ++j)
    {
        Speed_DP_col_nodes speed_DP_col_nodes;
        for (size_t i = 0; i < s_node_num; ++i)
        {
            Speed_DP_Single_node speed_single_node;
            speed_single_node.node_s = s_node[i];
            speed_single_node.node_t = j * speed_plan_t_dt;
            speed_DP_col_nodes.rol_nodes.push_back(speed_single_node);
        }
        DP_ST_nodes.col_nodes.push_back(speed_DP_col_nodes);
    }
    return DP_ST_nodes;
}


vector<double> EMPlanner::CalcSpeedDpCost(planning_msgs::ObstacleList::Ptr &obstacle_list_qp_path_sl, Speed_Plan_DP_ST_nodes &DP_ST_nodes)
{
    if (DP_ST_nodes.col_nodes.size() < 2 || DP_ST_nodes.col_nodes[1].rol_nodes.empty()) {
        return std::vector<double>();
    }
    const int max_jump_S = std::min(
        static_cast<int>(DP_ST_nodes.col_nodes[1].rol_nodes.size()) - 1,
        static_cast<int>(std::ceil(dp_vel_max * speed_plan_t_dt / ST_s_min_step)));  //向上取整 
    
    clock_t speed_time_start=clock();
    // 起点循环
    DP_ST_nodes.col_nodes[0].rol_nodes[0].is_possible = true; //起点为true
    DP_ST_nodes.col_nodes[0].rol_nodes[0].cost = 0.0;
    DP_ST_nodes.col_nodes[0].rol_nodes[0].node_s_dot2 = 0.0;
    DP_ST_nodes.col_nodes[0].rol_nodes[0].toThis_min_cost_index = 0;
    for (int i = 0; i <= max_jump_S; i++)
    {
        const double node_vel =
            (DP_ST_nodes.col_nodes[1].rol_nodes[i].node_s - DP_ST_nodes.col_nodes[0].rol_nodes[0].node_s) / speed_plan_t_dt;
        const double node_a =
            (node_vel - DP_ST_nodes.col_nodes[0].rol_nodes[0].node_s_dot) / speed_plan_t_dt;
        DP_ST_nodes.col_nodes[1].rol_nodes[i].node_s_dot = node_vel;
        DP_ST_nodes.col_nodes[1].rol_nodes[i].node_s_dot2 = node_a;
        if (node_vel <= dp_vel_max && std::abs(node_a) <= dp_a_max)
        {
            DP_ST_nodes.col_nodes[1].rol_nodes[i].is_possible = true;
            DP_ST_nodes.col_nodes[1].rol_nodes[i].speed_cost = w_SpeedDpPlan_ref_vel * pow(speed_reference - node_vel, 2);
            DP_ST_nodes.col_nodes[1].rol_nodes[i].a_cost = w_SpeedDpPlan_a * pow(node_a, 2);
            DP_ST_nodes.col_nodes[1].rol_nodes[i].obs_cost =
                w_SpeedDpPlan_obs * CalcSpeedDp_ObsCost(obstacle_list_qp_path_sl, DP_ST_nodes, 0, 1, 0, i);
            DP_ST_nodes.col_nodes[1].rol_nodes[i].cost = DP_ST_nodes.col_nodes[1].rol_nodes[i].obs_cost + DP_ST_nodes.col_nodes[1].rol_nodes[i].speed_cost + DP_ST_nodes.col_nodes[1].rol_nodes[i].a_cost;
            DP_ST_nodes.col_nodes[1].rol_nodes[i].toThis_min_cost_index=0;  //第一列的上一行的最优节点都是0，因为第0列只有0号节点
        }
        else
        {
            continue;
        }
    }
    clock_t time_end = clock();
    double time_diff = static_cast<double>(time_end - speed_time_start) / CLOCKS_PER_SEC;
    std::cout << "SpeedDpstart程序运行时间: " << time_diff << " 秒" << std::endl;
    speed_time_start=clock();
    int calc_num=0;
    // 计算剩下列cost损失
    #pragma omp for
    for (int i = 1; i < DP_ST_nodes.col_nodes.size()-1 ; i++) // 列数
    {
        for (int k = 0; k < ((i+1)*max_jump_S>DP_ST_nodes.col_nodes[i].rol_nodes.size()?DP_ST_nodes.col_nodes[i].rol_nodes.size():(i+1)*max_jump_S); k++) //下一列行数
        {
            double min_cost = std::numeric_limits<double>::max();
            int min_cost_index = -1;
            for(int j = (k-max_jump_S>0?k-max_jump_S:0); j <= k; j++)  // 开始列行数    //此处为省略计算，k-6为手动计算的速度最大节点，更改ST图时要重新计算更改
            {
                if(!DP_ST_nodes.col_nodes[i].rol_nodes[j].is_possible || k<j ) //上一列不可能节点和小于这列数的不考虑，即不考虑倒车
                {
                    continue;
                }

                calc_num++;
                double node_vel=(DP_ST_nodes.col_nodes[i+1].rol_nodes[k].node_s - DP_ST_nodes.col_nodes[i].rol_nodes[j].node_s) / speed_plan_t_dt;
                // cout<<"node_vel:"<<node_vel<<endl;
                double node_a=(node_vel - DP_ST_nodes.col_nodes[i].rol_nodes[j].node_s_dot) / speed_plan_t_dt;

                if (node_vel <= dp_vel_max && std::abs(node_a) <= dp_a_max)
                {
                    DP_ST_nodes.col_nodes[i+1].rol_nodes[k].is_possible = true;
                    double speed_cost = w_SpeedDpPlan_ref_vel * pow(speed_reference - node_vel, 2);
                    double a_cost = w_SpeedDpPlan_a * pow(node_a, 2);
                    double obs_cost= w_SpeedDpPlan_obs* CalcSpeedDp_ObsCost(obstacle_list_qp_path_sl, DP_ST_nodes, i, i+1, j,k);  
                    // cout<<"cost:::"<< DP_ST_nodes.col_nodes[i+1].rol_nodes[k].speed_cost<<" ,"<<DP_ST_nodes.col_nodes[i+1].rol_nodes[k].a_cost<<" ,"<<DP_ST_nodes.col_nodes[i+1].rol_nodes[k].obs_cost<<endl;
                    double forward_node_to_this_node_cost=speed_cost+a_cost+obs_cost;
                    double total_cost=forward_node_to_this_node_cost+DP_ST_nodes.col_nodes[i].rol_nodes[j].cost; //总代价等于上一节点代价加上列到此节点代价
                    if(total_cost<min_cost)  //选择最小的cost，并记录最小值cost的上一列序号
                    {
                        min_cost=total_cost;
                        min_cost_index=j;

                        // DP_ST_nodes.col_nodes[i+1].rol_nodes[k].a_cost=a_cost;
                        // DP_ST_nodes.col_nodes[i+1].rol_nodes[k].speed_cost=speed_cost;
                        // DP_ST_nodes.col_nodes[i+1].rol_nodes[k].obs_cost=obs_cost;
                        DP_ST_nodes.col_nodes[i+1].rol_nodes[k].cost=min_cost;
                        DP_ST_nodes.col_nodes[i+1].rol_nodes[k].node_s_dot=node_vel;
                        DP_ST_nodes.col_nodes[i+1].rol_nodes[k].node_s_dot2=node_a;
                        DP_ST_nodes.col_nodes[i+1].rol_nodes[k].toThis_min_cost_index=min_cost_index;

                    }
                }
                else
                {
                    continue;
                }
            }
        }
    //     clock_t time_end = clock();
    // double time_diff = static_cast<double>(time_end - time_start) / CLOCKS_PER_SEC;
    // std::cout << "DP——start  列节点运行时间: " << time_diff << " 秒" << std::endl;
    }
    clock_t time_end1 = clock();
    double time_diff1 = static_cast<double>(time_end1 - speed_time_start) / CLOCKS_PER_SEC;
    std::cout << "SpeedDpstart程序运行时间: " << time_diff1 << " 秒 计算次数：" <<calc_num<< std::endl;

    // for (int i = 0; i < DP_ST_nodes.col_nodes.size(); i++)
    // {
    //     for (int j = 0; j < DP_ST_nodes.col_nodes[i].rol_nodes.size(); j++)
    //     {
    //         cout<<i<<","<<j<<":"<<"||"<<DP_ST_nodes.col_nodes[i].rol_nodes[j].node_s_dot<<"||";
    //     }
    //     cout<<endl;
    // }
    
    // 优先寻找最后一列的最优节点；如果最后一列无可行节点，则回退到最后一个可行列，
    // 并在后续时刻保持该列的末端s，避免直接整条速度决策归零。
    double min_cost = std::numeric_limits<double>::max();
    int min_cost_last_index = 0;
    int min_cost_index = 0;
    int best_col_index = -1;
    for (int col = static_cast<int>(DP_ST_nodes.col_nodes.size()) - 1; col >= 0; --col)
    {
        double col_min_cost = std::numeric_limits<double>::max();
        int col_min_index = -1;
        for (int row = 0; row < DP_ST_nodes.col_nodes[col].rol_nodes.size(); ++row)
        {
            const auto& node = DP_ST_nodes.col_nodes[col].rol_nodes[row];
            if (!node.is_possible || !std::isfinite(node.cost) || node.cost >= 1e9)
            {
                continue;
            }
            if (node.cost < col_min_cost)
            {
                col_min_cost = node.cost;
                col_min_index = row;
            }
        }
        if (col_min_index >= 0)
        {
            best_col_index = col;
            min_cost = col_min_cost;
            min_cost_last_index = col_min_index;
            break;
        }
    }

    speed_dp_last_feasible_index_ = best_col_index;
    vector<double> Speed_DP_path_s(DP_ST_nodes.col_nodes.size(), 0.0);
    if(best_col_index < 0)
    {
        cout<<"DpSpeed无解,为初始0节点，速度规划为0"<<endl;
    }
    else
    {
        if (best_col_index < static_cast<int>(DP_ST_nodes.col_nodes.size()) - 1)
        {
            cout<<"DpSpeed末列无解, 回退到第 "<<best_col_index<<" 列最后可行节点"<<endl;
        }
        vector<int> Speed_DP_path_index(best_col_index + 1, 0);
        Speed_DP_path_index[best_col_index] = min_cost_last_index;
        min_cost_index=min_cost_last_index;

        for (int i = best_col_index; i > 0; --i)
        {
            min_cost_index = DP_ST_nodes.col_nodes[i].rol_nodes[min_cost_index].toThis_min_cost_index;
            min_cost_index = std::max(0, min_cost_index);
            Speed_DP_path_index[i - 1] = min_cost_index;
        }

        for (int i = 0; i <= best_col_index; ++i)
        {
            Speed_DP_path_s[i] = DP_ST_nodes.col_nodes[i].rol_nodes[Speed_DP_path_index[i]].node_s;
        }

        const double hold_s = Speed_DP_path_s[best_col_index];
        for (int i = best_col_index + 1; i < Speed_DP_path_s.size(); ++i)
        {
            Speed_DP_path_s[i] = hold_s;
        }
    }
    cout<<endl;

    return Speed_DP_path_s;
}

Speed_plan_points EMPlanner::CalcSpeedPlan_QpPath(planning_msgs::ObstacleList::Ptr &obstacle_list_qp_path_sl, Speed_Plan_DP_ST_nodes &DP_ST_nodes, vector<double> best_speed_path_s,double s_start)
{
    const int n = std::max(2, static_cast<int>(std::lround(plan_time / speed_plan_t_dt)) + 1);
    Eigen::VectorXd s_ub = Eigen::VectorXd::Constant(n, speed_plan_distance);
    Eigen::VectorXd s_lb = Eigen::VectorXd::Zero(n);
    auto sample_dp_s = [&](int sample_index) -> double {
        if (best_speed_path_s.empty())
        {
            return std::min(static_cast<double>(speed_reference) * sample_index * speed_plan_t_dt,
                            static_cast<double>(speed_plan_distance));
        }
        const int clamped_index =
            std::max(0, std::min(static_cast<int>(best_speed_path_s.size()) - 1, sample_index));
        return best_speed_path_s[clamped_index];
    };
    auto sample_obstacle_bounds =
        [&](const planning_msgs::Obstacle& obstacle, double sample_t, double& lower_s, double& upper_s) {
            lower_s = obstacle.speed_plan_s_in;
            if (obstacle.speed_plan_t_out > obstacle.speed_plan_t_in + 1e-6)
            {
                const double ratio =
                    (sample_t - obstacle.speed_plan_t_in) /
                    (obstacle.speed_plan_t_out - obstacle.speed_plan_t_in);
                const double clamped_ratio = std::max(0.0, std::min(1.0, ratio));
                lower_s =
                    obstacle.speed_plan_s_in +
                    clamped_ratio * (obstacle.speed_plan_s_out - obstacle.speed_plan_s_in);
            }
            upper_s = lower_s + (obstacle.max_s - obstacle.min_s);
        };
    // cout<<"obstacle_list_qp_path_sl->obstacles.size():"<<obstacle_list_qp_path_sl->obstacles.size()<<endl;
    // cout<<"s_start:"<<s_start<<endl;
    for (int i = 0; i < obstacle_list_qp_path_sl->obstacles.size(); i++)
    {
        if (!obstacle_list_qp_path_sl->obstacles[i].is_consider)
        {
            continue;
        }
        int index_s_t_in, index_s_t_out;

        index_s_t_in = static_cast<int>(std::ceil(
            obstacle_list_qp_path_sl->obstacles[i].speed_plan_t_in / speed_plan_t_dt));       // 找到障碍物的ST图Tmin
        index_s_t_out = static_cast<int>(std::floor(
            obstacle_list_qp_path_sl->obstacles[i].speed_plan_t_out / speed_plan_t_dt)); // 找到障碍物的ST图Tmax
        index_s_t_in = std::max(0, std::min(n - 1, index_s_t_in));
        index_s_t_out = std::max(0, std::min(n - 1, index_s_t_out));
        if (index_s_t_in > index_s_t_out)
        {
            continue;
        }
        int overtake_votes = 0;
        int follow_votes = 0;
        for (int j = index_s_t_in; j <= index_s_t_out; j++)
        {
            const double sample_t = j * speed_plan_t_dt;
            double obstacle_lower_s = 0.0;
            double obstacle_upper_s = 0.0;
            sample_obstacle_bounds(obstacle_list_qp_path_sl->obstacles[i], sample_t,
                                   obstacle_lower_s, obstacle_upper_s);
            const double dp_target_s = sample_dp_s(j);
            const double follow_upper_bound =
                std::max(0.01, obstacle_lower_s - static_cast<double>(SpeedQpPlan_SafeDistance));
            const double overtake_lower_bound =
                obstacle_upper_s + static_cast<double>(SpeedQpPlan_SafeDistance);
            if (dp_target_s >= overtake_lower_bound - 1e-6)
            {
                ++overtake_votes;
            }
            else if (dp_target_s <= follow_upper_bound + 1e-6)
            {
                ++follow_votes;
            }
            else
            {
                const double obstacle_mid_s = 0.5 * (obstacle_lower_s + obstacle_upper_s);
                if (dp_target_s >= obstacle_mid_s)
                {
                    ++overtake_votes;
                }
                else
                {
                    ++follow_votes;
                }
            }
        }

        bool use_overtake = overtake_votes > follow_votes;
        if (overtake_votes == follow_votes)
        {
            double obstacle_lower_s = 0.0;
            double obstacle_upper_s = 0.0;
            sample_obstacle_bounds(obstacle_list_qp_path_sl->obstacles[i],
                                   index_s_t_in * speed_plan_t_dt,
                                   obstacle_lower_s,
                                   obstacle_upper_s);
            use_overtake = sample_dp_s(index_s_t_in) >= 0.5 * (obstacle_lower_s + obstacle_upper_s);
        }

        for (int j = index_s_t_in; j <= index_s_t_out; ++j)
        {
            const double sample_t = j * speed_plan_t_dt;
            double obstacle_lower_s = 0.0;
            double obstacle_upper_s = 0.0;
            sample_obstacle_bounds(obstacle_list_qp_path_sl->obstacles[i], sample_t,
                                   obstacle_lower_s, obstacle_upper_s);
            if (use_overtake)
            {
                const double overtake_lower_bound =
                    obstacle_upper_s + static_cast<double>(SpeedQpPlan_SafeDistance);
                s_lb[j] = std::max(s_lb[j], overtake_lower_bound);
            }
            else
            {
                const double follow_upper_bound =
                    std::max(0.01, obstacle_lower_s - static_cast<double>(SpeedQpPlan_SafeDistance));
                if (s_ub[j] > follow_upper_bound)
                {
                    s_ub[j] = follow_upper_bound;
                }
            }
        }
    }

    for (int i = 0; i < n; ++i)
    {
        if (s_lb[i] > s_ub[i])
        {
            const double dp_target_s = sample_dp_s(i);
            const double projected_s = std::min(std::max(dp_target_s, s_ub[i]), s_lb[i]);
            s_lb[i] = projected_s;
            s_ub[i] = projected_s;
        }
    }
    
    // qp_speed计算
    Eigen::SparseMatrix<double> A(6 * n - 3, 3 * n);
    Eigen::SparseMatrix<double> H(3 * n, 3 * n);
    Eigen::SparseMatrix<double> Aeq_sub(2, 6);
    Eigen::SparseMatrix<double> A_sub1(3, 3);
    Eigen::SparseMatrix<double> A_sub2(1, 6);
    
    Eigen::VectorXd f = Eigen::VectorXd::Zero(3 * n, 1);
    Eigen::VectorXd lb = Eigen::VectorXd::Zero(6 * n - 3, 1);
    Eigen::VectorXd ub = Eigen::VectorXd::Zero(6 * n - 3, 1);
    // Eigen::VectorXd l_min = Eigen::VectorXd::Constant(n, min_l);
    // Eigen::VectorXd l_max = Eigen::VectorXd::Constant(n, max_l);

    Aeq_sub.reserve(9); // 预分配非零元素空间
    A_sub1.reserve(3);
    A_sub2.reserve(2);
    A.reserve(14 * n - 11);

    H.reserve(2 * n);

    // 设置 Aeq_sub 稀疏矩阵的值
    Aeq_sub.insert(0, 0) = 1;
    Aeq_sub.insert(0, 1) = speed_plan_t_dt;
    Aeq_sub.insert(0, 2) = pow(speed_plan_t_dt, 2) / 3;
    Aeq_sub.insert(0, 3) = -1;
    Aeq_sub.insert(0, 5) = pow(speed_plan_t_dt, 2) / 6;
    Aeq_sub.insert(1, 1) = 1;
    Aeq_sub.insert(1, 2) = speed_plan_t_dt / 2;
    Aeq_sub.insert(1, 4) = -1;
    Aeq_sub.insert(1, 5) = speed_plan_t_dt / 2;
     
    // 设置 A_sub 稀疏矩阵的值
    A_sub1.insert(0, 0) = 1;
    A_sub1.insert(1, 1) = 1;
    A_sub1.insert(2, 2) = 1;

    A_sub2.insert(0, 0) = -1;
    A_sub2.insert(0, 3) = 1;

    // 设置 A 稀疏矩阵的值
    for (int i = 0; i < n; i++)
    {
        A.insert(3 * i, 3 * i) = A_sub1.coeff(0, 0);
        A.insert(3 * i + 1, 3 * i + 1) = A_sub1.coeff(1, 1);
        A.insert(3 * i + 2, 3 * i + 2) = A_sub1.coeff(2, 2);
    }
    
    for (int i = 0; i < n - 1; i++)
    {
        A.insert(3 * n + 2 * i, 3 * i) = Aeq_sub.coeff(0, 0);
        A.insert(3 * n + 2 * i, 3 * i + 1) = Aeq_sub.coeff(0, 1);
        A.insert(3 * n + 2 * i, 3 * i + 2) = Aeq_sub.coeff(0, 2);
        A.insert(3 * n + 2 * i, 3 * i + 3) = Aeq_sub.coeff(0, 3);
        A.insert(3 * n + 2 * i, 3 * i + 5) = Aeq_sub.coeff(0, 5);
        A.insert(3 * n + 2 * i + 1, 3 * i + 1) = Aeq_sub.coeff(1, 1);
        A.insert(3 * n + 2 * i + 1, 3 * i + 2) = Aeq_sub.coeff(1, 2);
        A.insert(3 * n + 2 * i + 1, 3 * i + 4) = Aeq_sub.coeff(1, 4);
        A.insert(3 * n + 2 * i + 1, 3 * i + 5) = Aeq_sub.coeff(1, 5);

        A.insert((5 * n - 2) + i, 3 * i) = A_sub2.coeff(0, 0);
        A.insert((5 * n - 2) + i, 3 * i + 3) = A_sub2.coeff(0, 3);
        ub(5 * n - 2 + i) = 1e8;
    }
    // 设置 H 稀疏矩阵的值
    for (int i = 0; i < n; i++)
    {
        const double ref_s = sample_dp_s(i);
        if (w_SpeedQpPlan_ref_s > 1e-6)
        {
            H.insert(3 * i, 3 * i) = 2 * w_SpeedQpPlan_ref_s;
            f(3 * i) = -2 * ref_s * w_SpeedQpPlan_ref_s;
        }
        H.insert(3 * i + 1, 3 * i + 1) = 2 * (w_SpeedQpPlan_ref_vel);
        H.insert(3 * i + 2, 3 * i + 2) = 2 * (w_SpeedQpPlan_a);

        f(3 * i + 1) = -2 * speed_reference * w_SpeedQpPlan_ref_vel;
        lb(3 * i) = s_lb(i);  //这里的s_lb限制是绝对s，但是QP计算是相对s，所以减去自车s
        ub(3 * i) = s_ub(i);
       
        ub(3 * i + 1) = SpeedQP_v_max;
        lb(3 * i + 2) = -SpeedQP_a_max;
        ub(3 * i + 2) = SpeedQP_a_max;

    }
    lb(0)=0; //起点s设置
    ub(0)=0; //起点s设置
    lb(1)=real_vehicle_speed; // 起点速度与当前车速保持一致
    ub(1)=real_vehicle_speed; // 起点速度与当前车速保持一致
    
    A.makeCompressed();
    H.makeCompressed();

    // // osqp求解
    int NumberOfVariables = 3 * n;       // A矩阵的列数
    int NumberOfConstraints = 6 * n - 3; // A矩阵的行数
    // cout << "Path optimization progress --25% " << endl;
    // // 求解部分
    OsqpEigen::Solver solver;

    // // settings
    solver.settings()->setVerbosity(false); // 求解器信息输出控制
    solver.settings()->setWarmStart(true);  // 启用热启动
    // solver.settings()->setInitialGuessX(f); // 设置初始解向量,加速收敛

    // set the initial data of the QP solver
    // 矩阵A为m*n矩阵
    solver.data()->setNumberOfVariables(NumberOfVariables);     // 设置A矩阵的列数，即n
    solver.data()->setNumberOfConstraints(NumberOfConstraints); // 设置A矩阵的行数，即m

    if (!solver.data()->setHessianMatrix(H))
        // return 1; //设置P矩阵
        cout << "error1" << endl;
    if (!solver.data()->setGradient(f))
        // return 1; //设置q or f矩阵。当没有时设置为全0向量
        cout << "error2" << endl;
    if (!solver.data()->setLinearConstraintsMatrix(A))
        // return 1; //设置线性约束的A矩阵
        cout << "error3" << endl;
    if (!solver.data()->setLowerBound(lb))
    { // return 1; //设置下边界
        cout << "error4" << endl;
    }
    if (!solver.data()->setUpperBound(ub))
    { // return 1; //设置上边界
        cout << "error5" << endl;
    }

    // instantiate the solver
    if (!solver.initSolver())
        // return 1;
        cout << "speed_error6" << endl;
    Eigen::VectorXd QPSolution = Eigen::VectorXd::Zero(3 * n);

    // solve the QP problem

    vector<double> s(n, 0.0);
    vector<double> s_dot(n, 0.0);
    vector<double> s_dot2(n, 0.0);
    vector<double> s_time(n, 0.0);

    if (!solver.solve())
    {
        SpeedPlanRunningNormally = false;
        cout << "Speed Error Slove!" << endl;
    }
    else
    {
        SpeedPlanRunningNormally = true;
        QPSolution = solver.getSolution();
    }

    for (int i = 0; i < n; i++)
    {
        if (SpeedPlanRunningNormally)
        {
            s[i] = QPSolution(3 * i);
            s_dot[i] = QPSolution(3 * i + 1);
            s_dot2[i] = QPSolution(3 * i + 2);
        }
        s_time[i] = i*speed_plan_t_dt;
    }
    Speed_plan_points speed_plan_points;
    speed_plan_points.s=s;
    speed_plan_points.s_dot=s_dot;
    speed_plan_points.s_dot2=s_dot2;
    speed_plan_points.s_time=s_time;
    if(SpeedPlanRunningNormally && s_dot[0]<speed_reference-0.1)
        cout<<"减速！！！"<<endl;
    // cout<<"推荐速度>>>>> "<<s_dot[0]<<" <<<<<"<<endl;
    QP_path->car_direct=car_direct;
    if(car_direct==1)
    {
        calc_car_speed=s_dot[0];   
    }
    else
    {
        calc_car_speed=-0.5;
    }
        
    // cout<<"car_direct:"<<car_direct<<" QP_path->V_straight:"<<QP_path->V_straight<<endl;
    return speed_plan_points;
}


/**
 * @description:
 * @param {Speed_DP_col_nodes} &speed_DP_col_nodes ST节点的列
 * @param {int} rol_start 行数
 * @return {*}
 */
double EMPlanner::CalcSpeedDp_ObsCost(const planning_msgs::ObstacleList::Ptr &obstacle_list_qp_path_sl, Speed_Plan_DP_ST_nodes &DP_ST_nodes, int col_start, int col_end, int rol_start,int rol_end)
{
    const int n = 5; // 对列间连线做等分采样
    const double hard_gap = std::max(0.0, static_cast<double>(SpeedDpPlanMinObstacleDistance));
    const double soft_gap = std::max(hard_gap + 0.05, static_cast<double>(SpeedDpPlanMaxObstacleDistance));
    const double soft_margin = std::max(soft_gap - hard_gap, 1e-6);
    double cost = 0.0;
    for (int j = 0; j < n; j++) // 对候选边上的采样点做统一走廊代价评估
    {
        const double sample_ratio =
            (n <= 1) ? 0.0 : static_cast<double>(j) / static_cast<double>(n - 1);
        const double A_s =
            DP_ST_nodes.col_nodes[col_start].rol_nodes[rol_start].node_s +
            (DP_ST_nodes.col_nodes[col_end].rol_nodes[rol_end].node_s -
             DP_ST_nodes.col_nodes[col_start].rol_nodes[rol_start].node_s) *
                sample_ratio;
        const double A_t =
            DP_ST_nodes.col_nodes[col_start].rol_nodes[rol_start].node_t +
            (DP_ST_nodes.col_nodes[col_end].rol_nodes[rol_end].node_t -
             DP_ST_nodes.col_nodes[col_start].rol_nodes[rol_start].node_t) *
                sample_ratio;

        bool has_active_obstacle = false;
        bool inside_hard_interval = false;
        double min_clearance_to_hard_interval = std::numeric_limits<double>::max();
        for (int k = 0; k < obstacle_list_qp_path_sl->obstacles.size(); k++) // 障碍物循环
        {
            if (!obstacle_list_qp_path_sl->obstacles[k].is_consider) // 不满足的障碍物不考虑
            {
                continue;
            }
            const double obs_t_in = obstacle_list_qp_path_sl->obstacles[k].speed_plan_t_in;
            const double obs_t_out = obstacle_list_qp_path_sl->obstacles[k].speed_plan_t_out;
            if (obs_t_out < obs_t_in - 1e-6)
            {
                continue;
            }
            if (A_t < obs_t_in - 1e-6 || A_t > obs_t_out + 1e-6)
            {
                continue;
            }

            has_active_obstacle = true;
            double obstacle_lower_s = obstacle_list_qp_path_sl->obstacles[k].speed_plan_s_in;
            if (obs_t_out > obs_t_in + 1e-6)
            {
                const double ratio =
                    (A_t - obs_t_in) / (obs_t_out - obs_t_in);
                const double clamped_ratio = std::max(0.0, std::min(1.0, ratio));
                obstacle_lower_s =
                    obstacle_list_qp_path_sl->obstacles[k].speed_plan_s_in +
                    clamped_ratio *
                        (obstacle_list_qp_path_sl->obstacles[k].speed_plan_s_out -
                         obstacle_list_qp_path_sl->obstacles[k].speed_plan_s_in);
            }
            const double obstacle_upper_s =
                obstacle_lower_s +
                (obstacle_list_qp_path_sl->obstacles[k].max_s -
                 obstacle_list_qp_path_sl->obstacles[k].min_s);

            const double hard_lower_s = obstacle_lower_s - hard_gap;
            const double hard_upper_s = obstacle_upper_s + hard_gap;
            if (A_s >= hard_lower_s && A_s <= hard_upper_s)
            {
                inside_hard_interval = true;
                break;
            }

            const double clearance_to_hard_interval =
                (A_s < hard_lower_s) ? (hard_lower_s - A_s) : (A_s - hard_upper_s);
            min_clearance_to_hard_interval =
                std::min(min_clearance_to_hard_interval, clearance_to_hard_interval);
        }

        if (!has_active_obstacle)
        {
            continue;
        }
        if (inside_hard_interval)
        {
            cost += 1e8;
            continue;
        }

        if (min_clearance_to_hard_interval < soft_margin)
        {
            const double ratio = (soft_margin - min_clearance_to_hard_interval) / soft_margin;
            cost += ratio * ratio;
        }
    }
    return cost;
}

double EMPlanner::Speed_Dp_collision_cost(double s_gap)
{
    const double hard_gap = std::max(0.0, static_cast<double>(SpeedDpPlanMinObstacleDistance));
    const double soft_gap = std::max(hard_gap + 0.05, static_cast<double>(SpeedDpPlanMaxObstacleDistance));
    if (s_gap <= hard_gap)
    {
        return 1e8;
    }
    else if (s_gap < soft_gap)
    {
        const double denom = std::max(soft_gap - hard_gap, 1e-6);
        const double ratio = (soft_gap - s_gap) / denom;
        return ratio * ratio;
    }
    else
    {
        return 0;
    }
}

// 寻找起始点的sl
Eigen::Vector2d EMPlanner::Find_start_sl_FirstRun(Eigen::Vector2d host_point, planning_msgs::path_point host_projected_point, int index_match_start)
{
    float planning_start_s, planning_start_l;
    float host_s = host_projected_point.absolute_s;

    Eigen::Vector2d AB(path.points[index_match_start].tor.x, path.points[index_match_start].tor.y);
    Eigen::Vector2d AC(host_point(0) - path.points[index_match_start].x, host_point(1) - path.points[index_match_start].y);
    float host_l;

    Eigen::Vector2d tor_match(path.points[index_match_start].tor.x, path.points[index_match_start].tor.y);
    host_l = -1 * (AC.x() * tor_match.y() - AC.y() * tor_match.x());

    int index_host_to_path = 0;
    int index_host_plan_start_to_path = 0;

    for (int i = 0; i < DP_path_sl.s.size(); i++)
    {
        if (host_s < DP_path_sl.s(i))
        {
            index_host_to_path = i;
            break;
        }
    }

    planning_start_s = host_projected_point.absolute_s;

    planning_start_l = host_l;
    Eigen::Vector2d planning_start(planning_start_s, planning_start_l);
    return planning_start;
}

Eigen::Vector2d EMPlanner::Find_start_sl(Eigen::Vector2d host_point, planning_msgs::path_point host_projected_point, int index_match_start)
{
    // index_car_match_points = index2s(path, host_projected_point, index_car_match_points);
    float planning_start_s, planning_start_l;
    float host_s = host_projected_point.absolute_s;

    Eigen::Vector2d AB(path.points[index_match_start].tor.x, path.points[index_match_start].tor.y);
    Eigen::Vector2d AC(host_point(0) - path.points[index_match_start].x, host_point(1) - path.points[index_match_start].y);
    float host_l;

    Eigen::Vector2d tor_match(path.points[index_match_start].tor.x, path.points[index_match_start].tor.y);
    host_l = -1 * (AC.x() * tor_match.y() - AC.y() * tor_match.x());

    int index_host_to_path = 0;
    int index_host_plan_start_to_path = 0;

    for (int i = 0; i < DP_path_sl.s.size(); i++)
    {
        if (host_s < DP_path_sl.s(i))
        {
            index_host_to_path = i;
            break;
        }
    }
    // if (sqrt(pow(host_s - planning_Interpolate_path.s(index_host_to_path), 2) + pow(host_l - planning_Interpolate_path.l(index_host_to_path), 2)) > panning_start_max_error)
    // {
    planning_start_s = host_projected_point.absolute_s;
    // planning_start_l = CrossProduct(AB, AC) * sqrt(pow(host_point(0) - host_projected_point.x, 2) + pow(host_point(1) - host_projected_point.y, 2));
    planning_start_l = host_l;
    Eigen::Vector2d planning_start(planning_start_s, planning_start_l);
    // cout << "host:" << host_s << " ," << host_l << " planning_Interpolate_path:" << planning_Interpolate_path.s(index_host_to_path) << " ," << planning_Interpolate_path.l(index_host_to_path) << endl;
    // cout << "误差：" << sqrt(pow(host_s - planning_Interpolate_path.s(index_host_to_path), 2) + pow(host_l - planning_Interpolate_path.l(index_host_to_path), 2)) << " Replanning!!!" << endl;
    return planning_start;
    // }
    // else
    // {
    //     planning_start_s = host_projected_point.absolute_s + v_planning_start * v_planning_start_dt;
    //     for (int i = index_host_to_path; i < planning_Interpolate_path.s.size(); i++)
    //     {
    //         if (planning_start_s < planning_Interpolate_path.s(i))
    //         {
    //             index_host_plan_start_to_path = i;
    //             break;
    //         }
    //     }
    //     planning_start_l = planning_Interpolate_path.l(index_host_plan_start_to_path);
    //     Eigen::Vector2d planning_start(planning_start_s, planning_start_l);
    //     return planning_start;
    // }
}

// void EMPlanner::Plot_SL_Graph(Eigen::Vector2d host_forward_sl, const Frenet_path_points &DP_path_sl, const Frenet_path_points &QP_path_sl)
// {
//     plt::ion();
//     plt::figure(1);
//     plt::clf(); // 清空画布
//     int dp_num = DP_path_sl.s.size();
//     int qp_num = QP_path_sl.s.size();

//     vector<double> local_s_watch;
//     vector<double> local_l_watch;
//     vector<double> local_path_s_left_watch;
//     vector<double> local_path_l_left_watch;
//     vector<double> local_path_s_right_watch;
//     vector<double> local_path_l_right_watch;
//     vector<double> local_path_s_middle_watch;
//     vector<double> local_path_l_middle_watch;
//     std::vector<double> car_pose_s = {host_forward_sl(0), host_forward_sl(0), host_forward_sl(0), host_forward_sl(0) - 4.75, host_forward_sl(0) - 4.75, host_forward_sl(0)};
//     std::vector<double> car_pose_l = {host_forward_sl(1), host_forward_sl(1) - 1, host_forward_sl(1) + 1, host_forward_sl(1) + 1, host_forward_sl(1) - 1, host_forward_sl(1) - 1};

//     std::vector<double> dp_s(dp_num), dp_l(dp_num);
//     std::vector<double> qp_s(qp_num), qp_l(qp_num);

//     std::copy(DP_path_sl.s.data(), DP_path_sl.s.data() + DP_path_sl.s.size(), dp_s.begin());
//     std::copy(DP_path_sl.l.data(), DP_path_sl.l.data() + DP_path_sl.l.size(), dp_l.begin());
//     std::copy(QP_path_sl.s.data(), QP_path_sl.s.data() + QP_path_sl.s.size(), qp_s.begin());
//     std::copy(QP_path_sl.l.data(), QP_path_sl.l.data() + QP_path_sl.l.size(), qp_l.begin());

//     local_s_watch.reserve(2 * dp_num);
//     local_l_watch.reserve(2 * dp_num);
//     local_path_s_left_watch.reserve(2 * dp_num);
//     local_path_l_left_watch.reserve(2 * dp_num);
//     local_path_s_right_watch.reserve(2 * dp_num);
//     local_path_l_right_watch.reserve(2 * dp_num);
//     local_path_s_middle_watch.reserve(2 * dp_num);
//     local_path_l_middle_watch.reserve(2 * dp_num);

//     for (int i = 0; i < obstacle_list->obstacles.size(); i++)
//     {
//         if(obstacle_list->obstacles[i].s>host_forward_sl[0]+50||obstacle_list->obstacles[i].s<host_middle_sl[0])
//             continue;
//         std::vector<double> obs_list_s_watch = {obstacle_list->obstacles[i].max_s, obstacle_list->obstacles[i].max_s, obstacle_list->obstacles[i].min_s, obstacle_list->obstacles[i].min_s, obstacle_list->obstacles[i].max_s};
//         std::vector<double> obs_list_l_watch = {obstacle_list->obstacles[i].max_l, obstacle_list->obstacles[i].min_l, obstacle_list->obstacles[i].min_l, obstacle_list->obstacles[i].max_l, obstacle_list->obstacles[i].max_l};
//         plt::plot(obs_list_s_watch, obs_list_l_watch);
//     }

//     int index_local_path_watch = get_aim_point(40, path, index_middle_Car, true);
//     for (int i = index_middle_Car; i <= index_local_path_watch; i++) // 添加参考线，左右路宽
//     {
//         local_s_watch.push_back(path.points[i].absolute_s);
//         local_l_watch.push_back(0);
//         local_path_s_right_watch.push_back(path.points[i].absolute_s);
//         local_path_l_right_watch.push_back(-2.5);
//         local_path_s_left_watch.push_back(path.points[i].absolute_s);
//         local_path_l_left_watch.push_back(7.5);
//         local_path_s_middle_watch.push_back(path.points[i].absolute_s);
//         local_path_l_middle_watch.push_back(2.5);
//     }

//     // plt::plot(local_s_watch,local_l_watch,"--");

//     plt::named_plot("left_path", local_path_s_left_watch, local_path_l_left_watch, "black");
//     plt::named_plot("right_path", local_path_s_right_watch, local_path_l_right_watch, "black");

//     plt::named_plot("local", local_s_watch, local_l_watch, "--");
//     plt::named_plot("middle", local_path_s_middle_watch, local_path_l_middle_watch, "--");
//     plt::named_plot("Car", car_pose_s, car_pose_l, "red");
//     plt::named_plot("dp", dp_s, dp_l, "blue");
//     plt::named_plot("qp", qp_s, qp_l, "green");
//     plt::ylim(-20, 20);
//     plt::legend();
//     plt::title("SL");
//     plt::pause(0.0000001); // 等待0.001秒，让GUI有机会响应用户交互
//     plt::show();
// }

float EMPlanner::CrossProduct(Eigen::Vector2d AB, Eigen::Vector2d A, Eigen::Vector2d C)
{
    float Z; // 判断叉积结果
    Eigen::Vector2d AC = C - A;
    Z = AB.x() * AC.y() - AB.y() * AC.x();
    if (Z >= 0)
    {
        return 1.0;
    }
    else
    {
        return -1.0;
    }
}

float EMPlanner::CrossProduct(Eigen::Vector2d AB, Eigen::Vector2d AC)
{
    float Z; // 判断叉积结果
    Z = AB.x() * AC.y() - AB.y() * AC.x();
    if (Z >= 0)
    {
        // cout<<"车在法相量左侧"<<endl;
        return 1.0;
    }
    else
    {
        // cout<<"车在法相量右侧"<<endl;
        return -1.0;
    }
}

/**
 * @description: 计算在frenet坐标系下所有局部路径的点转化为XY坐标系
 * @param {Frenet_path_points} frenet_path_points
 * @param {car_path} path
 * @return {*}
 */
vector<Eigen::Vector2d> EMPlanner::FrenetToXY(const Frenet_path_points &frenet_path_points, planning_msgs::car_path &path)
{
    // cout<<"FrenetToXY start"<<endl;
    vector<Eigen::Vector2d> points_XY;
    points_XY.reserve(frenet_path_points.s.size());
    int pre_index2s_id = index_back_Car;
    for (int i = 0; i < frenet_path_points.s.size(); i++)
    {
        planning_msgs::path_point proj_point_;
        planning_msgs::path_point point_xy;

        proj_point_.absolute_s = frenet_path_points.s[i];
        // cout<<"proj_point_.absolute_s :"<<proj_point_.absolute_s <<endl;
        int index_match_points;
        index_match_points = index2s(path, proj_point_, pre_index2s_id);

        proj_point_ = find_projected_point_Frenet(path, index_match_points, frenet_path_points.s[i], frenet_path_points.l[i]);

        Eigen::Vector2d match_point_xy(proj_point_.x, proj_point_.y);
        Eigen::Vector2d point_nor(proj_point_.nor.x, proj_point_.nor.y);   //法相量
        Eigen::Vector2d point_temp_xy = match_point_xy + frenet_path_points.l(i) * point_nor;
        
        pre_index2s_id = index_match_points;

        points_XY.push_back(point_temp_xy);
    }

    // std::cout << "FrenetToXY程序运行时间: " << time_diff << " 秒" << std::endl;
    return points_XY;
}

/**
 * @description: 此函数已知在frenet坐标系下的点找到frenet坐标系下的投影点
 * @param {car_path} &path 全局路径
 * @param {int} index_match_point  匹配点序号
 * @param {Vector2d} host_point 自车坐标
 * @return {*}
 */
planning_msgs::path_point EMPlanner::find_projected_point_Frenet(planning_msgs::car_path &path, int index_match_point, float point_s, float point_l)
{
    planning_msgs::path_point projected_point;
    float match_points_x = path.points[index_match_point].x;
    float match_points_y = path.points[index_match_point].y;
    Eigen::Vector2d match_point(match_points_x, match_points_y);
    float match_points_heading = path.points[index_match_point].yaw;
    float match_points_kappa = path.points[index_match_point].kappa;
    float ds = point_s - path.points[index_match_point].absolute_s;
    
    Eigen::Vector2d point_tor(path.points[index_match_point].tor.x, path.points[index_match_point].tor.y);
    Eigen::Vector2d proj_point = match_point + ds * point_tor;
    float proj_points_heading = path.points[index_match_point].yaw + ds * path.points[index_match_point].kappa;
    float proj_points_kappa = path.points[index_match_point].kappa;
    projected_point.x = proj_point(0);
    projected_point.y = proj_point(1);
    projected_point.yaw = proj_points_heading;
    projected_point.tor.x = cos(proj_points_heading);
    projected_point.tor.y = sin(proj_points_heading);
    // cout<<"yaw、torx，tory："<<projected_point.yaw<<","<<projected_point.tor.x<<","<<projected_point.tor.y<<endl;
    projected_point.nor.x = -sin(proj_points_heading);
    projected_point.nor.y = cos(proj_points_heading);
    projected_point.kappa = proj_points_kappa;
    projected_point.absolute_s = point_s;

    return projected_point;
}

/**
 * @description: 此函数在frenet坐标系下找到投影点
 * @param {car_path} &path 全局路径
 * @param {int} index_match_point  匹配点序号
 * @param {Vector2d} host_point 自车坐标
 * @return {*}
 */
planning_msgs::path_point EMPlanner::find_projected_point_Frenet(planning_msgs::car_path &path, int index_match_point, Eigen::Vector2d host_point)
{
    planning_msgs::path_point projected_point;

    Eigen::Vector2f vec_d(host_point(0) - path.points[index_match_point].x, host_point(1) - path.points[index_match_point].y);
    Eigen::Vector2f vec_tor(path.points[index_match_point].tor.x, path.points[index_match_point].tor.y);
    Eigen::Vector2f vec_Rm(path.points[index_match_point].x, path.points[index_match_point].y);
    Eigen::Vector2f vec_Rr = vec_Rm + (vec_d.dot(vec_tor)) * vec_tor;
    projected_point.x = vec_Rr(0);
    projected_point.y = vec_Rr(1);
    projected_point.absolute_s = path.points[index_match_point].absolute_s + vec_d.dot(vec_tor);

    projected_point.l = 0; // 此处应该为投影点的l
    // projected_point.yaw = path.points[index_match_point].yaw + path.points[index_match_point].kappa * (vec_d.dot(vec_tor));
    projected_point.yaw = path.points[index_match_point].yaw;
    projected_point.yaw = revise_angle(projected_point.yaw);
    projected_point.tor.x = cos(projected_point.yaw);
    projected_point.tor.y = sin(projected_point.yaw);
    projected_point.kappa = path.points[index_match_point].kappa;
    return projected_point;
}

/**
 * @description: 增密min_path_nodes节点之间的点，相当于对s进行插值,每米20个s点
 * @param {Min_path_nodes} min_path_nodes
 * @return {*} 增密后的s点
 */
Frenet_path_points EMPlanner::InterpolatePoints(const Min_path_nodes &min_path_nodes,float host_start_s)
{
    Frenet_path_points frenet_path_points;

    int total_points_num = static_cast<int>(std::floor((min_path_nodes.nodes.size() - 1) * sample_s * sample_s_per_meters));
    Eigen::VectorXd ds_(total_points_num);
    Eigen::VectorXd l_(total_points_num);
    Eigen::VectorXd dl_(total_points_num);
    Eigen::VectorXd ddl_(total_points_num);
    Eigen::VectorXd dddl_(total_points_num);

    for (int i = 0; i < min_path_nodes.nodes.size() - 1; i++)
    {
        float start_l = min_path_nodes.nodes[i].node_l;
        float start_dl = 0;
        float start_ddl = 0;
        float end_l = min_path_nodes.nodes[i + 1].node_l;
        float end_dl = 0;
        float end_ddl = 0;
        float start_s = min_path_nodes.nodes[i].node_s;
        float end_s = min_path_nodes.nodes[i + 1].node_s;
        // cout<<i<<" start s l:"<<start_s<<" "<<start_l<<" ,end s l:"<<end_s<<" "<<end_l<<endl;
        // Eigen::VectorXd coeff = CalculateFiveDegreePolynomialCoefficients(start_l, start_dl, start_ddl, end_l, end_dl, end_ddl, start_s, end_s);
        Eigen::VectorXd coeff = CalculateThreeDegreePolynomialCoefficients(start_l, start_dl, start_ddl, end_l, end_dl, end_ddl, start_s, end_s);
        float a0 = coeff(0);
        float a1 = coeff(1);
        float a2 = coeff(2);
        float a3 = coeff(3);
        // float a4 = coeff(4);
        // float a5 = coeff(5);
        float points_num_float = sample_s * sample_s_per_meters;
        int points_num = static_cast<int>(std::floor(points_num_float)); // 每米20个点,这里丢掉了终点应该为sample_s * sample_s_per_meters+1

        Eigen::VectorXd I_ = Eigen::VectorXd::Ones(points_num);
        Eigen::VectorXd ds(points_num);
        Eigen::VectorXd l(points_num);
        Eigen::VectorXd dl(points_num);
        Eigen::VectorXd ddl(points_num);
        Eigen::VectorXd dddl(points_num);
        for (int j = 0; j < points_num; j++) // 0.1米一个点
        {
            ds(j) = start_s + j / points_num_float*sample_s;
        }
        // l = a0 * I_.array() + a1 * ds.array() + a2 * ds.array().pow(2) + a3 * ds.array().pow(3) + a4 * ds.array().pow(4) + a5 * ds.array().pow(5);
        // dl = a1 * I_.array() + 2 * a2 * ds.array() + 3 * a3 * ds.array().pow(2) + 4 * a4 * ds.array().pow(3) + 5 * a5 * ds.array().pow(4);
        // ddl = 2 * a2 * I_.array() + 6 * a3 * ds.array() + 12 * a4 * ds.array().pow(2) + 20 * a5 * ds.array().pow(3);
        // dddl = 6 * a3 * I_.array() + 24 * a4 * ds.array() + 60 * a5 * ds.array().pow(2);
        l = a0 * I_.array() + a1 * ds.array() + a2 * ds.array().pow(2) + a3 * ds.array().pow(3) ;
        dl = a1 * I_.array() + 2 * a2 * ds.array() + 3 * a3 * ds.array().pow(2) ;
        ddl = 2 * a2 * I_.array() + 6 * a3 * ds.array() ;
        dddl = 6 * a3 * I_.array();

        
        frenet_path_points.s = ds;

        ds_.segment(i * points_num, points_num) = ds.array()+host_start_s;
        l_.segment(i * points_num, points_num) = l;
        dl_.segment(i * points_num, points_num) = dl;
        ddl_.segment(i * points_num, points_num) = ddl;
        dddl_.segment(i * points_num, points_num) = dddl;
        
    }
    
    frenet_path_points.s = ds_;
    frenet_path_points.l = l_;
    frenet_path_points.dl = dl_;
    frenet_path_points.ddl = ddl_;
    frenet_path_points.dddl = dddl_;

    return frenet_path_points;
}

bool EMPlanner::BuildRlDpMinPath(float plan_start_s, float plan_start_l, Min_path_nodes &min_path_nodes)
{
    rl_dp_last_fail_reason_.clear();
    rl_dp_soft_fail_ = false;
    if (!use_rl_dp_) {
        rl_dp_last_fail_reason_ = rl_dp_disable_reason_.empty() ? "use_rl_dp disabled" : rl_dp_disable_reason_;
        return false;
    }
    if (!rl_dp_) {
        rl_dp_last_fail_reason_ = "rl_dp not initialized";
        return false;
    }
    if (rl_dp_s_samples_ <= 1 || rl_dp_l_samples_ <= 1) {
        rl_dp_last_fail_reason_ = "invalid rl_dp grid size";
        return false;
    }
    if (plan_start_l < rl_dp_l_min_ || plan_start_l > rl_dp_l_max_) {
        ROS_WARN_STREAM("RL_DP start l out of range: " << plan_start_l
                        << " not in [" << rl_dp_l_min_ << ", " << rl_dp_l_max_ << "]");
        rl_dp_last_fail_reason_ = "start_l out of range";
        return false;
    }

    std::vector<ObstacleCorners> obstacles;
    obstacles.reserve(obstacle_list->obstacles.size());
    std::vector<size_t> check_indices;
    check_indices.reserve(obstacle_list->obstacles.size());
    for (size_t obs_idx = 0; obs_idx < obstacle_list->obstacles.size(); ++obs_idx) {
        const auto& obs = obstacle_list->obstacles[obs_idx];
        if (obs.is_dynamic_obs) {
            continue;
        }
        float obs_min_s = static_cast<float>(obs.min_s - plan_start_s);
        float obs_max_s = static_cast<float>(obs.max_s - plan_start_s);
        float s_min_filter = rl_dp_s_min_ - 1.0f;
        if (obs_max_s < s_min_filter || obs_min_s > rl_dp_s_max_) {
            continue;
        }
        if (obs.max_l < rl_dp_l_min_ || obs.min_l > rl_dp_l_max_) {
            continue;
        }

        ObstacleCorners corners;
        for (size_t i = 0; i < corners.size(); ++i) {
            corners[i].s = static_cast<float>(obs.bounding_boxs_SL[i].x - plan_start_s);
            corners[i].l = static_cast<float>(obs.bounding_boxs_SL[i].y);
        }
        obstacles.push_back(corners);
        check_indices.push_back(obs_idx);
    }

    std::vector<int> path_indices;
    try {
        path_indices = rl_dp_->Plan(obstacles, plan_start_l);
    } catch (const std::exception& e) {
        ROS_WARN_STREAM("RL_DP plan failed: " << e.what());
        rl_dp_last_fail_reason_ = std::string("plan failed: ") + e.what();
        return false;
    }

    if (static_cast<int>(path_indices.size()) != rl_dp_s_samples_ || path_indices.size() < 2) {
        ROS_WARN_STREAM("RL_DP path size mismatch: expected " << rl_dp_s_samples_
                        << ", got " << path_indices.size());
        rl_dp_last_fail_reason_ = "path size mismatch";
        return false;
    }

    float s_step = (rl_dp_s_max_ - rl_dp_s_min_) / std::max(1, rl_dp_s_samples_ - 1);
    float l_step = (rl_dp_l_max_ - rl_dp_l_min_) / std::max(1, rl_dp_l_samples_ - 1);

    std::vector<float> rl_dp_l_values(path_indices.size(), 0.0f);
    for (size_t i = 0; i < path_indices.size(); ++i) {
        int l_index = std::max(0, std::min(rl_dp_l_samples_ - 1, path_indices[i]));
        rl_dp_l_values[i] = rl_dp_l_min_ + l_step * static_cast<float>(l_index);
    }

    int target_nodes = col_node_num + 1;
    if (target_nodes < 2) {
        rl_dp_last_fail_reason_ = "invalid target node count for QP";
        return false;
    }

    min_path_nodes.nodes.clear();
    min_path_nodes.nodes.reserve(static_cast<size_t>(target_nodes));
    for (int i = 0; i < target_nodes; ++i) {
        float node_s = static_cast<float>(i) * sample_s;
        float s_rel = node_s - rl_dp_s_min_;
        float idx_f = s_rel / s_step;
        int idx0 = static_cast<int>(std::floor(idx_f));
        if (idx0 < 0) {
            idx0 = 0;
        }
        if (idx0 >= static_cast<int>(path_indices.size()) - 1) {
            idx0 = static_cast<int>(path_indices.size()) - 2;
        }
        float s0 = rl_dp_s_min_ + s_step * static_cast<float>(idx0);
        float t = (s_step > 0.0f) ? (node_s - s0) / s_step : 0.0f;
        t = std::max(0.0f, std::min(1.0f, t));
        float l0 = rl_dp_l_values[static_cast<size_t>(idx0)];
        float l1 = rl_dp_l_values[static_cast<size_t>(idx0 + 1)];
        float node_l = (1.0f - t) * l0 + t * l1;
        if (i == 0) {
            node_l = plan_start_l;
        }
        Single_node node;
        node.node_s = node_s;
        node.node_l = node_l;
        min_path_nodes.nodes.push_back(node);
    }

    if (!check_indices.empty()) {
        auto node_hits_obstacle = [&](float abs_s,
                                      float node_l,
                                      const planning_msgs::Obstacle& obs) -> bool {
            if ((abs_s < obs.max_s) && (abs_s > obs.min_s)
                && (node_l < obs.max_l) && (node_l > obs.min_l)) {
                return true;
            }
            float ds = static_cast<float>(obs.s) - abs_s;
            float d01 = ds * ds + (node_l - static_cast<float>(obs.max_l))
                                 * (node_l - static_cast<float>(obs.max_l));
            float d02 = ds * ds + (node_l - static_cast<float>(obs.min_l))
                                 * (node_l - static_cast<float>(obs.min_l));
            float d1 = (static_cast<float>(obs.bounding_boxs_SL[0].x) - abs_s)
                       * (static_cast<float>(obs.bounding_boxs_SL[0].x) - abs_s)
                       + (node_l - static_cast<float>(obs.bounding_boxs_SL[0].y))
                         * (node_l - static_cast<float>(obs.bounding_boxs_SL[0].y));
            float d2 = (static_cast<float>(obs.bounding_boxs_SL[1].x) - abs_s)
                       * (static_cast<float>(obs.bounding_boxs_SL[1].x) - abs_s)
                       + (node_l - static_cast<float>(obs.bounding_boxs_SL[1].y))
                         * (node_l - static_cast<float>(obs.bounding_boxs_SL[1].y));
            float d3 = (static_cast<float>(obs.bounding_boxs_SL[2].x) - abs_s)
                       * (static_cast<float>(obs.bounding_boxs_SL[2].x) - abs_s)
                       + (node_l - static_cast<float>(obs.bounding_boxs_SL[2].y))
                         * (node_l - static_cast<float>(obs.bounding_boxs_SL[2].y));
            float d4 = (static_cast<float>(obs.bounding_boxs_SL[3].x) - abs_s)
                       * (static_cast<float>(obs.bounding_boxs_SL[3].x) - abs_s)
                       + (node_l - static_cast<float>(obs.bounding_boxs_SL[3].y))
                         * (node_l - static_cast<float>(obs.bounding_boxs_SL[3].y));
            float min_distance = d01;
            min_distance = std::min(min_distance, d02);
            min_distance = std::min(min_distance, d1);
            min_distance = std::min(min_distance, d2);
            min_distance = std::min(min_distance, d3);
            min_distance = std::min(min_distance, d4);
            return min_distance < dp_min_collision_distance_pow2;
        };

        bool node_collision = false;
        float collision_abs_s = 0.0f;
        float collision_l = 0.0f;
        size_t collision_obs_idx = 0;
        for (const auto& node : min_path_nodes.nodes) {
            float abs_s = plan_start_s + node.node_s;
            for (size_t obs_idx : check_indices) {
                const auto& obs = obstacle_list->obstacles[obs_idx];
                if (node_hits_obstacle(abs_s, node.node_l, obs)) {
                    node_collision = true;
                    collision_abs_s = abs_s;
                    collision_l = node.node_l;
                    collision_obs_idx = obs_idx;
                    break;
                }
            }
            if (node_collision) {
                break;
            }
        }
        if (node_collision) {
            rl_dp_soft_fail_ = true;
            rl_dp_last_fail_reason_ =
                "node collision s=" + std::to_string(collision_abs_s)
                + ", l=" + std::to_string(collision_l)
                + ", obs_idx=" + std::to_string(collision_obs_idx);
            return false;
        }
    }

    return true;
}

/**
 * @description: 计算最小代价的路径节点
 * @param {float} plan_star_s
 * @param {float} plan_star_l
 * @return {*} 返回最小代价的路径节点的sl
 */
Min_path_nodes EMPlanner::CalcNodeMinCost(float plan_star_s, float plan_star_l,float start_dl)
{
    Avoid_nodes avoid_nodes;
    Col_nodes col_node;
    avoid_nodes.nodes.reserve(col_node_num);
    col_node.row_nodes.reserve(row_node_num);
    for (int i = 0; i < row_node_num; i++) // 计算起点与第一列的cost，单独计算
    {
        Single_node single_node;
        float node_s = sample_s;
        float node_l = ((row_node_num - 1) / 2 - i) * sample_l;
        double cost = CalculateStarCost(plan_star_s, plan_star_l, start_dl, 0, node_s, node_l);
        single_node.node_s = node_s;
        single_node.node_l = node_l;
        single_node.toThisNodeMinCost = cost;      // 第一列最小cost就是起点到此点 ，不需要判断
        single_node.pre2this_min_cost_index = 0;   // 第一列最小cost序号就是起点 ，不需要判断
        col_node.row_nodes.push_back(single_node); // 将第一列节点加入
    }
    avoid_nodes.nodes.push_back(col_node); // 列加入总节点（未包含原点）

    #pragma omp for

    for (int j = 1; j < col_node_num; j++) // 计算第一列与第最后一列的cost
    {
        Col_nodes col_nodes;
        col_nodes.row_nodes.reserve(row_node_num);
        for (int i = 0; i < row_node_num; i++) // 行
        {
            float node_s = (j + 1) * sample_s;        // 该节点的s
            float node_l = ((row_node_num - 1) / 2 - i) * sample_l; // 无误 
            double min_cost = std::numeric_limits<double>::max();
            int min_cost_index;
            Single_node sigle_node;
            sigle_node.node_s = node_s;
            sigle_node.node_l = node_l;
            for (int k = 0; k < row_node_num; k++) // 遍历上一列
            {
                if((abs(k-i)*sample_l/ sample_s)>1.2)  //提前将不可能的节点删除，提高计算速度
                {
                    continue;
                }
                double cost = CalculateForwardCost(avoid_nodes.nodes[j - 1].row_nodes[k].node_s, avoid_nodes.nodes[j - 1].row_nodes[k].node_l, node_s, node_l,plan_star_s);
                cost = cost + avoid_nodes.nodes[j - 1].row_nodes[k].toThisNodeMinCost;
                if (cost < min_cost)
                {
                    min_cost = cost;
                    min_cost_index = k;
                }
            }
            
            sigle_node.toThisNodeMinCost = min_cost;
            sigle_node.pre2this_min_cost_index = min_cost_index;
            col_nodes.row_nodes.push_back(sigle_node);
            // cout<<"行:"<<i<<"，列："<<j<<" min_cost:"<<min_cost<<" min_cost_index:"<<min_cost_index<<endl;
        }

        avoid_nodes.nodes.push_back(col_nodes);
    }

    double path_min_cost = std::numeric_limits<double>::max();
    int min_cost_last_index;
    // 寻找全局最小代价路径
    for (int i = 0; i < row_node_num; i++)
    {
        if (avoid_nodes.nodes[col_node_num - 1].row_nodes[i].toThisNodeMinCost < path_min_cost)
        {
            path_min_cost = avoid_nodes.nodes[col_node_num - 1].row_nodes[i].toThisNodeMinCost;
            min_cost_last_index = i;
        }
    }
    cout<<"DP_Path总代价："<<path_min_cost<<endl;
    // 从后向前遍历
    vector<int> min_cost_index;
    min_cost_index.push_back(min_cost_last_index); // 最优路径最后一列的序号
    for (int i = col_node_num - 1; i > 0; i--)     // 倒序输入
    {
        min_cost_index.push_back(avoid_nodes.nodes[i].row_nodes[min_cost_last_index].pre2this_min_cost_index);
        min_cost_last_index = avoid_nodes.nodes[i].row_nodes[min_cost_last_index].pre2this_min_cost_index;
    }
    std::reverse(min_cost_index.begin(), min_cost_index.end()); // 序号反转，改为正序
    avoid_nodes.min_cost_path_indexs = min_cost_index;

    // for (int i = 0; i < min_cost_index.size(); i++)
    // {
    //     cout<<"index:"<<i<<"、min_index:"<<min_cost_index[i]<<" node_s:"<<avoid_nodes.nodes[i].row_nodes[min_cost_index[i]].node_s<<" node_l:"<<avoid_nodes.nodes[i].row_nodes[min_cost_index[i]].node_l<<endl;
    // }
    Single_node min_path_node_star;
    Min_path_nodes min_path_nodes;

    min_path_node_star.node_s = 0; // 加入起点
    min_path_node_star.node_l = plan_star_l;
    min_path_nodes.nodes.push_back(min_path_node_star);

    for (int i = 0; i < min_cost_index.size(); i++)
    {
        Single_node min_path_node;
        min_path_node.node_s = (avoid_nodes.nodes[i].row_nodes[min_cost_index[i]].node_s);
        min_path_node.node_l = (avoid_nodes.nodes[i].row_nodes[min_cost_index[i]].node_l);
        // cout<<"min_path_node_s:"<<min_path_node.node_s<<" l:"<<min_path_node.node_l<<endl;
        min_path_nodes.nodes.push_back(min_path_node);
    }
    return min_path_nodes;
}

double EMPlanner::CalculateForwardCost(float pre_node_s, float pre_node_l, float current_node_s, float current_node_l,float host_start_s)
{
    // cout<<"CalculateForwardCost_star"<<endl;
    float start_l = pre_node_l;
    float start_dl = 0;
    float start_ddl = 0;
    float end_l = current_node_l;
    float end_dl = 0;
    float end_ddl = 0;
    float start_s = pre_node_s;
    float end_s = current_node_s;
    Eigen::VectorXd coeff = CalculateThreeDegreePolynomialCoefficients(start_l, start_dl, start_ddl, end_l, end_dl, end_ddl, start_s, end_s);

    float a0 = coeff(0);
    float a1 = coeff(1);
    float a2 = coeff(2);
    float a3 = coeff(3);
    // float a4 = coeff(4);
    // float a5 = coeff(5);

    int points_num = static_cast<int>(std::floor(sample_s * sample_s_num)); // 输出80
    Eigen::VectorXd I_ = Eigen::VectorXd::Ones(points_num);
    Eigen::VectorXd ds = Eigen::VectorXd::LinSpaced(points_num, start_s, start_s + sample_s); // 生成等间距的向量
    Eigen::VectorXd ds_pow2 = ds.array().pow(2);
    Eigen::VectorXd ds_pow3 = ds.array().pow(3);
    // Eigen::VectorXd ds_pow4 = ds.array().pow(4);
    // Eigen::VectorXd ds_pow5 = ds.array().pow(5);

    // Eigen::VectorXd l = a0 * I_ + a1 * ds + a2 * ds_pow2 + a3 * ds_pow3 + a4 * ds_pow4 + a5 * ds_pow5;
    // Eigen::VectorXd dl = a1 * I_ + 2 * a2 * ds + 3 * a3 * ds_pow2 + 4 * a4 * ds_pow3 + 5 * a5 * ds_pow4;
    // Eigen::VectorXd ddl = 2 * a2 * I_ + 6 * a3 * ds + 12 * a4 * ds_pow2 + 20 * a5 * ds_pow3;
    // Eigen::VectorXd dddl = 6 * a3 * I_ + 24 * a4 * ds + 60 * a5 * ds_pow2;

    Eigen::VectorXd l = a0 * I_ + a1 * ds + a2 * ds_pow2 + a3 * ds_pow3;
    Eigen::VectorXd dl = a1 * I_ + 2 * a2 * ds + 3 * a3 * ds_pow2 ;
    Eigen::VectorXd ddl = 2 * a2 * I_ + 6 * a3 * ds ;
    Eigen::VectorXd dddl = 6 * a3 * I_;

    float cost_smooth =w_cost_smooth_total * (w_cost_smooth_dl * dl.squaredNorm() + w_cost_smooth_ddl * ddl.squaredNorm() + w_cost_smooth_dddl * dddl.squaredNorm()) / ds.size();
    if(dl.cwiseAbs().maxCoeff() > 1.4)  //为防止出现起始点转角过大而无法实现车辆运动学
    {
        cost_smooth+=1e6;
    };
    // cout<<"ds.size():"<<ds.size()<<endl;

    float cost_ref = w_cost_ref * l.squaredNorm();
    double cost_collision = 0;
    // double cost_right_boundary=0;
    // if(pre_node_l<DpPathBoundaryRightLimit||current_node_l<DpPathBoundaryRightLimit)
    // {
    //     cost_right_boundary=1e10;
    // }

    bool is_collsion=false;
    for (int i = 0; i < obstacle_list->obstacles.size(); i++)
    {
        if(obstacle_list->obstacles[i].is_dynamic_obs
        ||(car_direct==1&&(obstacle_list->obstacles[i].max_s<host_middle_sl(0)
        ||obstacle_list->obstacles[i].min_s>host_middle_sl(0)+sample_s*col_node_num))
        ||obstacle_list->obstacles[i].max_l>sample_l*row_node_num||obstacle_list->obstacles[i].min_l< -sample_l*row_node_num)  //规划SL范围外的障碍物均不考虑
        {
            // cout<<"obsssss:max_s"<<obstacle_list->obstacles[i].max_s<<","<<host_middle_sl(0)<<","<<obstacle_list->obstacles[i].min_s<<","<<host_middle_sl(0)+sample_s*col_node_num<<endl;
            continue;
        }
        for (int j = 0; j < points_num; j++) // 0.1米一个点
        {
            cost_collision += CalcObstacleCost(obstacle_list->obstacles[i], ds[j], l[j],host_start_s,is_collsion);
            if(is_collsion)
                break;
        }
        if(is_collsion)
            break;
    }
    cost_collision = cost_collision * w_cost_collision;
    double cost_all = 0;
    cost_all = cost_collision + cost_ref + cost_smooth ;
    // cout<<"cost_collision:"<<cost_collision<<" cost_ref:"<<cost_ref<<" cost_smooth:"<<cost_smooth<<endl;
    return cost_all;
}

double EMPlanner::CalculateStarCost(float begin_s, float begin_l, float begin_dl, float begin_ddl, float end_S, float end_L)
{
    // cout<<"CalculateStarCost"<<endl;
    float start_l = begin_l;
    float start_dl = begin_dl;
    float start_ddl = begin_ddl;
    float end_l = end_L;
    float end_dl = 0;
    float end_ddl = 0;
    float start_s = 0;
    float end_s = end_S;
    Eigen::VectorXd coeff = CalculateThreeDegreePolynomialCoefficients(start_l, start_dl, start_ddl, end_l, end_dl, end_ddl, start_s, end_s);

    float a0 = coeff(0);
    float a1 = coeff(1);
    float a2 = coeff(2);
    float a3 = coeff(3);
    // float a4 = coeff(4);
    // float a5 = coeff(5);

    int points_num = static_cast<int>(std::floor(sample_s * sample_s_num)); // 输出80
    Eigen::VectorXd I_ = Eigen::VectorXd::Ones(points_num);
    Eigen::VectorXd ds = Eigen::VectorXd::LinSpaced(points_num, start_s, start_s + sample_s); // 生成等间距的向量
    Eigen::VectorXd ds_pow2 = ds.array().pow(2);
    Eigen::VectorXd ds_pow3 = ds.array().pow(3);
    // Eigen::VectorXd ds_pow4 = ds.array().pow(4);
    // Eigen::VectorXd ds_pow5 = ds.array().pow(5);

    // Eigen::VectorXd l = a0 * I_ + a1 * ds + a2 * ds_pow2 + a3 * ds_pow3 + a4 * ds_pow4 + a5 * ds_pow5;
    // Eigen::VectorXd dl = a1 * I_ + 2 * a2 * ds + 3 * a3 * ds_pow2 + 4 * a4 * ds_pow3 + 5 * a5 * ds_pow4;
    // Eigen::VectorXd ddl = 2 * a2 * I_ + 6 * a3 * ds + 12 * a4 * ds_pow2 + 20 * a5 * ds_pow3;
    // Eigen::VectorXd dddl = 6 * a3 * I_ + 24 * a4 * ds + 60 * a5 * ds_pow2;

    Eigen::VectorXd l = a0 * I_ + a1 * ds + a2 * ds_pow2 + a3 * ds_pow3;
    Eigen::VectorXd dl = a1 * I_ + 2 * a2 * ds + 3 * a3 * ds_pow2 ;
    Eigen::VectorXd ddl = 2 * a2 * I_ + 6 * a3 * ds ;
    Eigen::VectorXd dddl = 6 * a3 * I_;

    float cost_smooth =w_cost_smooth_total * (w_cost_smooth_dl * dl.squaredNorm() + w_cost_smooth_ddl * ddl.squaredNorm() + w_cost_smooth_dddl * dddl.squaredNorm()) / ds.size();
    if(dl.cwiseAbs().maxCoeff() > 1.4)  //为防止出现起始点转角过大而无法实现车辆运动学
    {
        cost_smooth+=1e6;
    };

    float cost_ref = w_cost_ref * l.squaredNorm();
    double cost_collision = 0;
    // double cost_right_boundary=0;
    // if(begin_l<DpPathBoundaryRightLimit||end_L<DpPathBoundaryRightLimit)
    // {
    //     cost_right_boundary=1e10;
    // }
    bool is_collsion=false;
    for (int i = 0; i < obstacle_list->obstacles.size(); i++) 
    {
        if(obstacle_list->obstacles[i].is_dynamic_obs
        ||(car_direct==1&&(obstacle_list->obstacles[i].max_s<host_middle_sl(0)
        ||obstacle_list->obstacles[i].min_s>host_middle_sl(0)+sample_s*col_node_num))
        ||obstacle_list->obstacles[i].max_l>sample_l*row_node_num||obstacle_list->obstacles[i].min_l< -sample_l*row_node_num)
        {
            // cout<<"动态障碍物,或车前进时的车身后的障碍物。"<<endl;
            continue;
        }
        for (int j = 0; j < points_num; j++) // 0.1米一个点
        {
            // cout<<"1!"<<points_num<<endl;
            cost_collision += CalcObstacleCost(obstacle_list->obstacles[i], ds[j], l[j], begin_s , is_collsion);  //减少计算量，只考虑每个节点的起始节点，中间节点终止节点
            if(is_collsion)
                break;

            // cout<<"points_num:"<<points_num<<endl;
        }
        if(is_collsion)
            break;
    }
    cost_collision = cost_collision * w_cost_collision;

    double cost_all = 0;
    cost_all = cost_collision + cost_ref + cost_smooth ;
    // cout<<"start_cost_collision:"<<cost_collision<<" start_cost_ref:"<<cost_ref<<" start_cost_smooth:"<<cost_smooth<<endl;
    return cost_all;
}

double EMPlanner::calcDistance(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2) {
    return (p1 - p2).norm(); // Euclidean distance
}

double EMPlanner::CalcObstacleCost(planning_msgs::Obstacle &Obstacle_, float aim_s, float aim_l,float host_start_s,bool &is_collsion)
{
    const double abs_s = static_cast<double>(host_start_s) + static_cast<double>(aim_s);
    if ((abs_s < Obstacle_.max_s) && (abs_s > Obstacle_.min_s) &&
        (aim_l < Obstacle_.max_l) && (aim_l > Obstacle_.min_l)) {
        is_collsion = true;
        return 1e8;
    }

    const Eigen::Vector2d A(Obstacle_.bounding_boxs_SL[0].x - host_start_s, Obstacle_.bounding_boxs_SL[0].y);
    const Eigen::Vector2d B(Obstacle_.bounding_boxs_SL[1].x - host_start_s, Obstacle_.bounding_boxs_SL[1].y);
    const Eigen::Vector2d C(Obstacle_.bounding_boxs_SL[2].x - host_start_s, Obstacle_.bounding_boxs_SL[2].y);
    const Eigen::Vector2d D(Obstacle_.bounding_boxs_SL[3].x - host_start_s, Obstacle_.bounding_boxs_SL[3].y);
    const Eigen::Vector2d P(aim_s, aim_l);

    if (pointInsideQuadrilateral(A, B, C, D, P)) {
        is_collsion = true;
        return 1e8;
    }

    auto point_to_segment_dist_sq = [](const Eigen::Vector2d& p,
                                       const Eigen::Vector2d& s0,
                                       const Eigen::Vector2d& s1) -> double {
        const Eigen::Vector2d seg = s1 - s0;
        const double seg_len_sq = seg.squaredNorm();
        if (seg_len_sq <= 1e-12) {
            return (p - s0).squaredNorm();
        }
        double t = (p - s0).dot(seg) / seg_len_sq;
        t = std::max(0.0, std::min(1.0, t));
        const Eigen::Vector2d proj = s0 + t * seg;
        return (p - proj).squaredNorm();
    };

    double min_distance_sq = point_to_segment_dist_sq(P, A, B);
    min_distance_sq = std::min(min_distance_sq, point_to_segment_dist_sq(P, B, C));
    min_distance_sq = std::min(min_distance_sq, point_to_segment_dist_sq(P, C, D));
    min_distance_sq = std::min(min_distance_sq, point_to_segment_dist_sq(P, D, A));

    if (min_distance_sq >= static_cast<double>(dp_min_collision_distance_pow2)) {
        return 0.0;
    }

    is_collsion = true;
    return 1e8;
}

//判断障碍物与障碍物的四边形的位置，此函数为计算绕数
int EMPlanner::windingNumber(Eigen::Vector2d A, Eigen::Vector2d B, Eigen::Vector2d P) {
    // Calculate vector AB and AP
    double ABx = B(0) - A(0);
    double ABy = B(1) - A(1);
    double APx = P(0) - A(0);
    double APy = P(1) - A(1);
    // cout<<"ABx"<<ABx<<","<<ABy<<","<<APx<<","<<APy<<endl;
    // Calculate cross product AB x AP
    double crossProduct = ABx * APy - ABy * APx;
    
    // Determine the winding number based on the sign of the cross product
    if (crossProduct > 0) {
        return 1;   // P is to the left of AB
    } else if (crossProduct < 0) {
        return -1;  // P is to the right of AB
    } else {
        return 0;   // P is collinear with AB
    }
}
//判断障碍物与障碍物的四边形的位置,是否在障碍物边框内部
bool EMPlanner::pointInsideQuadrilateral(Eigen::Vector2d A, Eigen::Vector2d B, Eigen::Vector2d C, Eigen::Vector2d D, Eigen::Vector2d P) {
    // cout<<"ABCD:"<<A<<","<<B<<","<<C<<","<<D<<","<<P<<endl;
    double windingABP = windingNumber(A, B, P);
    double windingBCP = windingNumber(B, C, P);
    double windingCDP = windingNumber(C, D, P);
    double windingDAP = windingNumber(D, A, P);
    // cout<<"abcdp:"<<windingABP<<","<<windingBCP<<","<<windingCDP<<","<<windingDAP<<endl;
    
    // P is inside ABCD if all winding numbers are either all positive or all negative
    if ((windingABP > 0 && windingBCP > 0 && windingCDP > 0 && windingDAP > 0) ||
        (windingABP < 0 && windingBCP < 0 && windingCDP < 0 && windingDAP < 0) || ((windingABP*windingBCP*windingCDP*windingDAP)==0)) {
        return true;
    } else {
        return false;
    }
}

/**
 * @description: 求解五次多项式的系数
 * @param {float} star_l
 * @param {float} star_dl l一阶导数
 * @param {float} star_ddl 二阶导数
 * @param {float} end_l
 * @param {float} end_dl
 * @param {float} end_ddl
 * @param {float} star_s
 * @param {float} end_s
 * @return {*} 五次多项式系数
 */
Eigen::VectorXd EMPlanner::CalculateFiveDegreePolynomialCoefficients(float start_l, float start_dl, float start_ddl, float end_l, float end_dl, float end_ddl, float start_s, float end_s)
{
    float ss = start_s * start_s;
    float sss = ss * start_s;
    float ssss = sss * start_s;
    float sssss = ssss * start_s;

    float es = end_s * end_s;
    float ess = es * end_s;
    float esss = ess * end_s;
    float essss = esss * end_s;

    Eigen::MatrixXd A(6, 6);
    A << 1, start_s, ss, sss, ssss, sssss,
        0, 1, 2 * start_s, 3 * ss, 4 * sss, 5 * ssss,
        0, 0, 2, 6 * start_s, 12 * ss, 20 * sss,
        1, end_s, es, ess, esss, essss,
        0, 1, 2 * end_s, 3 * es, 4 * ess, 5 * esss,
        0, 0, 2, 6 * end_s, 12 * es, 20 * ess;
    Eigen::VectorXd B(6);
    B << start_l, start_dl, start_ddl, end_l, end_dl, end_ddl;
    // 求解线性方程组
    // 求解线性方程组
    // Eigen::VectorXd x = A.colPivHouseholderQr().solve(B);
    Eigen::VectorXd x = A.fullPivLu().solve(B);
    // 检查矩阵是否满秩，即方程组是否有唯一解
    // if(solver.rank() != A.rows()) {
    //     std::cerr << "矩阵不满秩，可能没有唯一解！" << std::endl;
    //     // 根据实际情况处理，比如返回一个默认解或处理特殊情况
    // }
    
    return x;
}

Eigen::VectorXd EMPlanner::CalculateThreeDegreePolynomialCoefficients(float start_l, float start_dl, float start_ddl, float end_l, float end_dl, float end_ddl, float start_s, float end_s)
{
    float ss = start_s * start_s;
    float sss = ss * start_s;

    float es = end_s * end_s;
    float ess = es * end_s;

    Eigen::MatrixXd A(4, 4);
    A << 1, start_s, ss, sss, 
        0, 1, 2 * start_s, 3 * ss, 
        1, end_s, es, ess,
        0, 1, 2 * end_s, 3 * es;
    Eigen::VectorXd B(4);
    B << start_l, start_dl, end_l, end_dl;
    // 求解线性方程组
    // 求解线性方程组
    // Eigen::VectorXd x = A.colPivHouseholderQr().solve(B);
    Eigen::VectorXd x = A.fullPivLu().solve(B);
    // 检查矩阵是否满秩，即方程组是否有唯一解
    // if(solver.rank() != A.rows()) {
    //     std::cerr << "矩阵不满秩，可能没有唯一解！" << std::endl;
    //     // 根据实际情况处理，比如返回一个默认解或处理特殊情况
    // }
    return x;
}


/**
 * @description: 通过优化后的路径来计算航向角
 * @param {VectorXd} QPSolution 优化后路径点
 * @param {vector<float>&} temp_phi 临时存储航向角变量
 * @return {*}
 */
void EMPlanner::Calculate_OptimizedPath_Heading(const Eigen::VectorXd &QPSolution, vector<float> &temp_phi)
{
    if (total_points_num < 21) {
        ROS_ERROR_STREAM("total_points_num too small for heading calculation: " << total_points_num);
        return;
    }

    // 计算航向角(路线斜率) 隔着两个点计算
    for (int i = 0; i < 2 * total_points_num - 12; i += 2)
    {
        float dif_x1 = QPSolution(i + 6) - QPSolution(i);
        float dif_x2 = QPSolution(i + 12) - QPSolution(i + 6);
        float dif_y1 = QPSolution(i + 7) - QPSolution(i + 1);
        float dif_y2 = QPSolution(i + 13) - QPSolution(i + 7);
        float phi_1 = atan2(dif_y1, dif_x1);
        float phi_2 = atan2(dif_y2, dif_x2);
        if (abs(phi_1 - phi_2) > 3.14)
        {
            temp_phi[(i / 2) + 3] = phi_2;
        }
        else if (phi_1 + phi_2 == 0)
        {
            temp_phi[(i / 2) + 3] = temp_phi[(i / 2) + 2];
        }
        else
        {
            temp_phi[(i / 2) + 3] = (phi_1 + phi_2) / 2;
        }
    }
    // temp_phi[0] = temp_phi[3];
    // temp_phi[1] = temp_phi[3];
    // temp_phi[2] = temp_phi[3];
    temp_phi[total_points_num - 1] = temp_phi[total_points_num - 4];
    temp_phi[total_points_num - 2] = temp_phi[total_points_num - 4];
    temp_phi[total_points_num - 3] = temp_phi[total_points_num - 4];
    for (int i = 0; i < 20; i++)// 因为计算出的头部偏航角不稳定和不准确，这里纯属暂时应对方法，应该当全局规划路线换为Astart等后，用多项式求导求出，而不是简单计算！？
    {
        temp_phi[i]=temp_phi[20];
    }

    for (int i = 1; i < total_points_num; i++)
    {
        if (temp_phi[i] == 0)
            temp_phi[i] = temp_phi[i - 1];
    }

    line_record_opt->points.reserve(2 * total_points_num);
    for (int i = 0; i < 2 * total_points_num; i += 2)
    {
        pcl::PointXYZI points_;
        points_.x = QPSolution(i);
        points_.y = QPSolution(i + 1);
        points_.z = temp_phi[i / 2];
        // points_.intensity = line_record_->points[i].intensity;
        line_record_opt->push_back(points_);
    }
}

/**
 * @description: 通过优化后的路径来计算航向角
 * @param {VectorXd} QPSolution 优化后路径点
 * @param {vector<float>&} temp_phi 临时存储航向角变量
 * @return {*}
 */
void EMPlanner::Calculate_OptimizedPath_Heading(const vector<Eigen::Vector2d> &QPSolution, vector<float> &temp_phi, pcl::PointCloud<pcl::PointXYZI>::Ptr &line_qp_path)
{
    // 计算航向角(路线斜率) 隔着两个点计算
    for (int i = 0; i < QPSolution.size() - 6; i++)
    {
        float dif_x1 = QPSolution[i + 3][0] - QPSolution[i][0];
        float dif_x2 = QPSolution[i + 6][0] - QPSolution[i + 3][0];
        float dif_y1 = QPSolution[i + 3][1] - QPSolution[i][1];
        float dif_y2 = QPSolution[i + 6][1] - QPSolution[i + 3][1];
        float phi_1 = atan2(dif_y1, dif_x1);
        float phi_2 = atan2(dif_y2, dif_x2);
        if (abs(phi_1 - phi_2) > 3.14)
        {
            temp_phi[i + 3] = phi_2;
        }
        else if (phi_1 + phi_2 == 0)
        {
            temp_phi[i + 3] = temp_phi[i + 2];
        }
        else
        {
            temp_phi[i + 3] = (phi_1 + phi_2) / 2;
        }
    }

    temp_phi[0] = temp_phi[3];
    temp_phi[1] = temp_phi[3];
    temp_phi[2] = temp_phi[3];
    temp_phi[QPSolution.size() - 1] = temp_phi[QPSolution.size() - 4];
    temp_phi[QPSolution.size() - 2] = temp_phi[QPSolution.size() - 4];
    temp_phi[QPSolution.size() - 3] = temp_phi[QPSolution.size() - 4];

    for (int i = 1; i < temp_phi.size(); i++)
    {
        if (temp_phi[i] == 0)
            temp_phi[i] = temp_phi[i - 1];
    }

    line_qp_path.reset(new pcl::PointCloud<pcl::PointXYZI>);
    line_qp_path->points.reserve(QPSolution.size());

    for (int i = 0; i < QPSolution.size(); i++)
    {
        pcl::PointXYZI points_;
        points_.x = QPSolution[i][0];
        points_.y = QPSolution[i][1];
        points_.z = temp_phi[i];
        // cout<<i<<" xyz"<<points_.x<<" "<<points_.y<<" "<<points_.z<<endl;
        line_qp_path->push_back(points_);
    }
}

/**
 * @description: 计算路径上的航向角，曲率，切、法向量、速度规划（暂时）
 * @param {car_path} &path  输入路径点
 * @return {*}
 */
void EMPlanner::Path_Parameter_Calculate(planning_msgs::car_path &path)
{
    cout << "Path_Parameter_Calculate" << endl;
    if (line_record->points.size() < 3) {
        ROS_ERROR_STREAM("Insufficient points in line_record: " << line_record->points.size());
        return;
    }

    path.points.clear();
    // ** 速度规划 ** /
    int num = 0;
    // 第一次push_back到path_points
    path.points.reserve(line_record->points.size() + 1);
    for (pcl::PointCloud<pcl::PointXYZI>::iterator it = line_record->begin(); it != line_record->end(); ++it)
    {
        // cout << "star" << num << endl;
        planning_msgs::path_point point_temp;
        point_temp.number = num;
        num++;
        point_temp.x = (*it).x;
        point_temp.y = (*it).y;
        point_temp.yaw = (*it).z;
        point_temp.theta = (*it).intensity;
        
        path.points.push_back(point_temp);
        // cout << "number,int,vel: " << path.number[num - 1] << " " << path.theta[num - 1] << " " << path.vel[num - 1] << endl;
    }

    // 曲率、ds计算
    float kappa_ = 0;
    path.points[0].ds = 0;
    path.points[0].absolute_s = 0;
    // path.points[0].kappa = 0;

    float calc_s_temp = 0; // 临时变量，用于计算绝对的纵向距离
    for (int i = 1; i != path.points.size(); ++i)
    {
        path.points[i].ds = (sqrt(pow((path.points[i].x - path.points[i - 1].x), 2) + pow((path.points[i].y - path.points[i - 1].y), 2)));
        calc_s_temp += path.points[i].ds;
        path.points[i].absolute_s = calc_s_temp;

        // float delta_angle = abs(path.points[i].yaw - path.points[i - 1].yaw);
        // if (abs(delta_angle) > M_PI)
        // {
        //     delta_angle = 2 * M_PI - abs(path.points[i].yaw) - abs(path.points[i - 1].yaw);
        // }
        // kappa_ = delta_angle / path.points[i].ds;
        // if (abs(kappa_) < 0.001) // 太小置0
        //     kappa_ = 0;
        // if (delta_angle < 0.0001)
        // {
        //     path.points[i].kappa = path.points[i - 1].kappa;
        // }
        // else
        // {
        //     path.points[i].kappa = kappa_;
        // }
    }
    // 曲率计算
    for (int i = 2; i < path.points.size() - 2; ++i)
    {
        float a = sqrt(pow(path.points[i + 2].x - path.points[i].x, 2) + pow(path.points[i + 2].y - path.points[i].y, 2));
        float b = sqrt(pow(path.points[i + 2].x - path.points[i - 2].x, 2) + pow(path.points[i + 2].y - path.points[i - 2].y, 2));
        float c = sqrt(pow(path.points[i].x - path.points[i - 2].x, 2) + pow(path.points[i].y - path.points[i - 2].y, 2));
        float temp_ = (pow(a, 2) + pow(c, 2) - pow(b, 2)) / (2 * a * c);
        if (temp_ > 1)
            temp_ = 1;
        if (temp_ < -1)
            temp_ = -1;
        float theta_b = acos(temp_);
        // cout<<"theta_b:"<<theta_b<<endl;
        path.points[i].kappa = 2 * sin(theta_b) / b;
    }
    path.points[0].kappa = path.points[2].kappa;
    path.points[1].kappa = path.points[2].kappa;
    path.points[path.points.size() - 2].kappa = path.points[path.points.size() - 3].kappa;
    path.points[path.points.size() - 1].kappa = path.points[path.points.size() - 3].kappa;
    // 法向量、切向量初始化
    for (int i = 0; i != path.points.size(); ++i)
    {
        path.points[i].tor.x = cos(path.points[i].yaw);
        path.points[i].tor.y = sin(path.points[i].yaw);
        path.points[i].nor.x = -sin(path.points[i].yaw);
        path.points[i].nor.y = cos(path.points[i].yaw);
    }
    for (int i = 1; i < path.points.size(); i++)
    {
        if (path.points[i].flag_turn == 0 && path.points[i - 1].flag_turn == 1)
        {
            get_front_point_and_setVel(path, i);
        }
    }
}

/**
 * @description: 计算路径上的航向角，曲率，切、法向量、速度规划（暂时）
 * @param {car_path} &path  输入路径点
 * @return {*}
 */
void EMPlanner::QP_Path_Publish(planning_msgs::car_path::Ptr &path, pcl::PointCloud<pcl::PointXYZI>::Ptr line_qp_Interpolation)
{
    // ** 速度规划 ** /
    int num = 0;
    // 第一次push_back到path_points
    path->points.reserve(line_qp_Interpolation->points.size() + 1);
    for (pcl::PointCloud<pcl::PointXYZI>::iterator it = line_qp_Interpolation->begin(); it != line_qp_Interpolation->end(); ++it)
    {
        planning_msgs::path_point point_temp;
        point_temp.number = num;
        point_temp.x = (*it).x;
        point_temp.y = (*it).y;
        point_temp.yaw = (*it).z;
        path->points.push_back(point_temp);
        num++;
    }
}

/**
 * @description: 均值插值
 * @param {Ptr} line_record_opt 输入路径点
 * @param {Ptr} line_record 输出插值路径点
 * @return {*}
 */
void EMPlanner::Mean_Interpolation(pcl::PointCloud<pcl::PointXYZI>::Ptr line_record_opt, pcl::PointCloud<pcl::PointXYZI>::Ptr line_record)
{
    pcl::PointXYZI points_next;
    line_record->points.reserve(line_record_opt->points.size() * (divide_num + 1));
    for (pcl::PointCloud<pcl::PointXYZI>::iterator it = line_record_opt->begin(); it != line_record_opt->end() - 1; it++)
    {
        line_record->push_back(*it);
        points_next = *(it + 1);
        for (int i = 0; i < divide_num; i++)
        {
            pcl::PointXYZI points;
            points.x = (i + 1) * (points_next.x - it->x) / (divide_num + 1) + it->x;
            points.y = (i + 1) * (points_next.y - it->y) / (divide_num + 1) + it->y;

            if (abs(points_next.z - it->z) > M_PI)
            {
                float abs_delta_z = 2 * M_PI - (abs(points_next.z) + abs(it->z));
                if (it->z > 0)
                {
                    points.z = it->z + (i + 1) * (abs_delta_z) / (divide_num + 1);
                }
                else
                {
                    points.z = it->z - (i + 1) * (abs_delta_z) / (divide_num + 1);
                }
                if (points.z > M_PI)
                {
                    points.z -= 2 * M_PI;
                }
                if (points.z < -M_PI)
                {
                    points.z += 2 * M_PI;
                }
            }
            else
            {
                points.z = (i + 1) * (points_next.z - it->z) / (divide_num + 1) + it->z;
            }
            points.intensity = (i + 1) * (points_next.intensity - it->intensity) / (divide_num + 1) + it->intensity;
            line_record->push_back(points);
        }
    }
    line_record->push_back(points_next);
}

/**
 * @description: 注意要加障碍物判断，否则障碍物太多，其次需要加障碍物信息，判断障碍物四个角点的最大最小sl信息,注意qp_path数量序号要和增密后的dp_path数量一致
 * @param {Obstacle_list} obstacle_list
 * @param {Frenet_path_points} &planning_Interpolate_path
 * @param {VectorXd} &l_min
 * @param {VectorXd} &l_max
 * @param {float} min_l
 * @param {float} max_l
 * @return {*}
 */
void EMPlanner::calculatePathBoundary(planning_msgs::ObstacleList::Ptr &obstacle_list, const Frenet_path_points &DP_path_sl, Eigen::VectorXd &l_min, Eigen::VectorXd &l_max, float min_l, float max_l,float delta_s, float start_l)
{
    // 先判断障碍物决策，并给障碍物打躲避方向标签
    for (int i = 0; i < obstacle_list->obstacles.size(); i++) 
    {
        if(obstacle_list->obstacles[i].is_dynamic_obs) //动态障碍物不考虑
        {
            continue;
        }
        if (obstacle_list->obstacles[i].min_s > DP_path_sl.s.tail(1)(0) || obstacle_list->obstacles[i].max_s < DP_path_sl.s(0)) //前后不考虑
        {
            continue;
        }
        int index_min = 0;
        int index_middle = 0;
        int index_max = 0;

        for (int j = 0; j < DP_path_sl.s.size(); j++)
        {
            if (obstacle_list->obstacles[i].min_s <= DP_path_sl.s(j))
            {
                index_min = j;
    
                break;
            }
            if(j==DP_path_sl.s.size()-1)
            {
                index_min=DP_path_sl.s.size()-1;
            }
        }
        for (int k = index_min; k < DP_path_sl.s.size(); k++)
        {
            if (obstacle_list->obstacles[i].s <= DP_path_sl.s(k))
            {
                index_middle = k;
                break;
            }
            if(k==DP_path_sl.s.size()-1)
            {
                index_middle=DP_path_sl.s.size()-1;
            }
        }
        for (int k = index_middle; k < DP_path_sl.s.size(); k++)
        {
            // cout<<"max_s:"<<obstacle_list->obstacles[i].max_s<<","<<DP_path_sl.s(k)<<endl;
            if (obstacle_list->obstacles[i].max_s <= DP_path_sl.s(k))
            {
                index_max = k;
                // cout<<"找到index_max"<<endl;
                break;
            }
            if(k==DP_path_sl.s.size()-1)
            {
                index_max=DP_path_sl.s.size()-1;
            }
        }
        obstacle_list->obstacles[i].index_s_min = index_min;
        obstacle_list->obstacles[i].index_s_max = index_max;
        
        if (DP_path_sl.l(index_middle) >= obstacle_list->obstacles[i].l) // 决策向左侧绕
        {
            obstacle_list->obstacles[i].is_left_avoid = true;
        }
        else
        {
            obstacle_list->obstacles[i].is_left_avoid = false;
        }
        // cout<<i<<" index_min_mid_max:"<<index_min<<" ,"<<index_middle<<" ,"<<index_max<<endl;
    }
    
    //计算qp路径的边界值
    for (int i = 0; i < obstacle_list->obstacles.size(); i++) 
    {
        if(obstacle_list->obstacles[i].is_dynamic_obs)  //动态障碍物不考虑
        {
            continue;
        }
        if (obstacle_list->obstacles[i].min_s > DP_path_sl.s.tail(1)(0) || obstacle_list->obstacles[i].max_s  < DP_path_sl.s(0))
        {
            continue;
        }
        if (obstacle_list->obstacles[i].is_left_avoid) // 决策向左侧绕
        {
            float distance;
            if(obstacle_list->obstacles[i].isLidarObs)
            {
                                // cout<<"lidar ";
                if(obstacle_list->obstacles[i].isMapObs)
                {
                    distance = obstacle_list->obstacles[i].max_l + car_width / 2 + safe_distance_wall;
                    // cout<<"wall"<<endl;
                }
                else
                {
                    distance = obstacle_list->obstacles[i].max_l + car_width / 2 + safe_distance;
                    // cout<<"not_wall"<<endl;
                }
            }
            else
            {
                // cout<<"camer ";
                distance = obstacle_list->obstacles[i].max_l + car_width / 2 + safe_distance;
            }
            // cout<<" 决策向左！max_l:"<<obstacle_list->obstacles[i].max_l<<" distance:"<<distance<<" index改："<<obstacle_list->obstacles[i].index_s_min<<","<<obstacle_list->obstacles[i].index_s_max<<endl;
            for (int j = obstacle_list->obstacles[i].index_s_min; j <= obstacle_list->obstacles[i].index_s_max; j++)
            {
                // cout<<"左侧distance="<<distance<<endl;
                if (obstacle_list->obstacles[i].max_l > l_min(j) && ((distance) < l_max(j)))
                {

                    l_min(j) = distance;
                }

            }
            
        }
        else // 决策向右侧绕
        {
            float distance;
            if(obstacle_list->obstacles[i].isLidarObs)
            {
                if (obstacle_list->obstacles[i].isMapObs)
                {
                    distance = obstacle_list->obstacles[i].min_l - car_width / 2-safe_distance_wall;
                    // cout<<"wall"<<endl;
                }
                else
                {
                    distance = obstacle_list->obstacles[i].min_l - car_width / 2-safe_distance;
                    // cout<<"not_wall"<<endl;
                }
            }
            else
            {
                // cout<<"camer ";
                distance = obstacle_list->obstacles[i].min_l - car_width / 2 - safe_distance;
            }
            // cout<<" 决策向右！min_l:"<<obstacle_list->obstacles[i].min_l<<" distance:"<<distance<<" index改："<<obstacle_list->obstacles[i].index_s_min<<","<<obstacle_list->obstacles[i].index_s_max<<endl;
            for (int j = obstacle_list->obstacles[i].index_s_min; j <= obstacle_list->obstacles[i].index_s_max; j++)
            {
                // cout<<"右侧distance="<<distance<<endl;
                
                if (obstacle_list->obstacles[i].min_l < l_max(j) && (distance > l_min(j)))
                {
                    l_max(j) = distance;
                }

            }
            
        }
    }

    if(start_l<l_min(0))
    {
        cout<<"车起始点在危险区域内！！"<<endl;
        l_min(0)=start_l-0.01;
        for (int i = 1; i < l_min.size(); i++)
        {
            float l=l_min(0)+(i-2)*delta_s*0.8;
            if(l_min(i)>l)
            {
                l_min(i)=l;
                cout<<i<<" min_l改变:"<<l_min(i)<<endl;
            }
            else
            {
                break;
            }
        }
    }
    else if(start_l>l_max(0))
    {
        cout<<"车起始点在危险区域内！！"<<endl;
        l_max(0)=start_l+0.01;
        for (int i = 1; i < l_max.size(); i++)
        {
            float l=l_max(0)-(i-2)*delta_s*0.8;
            if(l_max(i)<l)
            {
                l_max(i)=l;
            }
            else
            {
                break;
            }
        }
    }
    // cout<<"start_l:"<<start_l<<" d_s:"<<delta_s<<endl;
    // for (int i = 0; i < l_max.size(); i++)
    // {
    //     cout<<l_min(i)<<"< x <"<<l_max(i)<<endl;
    // }
    // cout<<"车起始点在危险区域内！！"<<endl;

    // max_watch = l_max;
    // min_watch = l_min;
}

/**
 * @description: 参考线平滑算法,其中可以设置x_lb、x_ub、y_ub、y_lb、w设置参数，分别为xy离原始路径的最大误差，w为各个优化权重
 * @param {Ptr} line_record_ 待平滑原始路线
 * @return {*} 求解出的路线
 */
Frenet_path_points EMPlanner::cacl_qp_path(float plan_start_s, float plan_start_l, float plan_start_dl, float plan_start_ddl)
{
    double min_l = -4.5;
    double max_l = 4.5;
    float max_dl = 1.2;
    float max_ddl = 1.4;

    int n = static_cast<int>(std::floor(col_node_num * sample_s * sample_s_per_meters)); //6*1.2*20=144
    float delta_s = sample_s*col_node_num/n;

    Eigen::SparseVector<double> qp_path_l(n);
    Eigen::SparseVector<double> qp_path_dl(n);
    Eigen::SparseVector<double> qp_path_ddl(n);
    Eigen::SparseMatrix<double> A(5 * n , 3 * n);
    Eigen::SparseMatrix<double> H(3 * n, 3 * n);
    Eigen::SparseMatrix<double> Aeq_sub(2, 6);
    Eigen::SparseMatrix<double> A_sub(3, 3);

    Eigen::VectorXd f = Eigen::VectorXd::Zero(3 * n, 1);
    Eigen::VectorXd lb = Eigen::VectorXd::Zero(5 * n , 1);
    Eigen::VectorXd ub = Eigen::VectorXd::Zero(5 * n , 1);
    Eigen::VectorXd l_min = Eigen::VectorXd::Constant(n, min_l);
    Eigen::VectorXd l_max = Eigen::VectorXd::Constant(n, max_l);
    calculatePathBoundary(obstacle_list, DP_path_sl, l_min, l_max, min_l, max_l,delta_s,plan_start_l); // 根据障碍物来重新计算边界
    //  cout<<l_max<<endl;
    Aeq_sub.reserve(9); // 预分配非零元素空间
    A_sub.reserve(3);
    A.reserve(12 * n - 6);
    H.reserve(3 * n);

    // 设置 Aeq_sub 稀疏矩阵的值
    Aeq_sub.insert(0, 0) = 1;
    Aeq_sub.insert(0, 1) = delta_s;
    Aeq_sub.insert(0, 2) = pow(delta_s, 2) / 3;
    Aeq_sub.insert(0, 3) = -1;
    Aeq_sub.insert(0, 5) = pow(delta_s, 2) / 6;
    Aeq_sub.insert(1, 1) = 1;
    Aeq_sub.insert(1, 2) = delta_s / 2;
    Aeq_sub.insert(1, 4) = -1;
    Aeq_sub.insert(1, 5) = delta_s / 2;

    // 设置 A_sub 稀疏矩阵的值
    A_sub.insert(0, 0) = 1;
    A_sub.insert(1, 1) = 1;
    A_sub.insert(2, 2) = 1;

    // 设置 A 稀疏矩阵的值
    for (int i = 0; i < n; i++)
    {
        A.insert(3 * i, 3 * i) = A_sub.coeff(0, 0);
        A.insert(3 * i + 1, 3 * i + 1) = A_sub.coeff(1, 1);
        A.insert(3 * i + 2, 3 * i + 2) = A_sub.coeff(2, 2);
    }
    for (int i = 0; i < n - 1; i++)
    {
        A.insert(3 * n + 2 * i, 3 * i) = Aeq_sub.coeff(0, 0);
        A.insert(3 * n + 2 * i, 3 * i + 1) = Aeq_sub.coeff(0, 1);
        A.insert(3 * n + 2 * i, 3 * i + 2) = Aeq_sub.coeff(0, 2);
        A.insert(3 * n + 2 * i, 3 * i + 3) = Aeq_sub.coeff(0, 3);
        A.insert(3 * n + 2 * i, 3 * i + 5) = Aeq_sub.coeff(0, 5);
        A.insert(3 * n + 2 * i + 1, 3 * i + 1) = Aeq_sub.coeff(1, 1);
        A.insert(3 * n + 2 * i + 1, 3 * i + 2) = Aeq_sub.coeff(1, 2);
        A.insert(3 * n + 2 * i + 1, 3 * i + 4) = Aeq_sub.coeff(1, 4);
        A.insert(3 * n + 2 * i + 1, 3 * i + 5) = Aeq_sub.coeff(1, 5);
    }
    A.insert(5 * n - 2, 0) = 1;
    A.insert(5 * n - 1, 1) = 1;
    // A.insert(5 * n, 2) = 1;

    // 设置 H 稀疏矩阵的值
    for (int i = 0; i < n; i++)
    {
        H.insert(3 * i, 3 * i) = 2 * (w_qp_l + w_qp_ref_dp);
        H.insert(3 * i + 1, 3 * i + 1) = 2 * (w_qp_dl);
        H.insert(3 * i + 2, 3 * i + 2) = 2 * (w_qp_ddl);
        lb(3 * i) = l_min(i);
        lb(3 * i + 1) = -max_dl + plan_start_dl;
        lb(3 * i + 2) = -max_ddl;
        ub(3 * i) = l_max(i);
        ub(3 * i + 1) = max_dl + plan_start_dl;
        ub(3 * i + 2) = max_ddl;
        f(3 * i) = -2 * DP_path_sl.l(i) * w_qp_ref_dp;
    }
    lb(5 * n - 2) = plan_start_l;
    ub(5 * n - 2) = plan_start_l;
    lb(5 * n - 1) = plan_start_dl;
    ub(5 * n - 1) = plan_start_dl;
    // lb(5 * n) = plan_start_ddl;
    // ub(5 * n) = plan_start_ddl;

    A.makeCompressed();
    H.makeCompressed();

    // // osqp求解
    int NumberOfVariables = 3 * n;       // A矩阵的列数
    int NumberOfConstraints = 5 * n ; // A矩阵的行数
    // cout << "Path optimization progress --25% " << endl;
    // // 求解部分
    OsqpEigen::Solver solver;

    // // settings
    solver.settings()->setVerbosity(false); // 求解器信息输出控制
    solver.settings()->setWarmStart(true);  // 启用热启动
    // solver.settings()->setInitialGuessX(f); // 设置初始解向量,加速收敛

    // set the initial data of the QP solver
    // 矩阵A为m*n矩阵
    solver.data()->setNumberOfVariables(NumberOfVariables);     // 设置A矩阵的列数，即n
    solver.data()->setNumberOfConstraints(NumberOfConstraints); // 设置A矩阵的行数，即m

    if (!solver.data()->setHessianMatrix(H))
        // return 1; //设置P矩阵
        cout << "error1" << endl;
    if (!solver.data()->setGradient(f))
        // return 1; //设置q or f矩阵。当没有时设置为全0向量
        cout << "error2" << endl;
    if (!solver.data()->setLinearConstraintsMatrix(A))
        // return 1; //设置线性约束的A矩阵
        cout << "error3" << endl;
    if (!solver.data()->setLowerBound(lb))
    { // return 1; //设置下边界
        cout << "error4" << endl;
    }
    if (!solver.data()->setUpperBound(ub))
    { // return 1; //设置上边界
        cout << "error5" << endl;
    }

    // instantiate the solver
    if (!solver.initSolver())
        // return 1;
        cout << "error6" << endl;
    Eigen::VectorXd QPSolution = Eigen::VectorXd::Zero(3 * n);

    // solve the QP problem

    if (!solver.solve())
    {
        QpPathRunningNormally = false;
        cout << "error_slove" << endl;
    }
    else
    {
        QpPathRunningNormally = true;
    }
    // get the controller input
    // clock_t time_start = clock();
    // clock_t time_end = clock();
    // time_start = clock();
    QPSolution = solver.getSolution();
    Eigen::VectorXd s_ = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd l_ = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd dl_ = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd ddl_ = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd dddl_ = Eigen::VectorXd::Zero(n);

    for (int i = 0; i < n; i++)
    {
        s_(i) = plan_start_s + i * delta_s;
        l_(i) = QPSolution(3 * i);
        dl_(i) = QPSolution(3 * i + 1);
        ddl_(i) = QPSolution(3 * i + 2);
    }
    // cout<<"dl_: "<<dl_<<endl;
    // cout<<"ddl_: "<<ddl_<<endl;

    Frenet_path_points qp_path;
    qp_path.s = s_;
    qp_path.l = l_;
    qp_path.dl = dl_;
    qp_path.ddl = ddl_;

    return qp_path;
}

/**
 * @description: 参考线平滑算法,其中可以设置x_lb、x_ub、y_ub、y_lb、w设置参数，分别为xy离原始路径的最大误差，w为各个优化权重
 * @param {Ptr} line_record_ 待平滑原始路线
 * @return {*} 求解出的路线
 */
Eigen::VectorXd EMPlanner::Smooth_Reference_Line(pcl::PointCloud<pcl::PointXYZI>::Ptr line_record_)
{
    float x_lb = -0.06; // x限制值
    float x_ub = 0.06;
    float y_ub = 0.06;
    float y_lb = -0.06;

    // float w_smooth = 5;
    // float w_length = 2;
    // float w_ref = 0.000001;
    float w_smooth = 2000;
    float w_length = 2000;
    float w_ref = 0.1;
    int n_total = line_record_->points.size();
    cout<<"num:"<<n_total<<endl;
    Eigen::VectorXd referenceline_x = Eigen::VectorXd::Zero(n_total);
    Eigen::VectorXd referenceline_y = Eigen::VectorXd::Zero(n_total);
    for (int i = 0; i < line_record_->points.size(); i++)
    {
        referenceline_x(i) = line_record_->points[i].x;
        referenceline_y(i) = line_record_->points[i].y;
    }
    cout << "Path optimization progress --10% " << endl;
    Eigen::SparseMatrix<double> A1(2 * n_total - 4, 2 * n_total);
    Eigen::SparseMatrix<double> A2(2 * n_total - 2, 2 * n_total);
    // 创建稀疏矩阵对象
    Eigen::SparseMatrix<double> A3(2 * n_total, 2 * n_total);
    // 遍历对角线上的元素，设置为1
    for (int i = 0; i < 2 * n_total; ++i)
    {
        A3.insert(i, i) = 1.0; // 在(i, i)位置插入1.0
    }
    // 将所有元素插入矩阵中
    A3.makeCompressed();
    Eigen::MatrixXd A_cons = Eigen::MatrixXd::Identity(2 * n_total, 2 * n_total);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2 * n_total, 2 * n_total);
    Eigen::VectorXd f = Eigen::VectorXd::Zero(2 * n_total);

    Eigen::VectorXd lb = Eigen::VectorXd::Zero(2 * n_total);
    Eigen::VectorXd ub = Eigen::VectorXd::Zero(2 * n_total);
    // 限制条件赋值
    for (int i = 0; i < n_total; i++)
    {
        f(2 * i) = referenceline_x(i);
        f(2 * i + 1) = referenceline_y(i);
        lb(2 * i) = f(2 * i) + x_lb;
        ub(2 * i) = f(2 * i) + x_ub;
        lb(2 * i + 1) = f(2 * i + 1) + y_lb;
        ub(2 * i + 1) = f(2 * i + 1) + y_ub;
    }
    // A1赋值
    for (int i = 0; i < 2 * n_total - 4; i++)
    {
        A1.coeffRef(i, i) = 1;
        A1.coeffRef(i, i + 2) = -2;
        A1.coeffRef(i, i + 4) = 1;
    }
    // A2赋值
    for (int i = 0; i < 2 * n_total - 2; i++)
    {
        A2.coeffRef(i, i) = 1;
        A2.coeffRef(i, i + 2) = -1;
    }
    H = 2 * (w_smooth * A1.transpose() * A1 + w_length * A2.transpose() * A2 + w_ref * A3);
    f = -2 * w_ref * f;
    // osqp求解

    int NumberOfVariables = 2 * n_total;   // A矩阵的列数
    int NumberOfConstraints = 2 * n_total; // A矩阵的行数
    cout << "Path optimization progress --25% " << endl;
    // 求解部分
    OsqpEigen::Solver solver;
    Eigen::SparseMatrix<double> H_osqp = H.sparseView(); // 密集矩阵转换为稀疏矩阵
    H_osqp.makeCompressed();                             // 压缩稀疏行 (CSR) 格式
    H_osqp.reserve(H.nonZeros());                        // 预分配非零元素数量
    cout << "Path optimization progress --50% " << endl;
    Eigen::SparseMatrix<double> linearMatrix = A_cons.sparseView();
    linearMatrix.makeCompressed();                 // 压缩稀疏行 (CSR) 格式
    linearMatrix.reserve(linearMatrix.nonZeros()); // 预分配非零元素数量
    // lb_osqp.setConstant(-OsqpEigen::INFTY);
    // ub_osqp.setConstant(+OsqpEigen::INFTY);
    // settings
    solver.settings()->setVerbosity(false); // 求解器信息输出控制
    // solver.settings()->setWarmStart(true); // 启用热启动
    // solver.settings()->setInitialGuessX(f); // 设置初始解向量,加速收敛
    // solver.settings()->setWarmStart(true);

    // set the initial data of the QP solver
    // 矩阵A为m*n矩阵
    solver.data()->setNumberOfVariables(NumberOfVariables);     // 设置A矩阵的列数，即n
    solver.data()->setNumberOfConstraints(NumberOfConstraints); // 设置A矩阵的行数，即m

    if (!solver.data()->setHessianMatrix(H_osqp))
        // return 1; //设置P矩阵
        cout << "error1" << endl;
    if (!solver.data()->setGradient(f))
        // return 1; //设置q or f矩阵。当没有时设置为全0向量
        cout << "error2" << endl;
    if (!solver.data()->setLinearConstraintsMatrix(linearMatrix))
        // return 1; //设置线性约束的A矩阵
        cout << "error3" << endl;
    if (!solver.data()->setLowerBound(lb))
    { // return 1; //设置下边界
        cout << "error4" << endl;
    }
    if (!solver.data()->setUpperBound(ub))
    { // return 1; //设置上边界
        cout << "error5" << endl;
    }

    // instantiate the solver
    if (!solver.initSolver())
        // return 1;
        cout << "error6" << endl;
    Eigen::VectorXd QPSolution;

    // solve the QP problem

    cout << "Path optimization progress --75% " << endl;
    if (!solver.solve())
    {
        cout << "error_slove" << endl;
    }
    // get the controller input
    // clock_t time_start = clock();
    // clock_t time_end = clock();
    // time_start = clock();
    QPSolution = solver.getSolution();
    cout << "Path optimization progress --100% " << endl;
    cout << "Path Optimization Successful!" << endl;
    return QPSolution;
}

// /**
//  * @description: 此函数在Cartesian坐标系下找到投影点
//  * @param {car_path} &path 全局路径
//  * @param {int} index_match_point  匹配点序号
//  * @param {Vector2d} host_point 自车坐标
//  * @return {*}
//  */
// planning_msgs::path_point EMPlanner::find_projected_point_Cartesian(planning_msgs::car_path &path, int index_match_point, planning_msgs::path_point host_point)
// {
//     planning_msgs::path_point projected_point;

//     planning_msgs::path_point match_point = path.points[index_match_point];
//     Eigen::Vector2d projected_xy;
//     Eigen::Vector2d match_tor(match_point.tor.x, match_point.tor.y);
//     Eigen::Vector2d match_xy(match_point.x, match_point.y);
//     float ds = host_point.absolute_s - match_point.absolute_s;
//     projected_xy = match_xy + ds * match_tor;
//     projected_point.x = projected_xy(0);
//     projected_point.y = projected_xy(1);

//     return projected_point;
// }
/**
 * @description: 由绝对s找到匹配点match_points
 * @param {car_path} path 全局路径
 * @param {path_point} point_ 目标点
 * @param {int} star_index 开始序号，默认为0
 * @return {*} 匹配点序号
 */
int EMPlanner::index2s(planning_msgs::car_path &path, planning_msgs::path_point point_, int pre_index2s_id)
{
    int star_ = pre_index2s_id;

    for (int i = star_; i < path.points.size() - 1; i++)
    {
        // cout<<"i:"<<i<<" point_.absolute_s:"<<point_.absolute_s<<" path.points[i + 1].absolute_s"<<path.points[i + 1].absolute_s<<" path.points[i].absolute_s"<<path.points[i].absolute_s <<endl;
        if (path.points[i + 1].absolute_s > point_.absolute_s && path.points[i].absolute_s <= point_.absolute_s)
        {
            return i;
        }
    }

    // cout << "到达终点！！！若无到达则为寻找匹配点失败！" << endl;
    return 0;
}

int EMPlanner::index2s_r(planning_msgs::car_path &path, planning_msgs::path_point point_, int pre_index2s_id)
{
    int star_ = pre_index2s_id;

    for (int i = star_; i > 0; i--)
    {
        // cout<<"i:"<<i<<" point_.absolute_s:"<<point_.absolute_s<<" path.points[i + 1].absolute_s"<<path.points[i + 1].absolute_s<<" path.points[i].absolute_s"<<path.points[i].absolute_s <<endl;
        if (path.points[i].absolute_s <= point_.absolute_s)
        {
            return i;
        }
    }

    // cout << "到达终点！！！若无到达则为寻找匹配点失败！" << endl;
    return 0;
}

int EMPlanner::index2s(planning_msgs::car_path &path, float absolute_s, int pre_index2s_id)
{
    int star_ = pre_index2s_id;

    for (int i = star_; i < path.points.size() - 1; i++)
    {
        if (path.points[i + 1].absolute_s > absolute_s && path.points[i].absolute_s <= absolute_s)
        {
            return i;
        }
    }

    // cout << "到达终点！！！若无到达则为寻找匹配点失败！" << endl;
    return 0;
}

int EMPlanner::index2s(const Frenet_path_points &path, float host_s)
{

    for (int i = 0; i < path.s.size(); i++)
    {
        if (path.s[i] >= host_s)
        {
            return i;
        }
    }

    // cout << "到达终点！！！若无到达则为寻找匹配点失败！" << endl;
    return 0;
}

float EMPlanner::revise_angle(float angle)
{
    if (angle > 2 * M_PI)
    {
        angle -= 2 * M_PI;
    }
    else if (angle < -2 * M_PI)
    {
        angle += 2 * M_PI;
    }
    else
    {
        return angle;
    }
    return angle;
}

/**
 * @description: 找到最近点的匹配点序号
 * @param {float} x
 * @param {float} y
 * @param {car_path} path_ 路径点信息
 * @param {int} star 默认为0,开始从哪个点寻找   (加速)
 * @param {int} pre_index 默认为0,上个时刻的序号,若使用次加速，star直接置零 （加速）
 * @return {*} 序号点
 */
int EMPlanner::get_closest_point(float x, float y, planning_msgs::car_path &path_, int star, int pre_index)
{
    float min_distance = std::numeric_limits<float>::max();
    int index = star;
    int index_forward_point = 0;
    int index_back_point = 0;

    if (pre_index != 0)
    {
        index_forward_point = get_aim_point(4, path_, pre_index, true);
        index_back_point = get_aim_point(4, path_, pre_index, false);
        for (int i = index_back_point; i < index_forward_point; i++)
        {
            float dx = x - path_.points[i].x;
            float dy = y - path_.points[i].y;
            float distance = sqrt(dx * dx + dy * dy);
            if (distance < min_distance)
            {
                min_distance = distance;
                index = i;
            }
        }
        // 终点是否循环判断
        if (index == path_.points.size() - 1)
        {
            float dx = x - path_.points[0].x;
            float dy = y - path_.points[0].y;
            float distance = sqrt(dx * dx + dy * dy);
            if (distance < min_distance)
            {
                min_distance = distance;
                index = 0;
            }
        }
    }
    else if (star != 0)
    {
        index_back_point = get_aim_point(2, path_, star, false);
        for (int i = index_back_point; i < path_.points.size(); i++)
        {
            float dx = x - path_.points[i].x;
            float dy = y - path_.points[i].y;
            float distance = sqrt(dx * dx + dy * dy);
            if (distance < min_distance)
            {
                min_distance = distance;
                index = i;
            }
        }
        // 终点是否循环判断
        if (index == path_.points.size() - 1)
        {
            float dx = x - path_.points[0].x;
            float dy = y - path_.points[0].y;
            float distance = sqrt(dx * dx + dy * dy);
            if (distance < min_distance)
            {
                min_distance = distance;
                index = 0;
            }
        }
    }
    else
    {
        for (int i = 0; i < path_.points.size(); i++)
        {
            float dx = x - path_.points[i].x;
            float dy = y - path_.points[i].y;
            float distance = sqrt(dx * dx + dy * dy);
            if (distance < min_distance)
            {
                // cout<<"distance:"<<distance<<" i:"<<i<<endl;
                min_distance = distance;
                index = i;
            }
        }
    }

    return index;
}

// /**
//  * @description: 找到最近点的匹配点序号
//  * @param {float} x
//  * @param {float} y
//  * @param {car_path} path_ 路径点信息
//  * @param {int} star 默认为0,开始从哪个点寻找   (加速)
//  * @param {int} pre_index 默认为0,上个时刻的序号,若使用次加速，star直接置零 （加速）
//  * @return {*} 序号点
//  */
// int EMPlanner::get_closest_point(float x, float y, const vector<Eigen::Vector2d> &qp_path_xy)
// {
//     float min_distance = std::numeric_limits<float>::max();
//     int index = 0;

//     for (int i = 0; i < qp_path_xy.size(); i++)
//     {
//         float dx = x - qp_path_xy[i](0);
//         float dy = y - qp_path_xy[i](1);
//         float distance = sqrt(dx * dx + dy * dy);
//         if (distance < min_distance)
//         {
//             min_distance = distance;
//             index = i;
//         }
//     }

//     return index;
// }

/**
 * @description:  找到路线中，符合离目标点一定距离的点的下标,distance想要的距离点，begin从那个点开始遍历，sequence顺查，倒查
 * @param {float} distance 想要的距离
 * @param {car_path} path_ 路径点
 * @param {int} begin 从那个点开始遍历
 * @param {bool} sequence 顺查，倒查
 * @return {*}
 */
int EMPlanner::get_aim_point(float distance, planning_msgs::car_path &path_, int begin, bool sequence)
{
    float num_dis = 0;
    int index = 0;
    if (sequence)
    {
        for (int i = begin; i < path_.points.size() - 1; i++)
        {
            num_dis += sqrt(pow((path_.points[i + 1].x - path_.points[i].x), 2) + pow((path_.points[i + 1].y - path_.points[i].y), 2));
            if (num_dis >= distance)
            {
                index = i + 1;
                break;
            }
        }
        if (num_dis < distance)
        {
            index = path_.points.size() - 1;
        }
        return index;
    }
    else
    {
        for (int i = begin; i > 0; i--)
        {
            num_dis += sqrt(pow((path_.points[i].x - path_.points[i - 1].x), 2) + pow((path_.points[i].y - path_.points[i - 1].y), 2));
            if (num_dis >= distance)
            {
                index = i - 1;
                break;
            }
        }
        if (num_dis < distance)
        {
            index = 0;
        }
        return index;
    }
}

void EMPlanner::get_back_point_and_setVel(planning_msgs::car_path &path_, int begin)
{
    float num_dis = 0;
    int index = 0;
    float star_slow_num = 0;
    float end_slow_num = 0;

    for (int i = begin; i > 0; i--)
    {
        num_dis += sqrt(pow((path_.points[i].x - path_.points[i - 1].x), 2) + pow((path_.points[i].y - path_.points[i - 1].y), 2));
        if (num_dis <= end_slow_distance)
        {
            end_slow_num = i;
        }
        if (num_dis > star_slow_distance)
        {
            star_slow_num = i;
            break;
        }
    }

    for (int i = begin; i > 0; i--)
    {
        if (i >= end_slow_num)
        {
            path_.points[i].vel = V_turn;
            if (path_.points[begin].theta > 0)
            {
                path.points[i].flag_right_turn = 0;
                path.points[i].flag_left_turn = 1;
            }
            else
            {
                path.points[i].flag_right_turn = 1;
                path.points[i].flag_left_turn = 0;
            }
            // cout << "number_test:" << i << endl;
        }
        else if (i < end_slow_num && i >= star_slow_num)
        {
            path_.points[i].vel = V_turn + (V_straight - V_turn) / (end_slow_num - star_slow_num) * (end_slow_num - i);
            if (path_.points[begin].theta > 0)
            {
                path.points[i].flag_right_turn = 0;
                path.points[i].flag_left_turn = 1;
            }
            else
            {
                path.points[i].flag_right_turn = 1;
                path.points[i].flag_left_turn = 0;
            }
            // cout << "number_test:" << i << endl;
        }
        else
        {
            break;
        }
    }
    flag_V_slow = false;
}

// void EMPlanner::get_back_point_and_setVel2(planning_msgs::car_path &path_, int begin)
// {
//     float num_dis = 0;
//     int index = 0;
//     float star_slow_num = 0;
//     float end_slow_num = 0;

//     for (int i = begin; i > 0; i--)
//     {
//         num_dis += path_.points[i].ds;
//         if (num_dis >= end_slow_distance)
//         {
//             end_slow_num = i;
//             break;
//         }
//     }

//     for (int i = end_slow_num; i > 0; i--)
//     {
//         num_dis += path_.points[i].ds;
//         if (num_dis >= star_slow_distance)
//         {
//             star_slow_num = i;
//             break;
//         }
//     }

//     for (int i = begin; i > 0; i--)
//     {
//         if (i >= end_slow_num)
//         {
//             path_.points[i].vel = V_turn;
//             if (path_.points[begin].theta > 0)
//             {
//                 path.points[i].flag_right_turn = 0;
//                 path.points[i].flag_left_turn = 1;
//             }
//             else
//             {
//                 path.points[i].flag_right_turn = 1;
//                 path.points[i].flag_left_turn = 0;
//             }
//             // cout << "number_test:" << i << endl;
//         }
//         else if (i < end_slow_num && i >= star_slow_num)
//         {
//             path_.points[i].vel = V_turn + (V_straight - V_turn) / (end_slow_num - star_slow_num) * (end_slow_num - i);
//             if (path_.points[begin].theta > 0)
//             {
//                 path.points[i].flag_right_turn = 0;
//                 path.points[i].flag_left_turn = 1;
//             }
//             else
//             {
//                 path.points[i].flag_right_turn = 1;
//                 path.points[i].flag_left_turn = 0;
//             }
//             // cout << "number_test:" << i << endl;
//         }
//         else
//         {
//             break;
//         }
//     }
//     flag_V_slow = false;
// }

void EMPlanner::get_front_point_and_setVel(planning_msgs::car_path &path_, int begin)
{
    float num_dis = 0;
    float star_speed_num = 0;
    float end_speed_num = 0;
    int index = 0;

    for (int i = begin; i < path_.points.size() - 1; i++)
    {
        num_dis += sqrt(pow((path_.points[i].x - path_.points[i + 1].x), 2) + pow((path_.points[i].y - path_.points[i + 1].y), 2));
        if (num_dis <= star_speed_distance)
        {
            star_speed_num = i;
        }
        else if (num_dis > end_speed_distance)
        {
            end_speed_num = i;
            break;
        }
    }
    for (int i = begin; i < path_.points.size() - 1; i++)
    {
        if (i <= star_speed_num)
        {
            path_.points[i].vel = V_turn;
            // cout << "number_test:" << i << endl;
        }
        else if (i > star_speed_num && i <= end_speed_num)
        {
            path_.points[i].vel = V_turn + (V_straight - V_turn) / (end_speed_num - star_speed_num) * (i - star_speed_num);
        }
    }
}

Min_path_nodes EMPlanner::CalcDpPathNodeMinCost_reverse(float plan_star_s, float plan_star_l)
{
    Avoid_nodes avoid_nodes;
    Col_nodes col_node;
    avoid_nodes.nodes.reserve(col_node_num_reverse);
    col_node.row_nodes.reserve(row_node_num_reverse);
    for (int i = 0; i < row_node_num_reverse; i++) // 计算起点与第一列的cost，单独计算
    {
        Single_node single_node;
        float node_s = sample_s_reverse;
        float node_l = ((row_node_num_reverse - 1) / 2 - i) * sample_l_reverse;
        double cost = CalculateDpPathStarCost_reverse(plan_star_s, plan_star_l, 0, 0, node_s, node_l);
        single_node.node_s = node_s;
        single_node.node_l = node_l;
        single_node.toThisNodeMinCost = cost;      // 第一列最小cost就是起点到此点 ，不需要判断
        single_node.pre2this_min_cost_index = 0;   // 第一列最小cost序号就是起点 ，不需要判断
        col_node.row_nodes.push_back(single_node); // 将第一列节点加入
    }
    avoid_nodes.nodes.push_back(col_node); // 列加入总节点（未包含原点）

#pragma omp for
    for (int j = 1; j < col_node_num_reverse; j++) // 计算第一列与第最后一列的cost
    {
        Col_nodes col_nodes;
        col_nodes.row_nodes.reserve(row_node_num_reverse);
        for (int i = 0; i < row_node_num_reverse; i++) // 行
        {
            float node_s =  (j + 1) * sample_s_reverse;                // 该节点的s
            float node_l = ((row_node_num_reverse - 1) / 2 - i) * sample_l_reverse; // 无误
            double min_cost = std::numeric_limits<double>::max();
            int min_cost_index;
            Single_node sigle_node;
            sigle_node.node_s = node_s;
            sigle_node.node_l = node_l;
            for (int k = 0; k < row_node_num_reverse; k++) // 遍历上一列
            {
                if(abs((k-i)*sample_l_reverse/ sample_s_reverse)>1.2)  //提前将不可能的节点删除，提高计算速度
                {
                    continue;
                }
                double cost = CalculateDpPathForwardCost_reverse(avoid_nodes.nodes[j - 1].row_nodes[k].node_s, avoid_nodes.nodes[j - 1].row_nodes[k].node_l, node_s, node_l,plan_star_s);
                cost = cost + avoid_nodes.nodes[j - 1].row_nodes[k].toThisNodeMinCost;
                if (cost < min_cost)
                {
                    min_cost = cost;
                    min_cost_index = k;
                }
            }
            sigle_node.toThisNodeMinCost = min_cost;
            sigle_node.pre2this_min_cost_index = min_cost_index;
            col_nodes.row_nodes.push_back(sigle_node);
            // cout<<"行:"<<i<<"，列："<<j<<" min_cost:"<<min_cost<<" min_cost_index:"<<min_cost_index<<endl;
        }

        avoid_nodes.nodes.push_back(col_nodes);
    }

    double path_min_cost = std::numeric_limits<double>::max();
    int min_cost_last_index;
    // 寻找全局最小代价路径
    for (int i = 0; i < row_node_num_reverse; i++)
    {
        if (avoid_nodes.nodes[col_node_num_reverse - 1].row_nodes[i].toThisNodeMinCost < path_min_cost)
        {
            path_min_cost = avoid_nodes.nodes[col_node_num_reverse - 1].row_nodes[i].toThisNodeMinCost;
            min_cost_last_index = i;
        }
    }

    // 从后向前遍历
    vector<int> min_cost_index;
    min_cost_index.push_back(min_cost_last_index);     // 最优路径最后一列的序号
    for (int i = col_node_num_reverse - 1; i > 0; i--) // 倒序输入
    {
        min_cost_index.push_back(avoid_nodes.nodes[i].row_nodes[min_cost_last_index].pre2this_min_cost_index);
        min_cost_last_index = avoid_nodes.nodes[i].row_nodes[min_cost_last_index].pre2this_min_cost_index;
    }
    std::reverse(min_cost_index.begin(), min_cost_index.end()); // 序号反转，改为正序
    avoid_nodes.min_cost_path_indexs = min_cost_index;

    // for (int i = 0; i < min_cost_index.size(); i++)
    // {
    //     cout<<i<<" node_s:"<<avoid_nodes.nodes[i].row_nodes[min_cost_index[i]].node_s<<" node_l"<<avoid_nodes.nodes[i].row_nodes[min_cost_index[i]].node_l<<endl;;
    // }
    Single_node min_path_node_star;
    Min_path_nodes min_path_nodes;

    min_path_node_star.node_s = 0; // 加入起点
    min_path_node_star.node_l = plan_star_l;
    min_path_nodes.nodes.push_back(min_path_node_star);

    for (int i = 0; i < min_cost_index.size(); i++)
    {
        Single_node min_path_node;
        min_path_node.node_s = (avoid_nodes.nodes[i].row_nodes[min_cost_index[i]].node_s);
        min_path_node.node_l = (avoid_nodes.nodes[i].row_nodes[min_cost_index[i]].node_l);
        // cout<<"min_path_node_s:"<<min_path_node.node_s<<" l:"<<min_path_node.node_l<<endl;
        min_path_nodes.nodes.push_back(min_path_node);
    }
    return min_path_nodes;
}

Frenet_path_points EMPlanner::calcQpPath_reverse(float plan_start_s, float plan_start_l, float plan_start_dl, float plan_start_ddl)
{

    double min_l = -3.5;
    double max_l = 3.5;
    float max_dl = 1.5;
    float max_ddl = 1.5;
    plan_start_dl=0;
    int n = static_cast<int>(std::floor(col_node_num_reverse * abs(sample_s_reverse) * sample_s_per_meters)); // 6*1.2*20=144
    float delta_s = sample_s_reverse * col_node_num / n;                                              // 这里暂时不清楚-s是否正确！？
    Eigen::SparseVector<double> qp_path_l(n);
    Eigen::SparseVector<double> qp_path_dl(n);
    Eigen::SparseVector<double> qp_path_ddl(n);
    Eigen::SparseMatrix<double> A(5 * n + 1, 3 * n);
    Eigen::SparseMatrix<double> H(3 * n, 3 * n);
    Eigen::SparseMatrix<double> Aeq_sub(2, 6);
    Eigen::SparseMatrix<double> A_sub(3, 3);

    Eigen::VectorXd f = Eigen::VectorXd::Zero(3 * n, 1);
    Eigen::VectorXd lb = Eigen::VectorXd::Zero(5 * n + 1, 1);
    Eigen::VectorXd ub = Eigen::VectorXd::Zero(5 * n + 1, 1);
    Eigen::VectorXd l_min = Eigen::VectorXd::Constant(n, min_l);
    Eigen::VectorXd l_max = Eigen::VectorXd::Constant(n, max_l);
    
    calculatePathBoundary(obstacle_list, DP_path_sl_r, l_min, l_max, min_l, max_l,delta_s,plan_start_l); // 根据障碍物来重新计算边界
   
    Aeq_sub.reserve(9); // 预分配非零元素空间
    A_sub.reserve(3);
    A.reserve(12 * n - 6);
    H.reserve(3 * n);

    // 设置 Aeq_sub 稀疏矩阵的值
    Aeq_sub.insert(0, 0) = 1;
    Aeq_sub.insert(0, 1) = delta_s;
    Aeq_sub.insert(0, 2) = pow(delta_s, 2) / 3;
    Aeq_sub.insert(0, 3) = -1;
    Aeq_sub.insert(0, 5) = pow(delta_s, 2) / 6;
    Aeq_sub.insert(1, 1) = 1;
    Aeq_sub.insert(1, 2) = delta_s / 2;
    Aeq_sub.insert(1, 4) = -1;
    Aeq_sub.insert(1, 5) = delta_s / 2;

    // 设置 A_sub 稀疏矩阵的值
    A_sub.insert(0, 0) = 1;
    A_sub.insert(1, 1) = 1;
    A_sub.insert(2, 2) = 1;
    // 设置 A 稀疏矩阵的值
    for (int i = 0; i < n; i++)
    {
        A.insert(3 * i, 3 * i) = A_sub.coeff(0, 0);
        A.insert(3 * i + 1, 3 * i + 1) = A_sub.coeff(1, 1);
        A.insert(3 * i + 2, 3 * i + 2) = A_sub.coeff(2, 2);
    }
    for (int i = 0; i < n - 1; i++)
    {
        A.insert(3 * n + 2 * i, 3 * i) = Aeq_sub.coeff(0, 0);
        A.insert(3 * n + 2 * i, 3 * i + 1) = Aeq_sub.coeff(0, 1);
        A.insert(3 * n + 2 * i, 3 * i + 2) = Aeq_sub.coeff(0, 2);
        A.insert(3 * n + 2 * i, 3 * i + 3) = Aeq_sub.coeff(0, 3);
        A.insert(3 * n + 2 * i, 3 * i + 5) = Aeq_sub.coeff(0, 5);
        A.insert(3 * n + 2 * i + 1, 3 * i + 1) = Aeq_sub.coeff(1, 1);
        A.insert(3 * n + 2 * i + 1, 3 * i + 2) = Aeq_sub.coeff(1, 2);
        A.insert(3 * n + 2 * i + 1, 3 * i + 4) = Aeq_sub.coeff(1, 4);
        A.insert(3 * n + 2 * i + 1, 3 * i + 5) = Aeq_sub.coeff(1, 5);
    }
    A.insert(5 * n - 2, 0) = 1;
    A.insert(5 * n - 1, 1) = 1;
    A.insert(5 * n, 2) = 1;

    // 设置 H 稀疏矩阵的值
    for (int i = 0; i < n; i++)
    {

        H.insert(3 * i, 3 * i) = 2 * (w_qp_l + w_qp_ref_dp);
        
        H.insert(3 * i + 1, 3 * i + 1) = 2 * (w_qp_dl);
        H.insert(3 * i + 2, 3 * i + 2) = 2 * (w_qp_ddl);
        lb(3 * i) = l_min(i);
        lb(3 * i + 1) = -max_dl + plan_start_dl;
        lb(3 * i + 2) = -max_ddl;
        
        ub(3 * i) = l_max(i);
        
        ub(3 * i + 1) = max_dl + plan_start_dl;
        
        ub(3 * i + 2) = max_ddl;
        
        f(3 * i) = -2 * DP_path_sl.l(i) * w_qp_ref_dp;

    }
   
    lb(5 * n - 2) = plan_start_l;
    ub(5 * n - 2) = plan_start_l;
    lb(5 * n - 1) = plan_start_dl;
    ub(5 * n - 1) = plan_start_dl;
    lb(5 * n) = plan_start_ddl;
    ub(5 * n) = plan_start_ddl;
    
    A.makeCompressed();
    H.makeCompressed();

    // // osqp求解
    int NumberOfVariables = 3 * n;       // A矩阵的列数
    int NumberOfConstraints = 5 * n + 1; // A矩阵的行数
    // cout << "Path optimization progress --25% " << endl;
    // // 求解部分
    OsqpEigen::Solver solver;

    // // settings
    solver.settings()->setVerbosity(false); // 求解器信息输出控制
    solver.settings()->setWarmStart(true);  // 启用热启动
    // solver.settings()->setInitialGuessX(f); // 设置初始解向量,加速收敛

    // set the initial data of the QP solver
    // 矩阵A为m*n矩阵
    solver.data()->setNumberOfVariables(NumberOfVariables);     // 设置A矩阵的列数，即n
    solver.data()->setNumberOfConstraints(NumberOfConstraints); // 设置A矩阵的行数，即m

    if (!solver.data()->setHessianMatrix(H))
        // return 1; //设置P矩阵
        cout << "error1" << endl;
    if (!solver.data()->setGradient(f))
        // return 1; //设置q or f矩阵。当没有时设置为全0向量
        cout << "error2" << endl;
    if (!solver.data()->setLinearConstraintsMatrix(A))
        // return 1; //设置线性约束的A矩阵
        cout << "error3" << endl;
    if (!solver.data()->setLowerBound(lb))
    { // return 1; //设置下边界
        cout << "error4" << endl;
    }
    if (!solver.data()->setUpperBound(ub))
    { // return 1; //设置上边界
        cout << "error5" << endl;
    }

    // instantiate the solver
    if (!solver.initSolver())
        // return 1;
        cout << "error6" << endl;
    Eigen::VectorXd QPSolution = Eigen::VectorXd::Zero(3 * n);
    // solve the QP problem

    if (!solver.solve())
    {
        reverseQpPathRunningNormally = false;
        cout << "error_slove" << endl;
    }
    else
    {
        reverseQpPathRunningNormally = true;
    }
    // get the controller input
    // clock_t time_start = clock();
    // clock_t time_end = clock();
    // time_start = clock();
    QPSolution = solver.getSolution();
    Eigen::VectorXd s_ = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd l_ = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd dl_ = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd ddl_ = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd dddl_ = Eigen::VectorXd::Zero(n);

    for (int i = 0; i < n; i++)
    {
        s_(i) = plan_start_s + i * delta_s;
        l_(i) = QPSolution(3 * i);
        dl_(i) = QPSolution(3 * i + 1);
        ddl_(i) = QPSolution(3 * i + 2);
    }
    // cout<<"dl_: "<<dl_<<endl;
    // cout<<"ddl_: "<<ddl_<<endl;
 
    Frenet_path_points qp_path;
    qp_path.s = s_;
    qp_path.l = l_;
    qp_path.dl = dl_;
    qp_path.ddl = ddl_;

    return qp_path;
}

/**
 * @description: 通过优化后的路径来计算航向角
 * @param {VectorXd} QPSolution 优化后路径点
 * @param {vector<float>&} temp_phi 临时存储航向角变量
 * @return {*}
 */
void EMPlanner::Calculate_OptimizedPath_Heading_reverse(const vector<Eigen::Vector2d> &QPSolution, vector<float> &temp_phi, pcl::PointCloud<pcl::PointXYZI>::Ptr &line_qp_path)
{
    // 计算航向角(路线斜率) 隔着两个点计算
    for (int i = 0; i < QPSolution.size() - 6; i++)
    {
        //这里计算反向偏航角度
        float dif_x1 = QPSolution[i][0] - QPSolution[i + 3][0];
        float dif_x2 = QPSolution[i + 3][0] - QPSolution[i + 6][0];
        float dif_y1 = QPSolution[i][1] - QPSolution[i + 3][1];
        float dif_y2 = QPSolution[i + 3][1] - QPSolution[i + 6][1];
        float phi_1 = atan2(dif_y1, dif_x1);
        float phi_2 = atan2(dif_y2, dif_x2);
        if (abs(phi_1 - phi_2) > 3.14)
        {
            temp_phi[i + 3] = phi_2;
        }
        else if (phi_1 + phi_2 == 0)
        {
            temp_phi[i + 3] = temp_phi[i + 2];
        }
        else
        {
            temp_phi[i + 3] = (phi_1 + phi_2) / 2;
        }
    }

    temp_phi[0] = temp_phi[3];
    temp_phi[1] = temp_phi[3];
    temp_phi[2] = temp_phi[3];
    temp_phi[QPSolution.size() - 1] = temp_phi[QPSolution.size() - 4];
    temp_phi[QPSolution.size() - 2] = temp_phi[QPSolution.size() - 4];
    temp_phi[QPSolution.size() - 3] = temp_phi[QPSolution.size() - 4];

    for (int i = 1; i < temp_phi.size(); i++)
    {
        if (temp_phi[i] == 0)
            temp_phi[i] = temp_phi[i - 1];
    }

    line_qp_path.reset(new pcl::PointCloud<pcl::PointXYZI>);
    line_qp_path->points.reserve(QPSolution.size());

    for (int i = 0; i < QPSolution.size(); i++)
    {
        pcl::PointXYZI points_;
        points_.x = QPSolution[i][0];
        points_.y = QPSolution[i][1];
        points_.z = temp_phi[i];
        // cout<<i<<" xyz"<<points_.x<<" "<<points_.y<<" "<<points_.z<<endl;
        line_qp_path->push_back(points_);
    }
}


/**
 * @description: 增密min_path_nodes节点之间的点，相当于对s进行插值,每米20个s点
 * @param {Min_path_nodes} min_path_nodes
 * @return {*} 增密后的s点
 */
Frenet_path_points EMPlanner::InterpolateDpPathPoints_reverse(const Min_path_nodes &min_path_nodes,float host_start_s)
{
    Frenet_path_points frenet_path_points;
    int total_points_num = static_cast<int>(std::floor((min_path_nodes.nodes.size() - 1) * abs(sample_s_reverse) * sample_s_per_meters));
    Eigen::VectorXd ds_(total_points_num);
    Eigen::VectorXd l_(total_points_num);
    Eigen::VectorXd dl_(total_points_num);
    Eigen::VectorXd ddl_(total_points_num);
    Eigen::VectorXd dddl_(total_points_num);

    for (int i = 0; i < min_path_nodes.nodes.size() - 1; i++)
    {
        float start_l = min_path_nodes.nodes[i].node_l;
        float start_dl = 0;
        float start_ddl = 0;
        float end_l = min_path_nodes.nodes[i + 1].node_l;
        float end_dl = 0;
        float end_ddl = 0;
        float start_s = min_path_nodes.nodes[i].node_s;
        float end_s = min_path_nodes.nodes[i + 1].node_s;
        // cout<<i<<" start s l:"<<start_s<<" "<<start_l<<" ,end s l:"<<end_s<<" "<<end_l<<endl;
        // Eigen::VectorXd coeff = CalculateFiveDegreePolynomialCoefficients(start_l, start_dl, start_ddl, end_l, end_dl, end_ddl, start_s, end_s);
        Eigen::VectorXd coeff = CalculateThreeDegreePolynomialCoefficients(start_l, start_dl, start_ddl, end_l, end_dl, end_ddl, start_s, end_s);
        float a0 = coeff(0);
        float a1 = coeff(1);
        float a2 = coeff(2);
        float a3 = coeff(3);
        // float a4 = coeff(4);
        // float a5 = coeff(5);
        float points_num_float = abs(sample_s_reverse) * sample_s_per_meters;
        int points_num = static_cast<int>(std::floor(points_num_float)); // 每米20个点,这里丢掉了终点应该为sample_s * sample_s_per_meters+1

        Eigen::VectorXd I_ = Eigen::VectorXd::Ones(points_num);
        Eigen::VectorXd ds(points_num);
        Eigen::VectorXd l(points_num);
        Eigen::VectorXd dl(points_num);
        Eigen::VectorXd ddl(points_num);
        Eigen::VectorXd dddl(points_num);
        for (int j = 0; j < points_num; j++) // 0.1米一个点
        {
            ds(j) = start_s + j / (points_num_float)*sample_s_reverse;
        }
        // l = a0 * I_.array() + a1 * ds.array() + a2 * ds.array().pow(2) + a3 * ds.array().pow(3) + a4 * ds.array().pow(4) + a5 * ds.array().pow(5);
        // dl = a1 * I_.array() + 2 * a2 * ds.array() + 3 * a3 * ds.array().pow(2) + 4 * a4 * ds.array().pow(3) + 5 * a5 * ds.array().pow(4);
        // ddl = 2 * a2 * I_.array() + 6 * a3 * ds.array() + 12 * a4 * ds.array().pow(2) + 20 * a5 * ds.array().pow(3);
        // dddl = 6 * a3 * I_.array() + 24 * a4 * ds.array() + 60 * a5 * ds.array().pow(2);
        l = a0 * I_.array() + a1 * ds.array() + a2 * ds.array().pow(2) + a3 * ds.array().pow(3);
        dl = a1 * I_.array() + 2 * a2 * ds.array() + 3 * a3 * ds.array().pow(2);
        ddl = 2 * a2 * I_.array() + 6 * a3 * ds.array();
        dddl = 6 * a3 * I_.array();

        frenet_path_points.s = ds;

        ds_.segment(i * points_num, points_num) = ds.array()+host_start_s;
        l_.segment(i * points_num, points_num) = l;
        dl_.segment(i * points_num, points_num) = dl;
        ddl_.segment(i * points_num, points_num) = ddl;
        dddl_.segment(i * points_num, points_num) = dddl;
    }

    frenet_path_points.s = ds_;
    frenet_path_points.l = l_;
    frenet_path_points.dl = dl_;
    frenet_path_points.ddl = ddl_;
    frenet_path_points.dddl = dddl_;

    return frenet_path_points;
}

double EMPlanner::CalculateDpPathForwardCost_reverse(float pre_node_s, float pre_node_l, float current_node_s, float current_node_l,float host_start_s)
{
    // cout<<"CalculateForwardCost_star"<<endl;
    float start_l = pre_node_l;
    float start_dl = 0;
    float start_ddl = 0;
    float end_l = current_node_l;
    float end_dl = 0;
    float end_ddl = 0;
    float start_s = pre_node_s;
    float end_s = current_node_s;
    Eigen::VectorXd coeff = CalculateThreeDegreePolynomialCoefficients(start_l, start_dl, start_ddl, end_l, end_dl, end_ddl, start_s, end_s);

    float a0 = coeff(0);
    float a1 = coeff(1);
    float a2 = coeff(2);
    float a3 = coeff(3);
    // float a4 = coeff(4);
    // float a5 = coeff(5);

    int points_num = static_cast<int>(std::floor(abs(sample_s_reverse) * sample_s_num)); // 输出80
    Eigen::VectorXd I_ = Eigen::VectorXd::Ones(points_num);
    Eigen::VectorXd ds = Eigen::VectorXd::LinSpaced(points_num, start_s, start_s + sample_s_reverse); // 生成等间距的向量
    Eigen::VectorXd ds_pow2 = ds.array().pow(2);
    Eigen::VectorXd ds_pow3 = ds.array().pow(3);
    // Eigen::VectorXd ds_pow4 = ds.array().pow(4);
    // Eigen::VectorXd ds_pow5 = ds.array().pow(5);

    // Eigen::VectorXd l = a0 * I_ + a1 * ds + a2 * ds_pow2 + a3 * ds_pow3 + a4 * ds_pow4 + a5 * ds_pow5;
    // Eigen::VectorXd dl = a1 * I_ + 2 * a2 * ds + 3 * a3 * ds_pow2 + 4 * a4 * ds_pow3 + 5 * a5 * ds_pow4;
    // Eigen::VectorXd ddl = 2 * a2 * I_ + 6 * a3 * ds + 12 * a4 * ds_pow2 + 20 * a5 * ds_pow3;
    // Eigen::VectorXd dddl = 6 * a3 * I_ + 24 * a4 * ds + 60 * a5 * ds_pow2;

    Eigen::VectorXd l = a0 * I_ + a1 * ds + a2 * ds_pow2 + a3 * ds_pow3;
    Eigen::VectorXd dl = a1 * I_ + 2 * a2 * ds + 3 * a3 * ds_pow2;
    Eigen::VectorXd ddl = 2 * a2 * I_ + 6 * a3 * ds;
    Eigen::VectorXd dddl = 6 * a3 * I_;

    float cost_smooth = w_cost_smooth_dl_r * dl.squaredNorm() + w_cost_smooth_ddl_r * ddl.squaredNorm() + w_cost_smooth_dddl_r * dddl.squaredNorm();

    float cost_ref = w_cost_ref_r * l.squaredNorm();
    double cost_collision = 0;

    bool is_collsion=false;
    // for (int i = 0; i < obstacle_list->obstacles.size(); i++)
    // {
    //     cout<<"倒车obs："<< -(obstacle_list->obstacles[i].s-host_start_s)<<","<< -obstacle_list->obstacles[i].l<<endl;
    //     if (obstacle_list->obstacles[i].is_dynamic_obs)
    //     {
    //         continue;
    //     }
    //     for (int j = 0; j < points_num; j++) // 0.1米一个点
    //     {
    //         cost_collision += CalcObstacleCost(obstacle_list->obstacles[i], ds[j], l[j],host_start_s,is_collsion);
    //         if (is_collsion)
    //             break;
    //     }
    //     if (is_collsion)
    //         break;
    // }
    cost_collision = cost_collision * w_cost_collision;
    double cost_all = 0;
    cost_all = cost_collision + cost_ref + cost_smooth;
    // cout<<"cost_collision:"<<cost_collision<<" cost_ref:"<<cost_ref<<" cost_smooth:"<<cost_smooth<<endl;
    return cost_all;
}

double EMPlanner::CalculateDpPathStarCost_reverse(float begin_s, float begin_l, float begin_dl, float begin_ddl, float end_S, float end_L)
{
    clock_t time_start = clock();
    float start_l = begin_l;
    float start_dl = begin_dl;
    float start_ddl = begin_ddl;
    float end_l = end_L;
    float end_dl = 0;
    float end_ddl = 0;
    float start_s = 0;
    float end_s = end_S;
    Eigen::VectorXd coeff = CalculateThreeDegreePolynomialCoefficients(start_l, start_dl, start_ddl, end_l, end_dl, end_ddl, start_s, end_s);

    float a0 = coeff(0);
    float a1 = coeff(1);
    float a2 = coeff(2);
    float a3 = coeff(3);
    // float a4 = coeff(4);
    // float a5 = coeff(5);

    int points_num = static_cast<int>(std::floor(abs(sample_s_reverse) * sample_s_num)); // 输出80
    Eigen::VectorXd I_ = Eigen::VectorXd::Ones(points_num);
    Eigen::VectorXd ds = Eigen::VectorXd::LinSpaced(points_num, start_s, start_s + sample_s_reverse); // 生成等间距的向量
    Eigen::VectorXd ds_pow2 = ds.array().pow(2);
    Eigen::VectorXd ds_pow3 = ds.array().pow(3);
    // Eigen::VectorXd ds_pow4 = ds.array().pow(4);
    // Eigen::VectorXd ds_pow5 = ds.array().pow(5);

    // Eigen::VectorXd l = a0 * I_ + a1 * ds + a2 * ds_pow2 + a3 * ds_pow3 + a4 * ds_pow4 + a5 * ds_pow5;
    // Eigen::VectorXd dl = a1 * I_ + 2 * a2 * ds + 3 * a3 * ds_pow2 + 4 * a4 * ds_pow3 + 5 * a5 * ds_pow4;
    // Eigen::VectorXd ddl = 2 * a2 * I_ + 6 * a3 * ds + 12 * a4 * ds_pow2 + 20 * a5 * ds_pow3;
    // Eigen::VectorXd dddl = 6 * a3 * I_ + 24 * a4 * ds + 60 * a5 * ds_pow2;

    Eigen::VectorXd l = a0 * I_ + a1 * ds + a2 * ds_pow2 + a3 * ds_pow3;
    Eigen::VectorXd dl = a1 * I_ + 2 * a2 * ds + 3 * a3 * ds_pow2;
    Eigen::VectorXd ddl = 2 * a2 * I_ + 6 * a3 * ds;
    Eigen::VectorXd dddl = 6 * a3 * I_;

    float cost_smooth = w_cost_smooth_dl * dl.squaredNorm() + w_cost_smooth_ddl * ddl.squaredNorm() + w_cost_smooth_dddl * dddl.squaredNorm();
    float cost_ref = w_cost_ref * l.squaredNorm();
    double cost_collision = 0;
    bool is_collsion=false;
    for (int i = 0; i < obstacle_list->obstacles.size(); i++)
    {
        if (obstacle_list->obstacles[i].is_dynamic_obs)
        {
            continue;
        }
        for (int j = 0; j < points_num; j++) // 0.1米一个点
        {
            cost_collision += CalcObstacleCost(obstacle_list->obstacles[i], ds[j], l[j],begin_s,is_collsion);
            if (is_collsion)
                break;
            
        }
        if (is_collsion)
            break;
    }

    cost_collision = cost_collision * w_cost_collision;
    double cost_all = 0;
    cost_all = cost_collision + cost_ref + cost_smooth;
    // cout<<"start_cost_collision:"<<cost_collision<<" start_cost_ref:"<<cost_ref<<" start_cost_smooth:"<<cost_smooth<<endl;
    return cost_all;
}


/**
 * @description: 计算在frenet坐标系下所有局部路径的点转化为XY坐标系
 * @param {Frenet_path_points} frenet_path_points
 * @param {car_path} path
 * @return {*}
 */
vector<Eigen::Vector2d> EMPlanner::FrenetToXY_r(const Frenet_path_points &frenet_path_points, planning_msgs::car_path &path)
{
    // cout<<"FrenetToXY start"<<endl;
    vector<Eigen::Vector2d> points_XY;
    points_XY.reserve(frenet_path_points.s.size());
    int pre_index2s_id = index_middle_Car;
    for (int i = 0; i < frenet_path_points.s.size(); i++)
    {
        planning_msgs::path_point proj_point_;
        planning_msgs::path_point point_xy;

        proj_point_.absolute_s = frenet_path_points.s[i];
        // cout<<"proj_point_.absolute_s :"<<proj_point_.absolute_s <<endl;
        int index_match_points;
        index_match_points = index2s_r(path, proj_point_, pre_index2s_id);
        // cout<<"index_match_points:"<<index_match_points<<endl;
        proj_point_ = find_projected_point_Frenet(path, index_match_points, frenet_path_points.s[i], frenet_path_points.l[i]);
        Eigen::Vector2d match_point_xy(proj_point_.x, proj_point_.y);
        Eigen::Vector2d point_nor(proj_point_.nor.x, proj_point_.nor.y);   //法相量
        Eigen::Vector2d point_temp_xy = match_point_xy + frenet_path_points.l(i) * point_nor;
        // cout<<"match_point_xy:"<<match_point_xy(0)<<" "<<match_point_xy(1)<<";l:"<<frenet_path_points.l(i)<<" nor:"<<point_nor(0)<<", "<<point_nor(1)<<";real_points:"<<point_temp_xy(0)<<","<<point_temp_xy(1)<<endl;
        // cout<<"point_temp_xy:"<<point_temp_xy<<" ";
        pre_index2s_id = index_match_points;

        points_XY.push_back(point_temp_xy);
    }

    // std::cout << "FrenetToXY程序运行时间: " << time_diff << " 秒" << std::endl;
    return points_XY;
}
