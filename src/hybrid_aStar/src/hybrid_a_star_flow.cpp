#include "hybrid_a_star/hybrid_a_star_flow.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <cmath>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <planning_msgs/car_path.h>
#include <planning_msgs/car_info.h>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <unistd.h>
#include <ros/topic.h>

__attribute__((unused)) double Mod2Pi(const double &x) {  // 这是一个 GCC 和 Clang 编译器的扩展属性，这个函数可能不会被使用，防止编译器发出未使用函数的警告。
    double v = fmod(x, 2 * M_PI);

    if (v < -M_PI) {
        v += 2.0 * M_PI;
    } else if (v > M_PI) {
        v -= 2.0 * M_PI;
    }
    return v;
}

HybridAStarFlow::HybridAStarFlow(ros::NodeHandle &nh) {
    // steering_angle：从ROS参数服务器获取转向角度，默认值为10度。
    // steering_angle_discrete_num：转向角度的离散化数量，默认值为1。
    // wheel_base：车辆的轮距，默认值为1.0米。
    // segment_length：每个路径段的长度，默认值为1.6米。
    // segment_length_discrete_num：路径段长度的离散化数量，默认值为8。
    // steering_penalty：转向操作的成本惩罚，默认值为1.05。
    // steering_change_penalty：改变转向方向的成本惩罚，默认值为1.5。
    // reversing_penalty：倒车操作的成本惩罚，默认值为2.0。
    // shot_distance：搜索过程中每次向前探索的距离，默认值为5.0米。

    // <param name="planner/steering_angle" value="15.0"/>
    // <param name="planner/steering_angle_discrete_num" value="1"/>
    // <param name="planner/wheel_base" value="0.8"/>
    // <param name="planner/segment_length" value="1.6"/>
    // <param name="planner/segment_length_discrete_num" value="8"/>
    // <param name="planner/steering_penalty" value="2.0"/>
    // <param name="planner/reversing_penalty" value="3.0"/>
    // <param name="planner/steering_change_penalty" value="2.0"/>
    // <param name="planner/shot_distance" value="5.0"/>

    double steering_angle = nh.param("planner/steering_angle", 10);  //从ROS参数服务器获取转向角度，默认值为10度。 launch 文件读取参数
    int steering_angle_discrete_num = nh.param("planner/steering_angle_discrete_num", 1);
    double wheel_base = nh.param("planner/wheel_base", 0.8);
    double segment_length = nh.param("planner/segment_length", 1.6);
    int segment_length_discrete_num = nh.param("planner/segment_length_discrete_num", 8);
    double steering_penalty = nh.param("planner/steering_penalty", 1.05);
    double steering_change_penalty = nh.param("planner/steering_change_penalty", 1.5);
    double reversing_penalty = nh.param("planner/reversing_penalty", 2.0);
    double shot_distance = nh.param("planner/shot_distance", 5.0);
    grid_obstacle_resolution_ = nh.param("planner/grid_obstacle_resolution", 0.25);  //障碍物栅格分辨率
    grid_state_resolution_ = nh.param("planner/grid_state_resolution", 1.0);   //状态栅格分辨率

    const std::string hybrid_astar_pkg_path = ros::package::getPath("hybrid_a_star");
    const std::string default_frontend_script =
        hybrid_astar_pkg_path + "/model_base_astar/neural-astar/scripts/run_transformer_guided_astar_frontend.py";
    const std::string default_frontend_ckpt =
        hybrid_astar_pkg_path + "/model_base_astar/neural-astar/outputs/"
        "model_guidance_grid_mpd_unet_transformer_v3_gatedskip_formal_v1/best.pt";
    const std::string default_frontend_onnx =
        hybrid_astar_pkg_path + "/model_base_astar/neural-astar/outputs/"
        "model_guidance_grid_mpd_unet_transformer_v3_gatedskip_formal_v1/best_cost_map.onnx";

    use_transformer_guided_frontend_ = nh.param("planner/use_transformer_guided_frontend", false);
    fallback_to_hybrid_astar_ = nh.param("planner/fallback_to_hybrid_astar", true);
    guided_frontend_backend_ = nh.param<std::string>("planner/guided_frontend_backend", "python");
    guided_frontend_python_ = nh.param<std::string>("planner/guided_frontend_python", "python3");
    guided_frontend_script_path_ =
        nh.param<std::string>("planner/guided_frontend_script", default_frontend_script);
    guided_frontend_ckpt_path_ =
        nh.param<std::string>("planner/guided_frontend_ckpt", default_frontend_ckpt);
    guided_frontend_onnx_path_ =
        nh.param<std::string>("planner/guided_frontend_onnx", default_frontend_onnx);
    guided_frontend_device_ = nh.param<std::string>("planner/guided_frontend_device", "cpu");
    guided_frontend_lambda_ = nh.param("planner/guided_frontend_lambda", 1.0);
    guided_frontend_heuristic_weight_ = nh.param("planner/guided_frontend_heuristic_weight", 1.0);
    guided_frontend_heuristic_mode_ =
        nh.param<std::string>("planner/guided_frontend_heuristic_mode", "octile");
    guided_frontend_integration_mode_ =
        nh.param<std::string>("planner/guided_frontend_integration_mode", "g_cost");
    guided_frontend_bonus_threshold_ = nh.param("planner/guided_frontend_bonus_threshold", 0.5);
    guided_frontend_allow_corner_cut_ = nh.param("planner/guided_frontend_allow_corner_cut", false);
    guided_frontend_invert_guidance_cost_ = nh.param("planner/guided_frontend_invert_guidance_cost", false);
    guided_frontend_onnx_intra_threads_ = nh.param("planner/guided_frontend_onnx_intra_threads", 1);
    guided_frontend_onnx_inter_threads_ = nh.param("planner/guided_frontend_onnx_inter_threads", 1);
    static_map_source_ = nh.param<std::string>("planner/static_map_source", "pcd");
    static_map_topic_ = nh.param<std::string>("planner/static_map_topic", "/guided_frontend_random_map");
    static_map_timeout_ = nh.param("planner/static_map_timeout", 3.0);
    static_map_fallback_to_pcd_ = nh.param("planner/static_map_fallback_to_pcd", true);

    if (guided_frontend_backend_ == "onnx") {
        try {
            guided_frontend_onnx_.reset(new guided_frontend::GuidanceCostMapOnnx(
                guided_frontend_onnx_path_,
                guided_frontend_onnx_intra_threads_,
                guided_frontend_onnx_inter_threads_));
            ROS_INFO_STREAM("Initialized ONNX guided frontend model: " << guided_frontend_onnx_path_);
        } catch (const std::exception& e) {
            ROS_WARN_STREAM("Failed to initialize ONNX guided frontend: " << e.what());
            if (!fallback_to_hybrid_astar_) {
                ROS_WARN("Planner fallback is disabled; guided frontend requests may fail.");
            }
        }
    }
    
    kinodynamic_astar_searcher_ptr_ = std::make_shared<HybridAStar>(
            steering_angle, steering_angle_discrete_num, segment_length, segment_length_discrete_num, wheel_base,
            steering_penalty, reversing_penalty, steering_change_penalty, shot_distance);

    // costmap_sub_ptr_ = std::make_shared<CostMapSubscriber>(nh, "/map", 1);
    init_pose_sub_ptr_ = std::make_shared<InitPoseSubscriber2D>(nh, "/initialpose", 1);
    goal_pose_sub_ptr_ = std::make_shared<GoalPoseSubscriber2D>(nh, "/move_base_simple/goal", 1);
    init_pose_car_position_sub_ptr_ = std::make_shared<InitPoseSubscriber2D>(nh, "/car_pos", 1);

    path_pub_ = nh.advertise<nav_msgs::Path>("searched_path", 1);
    path_pub_planning_path = nh.advertise<planning_msgs::hybrid_astar_paths>("/hybrid_astar_paths", 1);
    searched_tree_pub_ = nh.advertise<visualization_msgs::Marker>("searched_tree", 1);
    vehicle_path_pub_ = nh.advertise<visualization_msgs::MarkerArray>("vehicle_path", 1);
    line_pub_local_path = nh.advertise<sensor_msgs::PointCloud2>("hybrid_astar_path_watch", 100);
    line_pub_smoothed_path = nh.advertise<sensor_msgs::PointCloud2>("hybrid_astar_smoothed_path_watch", 100);
    curvature_marker_pub_ = nh.advertise<visualization_msgs::MarkerArray>("hybrid_astar_curvature", 1);
    InitStaticMap();  
    ros::spinOnce();
    has_map_ = false;
}

bool HybridAStarFlow::ExportGuidedFrontendRequest(
    const Vec4d &start_state,
    const Vec4d &goal_state,
    const std::string &request_json_path) const {
    std::ofstream out(request_json_path);
    if (!out.is_open()) {
        ROS_ERROR_STREAM("Failed to open guided frontend request file: " << request_json_path);
        return false;
    }

    out << "{\n";
    out << "  \"width\": " << grid_obstacle_width_ << ",\n";
    out << "  \"height\": " << grid_obstacle_height_ << ",\n";
    out << "  \"resolution\": " << grid_obstacle_resolution_ << ",\n";
    out << "  \"origin_x\": " << map_min_x_ << ",\n";
    out << "  \"origin_y\": " << map_min_y_ << ",\n";
    out << "  \"start_world\": [" << start_state.x() << ", " << start_state.y() << "],\n";
    out << "  \"goal_world\": [" << goal_state.x() << ", " << goal_state.y() << "],\n";
    out << "  \"start_yaw\": " << start_state.z() << ",\n";
    out << "  \"goal_yaw\": " << goal_state.z() << ",\n";
    out << "  \"occupancy\": [\n";
    for (int y = 0; y < grid_obstacle_height_; ++y) {
        out << "    [";
        for (int x = 0; x < grid_obstacle_width_; ++x) {
            out << grid_static_occupancy[x][y];
            if (x + 1 < grid_obstacle_width_) {
                out << ", ";
            }
        }
        out << "]";
        if (y + 1 < grid_obstacle_height_) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    return true;
}

bool HybridAStarFlow::LoadGuidedFrontendPath(const std::string &path_csv, VectorVec4d &path_original) const {
    std::ifstream in(path_csv);
    if (!in.is_open()) {
        ROS_ERROR_STREAM("Failed to open guided frontend output: " << path_csv);
        return false;
    }

    path_original.clear();
    std::string line;
    bool is_first_line = true;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (is_first_line) {
            is_first_line = false;
            if (line.find("x") != std::string::npos && line.find("y") != std::string::npos) {
                continue;
            }
        }
        std::stringstream ss(line);
        std::string x_str;
        std::string y_str;
        if (!std::getline(ss, x_str, ',') || !std::getline(ss, y_str, ',')) {
            continue;
        }

        Vec4d point;
        point << std::stod(x_str), std::stod(y_str), 0.0, 1.0;
        path_original.emplace_back(point);
    }
    if (path_original.size() < 2) {
        return false;
    }
    for (size_t i = 0; i + 1 < path_original.size(); ++i) {
        const double dx = path_original[i + 1].x() - path_original[i].x();
        const double dy = path_original[i + 1].y() - path_original[i].y();
        path_original[i].z() = std::atan2(dy, dx);
    }
    path_original.back().z() = path_original[path_original.size() - 2].z();
    return true;
}

bool HybridAStarFlow::PlanWithTransformerGuidedAstarPython(
    const Vec4d &start_state,
    const Vec4d &goal_state,
    VectorVec4d &path_original,
    VectorVec4d &path_smoothed) {
    if (guided_frontend_script_path_.empty() || guided_frontend_ckpt_path_.empty()) {
        ROS_ERROR("Guided frontend script/checkpoint path is empty.");
        return false;
    }

    const std::string hybrid_astar_pkg_path = ros::package::getPath("hybrid_a_star");
    const std::string frontend_pythonpath = hybrid_astar_pkg_path + "/model_base_astar/neural-astar/src";
    const std::string tmp_base =
        "/tmp/hybrid_a_star_guided_frontend_" + std::to_string(static_cast<long long>(::getpid()));
    const std::string request_json = tmp_base + "_request.json";
    const std::string output_csv = tmp_base + "_path.csv";

    if (!ExportGuidedFrontendRequest(start_state, goal_state, request_json)) {
        return false;
    }

    std::stringstream cmd;
    cmd << "export PYTHONPATH=\"" << frontend_pythonpath << ":${PYTHONPATH}\" && ";
    cmd << "\"" << guided_frontend_python_ << "\" "
        << "\"" << guided_frontend_script_path_ << "\" "
        << "--input-json " << "\"" << request_json << "\" "
        << "--output-csv " << "\"" << output_csv << "\" "
        << "--ckpt " << "\"" << guided_frontend_ckpt_path_ << "\" "
        << "--device " << "\"" << guided_frontend_device_ << "\" "
        << "--lambda-guidance " << guided_frontend_lambda_ << " "
        << "--heuristic-mode " << "\"" << guided_frontend_heuristic_mode_ << "\" "
        << "--heuristic-weight " << guided_frontend_heuristic_weight_ << " "
        << "--guidance-integration-mode " << "\"" << guided_frontend_integration_mode_ << "\" "
        << "--guidance-bonus-threshold " << guided_frontend_bonus_threshold_ << " ";
    if (guided_frontend_allow_corner_cut_) {
        cmd << "--allow-corner-cut ";
    }
    if (guided_frontend_invert_guidance_cost_) {
        cmd << "--invert-guidance-cost ";
    }

    const int status = std::system(cmd.str().c_str());
    if (status != 0) {
        ROS_ERROR_STREAM("Guided frontend command failed with code " << status);
        std::remove(request_json.c_str());
        std::remove(output_csv.c_str());
        return false;
    }

    const bool load_ok = LoadGuidedFrontendPath(output_csv, path_original);
    std::remove(request_json.c_str());
    std::remove(output_csv.c_str());
    if (!load_ok) {
        ROS_ERROR("Guided frontend produced no valid path.");
        return false;
    }

    path_smoothed = kinodynamic_astar_searcher_ptr_->SmoothPath(path_original);
    return !path_smoothed.empty();
}

bool HybridAStarFlow::PlanWithTransformerGuidedAstarOnnx(
    const Vec4d &start_state,
    const Vec4d &goal_state,
    VectorVec4d &path_original,
    VectorVec4d &path_smoothed) {
    if (!guided_frontend_onnx_) {
        ROS_ERROR("ONNX guided frontend is not initialized.");
        return false;
    }

    Vec2i start_xy = kinodynamic_astar_searcher_ptr_->Coordinate2MapGridIndex(start_state.head(2));
    Vec2i goal_xy = kinodynamic_astar_searcher_ptr_->Coordinate2MapGridIndex(goal_state.head(2));
    start_xy.x() = std::min(std::max(start_xy.x(), 0), grid_obstacle_width_ - 1);
    start_xy.y() = std::min(std::max(start_xy.y(), 0), grid_obstacle_height_ - 1);
    goal_xy.x() = std::min(std::max(goal_xy.x(), 0), grid_obstacle_width_ - 1);
    goal_xy.y() = std::min(std::max(goal_xy.y(), 0), grid_obstacle_height_ - 1);

    std::vector<int> occupancy_row_major;
    occupancy_row_major.reserve(static_cast<std::size_t>(grid_obstacle_width_ * grid_obstacle_height_));
    for (int y = 0; y < grid_obstacle_height_; ++y) {
        for (int x = 0; x < grid_obstacle_width_; ++x) {
            occupancy_row_major.push_back(grid_static_occupancy[x][y]);
        }
    }

    std::vector<float> guidance_cost;
    try {
        guidance_cost = guided_frontend_onnx_->Infer(
            occupancy_row_major,
            grid_obstacle_width_,
            grid_obstacle_height_,
            start_xy,
            goal_xy,
            static_cast<float>(start_state.z()),
            static_cast<float>(goal_state.z()));
    } catch (const std::exception& e) {
        ROS_ERROR_STREAM("ONNX guided frontend inference failed: " << e.what());
        return false;
    }

    guided_frontend::GridAstarOptions astar_options;
    astar_options.lambda_guidance = guided_frontend_lambda_;
    astar_options.heuristic_weight = guided_frontend_heuristic_weight_;
    astar_options.heuristic_mode = guided_frontend_heuristic_mode_;
    astar_options.guidance_integration_mode = guided_frontend_integration_mode_;
    astar_options.guidance_bonus_threshold = guided_frontend_bonus_threshold_;
    astar_options.allow_corner_cut = guided_frontend_allow_corner_cut_;

    guided_frontend::GridAstarResult search_result = guided_frontend::RunGuidedGridAstar(
        occupancy_row_major,
        grid_obstacle_width_,
        grid_obstacle_height_,
        start_xy,
        goal_xy,
        guidance_cost,
        astar_options);
    if (!search_result.success || search_result.path_xy.size() < 2) {
        ROS_ERROR("ONNX guided grid A* failed to find a valid path.");
        return false;
    }

    path_original.clear();
    path_original.reserve(search_result.path_xy.size());
    for (std::size_t i = 0; i < search_result.path_xy.size(); ++i) {
        Vec4d point = Vec4d::Zero();
        if (i == 0) {
            point.x() = start_state.x();
            point.y() = start_state.y();
        } else if (i + 1 == search_result.path_xy.size()) {
            point.x() = goal_state.x();
            point.y() = goal_state.y();
        } else {
            const Vec2i& grid_xy = search_result.path_xy[i];
            point.x() = map_min_x_ + (static_cast<double>(grid_xy.x()) + 0.5) * grid_obstacle_resolution_;
            point.y() = map_min_y_ + (static_cast<double>(grid_xy.y()) + 0.5) * grid_obstacle_resolution_;
        }
        point.w() = 1.0;
        path_original.emplace_back(point);
    }

    for (std::size_t i = 0; i + 1 < path_original.size(); ++i) {
        const double dx = path_original[i + 1].x() - path_original[i].x();
        const double dy = path_original[i + 1].y() - path_original[i].y();
        path_original[i].z() = std::atan2(dy, dx);
    }
    path_original.back().z() = path_original[path_original.size() - 2].z();

    path_smoothed = kinodynamic_astar_searcher_ptr_->SmoothPath(path_original);
    return !path_smoothed.empty();
}

bool HybridAStarFlow::PlanWithTransformerGuidedAstar(
    const Vec4d &start_state,
    const Vec4d &goal_state,
    VectorVec4d &path_original,
    VectorVec4d &path_smoothed) {
    if (guided_frontend_backend_ == "onnx") {
        return PlanWithTransformerGuidedAstarOnnx(start_state, goal_state, path_original, path_smoothed);
    }
    return PlanWithTransformerGuidedAstarPython(start_state, goal_state, path_original, path_smoothed);
}

bool HybridAStarFlow::PlanWithCurrentFrontend(
    const Vec4d &start_state,
    const Vec4d &goal_state,
    VectorVec4d &path_original,
    VectorVec4d &path_smoothed) {
    if (use_transformer_guided_frontend_) {
        if (PlanWithTransformerGuidedAstar(start_state, goal_state, path_original, path_smoothed)) {
            last_plan_used_transformer_frontend_ = true;
            return true;
        }
        if (!fallback_to_hybrid_astar_) {
            return false;
        }
        ROS_WARN("Transformer-guided frontend failed, fallback to classic Hybrid A*.");
    }

    if (!kinodynamic_astar_searcher_ptr_->Search(start_state, goal_state)) {
        return false;
    }
    last_plan_used_transformer_frontend_ = false;
    path_smoothed = kinodynamic_astar_searcher_ptr_->GetPath(path_original);
    return !path_smoothed.empty();
}

void HybridAStarFlow::PublishPlanningOutputs(const VectorVec4d &path_original, const VectorVec4d &path_smoothed) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr raw_path_watch(new pcl::PointCloud<pcl::PointXYZI>);
    for (const auto &pose : path_original) {
        pcl::PointXYZI xy;
        xy.x = pose.x();
        xy.y = pose.y();
        raw_path_watch->points.push_back(xy);
    }
    sensor_msgs::PointCloud2 ss_local_path;
    pcl::toROSMsg(*raw_path_watch, ss_local_path);
    ss_local_path.header.frame_id = "velodyne";
    line_pub_local_path.publish(ss_local_path);

    pcl::PointCloud<pcl::PointXYZI>::Ptr smoothed_path_watch(new pcl::PointCloud<pcl::PointXYZI>);
    for (const auto &pose : path_smoothed) {
        pcl::PointXYZI xy;
        xy.x = pose.x();
        xy.y = pose.y();
        smoothed_path_watch->points.push_back(xy);
    }
    sensor_msgs::PointCloud2 ss_smoothed_path_watch;
    pcl::toROSMsg(*smoothed_path_watch, ss_smoothed_path_watch);
    ss_smoothed_path_watch.header.frame_id = "velodyne";
    line_pub_smoothed_path.publish(ss_smoothed_path_watch);

    PublishPathToControl(path_smoothed);
    PublishPath(path_original);
    PublishVehiclePath(path_smoothed, 2.0, 1.0, 5u);
    PublishCurvatureProfiles(path_original, path_smoothed);
}

void HybridAStarFlow::Run(ros::NodeHandle &nh,planning_msgs::car_scene &car_scene) {
    ReadData();
    std::cout<<"RUN!!!"<<std::endl;
    if (!has_map_) {
        // if (costmap_deque_.empty()) {
        //     return;
        // }
        // std::cout<<"map_size:"<<costmap_deque_.size()<<std::endl;
        // current_costmap_ptr_ = costmap_deque_.front();
        // costmap_deque_.pop_front();

        kinodynamic_astar_searcher_ptr_->Init(                  //给出地图的长宽
                map_min_x_,
                // 1.0 * current_costmap_ptr_->info.width * current_costmap_ptr_->info.resolution,
                map_max_x_,
                map_min_y_,
                // 1.0 * current_costmap_ptr_->info.height * current_costmap_ptr_->info.resolution,
                map_max_y_,
                grid_state_resolution_, grid_obstacle_resolution_    //这里状态采样为1.0有疑问
        );
        //  std::cout<<"Init_end:"<<std::endl;
        // std::cout<<"Init:"<<current_costmap_ptr_->info.origin.position.x<<","<<current_costmap_ptr_->info.origin.position.y<<",1111"<<1.0 * current_costmap_ptr_->info.width * current_costmap_ptr_->info.resolution
        // <<","<<1.0 * current_costmap_ptr_->info.height * current_costmap_ptr_->info.resolution<<","<<grid_obstacle_resolution_<<std::endl;

        for (unsigned int w = 0; w < grid_obstacle_width_; ++w) {
            for (unsigned int h = 0; h < grid_obstacle_height_; ++h) {
                if (grid_static_occupancy[w][h]) {
                    kinodynamic_astar_searcher_ptr_->SetObstacle(w, h);
                }
            }
        }
        has_map_ = true;
    }
    // costmap_deque_.clear();

    // if (sqrt(pow(path_end_x-car_pose_flow_ptr->x,2)+pow(path_end_y-car_pose_flow_ptr->y,2)) < 1)   //超越多少距离才算到达终点
    // {
    //     next_plan=true;
    // }
    // else
    // {
    //     next_plan=false;
    // }
    // std::cout<<"next_plan:"<<next_plan<<" distance:"<<sqrt(pow(path_end_x-car_pose_flow_ptr->x,2)+pow(path_end_y-car_pose_flow_ptr->y,2)) <<std::endl;
    // std::cout<<"pub_path_size:"<<pub_path.size()<<std::endl;
    if (pub_path.size()>0)
    {
        PublishPathToControl(pub_path);

    }
    // while (HasGoalPose()&&next_plan) {
  
    // while (HasGoalPose()){
    if (HasGoalPose()){
        InitPoseData();
        grid_static_occupancy=grid_static_occupancy_map;
        // std::cout<<"持续运行中！！！"<<std::endl;
        double goal_yaw = tf::getYaw(current_goal_pose_ptr_->pose.orientation);

        Vec4d start_state = Vec4d(
                car_pose_flow_ptr->x,
                car_pose_flow_ptr->y,
                car_pose_flow_ptr->theta,0
        );
        Vec4d goal_state = Vec4d(
                current_goal_pose_ptr_->pose.position.x,
                current_goal_pose_ptr_->pose.position.y,
                goal_yaw,0
        );

        goal_state_last=goal_state;

        VectorVec4d path_original;
        VectorVec4d path_smoothed;
        if (PlanWithCurrentFrontend(start_state, goal_state, path_original, path_smoothed)) {
            path_end_x = path_smoothed.back().x();
            path_end_y = path_smoothed.back().y();
            pub_path = path_smoothed;
            path_type++;

            PublishPlanningOutputs(path_original, path_smoothed);
            if (!last_plan_used_transformer_frontend_) {
                PublishSearchedTree(kinodynamic_astar_searcher_ptr_->GetSearchedTree());
                kinodynamic_astar_searcher_ptr_->node_points_watch.reset(new pcl::PointCloud<pcl::PointXYZI>);
            }
        }
        // debug
        // std::cout << "visited nodes: " << kinodynamic_astar_searcher_ptr_->GetVisitedNodesNumber() << std::endl;
        kinodynamic_astar_searcher_ptr_->Reset();
    }
    else if (car_scene.task_type==11&&(!goal_state_last.isZero()))//重规划
    {
        std::cout<<"重规划！！！"<<std::endl;
        car_scene.task_type=10;
        updateOccupancyHybridAStar(nh);
        for (unsigned int w = 0; w < grid_obstacle_width_; ++w) {
            for (unsigned int h = 0; h < grid_obstacle_height_; ++h) {
                if (grid_static_occupancy[w][h]) {
                    kinodynamic_astar_searcher_ptr_->SetObstacle(w, h);
                }
            }
        }
        Vec4d start_state = Vec4d(
                car_pose_flow_ptr->x,
                car_pose_flow_ptr->y,
                car_pose_flow_ptr->theta,0
        );
        Vec4d goal_state = goal_state_last;

        VectorVec4d path_original;
        VectorVec4d path_smoothed;
        if (PlanWithCurrentFrontend(start_state, goal_state, path_original, path_smoothed)) {
            path_end_x = path_smoothed.back().x();
            path_end_y = path_smoothed.back().y();
            pub_path = path_smoothed;
            path_type++;

            PublishPlanningOutputs(path_original, path_smoothed);
            if (!last_plan_used_transformer_frontend_) {
                PublishSearchedTree(kinodynamic_astar_searcher_ptr_->GetSearchedTree());
                kinodynamic_astar_searcher_ptr_->node_points_watch.reset(new pcl::PointCloud<pcl::PointXYZI>);
            }

            static tf::TransformBroadcaster transform_broadcaster;
            // for (const auto &pose: path_ros.poses) {
            //     std::cout<<"pose_size:"<<path_ros.poses.size()<<std::endl;
            //     tf::Transform transform;
            //     transform.setOrigin(tf::Vector3(pose.pose.position.x, pose.pose.position.y, 0.0));

            //     tf::Quaternion q;
            //     q.setX(pose.pose.orientation.x);
            //     q.setY(pose.pose.orientation.y);
            //     q.setZ(pose.pose.orientation.z);
            //     q.setW(pose.pose.orientation.w);
            //     transform.setRotation(q);

            //     transform_broadcaster.sendTransform(tf::StampedTransform(transform,
            //                                                              ros::Time::now(), "velodyne",
            //                                                              "ground_link")
            //     );
            //     std::cout<<"sleep_start"<<std::endl;
            //     ros::Duration(0.05).sleep();
            //     std::cout<<"sleep_endl"<<std::endl;
            // }
        }
        // debug
        // std::cout << "visited nodes: " << kinodynamic_astar_searcher_ptr_->GetVisitedNodesNumber() << std::endl;
        kinodynamic_astar_searcher_ptr_->Reset();
    }
}

//读取起始点和目标点
void HybridAStarFlow::ReadData() {
    // costmap_sub_ptr_->ParseData(costmap_deque_);
    car_pose_flow_ptr.reset(new geometry_msgs::Pose2D());

    init_pose_sub_ptr_->ParseData(init_pose_deque_);
    init_pose_car_position_sub_ptr_->ParseData(car_pose_flow_ptr);
    goal_pose_sub_ptr_->ParseData(goal_pose_deque_);

    if (!init_pose_deque_.empty()) {
        current_init_pose_ptr_ = init_pose_deque_.back();
        init_pose_deque_.clear();
        car_pose_flow_ptr->x = current_init_pose_ptr_->pose.pose.position.x;
        car_pose_flow_ptr->y = current_init_pose_ptr_->pose.pose.position.y;
        car_pose_flow_ptr->theta = tf::getYaw(current_init_pose_ptr_->pose.pose.orientation);
    }
}

// //重新规划时机
// bool HybridAStarFlow::NextPlan(const VectorVec4d &path) {

//     if(path.empty())
//     {
//         return true;
//     }
//     // std::cout<<"path.back():"<<path.back().x()<<","<<path.back().y()<<" car_pose_flow_ptr:"<<car_pose_flow_ptr->x<<","<<car_pose_flow_ptr->y<<std::endl;
//     if(sqrt(pow(path.back().x()-car_pose_flow_ptr->x,2)+pow(path.back().y()-car_pose_flow_ptr->y,2)) < 1)  //这里暂时设置到路线终点0.1则认为到达终点
//     {
//         return true;
//     }
//     return false;
// }

//取出起始点和目标点
void HybridAStarFlow::InitPoseData() {
    // current_init_pose_ptr_ = init_pose_deque_.front();
    // init_pose_deque_.pop_front();

    current_goal_pose_ptr_ = goal_pose_deque_.front();
    goal_pose_deque_.pop_front();
}

bool HybridAStarFlow::HasGoalPose() {
    // std::cout<<"HasGoalPose"<<std::endl;
    return !goal_pose_deque_.empty();
}

bool HybridAStarFlow::HasStartPose() {
    // std::cout<<"HasStartPose"<<std::endl;
    return !init_pose_deque_.empty();
}

void HybridAStarFlow::PublishPath(const VectorVec4d &path) {
    nav_msgs::Path nav_path;
    std::cout<<"path_size:"<<path.size()<<std::endl;
    planning_msgs::car_path carPath;
    geometry_msgs::PoseStamped pose_stamped;
    for (const auto &pose: path) {
        pose_stamped.header.frame_id = "velodyne";
        pose_stamped.pose.position.x = pose.x();
        pose_stamped.pose.position.y = pose.y();
        pose_stamped.pose.position.z = 0.0;
        pose_stamped.pose.orientation = tf::createQuaternionMsgFromYaw(pose.z());

        planning_msgs::path_point carPathPoint;
        
        carPathPoint.x = pose.x();
        carPathPoint.y = pose.y();
        carPathPoint.yaw = pose.z();
        carPath.points.push_back(carPathPoint);

        nav_path.poses.emplace_back(pose_stamped);
    }
    nav_path.header.frame_id = "velodyne";
    nav_path.header.stamp = timestamp_;
    path_pub_.publish(nav_path);
    // for (int i = 0; i < path.size(); i++)
    // {
    //     std::cout<<"x,y,yaw,dir:"<<path[i].x()<<","<<path[i].y()<<","<<path[i].z()<<","<<path[i].w()<<std::endl;
    // }
    // for (int i = 1; i < path.size(); i++)
    // {
    //     std::cout<<sqrt(pow(path[i][0]-path[i-1][0],2)+pow(path[i][1]-path[i-1][1],2))<<std::endl;
    // }
}

void HybridAStarFlow::PublishPathToControl(const VectorVec4d &path) {

    // planning_msgs::car_path::Ptr PubPathPoints(new planning_msgs::car_path); 
    // PubPathPoints->path_type=path_type;
    // pcl::PointCloud<pcl::PointXYZI>::Ptr local_path(new pcl::PointCloud<pcl::PointXYZI>);

    // for (int i = 0; i < path.size(); i++)
    // {
    //     planning_msgs::path_point PathPoint;
    //     pcl::PointXYZI xyv;
    //     PathPoint.x=path[i].x();
    //     PathPoint.y=path[i].y();
    //     PathPoint.yaw=path[i].z();
    //     PathPoint.vel=path[i].w()*plan_vel;  //速度设置为0.4
    //     xyv.x=PathPoint.x;xyv.y=PathPoint.y;xyv.intensity=PathPoint.vel;
    //     // if (xyv.intensity<0)
    //     // {
    //         local_path->points.push_back(xyv);
    //     // }
        
    //     PubPathPoints->points.push_back(PathPoint);
    //     std::cout<<"vel "<<i<<":"<<PathPoint.x<<","<<PathPoint.y<<","<<PathPoint.yaw<<","<<PathPoint.vel<<std::endl;
    // }
    // // if (path.size()>1.0)
    // // {
    // //     PubPathPoints->points[0].vel=path[1].w()*plan_vel;
    // // }
    // //终点停车判断
    // if (sqrt(pow(init_pose_car_position_sub_ptr_->car_position_ptr->x-PubPathPoints->points.back().x,2)+pow(init_pose_car_position_sub_ptr_->car_position_ptr->y-PubPathPoints->points.back().y,2))<0.5)
    // {
    //     PubPathPoints->ReachTarget=true;()
    // }
    // else
    // {
    //     PubPathPoints->ReachTarget=false;
    // }

    //     // DeBug watch
    //     sensor_msgs::PointCloud2 ss_local_path;
    //     pcl::toROSMsg(*local_path, ss_local_path);
    //     ss_local_path.header.frame_id = "velodyne";
    //     line_pub_local_path.publish(ss_local_path);
    // std::cout<<"vel:"<<PubPathPoints->points[0].x<<","<<PubPathPoints->points[0].y<<","<<PubPathPoints->points[0].yaw<<","<<PubPathPoints->points[0].vel<<std::endl;
    // path_pub_planning_path.publish(PubPathPoints);

    planning_msgs::hybrid_astar_paths::Ptr PubPaths(new planning_msgs::hybrid_astar_paths); 
    PubPaths->path_type=path_type;
    pcl::PointCloud<pcl::PointXYZI>::Ptr local_path(new pcl::PointCloud<pcl::PointXYZI>);

    int change_path=0;
    int i = 0 ;

    while (i<path.size())
    {
        planning_msgs::hybrid_astar_path PubPath;
        change_path=path[i].w();
        if (path[i].w()>0)
        {
            PubPath.direct=1;
        }
        else
        {
            PubPath.direct=-1;
        }
        // std::cout<<"!!!!!!!!PubPath.direct:"<<PubPath.direct <<"!!!!!!!!!"<<std::endl;
        for (; i < path.size(); i++)
        {
            if (path[i].w()!=change_path)
            {
                // PubPath.single_path.back().vel=0.0;  //每段路径的终点设置为0
                break;
            }
            
            planning_msgs::hybrid_astar_path_point PubPath_point;

            PubPath_point.x=path[i].x();
            PubPath_point.y=path[i].y();
            PubPath_point.yaw=path[i].z();
            PubPath_point.vel=path[i].w()*plan_vel;  //速度设置为0.4
            PubPath.single_path.push_back(PubPath_point);
            // std::cout<<"vel11:"<<PubPath_point.x<<","<<PubPath_point.y<<","<<PubPath_point.yaw<<","<<PubPath_point.vel<<std::endl;
        }
        // std::cout<<std::endl;
        PubPaths->Path.push_back(PubPath);
    }

    // //debug
    // for (int j = 0; j < PubPaths->Path.size(); j++)
    // {
    //     std::cout<<"!!!!!!!!:"<<PubPaths->Path[j].direct<<std::endl;
    //     for (int k = 0; k < PubPaths->Path[j].single_path.size(); k++)
    //     {
    //         std::cout<<"vel12:"<<PubPaths->Path[j].single_path[k].x<<","<<PubPaths->Path[j].single_path[k].y<<","<<PubPaths->Path[j].single_path[k].yaw<<","<<PubPaths->Path[j].single_path[k].vel<<std::endl;
    //     }
    // }
    
    //观测
    for (int i = 0; i < path.size(); i++)
    {
        pcl::PointXYZI xyv;
        xyv.x=path[i].x();xyv.y=path[i].y();
        local_path->points.push_back(xyv);
    }
    //终点停车判断
    if (sqrt(pow(init_pose_car_position_sub_ptr_->car_position_ptr->x-path.back().x(),2)+pow(init_pose_car_position_sub_ptr_->car_position_ptr->y-path.back().y(),2))<0.25)
    {
        PubPaths->ReachTarget=true;
    }
    else
    {
        PubPaths->ReachTarget=false;
    }

    // DeBug watch
    sensor_msgs::PointCloud2 ss_local_path;
    pcl::toROSMsg(*local_path, ss_local_path);
    ss_local_path.header.frame_id = "velodyne";
    line_pub_local_path.publish(ss_local_path);
    // std::cout<<"vel:"<<PubPathPoints->points[0].x<<","<<PubPathPoints->points[0].y<<","<<PubPathPoints->points[0].yaw<<","<<PubPathPoints->points[0].vel<<std::endl;
    path_pub_planning_path.publish(PubPaths);

}
//发布车辆路线
void HybridAStarFlow::PublishVehiclePath(const VectorVec4d &path, double width,
                                         double length, unsigned int vehicle_interval = 5u) {
    visualization_msgs::MarkerArray vehicle_array;

    for (unsigned int i = 0; i < path.size(); i += vehicle_interval) {
        visualization_msgs::Marker vehicle;

        if (i == 0) {    //这段代码可以参考
            vehicle.action = 3;
        }

        vehicle.header.frame_id = "velodyne";
        vehicle.header.stamp = ros::Time::now();
        vehicle.type = visualization_msgs::Marker::CUBE;
        vehicle.id = static_cast<int>(i / vehicle_interval);
        vehicle.scale.x = width;
        vehicle.scale.y = length;
        vehicle.scale.z = 0.01;
        
        // 设置半透明效果
        vehicle.color.a = 0.05;  // 半透明（0.0 完全透明，1.0 完全不透明）

        // 设置为浅绿色
        vehicle.color.r = 0.0;  // 红色分量
        vehicle.color.g = 0.5;  // 绿色分量
        vehicle.color.b = 0.2;  // 蓝色分量

        vehicle.pose.position.x = path[i].x();
        vehicle.pose.position.y = path[i].y();
        vehicle.pose.position.z = 0.0;

        vehicle.pose.orientation = tf::createQuaternionMsgFromYaw(path[i].z());
        vehicle_array.markers.emplace_back(vehicle);
    }

    vehicle_path_pub_.publish(vehicle_array);
}

void HybridAStarFlow::ComputeCurvatureProfile(const VectorVec4d &path, std::vector<double> &s_out,
                                 std::vector<double> &kappa_out) {
    s_out.clear();
    kappa_out.clear();
    if (path.size() < 2) {
        return;
    }

    const size_t n = path.size();
    s_out.resize(n, 0.0);
    kappa_out.resize(n, 0.0);

    for (size_t i = 1; i < n; ++i) {
        const double dx = path[i].x() - path[i - 1].x();
        const double dy = path[i].y() - path[i - 1].y();
        s_out[i] = s_out[i - 1] + std::sqrt(dx * dx + dy * dy);
    }

    // 基于相邻弧段的稳健签名曲率：k = 2 * cross(v1,v2) / (|v1|*|v2|*(|v1|+|v2|))
    constexpr double kEps = 1e-4;
    for (size_t i = 1; i + 1 < n; ++i) {
        const double vx1 = path[i].x() - path[i - 1].x();
        const double vy1 = path[i].y() - path[i - 1].y();
        const double vx2 = path[i + 1].x() - path[i].x();
        const double vy2 = path[i + 1].y() - path[i].y();
        const double ds1 = std::hypot(vx1, vy1);
        const double ds2 = std::hypot(vx2, vy2);
        const double denom = ds1 * ds2 * (ds1 + ds2);
        if (denom < kEps) {
            kappa_out[i] = 0.0;
            continue;
        }
        const double cross = vx1 * vy2 - vy1 * vx2;
        kappa_out[i] = 2.0 * cross / denom;
    }

    // 边界赋值
    if (n >= 2) {
        kappa_out.front() = kappa_out[1];
        kappa_out.back() = kappa_out[n - 2];
    }

    // 5 点中值滤波抑制尖点
    std::vector<double> median_kappa = kappa_out;
    if (n > 4) {
        for (size_t i = 2; i + 2 < n; ++i) {
            double window[5] = {kappa_out[i - 2], kappa_out[i - 1], kappa_out[i],
                                kappa_out[i + 1], kappa_out[i + 2]};
            std::sort(window, window + 5);
            median_kappa[i] = window[2];
        }
        kappa_out.swap(median_kappa);
    }

    // 再次进行轻微的移动平均平滑
    std::vector<double> smooth_kappa = kappa_out;
    for (size_t i = 1; i + 1 < n; ++i) {
        smooth_kappa[i] = 0.25 * kappa_out[i - 1] + 0.5 * kappa_out[i] + 0.25 * kappa_out[i + 1];
    }
    kappa_out.swap(smooth_kappa);
}

void HybridAStarFlow::PublishCurvatureProfiles(const VectorVec4d &raw_path,
                                               const VectorVec4d &smoothed_path) {
    std::vector<double> raw_s;
    std::vector<double> raw_kappa;
    std::vector<double> smooth_s;
    std::vector<double> smooth_kappa;

    ComputeCurvatureProfile(raw_path, raw_s, raw_kappa);
    ComputeCurvatureProfile(smoothed_path, smooth_s, smooth_kappa);

    visualization_msgs::MarkerArray curvature_markers;
    auto make_marker = [](const std::vector<double> &s, const std::vector<double> &kappa,
                          const std::string &ns, int id,
                          double r, double g, double b) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "velodyne";
        marker.header.stamp = ros::Time::now();
        marker.ns = ns;
        marker.id = id;
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = 0.05;
        marker.color.a = 1.0;
        marker.color.r = r;
        marker.color.g = g;
        marker.color.b = b;
        marker.pose.orientation.w = 1.0;

        for (size_t i = 0; i < s.size() && i < kappa.size(); ++i) {
            geometry_msgs::Point pt;
            pt.x = s[i];
            pt.y = kappa[i];
            pt.z = 0.0;
            marker.points.emplace_back(pt);
        }
        return marker;
    };

    if (!raw_s.empty() && !raw_kappa.empty()) {
        curvature_markers.markers.emplace_back(
                make_marker(raw_s, raw_kappa, "raw_curvature", 0, 0.9, 0.2, 0.2));
    }
    if (!smooth_s.empty() && !smooth_kappa.empty()) {
        curvature_markers.markers.emplace_back(
                make_marker(smooth_s, smooth_kappa, "smoothed_curvature", 1, 0.2, 0.7, 0.3));
    }

    if (curvature_markers.markers.empty()) {
        visualization_msgs::Marker clear_marker;
        clear_marker.action = visualization_msgs::Marker::DELETEALL;
        curvature_markers.markers.emplace_back(clear_marker);
    }

    curvature_marker_pub_.publish(curvature_markers);
}

void HybridAStarFlow::InitStaticMap()
{ 
    if (static_map_source_ == "occupancy_grid") {
        if (InitStaticMapFromOccupancyGrid()) {
            return;
        }
        if (!static_map_fallback_to_pcd_) {
            ROS_FATAL_STREAM("Failed to initialize occupancy-grid static map from topic " << static_map_topic_
                             << " and fallback is disabled.");
            throw std::runtime_error("InitStaticMapFromOccupancyGrid failed");
        }
        ROS_WARN_STREAM("Falling back to PCD static map after occupancy-grid init failed.");
    }

    clock_t time_start = clock();
    std::string ugv_path = ros::package::getPath("ugv_position");

    std::string yaml_path1 =ugv_path+"/config.yaml";
    YAML::Node config_path = YAML::LoadFile(yaml_path1);
        
    std::string pcd_astar_path = ugv_path+"/"+config_path["ASTAR_PCD_PATH"].as<std::string>();
    pcl::PointCloud<pcl::PointXYZI>::Ptr pclStaticMapPointCloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile(pcd_astar_path, *pclStaticMapPointCloud);
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();

    for (const auto& point : pclStaticMapPointCloud->points) {
        if (point.x < min_x) min_x = point.x;
        if (point.x > max_x) max_x = point.x;
        if (point.y < min_y) min_y = point.y;
        if (point.y > max_y) max_y = point.y;
    }
    map_max_x_ = max_x;
    map_min_x_ = min_x;
    map_max_y_ = max_y;
    map_min_y_ = min_y;
    // std::cout<<"max_x y:"<<max_x<<","<<max_y<<" min x y:"<<min_x<<","<<min_y<<std::endl;

    int num_grids_x = std::floor((max_x - min_x) / grid_obstacle_resolution_);
    int num_grids_y = std::floor((max_y - min_y) / grid_obstacle_resolution_);
    grid_obstacle_width_  = num_grids_x;
    grid_obstacle_height_ = num_grids_y;

    // std::cout<<"map_x:"<<num_grids_x<<" map_y:"<<num_grids_y<<std::endl;

    // 创建一个二维数组来表示网格占用情况，0表示未占用，1表示已占用
    grid_static_occupancy = std::vector<std::vector<int>>(num_grids_x, std::vector<int>(num_grids_y, 0));
    grid_static_occupancy_map = std::vector<std::vector<int>>(num_grids_x, std::vector<int>(num_grids_y, 0));
    grid_point_count=std::vector<std::vector<int>>(num_grids_x, std::vector<int>(num_grids_y, 0));

    // 遍历点云数据，根据点的 x, y 坐标计算其所在的网格，并统计每个网格的点数
    for (const auto& point : pclStaticMapPointCloud->points) {
        if (point.x >= min_x && point.x <= max_x && point.y >= min_y && point.y <= max_y) {
            int grid_x = std::floor((point.x - min_x) / grid_obstacle_resolution_);
            int grid_y = std::floor((point.y - min_y) / grid_obstacle_resolution_);
    //         // 确保索引不越界
            if (grid_x >= 0 && grid_x < num_grids_x && grid_y >= 0 && grid_y < num_grids_y) {

                // 统计该网格内的点数 ,只有当网格内的点数大于等于2时才标记为占用
                // grid_point_count[grid_x][grid_y]++;
                // if (grid_point_count[grid_x][grid_y] >= 2) {
                    grid_static_occupancy[grid_x][grid_y] = 1;  // 标记为占用
                // }
            }
        }
    }
    grid_static_occupancy_map=grid_static_occupancy;
    clock_t time_end2 = clock();
    double time_diff2 = static_cast<double>(time_end2 - time_start) / CLOCKS_PER_SEC;
    std::cout<<"InitStaticMap加载耗时ms："<<time_diff2<<std::endl;
}

bool HybridAStarFlow::InitStaticMapFromOccupancyGrid()
{
    clock_t time_start = clock();
    ros::NodeHandle nh;
    nav_msgs::OccupancyGridConstPtr map_msg =
        ros::topic::waitForMessage<nav_msgs::OccupancyGrid>(
            static_map_topic_, nh, ros::Duration(static_map_timeout_));
    if (!map_msg) {
        ROS_ERROR_STREAM("Timed out waiting for occupancy grid on topic: " << static_map_topic_);
        return false;
    }

    if (map_msg->info.width == 0 || map_msg->info.height == 0) {
        ROS_ERROR_STREAM("Received empty occupancy grid on topic: " << static_map_topic_);
        return false;
    }

    grid_obstacle_resolution_ = map_msg->info.resolution;
    grid_obstacle_width_ = static_cast<int>(map_msg->info.width);
    grid_obstacle_height_ = static_cast<int>(map_msg->info.height);
    map_min_x_ = map_msg->info.origin.position.x;
    map_min_y_ = map_msg->info.origin.position.y;
    map_max_x_ = map_min_x_ + grid_obstacle_width_ * grid_obstacle_resolution_;
    map_max_y_ = map_min_y_ + grid_obstacle_height_ * grid_obstacle_resolution_;

    grid_static_occupancy = std::vector<std::vector<int>>(
        grid_obstacle_width_, std::vector<int>(grid_obstacle_height_, 0));
    grid_static_occupancy_map = std::vector<std::vector<int>>(
        grid_obstacle_width_, std::vector<int>(grid_obstacle_height_, 0));
    grid_point_count = std::vector<std::vector<int>>(
        grid_obstacle_width_, std::vector<int>(grid_obstacle_height_, 0));

    const std::size_t expected_size =
        static_cast<std::size_t>(grid_obstacle_width_) * static_cast<std::size_t>(grid_obstacle_height_);
    if (map_msg->data.size() != expected_size) {
        ROS_ERROR_STREAM("Occupancy grid size mismatch on topic " << static_map_topic_
                         << ": expected " << expected_size << ", got " << map_msg->data.size());
        return false;
    }

    for (int y = 0; y < grid_obstacle_height_; ++y) {
        for (int x = 0; x < grid_obstacle_width_; ++x) {
            const std::size_t index =
                static_cast<std::size_t>(y) * static_cast<std::size_t>(grid_obstacle_width_) +
                static_cast<std::size_t>(x);
            const int8_t occ = map_msg->data[index];
            grid_static_occupancy[x][y] = (occ > 50) ? 1 : 0;
        }
    }
    grid_static_occupancy_map = grid_static_occupancy;

    clock_t time_end = clock();
    double time_diff = static_cast<double>(time_end - time_start) / CLOCKS_PER_SEC;
    ROS_INFO_STREAM("InitStaticMap loaded occupancy-grid map from " << static_map_topic_
                    << " in " << time_diff << " s"
                    << ", size=" << grid_obstacle_width_ << "x" << grid_obstacle_height_
                    << ", resolution=" << grid_obstacle_resolution_);
    return true;
}


void HybridAStarFlow::updateOccupancyHybridAStar(ros::NodeHandle &nh)
{ 
    // 创建一个临时的订阅器
    bool message_received = false;
    ros::Subscriber temp_sub = nh.subscribe<sensor_msgs::PointCloud2>(
        "/matching_point1", 1, [this, &message_received](const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
            message_received = true; // 标记收到消息
            pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_dynamic_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::fromROSMsg(*cloud_msg, *pcl_dynamic_cloud);

            // 遍历点云数据并更新占据地图
            std::cout << "pcl_dynamic_cloud->points_size:"<< pcl_dynamic_cloud->points.size()<< std::endl;
            for (const auto& point : pcl_dynamic_cloud->points) {
                if (point.x >= map_min_x_ && point.x <= map_max_x_ && 
                    point.y >= map_min_y_ && point.y <= map_max_y_) {
                    int grid_x = std::floor((point.x - map_min_x_) / grid_obstacle_resolution_);
                    int grid_y = std::floor((point.y - map_min_y_) / grid_obstacle_resolution_);
                    
                    if (grid_x >= 0 && grid_x < grid_obstacle_width_ && 
                        grid_y >= 0 && grid_y < grid_obstacle_height_) {
                        grid_static_occupancy[grid_x][grid_y] = 1;  // 标记为占用
                    }
                }
            }

            std::cout << "Occupancy map refreshed successfully with new obstacle data." << std::endl;
        });

    // 等待消息一次
    ros::Duration timeout(1.0); // 设置一个合理的超时时间
    ros::Time start_time = ros::Time::now();
    while (ros::ok() && (ros::Time::now() - start_time < timeout)) {
        ros::spinOnce(); // 处理回调
        if (message_received) {
            break; // 如果收到消息，直接退出循环
        }
    }
}

void HybridAStarFlow::PublishSearchedTree(const VectorVec4d &searched_tree) {   //搜索树显示
    visualization_msgs::Marker tree_list;
    tree_list.header.frame_id = "velodyne";
    tree_list.header.stamp = ros::Time::now();
    tree_list.type = visualization_msgs::Marker::LINE_LIST;
    tree_list.action = visualization_msgs::Marker::ADD;
    tree_list.ns = "searched_tree";
    tree_list.scale.x = 0.02;

    tree_list.color.a = 1.0;
    tree_list.color.r = 0;
    tree_list.color.g = 0;
    tree_list.color.b = 0;

    tree_list.pose.orientation.w = 1.0;
    tree_list.pose.orientation.x = 0.0;
    tree_list.pose.orientation.y = 0.0;
    tree_list.pose.orientation.z = 0.0;

    geometry_msgs::Point point;
    for (const auto &i: searched_tree) {
        point.x = i.x();
        point.y = i.y();
        point.z = 0.0;
        tree_list.points.emplace_back(point);

        point.x = i.z();
        point.y = i.w();
        point.z = 0.0;
        tree_list.points.emplace_back(point);
    }

    searched_tree_pub_.publish(tree_list);
}
