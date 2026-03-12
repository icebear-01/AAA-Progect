/*
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2024-04-23 15:04:24
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2024-12-27 09:03:30
 * @FilePath: /src/planning/src/emplanner/include/emplanner.hpp
 * @Description: 
 * 
 * Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
 */
#ifndef __PLANNING_H  
#define __PLANNING_H
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unistd.h>
#include "yaml-cpp/yaml.h"
#include <pthread.h>
#include <limits.h>
#include <cfloat>
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <sensor_msgs/PointCloud2.h>
#include <vector>
#include <memory>
#include <cmath>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <geometry_msgs/Pose2D.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <planning_msgs/car_path.h>
#include <planning_msgs/car_info.h>
#include <planning_msgs/path_point.h>
#include <planning_msgs/point.h>
#include <planning_msgs/Obstacle.h>
#include <planning_msgs/ObstacleList.h>
#include <planning_msgs/car_scene.h>
#include <obj_msgs/ObstacleList.h>
#include <ros/package.h>
#include "OsqpEigen/OsqpEigen.h"
#include <std_msgs/Float32.h>


// #include "matplotlibcpp.h"

using namespace std;

struct path_point
{
    float V_straight;
    float V_turn;
    int flag_turn;       // ##判断是否是转弯状态
    int flag_right_turn; // ##判断是否是转弯状态
    int flag_left_turn;  // ##判断是否是转弯状态
    int number;          // ##从0开始
    float x;
    float y;
    float yaw;
    float theta;
    float vel;
    float s;
    float l=0;
    float ds;
    float absolute_s; // ##相对原点对应的s
    float relative_s; // ##相对点对应的s
    float kappa;      // ##曲率
    Eigen::Vector2d tor;        // 法向量
    Eigen::Vector2d nor;        // 切向量
};

struct car_path
{
    vector<planning_msgs::path_point> points;
    float V_straight;
    float V_turn;
};
struct Frenet_point //路径规划节点间的增密点
{
    float s; 
    float l;
    float dl;
    float ddl;
    float dddl;
};

struct Frenet_path_points //路径规划节点间的增密点
{
    Eigen::VectorXd s;  //此处使用VectorXd类型为方便五次多项式计算及qp计算方便，但不利于后续计算
    Eigen::VectorXd l;
    Eigen::VectorXd dl;
    Eigen::VectorXd ddl;
    Eigen::VectorXd dddl;
};

struct Speed_plan_points //路径规划节点间的增密点
{
    vector<double> s;
    vector<double> s_dot;
    vector<double> s_dot2;
    vector<double> s_time;
};

struct Single_node  //路径规划单个节点
{
    float node_s; 
    float node_l;
    double toThisNodeMinCost; //上一列到这点的最小代价
    int pre2this_min_cost_index; //上一列到这点的最小代价的列序号
};
struct Col_nodes //路径规划单个列所有节点
{
    vector<Single_node> row_nodes;
};
struct Avoid_nodes//路径规划所有节点
{
    vector<int> min_cost_path_indexs;
    vector<Col_nodes> nodes;
};
struct Min_path_nodes//路径规划代价最小节点集合
{
    vector<Single_node> nodes;
};

struct Speed_DP_Single_node
{
    float node_s; 
    float node_t;
    float node_s_dot=0.0010;
    float node_s_dot2; //加速度
    double speed_cost;
    double a_cost; 
    double obs_cost=0;
    double cost=1e10;
    bool is_possible=false; //判断节点是否有可能被选取
    double toThisNodeMinCost; //上一列到这点的最小代价
    int toThis_min_cost_index; //上一列到这点的最小代价的列序号
};

struct Speed_DP_col_nodes
{
    vector<Speed_DP_Single_node> rol_nodes;
    // int PreToNextColMinIndex; //此行所选择的最小代价节点
    // int ThisColSeclectIndex; //此行所选择的最小代价节点
};

struct Speed_Plan_DP_ST_nodes
{
    vector<Speed_DP_col_nodes> col_nodes;
};

struct Obstacle
{
    int num;
    float s;
    float l;
    float max_s;
    float max_l;
    float min_s;
    float min_l;
    float index_min;
    float index_max;
    float distance;
};
//障碍物信息(暂时)
struct Obstacle_list
{
    vector<Obstacle> obstacle;
};   

class RL_DP;

class EMPlanner
{
public:
    void Plan(planning_msgs::car_scene &car_scene);
    planning_msgs::car_path::Ptr QP_path;   //全局路径 
    bool reach_end;
    EMPlanner(ros::NodeHandle &nh,planning_msgs::car_scene car_scene) ;
    ~EMPlanner();
    void InjectSimCarState(const planning_msgs::car_info &car_pose);
    void InjectSimObstacleList(const planning_msgs::ObstacleList &obstacle_list_msg);
    void InjectSimCarStop(float flag_car_stop);
    void ReinitializeForCurrentInputs();
    bool HasLatestPlanResult() const;
    bool GetLatestQpRunningNormally() const;
    const std::string& GetLatestDpSource() const;
    const Frenet_path_points& GetLatestDpPathSL() const;
    const Frenet_path_points& GetLatestQpPathSL() const;
    const std::vector<Eigen::Vector2d>& GetLatestDpPathXY() const;
    const std::vector<Eigen::Vector2d>& GetLatestQpPathXY() const;
    const planning_msgs::car_path& GetLatestQpPathMsg() const;
    double GetLatestPlannerCycleMs() const;
    double GetLatestDpSamplingMs() const;
    double GetLatestQpOptimizationMs() const;
    double GetLatestSpeedPlanningMs() const;
    bool HasLatestSpeedPlanResult() const;
    const Speed_plan_points& GetLatestSpeedQpPoints() const;
    const std::vector<double>& GetLatestSpeedDpPathS() const;
    int GetLatestSpeedDpLastFeasibleIndex() const;
    const Speed_Plan_DP_ST_nodes& GetLatestSpeedDpStNodes() const;
    const planning_msgs::ObstacleList& GetLatestSpeedObstacleListSL() const;
private:
    // fstream &file_;
    ros::Publisher line_pub;
    ros::Publisher line_pub_watch;
    ros::Publisher line_pub_path;
    ros::Publisher line_pub_local_path;
    ros::Publisher line_pub_local_qp_path;
    ros::Publisher line_pub_local_qp_path_watch;
    
    ros::Publisher obs_watch_pub;
    ros::Subscriber sub_car_stop; // 订阅紧急停车
    ros::Subscriber sub_car_info; // 订阅规划路线信息
    ros::Subscriber sub_location;
    ros::Subscriber sub_obstacle_list_lidar; // 订阅规划路线信息
    ros::Subscriber sub_obstacle_list_vision; // 订阅规划路线信息
    
    void First_run();
    planning_msgs::car_scene car_scene_;  //车辆场景
    void callBack_location(const planning_msgs::car_info::ConstPtr &car_pose);
    void callBack_obstacleList_lidar(const planning_msgs::ObstacleList::ConstPtr &obstacle_list);
    void callBack_obstacle_vision_List(const obj_msgs::ObstacleList::ConstPtr &Obstacle_list_vision);
    void callBack_carInfo(const planning_msgs::car_info::ConstPtr &car_info);
    void callBack_car_stop(const std_msgs::Float32::ConstPtr& flag_car_stop);

    bool car_pos_is_arrive=false; //判断定位信息是否到达，否则会出现错误

    Eigen::Vector2d host_middle_sl; //车中点的SL(实时)
    Eigen::Vector2d host_forward_sl; //车前点的SL(实时)
    Eigen::Vector2d host_back_sl; //车后点的SL(实时)

    float real_vehicle_speed=0.0;

    //车辆参数
    float L=0.33; //车中到车前轴的距离
    float carLength=0.98;
    float car_width=0.75;
    float lidarToCarHead=0.55;
    int Gear=1; //驾驶方向，前进为1,倒车为-1

    //障碍物变量条件
    float dynamic_vel_threshold=0.0; // m/s, <=0 means any non-zero obstacle velocity is dynamic
    float road_obsracle_limit_left=4.0;  //障碍物超过左右道路范围则不考虑
    float road_obsracle_limit_right=-4.0;  //障碍物超过左右道路范围则不考虑
    float obstacleConsiderDistance=0.5;
    float st_lateral_limit=1.0;  // ST图只考虑局部路径左右1m内的切入带

    //Path_Planner
    float sample_s=0.8;
    float sample_l=0.35;
    float sample_s_num=10; //每一米几个点计算cost
    float sample_s_per_meters=20;//每一米增密几个点

    int col_node_num=6; //几列
    int row_node_num=11; //几行

    float w_cost_smooth_dl=2;
    float w_cost_smooth_ddl=1;
    float w_cost_smooth_dddl=2;
    float w_cost_smooth_total=20;
    float w_cost_ref=2000;
    float w_cost_collision=1;

    float w_qp_l = 800;
    float w_qp_dl = 200;
    float w_qp_ddl = 600;
    float w_qp_ref_dp = 50;

    Eigen::VectorXd min_watch;
    Eigen::VectorXd max_watch;

    float planning_star_s;
    float planning_star_l;

    int index_car_match_points=0;

    float dp_min_collision_distance_pow2=(car_width/2)*(car_width/2); //障碍物最小碰撞距离的平方(小于此距离，代价无限大)
    float max_collision_distance=car_width; //障碍物最大碰撞距离（影响距离，超过此距离认为无影响）

    // float DpPathBoundaryCost=0;
    // float DpPathBoundaryRightLimit=-0.3;

    planning_msgs::ObstacleList::Ptr obstacle_list;
    planning_msgs::ObstacleList::Ptr obstacle_list_qp_path_sl;
    planning_msgs::ObstacleList::Ptr obstacle_list_swap;

    planning_msgs::ObstacleList::Ptr obstacle_list_vision;
    planning_msgs::ObstacleList::Ptr obstacle_list_lidar;

    planning_msgs::car_path path;   //全局路径     
    
    planning_msgs::car_path::Ptr QP_path_last_success_run;   //全局路径   

    float err_max=0;
    float planning_frist_star_l;
    float planning_frist_star_s;

    float L3=3;
    float L4=4.75; //车后点

    float v_planning_start=1;
    float v_planning_start_dt=0;

    Frenet_path_points DP_path_sl; //增密后的sl路径坐标
    Frenet_path_points QP_path_sl;

    vector<Eigen::Vector2d> planning_dp_path_xy;
    vector<Eigen::Vector2d> planning_qp_path_xy;
    vector<Eigen::Vector2d> planning_qp_path_xy_last_success_run;

    Frenet_path_points DP_path_sl_r; //倒车增密后的sl路径坐标
    Frenet_path_points QP_path_sl_r;


    float safe_distance=0.2;//添加车辆碰撞安全距离
    float safe_distance_wall=0.15; //添加车辆碰撞安全距离
    bool QpPathRunningNormally=true;

    Frenet_path_points QP_path_sl_global;

    //Vel_Planner
    bool flag_obstacle_too_close=false;

    float vel_deceleration_rate=0.7;
    float ST_s_min_step=0.2;    //ST图的s最小分辨率
    float plan_time=4; // s
    float speed_reference=0.5; // m/s
    float w_SpeedDpPlan_ref_vel=500.0;
    float w_SpeedDpPlan_a=1.0;
    float w_SpeedDpPlan_obs=100.0;
    float SpeedDpPlanMinObstacleDistance=0.4;
    float SpeedDpPlanMaxObstacleDistance=0.6;

    float flagCarStop=0;

    float calc_car_speed=0;

    float speed_plan_t_dt=0.5; //速度采样间隔  

    float speed_plan_distance=5.0;  //最大速度规划长度

    float w_SpeedQpPlan_ref_s=0.0;
    float w_SpeedQpPlan_ref_vel=1.0;
    float w_SpeedQpPlan_a=1.0;
    float SpeedQpPlan_SafeDistance=0.01;

    float SpeedQP_v_max = 1.2;
    float SpeedQP_a_max = 0.5;

    bool SpeedPlanRunningNormally=true;

    float dp_vel_max=1.2; //允许车辆的最大速度
    float dp_a_max=0.6; //允许车辆的最大加速度

    float END_Stop_Distance=1.0;

    // 速度控制
    float V_straight = 1;
    float V_turn = 1;
    float end_slow_distance = 11;
    float star_slow_distance = 18; // 减速缓冲距离从star_slow_distance至end_slow_distance，速度由V_straight减到V_turn
    float end_speed_distance = 10;
    float star_speed_distance = 6; // 加速缓冲距离从star_speed_distance至star_speed_distance，速度由V_turn加到V_straight

    float sMinDistance=0.01;  //此处为防止出现speed_qp无解

    float index_closest_Car; //车身替代点
    int index_middle_Car;
    int index_back_Car;
    float index_middle_car=0; //车中点
    geometry_msgs::Pose2D::Ptr Car_Pose; // 车辆位姿信息(车前轴的坐标)
    geometry_msgs::Pose2D::Ptr Car_Pose_middle; // 车辆位姿信息(车中轴的坐标)
    geometry_msgs::Pose2D::Ptr Car_Pose_back; // 车辆位姿信息(车后轴的坐标)

    int divide_num = 2; // 数据中插值多少点
    int total_points_num; //原始数据中共多少点

    bool flag_V_slow = true;

    pcl::PointCloud<pcl::PointXYZI>::Ptr line_record; //插值后的点
    pcl::PointCloud<pcl::PointXYZI>::Ptr line_record_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr line_record_opt;
//  pcl::PointCloud<pcl::PointXYZI>::Ptr line_qp_path(new pcl::PointCloud<pcl::PointXYZI>);  //存储计算osqp求解出的解
    pcl::PointCloud<pcl::PointXYZI>::Ptr line_qp_Interpolation; //增密后的点
    pcl::PointCloud<pcl::PointXYZI>::Ptr line_record_watch; // 此处设置了z为0的点，方便rviz观察
    pcl::PointCloud<pcl::PointXYZI>::Ptr local_path; // 此处设置了z为0的点，方便rviz观察
    pcl::PointCloud<pcl::PointXYZI>::Ptr local_qp_path; // qp——path rviz观测值
    pcl::PointCloud<pcl::PointXYZI>::Ptr obs_watch; // 障碍物观测

    //**倒车模块函数
    int car_direct=1; //1:前进  -1：倒车
    int car_wait=0; //车辆是否等待
    const float sample_s_reverse=-0.5;
    const float sample_l_reverse=0.3;

    const float row_node_num_reverse=5;
    const float col_node_num_reverse=5;

    int sum_forward_reverse_num=0; //倒车判断积累值
    int sum_back_reverse_num=0; //前进判断积累值

    int sum_flag_stop_forward_num=0;
    int sum_flag_stop_back_num=0;
    int sum_flag_cancel_reverse_num=0; //取消后退规划

    bool flag_is_first_reverse=true;
    bool flagCalcBackDistance=false;
    Eigen::Vector2d firstReversePosition; //第一次停车点
    float reverseDistance=0;
    const float minReverseDistance=0.6;   //最小停车范围
    const float maxReverseDistance=1.5;   //最大倒车距离

    bool reverseQpPathRunningNormally = true;

    const float w_cost_smooth_dl_r=20;
    const float w_cost_smooth_ddl_r=10;
    const float w_cost_smooth_dddl_r=20;
    const float w_cost_ref_r=100;
    const float w_cost_collision_r=1;

    // RL_DP
    bool use_rl_dp_=true;
    std::string rl_dp_model_path_;
    std::string rl_dp_disable_reason_;
    std::string rl_dp_last_fail_reason_;
    bool rl_dp_soft_fail_ = false;
    int rl_dp_fail_count_ = 0;
    std::unique_ptr<RL_DP> rl_dp_;
    float rl_dp_vehicle_scale_ = 1.0f;
    int rl_dp_s_samples_=0;
    int rl_dp_l_samples_=0;
    float rl_dp_s_min_=0.0f;
    float rl_dp_s_max_=0.0f;
    float rl_dp_l_min_=0.0f;
    float rl_dp_l_max_=0.0f;

    // Timing statistics
    double dp_time_total_ms_ = 0.0;
    double dp_time_max_ms_ = 0.0;
    double planner_time_total_ms_ = 0.0;
    double planner_time_max_ms_ = 0.0;
    long long timing_sample_count_ = 0;
    int timing_print_every_ = 1;
    bool has_latest_plan_result_ = false;
    bool latest_qp_running_normally_ = false;
    std::string latest_dp_source_ = "unknown";
    double latest_planner_cycle_ms_ = 0.0;
    double latest_dp_sampling_ms_ = 0.0;
    double latest_qp_optimization_ms_ = 0.0;
    double latest_speed_planning_ms_ = 0.0;
    bool latest_speed_plan_available_ = false;
    Frenet_path_points latest_dp_path_sl_;
    Frenet_path_points latest_qp_path_sl_;
    std::vector<Eigen::Vector2d> latest_dp_path_xy_;
    std::vector<Eigen::Vector2d> latest_qp_path_xy_;
    planning_msgs::car_path latest_qp_path_msg_;
    Speed_plan_points latest_speed_qp_points_;
    std::vector<double> latest_speed_dp_path_s_;
    int latest_speed_dp_last_feasible_index_ = -1;
    Speed_Plan_DP_ST_nodes latest_speed_dp_st_nodes_;
    planning_msgs::ObstacleList latest_speed_obstacle_list_qp_path_sl_;
    int speed_dp_last_feasible_index_ = -1;

    //Path_plan
    Eigen::VectorXd Smooth_Reference_Line(pcl::PointCloud<pcl::PointXYZI>::Ptr line_record_);
    void Mean_Interpolation(pcl::PointCloud<pcl::PointXYZI>::Ptr line_record_opt,pcl::PointCloud<pcl::PointXYZI>::Ptr line_record);
    void Path_Parameter_Calculate(planning_msgs::car_path &path);
    pcl::PointCloud<pcl::PointXYZI>::Ptr Calculate_OptimizedPath_Heading_not_frist(const vector<Eigen::Vector2d> &QPSolution, vector<float> &temp_phi);//通过优化后的路径来计算航向角
    void Calculate_OptimizedPath_Heading(const Eigen::VectorXd &QPSolution, vector<float> &temp_phi);//通过优化后的路径来计算航向角
    void get_back_point_and_setVel(planning_msgs::car_path &path_, int begin);
    void get_front_point_and_setVel(planning_msgs::car_path &path_, int begin);
    int get_closest_point(float x, float y,planning_msgs::car_path &path_, int star=0, int pre_index=0); //寻找最近点
    int get_aim_point(float distance, planning_msgs::car_path &path_, int begin, bool sequence); //搜索距离一定距离的点
    float revise_angle(float angle);
    int index2s(planning_msgs::car_path &path, planning_msgs::path_point point_, int pre_index2s_id=0);
    int index2s(planning_msgs::car_path &path, float absolute_s, int pre_index2s_id=0);
    Eigen::VectorXd CalculateFiveDegreePolynomialCoefficients(float start_l, float start_dl, float start_ddl, float end_l, float end_dl, float end_ddl, float start_s, float end_s);
    Eigen::VectorXd CalculateThreeDegreePolynomialCoefficients(float start_l, float start_dl, float start_ddl, float end_l, float end_dl, float end_ddl, float start_s, float end_s);

    double CalculateStarCost(float begin_s, float begin_l, float begin_dl, float begin_ddl,float end_S,float end_L);
    double CalculateForwardCost(float pre_node_s, float pre_node_l, float current_node_s, float current_node_l,float host_start_s);
    double CalcObstacleCost(planning_msgs::Obstacle &Obstacle_, float aim_s, float aim_l,float host_start_s,bool &is_collsion);
    vector<Eigen::Vector2d>  FrenetToXY(const Frenet_path_points &frenet_path_points, planning_msgs::car_path &path);
    planning_msgs::path_point find_projected_point_Frenet(planning_msgs::car_path &path, int index_match_point,float point_s,float point_l);
    planning_msgs::path_point find_projected_point_Frenet(planning_msgs::car_path &path, int index_match_point, Eigen::Vector2d host_point);
    Frenet_path_points InterpolatePoints(const Min_path_nodes &min_path_nodes,float host_start_s);
    Min_path_nodes CalcNodeMinCost(float plan_star_s, float plan_star_l,float start_dl);
    bool BuildRlDpMinPath(float plan_start_s, float plan_start_l, Min_path_nodes &min_path_nodes);
    float CrossProduct(Eigen::Vector2d AB,Eigen::Vector2d AC);
    float CrossProduct(Eigen::Vector2d AB,Eigen::Vector2d A,Eigen::Vector2d C);
    Eigen::Vector2d Find_start_sl(Eigen::Vector2d host_point, planning_msgs::path_point host_projected_point,int index_match_start);
    void calculatePathBoundary(planning_msgs::ObstacleList::Ptr &obstacle_list, const Frenet_path_points &DP_path_sl, Eigen::VectorXd &l_min, Eigen::VectorXd &l_max, float min_l, float max_l,float delta_s, float start_l);
    Frenet_path_points cacl_qp_path(float plan_start_s,float plan_start_l, float plan_start_dl, float plan_start_ddl);
    void Calculate_OptimizedPath_Heading(const vector<Eigen::Vector2d> &QPSolution, vector<float> &temp_phi);
    void QP_Path_Publish(planning_msgs::car_path::Ptr &path, pcl::PointCloud<pcl::PointXYZI>::Ptr line_qp_Interpolation);
    void Obstacle_list_Initialization(planning_msgs::ObstacleList::Ptr &Obstacle_list);
    void Obstacle_list_Initialization_vision(planning_msgs::ObstacleList::Ptr &Obstacle_list);

    Eigen::Vector2d calc_obstacle_sl(Eigen::Vector2d obstacle_point);
    void spliceTracks(Frenet_path_points path_history, Frenet_path_points &path_now, float s_splice_start, float s_splice_end);
    void Calculate_OptimizedPath_Heading(const vector<Eigen::Vector2d> &QPSolution, vector<float> &temp_phi,pcl::PointCloud<pcl::PointXYZI>::Ptr &line_qp_path);
    void Plot_SL_Graph(Eigen::Vector2d host_forward_sl,const Frenet_path_points &DP_path_sl,const Frenet_path_points &QP_path_sl);

    int windingNumber(Eigen::Vector2d A, Eigen::Vector2d B, Eigen::Vector2d P) ;
    bool pointInsideQuadrilateral(Eigen::Vector2d A, Eigen::Vector2d B, Eigen::Vector2d C, Eigen::Vector2d D, Eigen::Vector2d P);
    double calcDistance(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2);
    //speed_plan
    void Speed_plan_calc_obs_ST(planning_msgs::ObstacleList::Ptr &obstacle_list_qp_path_sl);
    Speed_Plan_DP_ST_nodes CalcSpeedPlanDp_StNodes(float plan_start_s);
    vector<double> CalcSpeedDpCost(planning_msgs::ObstacleList::Ptr &obstacle_list_qp_path_sl, Speed_Plan_DP_ST_nodes &DP_ST_nodes);
    Speed_plan_points CalcSpeedPlan_QpPath(planning_msgs::ObstacleList::Ptr &obstacle_list_qp_path_sl, Speed_Plan_DP_ST_nodes &DP_ST_nodes, vector<double> best_speed_path_s,double s_start);
    void Obstacle_list_Initialization_qp_path(planning_msgs::ObstacleList::Ptr &Obstacle_list,Eigen::Vector2d planning_frist_star);
    Eigen::Vector2d calc_obstacle_sl(Eigen::Vector2d obstacle_point,planning_msgs::car_path::Ptr &path,Eigen::Vector2d planning_frist_star);
    double Speed_Dp_collision_cost(double min_dis);
    double CalcSpeedDp_ObsCost(const planning_msgs::ObstacleList::Ptr &obstacle_list_qp_path_sl, Speed_Plan_DP_ST_nodes &DP_ST_nodes, int col_start, int col_end, int rol_start,int rol_end);
    void Plot_ST_Graph(Speed_plan_points speed_qp_points,vector<double> Speed_DP_path_s,planning_msgs::ObstacleList::Ptr &obstacle_list_qp_path_sl);
    void Calc_QP_path_param(planning_msgs::car_path::Ptr &QP_path);

    //Temp临时函数声明
    Eigen::Vector2d Find_start_sl_FirstRun(Eigen::Vector2d host_point, planning_msgs::path_point host_projected_point,int index_match_start);
    int index2s(const Frenet_path_points &path, float host_s);
    Eigen::Vector2d ObstacleSL2XY(planning_msgs::Obstacle obs);

    //DP_PATH
    Min_path_nodes CalcDpPathNodeMinCost_reverse(float plan_star_s, float plan_star_l);
    double CalculateDpPathStarCost_reverse(float begin_s, float begin_l, float begin_dl, float begin_ddl, float end_S, float end_L);
    double CalculateDpPathForwardCost_reverse(float pre_node_s, float pre_node_l, float current_node_s, float current_node_l,float host_start_s);
    Frenet_path_points InterpolateDpPathPoints_reverse(const Min_path_nodes &min_path_nodes,float host_start_s);
    //QP_PATH
    Frenet_path_points calcQpPath_reverse(float plan_start_s, float plan_start_l, float plan_start_dl, float plan_start_ddl);
    int index2s_r(planning_msgs::car_path &path, planning_msgs::path_point point_, int pre_index2s_id);
    void Calculate_OptimizedPath_Heading_reverse(const vector<Eigen::Vector2d> &QPSolution, vector<float> &temp_phi, pcl::PointCloud<pcl::PointXYZI>::Ptr &line_qp_path);

    vector<Eigen::Vector2d> FrenetToXY_r(const Frenet_path_points &frenet_path_points, planning_msgs::car_path &path);
};



#endif
