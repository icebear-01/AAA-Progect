#ifndef LATTICE_PLANNER__H
#define LATTICE_PLANNER__H
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <message_filters/subscriber.h>
#include <sensor_msgs/PointCloud2.h>
#include <ros/package.h>
#include <Eigen/Dense>
#include "yaml-cpp/yaml.h"
#include <planning_msgs/car_info.h>
#include <planning_msgs/car_path.h>
#include "planning_msgs/Obstacle.h"
#include "planning_msgs/ObstacleList.h"
#include "obj_msgs/ObstacleList.h"
using namespace std;
pcl::PointCloud<pcl::PointXYZI>::Ptr line_global_astar(new pcl::PointCloud<pcl::PointXYZI>); //插值后的点
pcl::PointCloud<pcl::PointXYZI>::Ptr line_record(new pcl::PointCloud<pcl::PointXYZI>); //插值后的点
pcl::PointCloud<pcl::PointXYZI>::Ptr line_record_watch(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr LateralTrajectory(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr line_record_(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr line_record_opt(new pcl::PointCloud<pcl::PointXYZI>);
planning_msgs::ObstacleList::Ptr obstacleList_lidar(new planning_msgs::ObstacleList);
planning_msgs::ObstacleList::Ptr obstacleList_camera(new planning_msgs::ObstacleList);
planning_msgs::ObstacleList::Ptr obstacleList_total(new planning_msgs::ObstacleList);

planning_msgs::ObstacleList::Ptr obstacleList_long(new planning_msgs::ObstacleList);//用于更新A*全局路径规划
planning_msgs::car_path global_path; //通过txt读取的全局路径规划
planning_msgs::car_path ReferenceLine; //发布的局部路径规划

int divide_num = 3; // 数据中插值多少点
int total_points_num;//原始数据中共多少点

bool flag_line_change = false;
// 速度控制
float V_straight = 0.5;
float target_v = 0;
float V_turn = 1;
float end_slow_distance = 11;
float star_slow_distance = 18; // 减速缓冲距离从star_slow_distance至end_slow_distance，速度由V_straight减到V_turn
float end_speed_distance = 10;
float star_speed_distance = 6; // 加速缓冲距离从star_speed_distance至star_speed_distance，速度由V_turn加到V_straight
float speed_limit = 1.0;
//lattice参数
planning_msgs::path_point car_point; //车自身点
planning_msgs::car_info::Ptr car_info(new planning_msgs::car_info);
float car_ddaw = 0; //车身与匹配点的偏航角的差值
float front_plan_length = 6.0; //前面障碍物规划的长度
float back_plan_length = 0.0; //后面障碍物规划的长度
float lane_width = 4.0; //路宽
float trajectory_time_length = 2.0; //ST图时间长度

float long_flag = 1.0; //判断障碍物是否过长
float longitudinal_safe_distance = 0.5;

float K_offsets = 2.0; //横向cost的放大系数
float K_comfort = 1.0; //舒适度cost的放大系数
float obstacle_safe_range = 0.25+0.2;

float sampling_max_l = 2.0; //横向采样最大的L

float curv_max = 1/1.6; //线段的最大曲率；


enum State{
  normal,
  replanning
};
State car_state;

struct PointXY{
    float x;
    float y;
};
PointXY goal_point;
struct STPoint{
    float s;
    float t;
};

struct ST_graph_single{
    bool isStatic;
    vector<STPoint> Max_s_point;
    vector<STPoint> Min_s_point;
    STPoint left_upper_point;
    STPoint left_bottom_point;
    STPoint right_upper_point;
    STPoint right_bottom_point;
    float s_vel;
};

struct ST_graph{
    vector<ST_graph_single> ST_graph_node;
};


struct trajectory1d{
    std::array<double, 4> coef4_ = {{0.0, 0.0, 0.0, 0.0}}; //三次多项式系数
    std::array<double, 5> coef5_ = {{0.0, 0.0, 0.0, 0.0, 0.0}}; //四次多项式系数
    std::array<double, 2> start_condition_ = {{0.0, 0.0}}; //初始状态的s，ds 或l dl/ds
    std::array<double, 2> end_condition_ = {{0.0, 0.0}}; //结束状态的s，ds 或l dl/ds
    double target_vel;
    double target_time; //用于纵向采样 s,t
    double target_s; //用于横向采样 l,s
    // double target_l;
    float cost_crash = 0;
    float cost_lateral_offsets = 0;
    float cost_lateral_comfort = 0;
    float cost = 0;
};

trajectory1d trajectory_accessible;//最近一次可通行的点
typedef vector<trajectory1d> Trajectory1DBundle;
class lattice_planner
{
public:
    fstream file_;
    string FILE_NAME;
    ros::Subscriber line_astar_sub;
    ros::Subscriber car_pos_sub;
    ros::Subscriber car_info_sub;
    ros::Subscriber obstacleList_lidar_sub;
    ros::Subscriber obstacleList_camera_sub;
    ros::Publisher line_record_pub;
    ros::Publisher local_path_pub;
    ros::Publisher referenceline_pub;
    ros::Publisher long_obstacle_pub;
    void read_path(string file);
    void Plan(planning_msgs::car_info::ConstPtr car_pos);
    void pub_cloud(pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud, ros::Publisher Publisher);
    void pub_referenceline(planning_msgs::car_path reference, ros::Publisher Publisher);
    void pub_goal_point(planning_msgs::ObstacleList ObstacleList_long, ros::Publisher Publisher);
    void callBack_Info(const planning_msgs::car_info::ConstPtr &car_info);
    void callBack_Pos(const planning_msgs::car_info::ConstPtr &car_pose);
    void callBack_line_astar(const sensor_msgs::PointCloud2 &line_astar);
    void callBack_obstacleLidar(const planning_msgs::ObstacleList::ConstPtr &obstacleList_lidar);
    void callBack_obstacleCamera(const obj_msgs::ObstacleList::ConstPtr &obstacleList_camera);
    lattice_planner(ros::NodeHandle &nh);
    ~lattice_planner();
};

void Calculate_OptimizedPath_Heading(const Eigen::VectorXd &QPSolution, vector<float> &temp_phi);
void Mean_Interpolation(pcl::PointCloud<pcl::PointXYZI>::Ptr line_record_opt, pcl::PointCloud<pcl::PointXYZI>::Ptr line_recordss);
void Path_Parameter_Calculate(planning_msgs::car_path &path);
void get_front_point_and_setVel(planning_msgs::car_path &path_, int begin);
std::array<double, 4> ComputeCoefficients(double x0, double dx0, double dx1, double param); //默认最后加速度为零
std::array<double, 4> ComputeCoefficients_lateral(double x0, double dx0, double x1, double t); //默认最后dl/ds为零
std::array<double, 5> ComputeCoefficients(double x0, double dx0, double x1, double dx1, double t);
float ComputeS(trajectory1d trajectory1d);

planning_msgs::path_point MatchToPath(planning_msgs::car_path global_path , float x ,float y);
planning_msgs::path_point MatchToPath(planning_msgs::car_path global_path , float s);
float cal_distance_square(planning_msgs::path_point point , float x , float y);
planning_msgs::path_point FindProjectionPoint(planning_msgs::path_point p0,planning_msgs::path_point p1, float x, float y);
planning_msgs::path_point InterpolateUsingLinearApproximation(const planning_msgs::path_point &p0,
                                                              const planning_msgs::path_point &p1,
                                                              const float s) ;
float slerp(const float a0, const float t0, const float a1, const float t1, const float t);
float NormalizeAngle(const float angle) ;
void cartesian_to_frenet(const planning_msgs::path_point matched_point,planning_msgs::path_point &car_point,
                                             std::array<double, 2> &init_s,std::array<double, 2> &init_d);
ST_graph obstacleList_Preprocessing(planning_msgs::ObstacleList &ObstacleList_static);
void SetStaticObstacle(planning_msgs::Obstacle &Obstacle, planning_msgs::car_path global_path, ST_graph &STGraph);
void SetDynamicObstacle(planning_msgs::Obstacle &Obstacle, planning_msgs::car_path global_path, ST_graph &STGraph);
void Obstacle_init(planning_msgs::Obstacle &Obstacle, planning_msgs::car_path global_path,bool static_flag);
void ComputeObstacleBoundary(planning_msgs::Obstacle &Obstacle, planning_msgs::car_path global_path);
PointXY getCpFromBbox(const planning_msgs::Obstacle Obstacle);

void GenerateTrajectoryBundles(std::array<double, 2> init_s,std::array<double, 2> init_d,ST_graph STGraph,
                              Trajectory1DBundle &lon_trajectory1d_bundle,Trajectory1DBundle &lat_trajectory1d_bundle);
void GenerateLongitudinalTrajectoryBundle(std::array<double, 2> init_s,ST_graph STGraph,Trajectory1DBundle &lon_trajectory1d_bundle);
void GenerateSpeedProfilesForCruising(std::array<double, 2> init_s,Trajectory1DBundle &lon_trajectory1d_bundle);
void GenerateSpeedProfilesForPathTimeObstacles(std::array<double, 2> init_s,ST_graph STGraph,Trajectory1DBundle &lon_trajectory1d_bundle);

void GenerateLateralTrajectoryBundle(std::array<double, 2> init_d,Trajectory1DBundle &lat_trajectory1d_bundle);
void ShowReferenceLine(planning_msgs::car_path ReferenceLine,pcl::PointCloud<pcl::PointXYZI>::Ptr LateralTrajectory);
void ShowLateralTrajectoryBundle(Trajectory1DBundle lat_trajectory1d_bundle,pcl::PointCloud<pcl::PointXYZI>::Ptr LateralTrajectory);
void ShowBestLateralTrajectory(trajectory1d trajectory1d,pcl::PointCloud<pcl::PointXYZI>::Ptr LateralTrajectory);
void trajectory_evaluate(Trajectory1DBundle &lat_trajectory1d_bundle , planning_msgs::ObstacleList ObstacleList_static);
void trajectory_feasibility_test(Trajectory1DBundle &lat_trajectory1d_bundle);
float Calcost_lateral_offsets(trajectory1d trajectory1d);
float Calcost_lateral_comfort(trajectory1d trajectory1d);
float Calcost_crash(trajectory1d trajectory1d,planning_msgs::ObstacleList ObstacleList_static);
void GenerateReferenceLine(trajectory1d trajectory1d,planning_msgs::car_path &ReferenceLine);
bool get_newGoalPoint();
#endif