/*************************************
 Astar算法
 2021.9.30
 *************************************/
#include <ros/ros.h>
#include <string.h>
#include "std_msgs/String.h"
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_broadcaster.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/ndt.h>//NDT配准类对应头文件
#include <pcl/visualization/pcl_visualizer.h>//可视化头文件
#include <boost/thread/thread.hpp>//多线程相关头文件
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <sensor_msgs/PointCloud2.h>
#include <iostream>              //标准C++库中的输入输出的头文件
#include <pcl/visualization/cloud_viewer.h>  //点类型相关定义
#include "stdlib.h"
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <tf/transform_listener.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/passthrough.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <tf/transform_datatypes.h>
#include <nav_msgs/Odometry.h>
#include <pcl/filters/filter.h>
#include <sstream>
#include <fstream>
#include <string>
#include <chrono>
#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
//#include <velodyne_pointcloud/point_types.h>
//#include <velodyne_pointcloud/rawdata.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Pose2D.h>
#include <tf/tf.h>
#include <thread>
#include <pcl/io/io.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <vector>
#include <list>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <tf/transform_listener.h>
#include <pcl_ros/impl/transforms.hpp>
#include <sensor_msgs/point_cloud_conversion.h>
#include <pcl/io/pcd_io.h>
#include <a_star/spline.h>
#include <sensor_msgs/Imu.h>
#include "yaml-cpp/yaml.h"
#include <planning_msgs/car_info.h>
#include "planning_msgs/Obstacle.h"
#include "planning_msgs/ObstacleList.h"
#include <ros/package.h>
#include <memory>
using namespace std;

const std::string RED = "\033[31m";//红色
const std::string RESET = "\033[0m";

const int kCost1 = 10; //直移一格消耗，常量限定符；
const int kCost2 = 14; //斜移一格消耗
#define ugv_half_disgonal 0.2    //无人车质心到无人车边缘最远距离，宏定义；
#define ugv_safe_dis 0.2          //安全距离
#define wheeldistance 0.65
#define FRONT 0.327
#define BACK 0.653
#define LEFTRIGHT 0.359
#define MIN_DISTANCE 0.6
#define max_distance_ 50
#define CLIP_HEIGHT 0.2
#define RADIAL_DIVIDER_ANGLE 0.18
#define SENSOR_HEIGHT 1.05
#define concentric_divider_distance_ 0.01 //0.1 meters default
#define min_height_threshold_ 0.05
#define local_max_slope_ 18
//max slope of the ground between points, degree
#define general_max_slope_ 15 //max slope of the ground in entire point cloud, degree
#define reclass_distance_threshold_ 0.1
#define points_number 3
#define LEAF_SIZE 0.2
ros::Publisher bound_pub,cluster_map_pub,line_pub,line_pub_test;
pcl::PointCloud<pcl::PointXYZI>::Ptr clusterCloud(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr bound_1(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr line_global(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr line_global_test(new pcl::PointCloud<pcl::PointXYZI>);

sensor_msgs::PointCloud2 clusterCloud_output;
sensor_msgs::PointCloud2 bound_output;
sensor_msgs::PointCloud2 line_global_out;
sensor_msgs::PointCloud2 line_global_out_test;

std::vector<std::vector<int>> MapData;
struct Point
{
  int x, y; //点坐标，这里为了方便按照C++的数组来计算，x代表横排，y代表竖列
  int F, G, H; //F=G+H
  double timein, timeout;//进出每个节点的时间
  Point *parent; //parent的坐标，这里没有用指针，从而简化代码
  Point(int _x, int _y) :x(_x), y(_y), F(0), G(0), H(0), timein(0), timeout(0), parent(NULL)  //变量初始化
  {
  }
};

enum State{
  normal,
  replanning
};
State car_state;
class astar
{
  private:
    // ros::NodeHandle nh;//创建节点句柄
    std_msgs::Header point_cloud_header_;
    ros::Subscriber  sub_Goal, sub_Start, sub_carPos ,sub_reGoal;//订阅
    tf::TransformListener *tf_listener;//定义一个监听器

    std::vector<pcl::PointIndices> local_indices;
    std::vector<std::vector<int>> maze;
    std::vector<double> seg_distance_, cluster_distance_, h, vec_derta_d, s;
    std::list<Point *> openList;  //开启列表
    std::list<Point *> closeList; //关闭列表

    pcl::PointXYZI goal_point;
    pcl::PointXYZI initial_point;

    float min_x_map = 0;
    float min_y_map = 0;
    float max_x_map = 0;
    float max_y_map = 0;
    int numGrid_x = 0;
    int numGrid_y = 0;
    const float grid_size = 0.25;

    

    int LOCK_setgoal=0, LOCK_setstart=0, LOCK_global_path=0, socket_fd;
    int client_sockfd, countSet = 500;
    double grid_L, map_range_min_x, map_range_min_y, map_range_max_x, map_range_max_y;
    ros::Time time_1, time_2;
    tk::spline s1;
    tk::spline s2;

    int grid_num_x, grid_num_y;
    pcl::PointCloud<pcl::PointXYZI>::Ptr line;
    pcl::PointCloud<pcl::PointXYZI>::Ptr map_Astar;


    //这里我们定义了一个结构体 Detected_Obj ，用于存储检测到的障碍物的信息
    struct Detected_Obj
    {
      jsk_recognition_msgs::BoundingBox bounding_box_;

      pcl::PointXYZI min_point_;
      pcl::PointXYZI max_point_;
      pcl::PointXYZI centroid_;
    };

     struct location_posture
    {
        pcl::PointXYZI car_pos;
        double roll;
        double pitch;
        double yaw;
        double fitness_score;
        Eigen::Matrix4f transfer_matrix;//4*4的float型矩阵；
    };
    location_posture car_position_posture;

    struct AStarTimer
    {
        double Timerin;
        double Timerout;
    };

  public:
      astar(ros::NodeHandle nh)
      {
        sub_Goal = nh.subscribe("/move_base_simple/goal", 1, &astar::Goalcallback, this); //设定A*的目标位置
        sub_Start = nh.subscribe("/initialpose", 1, &astar::Startcallback, this); //设定A*的初始位置
        sub_carPos = nh.subscribe<planning_msgs::car_info>("/car_pos",10, &astar::Poscallback, this);
        sub_reGoal = nh.subscribe<planning_msgs::ObstacleList>("long_obstacleList",10, &astar::reGoalcallback, this);
         parameter_initialization();//初始化的函数
         environment_modeling();//建立环境模型的函数
         
        cout<<"environment modeling successful!"<<endl;
        time_1= ros::Time::now();
        time_2= ros::Time::now();
        
      }

    void parameter_initialization();

    void componentClustering(pcl::PointCloud<pcl::PointXYZI>::Ptr elevatedCloud,std::vector<std::vector<int>> &cartesianData);

    void merge_Clustered(std::vector<std::vector<int>> &cartesianData) ;

    void expansion_Clustered(std::vector<std::vector<int>> &cartesianData);

    void makeClusteredCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr& elevatedCloud,std::vector<std::vector<int>> &cartesianData,
                            pcl::PointCloud<pcl::PointXYZI>::Ptr& clusterCloud);

    void makeClusteredCloud(std::vector<std::vector<int>> &cartesianData,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr& clusterCloud); 

    void insertPoints(const pcl::PointXYZI& start, const pcl::PointXYZI& end, float step,
                             pcl::PointCloud<pcl::PointXYZI>::Ptr& points);
                                         
    void Poscallback(const planning_msgs::car_info::ConstPtr &car_pose);

    void Goalcallback(const geometry_msgs::PoseStamped::ConstPtr& end);

    void Startcallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& initial);

    void reGoalcallback(const planning_msgs::ObstacleList::ConstPtr& long_obstacleList);

    Point *findPath(Point &startPoint, Point &endPoint, bool isIgnoreCorner);

    std::vector<Point *> getSurroundPoints(const Point *point, bool isIgnoreCorner) const;

    bool isCanreach(const Point *point, const Point *target, bool isIgnoreCorner) const; //判断某点是否可以用于下一步判断

    Point *isInList(const std::list<Point *> &list, const Point *point) const; //判断开启/关闭列表中是否包含某点

    Point *getLeastFpoint(); //从开启列表中返回F值最小的节点

    //计算FGH值
    int calcG(Point *temp_start, Point *point);
    int calcH(Point *point, Point *end);
    int calcF(Point *point);

    void InitAstar(std::vector<std::vector<int>> &_maze);

    std::list<Point *> GetPath(Point &startPoint, Point &endPoint, bool isIgnoreCorner);

    void environment_modeling();
    
     void Astar(pcl::PointXYZI startPt, pcl::PointXYZI endPt);
    
    void pub_cloud(pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud, ros::Publisher Publisher);
    
    ~astar(){
       
    }
};

void astar::parameter_initialization()    //参数初始化
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr temp8(new pcl::PointCloud<pcl::PointXYZI>);
    line=temp8;
    pcl::PointCloud<pcl::PointXYZI>::Ptr temp14(new pcl::PointCloud<pcl::PointXYZI>);
    map_Astar=temp14;

}
  
   //环境建模，标示地图已有障碍物
void astar::environment_modeling()
{ 
  cout <<"地图初始化"<<endl;

    char current_absolute_path[100000];
    if (NULL == realpath("./", current_absolute_path))
    {
        printf("***Error***\n");
        exit(-1);
    }
    strcat(current_absolute_path, "/");
    printf("current absolute path:%s\n", current_absolute_path);
    //////////////////////////////////////////////////////////////////////////////////////////////////////

    std::string ugv_path = ros::package::getPath("ugv_position");

    std::string yaml_path1 =ugv_path+"/config.yaml";
    YAML::Node config_path = YAML::LoadFile(yaml_path1);
        //加载地图
    std::string pcd_astar_path = ugv_path+"/"+config_path["ASTAR_PCD_PATH"].as<std::string>();

    pcl::io::loadPCDFile(pcd_astar_path, *map_Astar);
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = -std::numeric_limits<float>::max();
    float max_y = -std::numeric_limits<float>::max();

    for(int i=0; i<map_Astar->points.size(); i++)//i小于地图点云数量进行循环
    {
         if(map_Astar->points[i].x<min_x)//遍历出地图上点的最小x坐标
         {
            min_x=map_Astar->points[i].x;
            map_range_min_x=map_Astar->points[i].x;
         }
           if(map_Astar->points[i].y<min_y)//遍历出地图上点的最小y坐标
         {
            min_y=map_Astar->points[i].y;
            map_range_min_y=map_Astar->points[i].y;
         }
         if(map_Astar->points[i].x>max_x)//遍历出地图上点的最大x坐标
         {
           max_x=map_Astar->points[i].x;
           map_range_max_x=map_Astar->points[i].x;
         }
           if(map_Astar->points[i].y>max_y)//遍历出地图上点的最大y坐标
         {
           max_y=map_Astar->points[i].y;
           map_range_max_y=map_Astar->points[i].y;
         }
    }
    cout <<"map_range_min_x map_range_min_x map_range_min_x map_range_min_x:"
         <<map_range_min_x<<" "<<map_range_min_y<<" "<<map_range_max_x<<" "<<map_range_max_y<<endl;
    // map_range_min_x = floor(map_range_min_x/grid_size)*grid_size;
    // map_range_min_y = floor(map_range_min_y/grid_size)*grid_size;
    // map_range_max_x = floor(map_range_max_x/grid_size)*grid_size;
    // map_range_max_y = floor(map_range_max_y/grid_size)*grid_size;

    // min_x_map = floor(min_x/grid_size);
    // min_y_map = floor(min_y/grid_size);
    // max_x_map = floor(max_x/grid_size);
    // max_y_map = floor(max_y/grid_size);

    min_x_map = min_x/grid_size;
    min_y_map = min_y/grid_size;
    max_x_map = max_x/grid_size;
    max_y_map = max_y/grid_size;
  
    numGrid_x = max_x_map - min_x_map;
    numGrid_y = max_y_map - min_y_map;
    grid_L = 0.25;
    
    std::vector<std::vector<int>> cartesianData(numGrid_x, std::vector<int>(numGrid_y, 0));
    componentClustering(map_Astar, cartesianData);
    MapData = cartesianData;
    // std::cout << std::endl;
    // for (size_t i = 0; i < numGrid_x; ++i) {  
    //   for (size_t j = 0; j < numGrid_y; ++j) {  
    //       // 使用setw设置宽度，并输出元素  
    //       if (MapData[i][j])
    //       {
    //         std::cout << RED<< std::setw(2) << MapData[i][j]<< RESET;  
    //       }else{
    //         std::cout << std::setw(2) << MapData[i][j];  
    //       }   
    //   }  
    //   std::cout << std::endl; // 换行  
    // }
    // std::cout << std::endl;
    expansion_Clustered(cartesianData);
    makeClusteredCloud(cartesianData,clusterCloud);
    //     for (size_t i = 0; i < grid_num_x; ++i) {  
    //     for (size_t j = 0; j < grid_num_y; ++j) {  
    //         // 使用setw设置宽度，并输出元素  
    //         std::cout << std::setw(2) << env_modle[i][j];  
    //     }  
    //     std::cout << std::endl; // 换行  
    //     }
    // cout<<endl;
    //    for (size_t i = 0; i < numGrid_x; ++i) {  
    //   for (size_t j = 0; j < numGrid_y; ++j) {  
    //       // 使用setw设置宽度，并输出元素  
    //       if (cartesianData[i][j])
    //       {
    //         std::cout << RED<< std::setw(2) << cartesianData[i][j]<< RESET;  
    //       }else{
    //         std::cout << std::setw(2) << cartesianData[i][j];  
    //       }   
    //   }  
    //   std::cout << std::endl; // 换行  
    // }
    InitAstar(cartesianData);//把栅格化给A*算法
    printf("obj_list_map is completed\n");  
    
     
}

void astar::componentClustering(pcl::PointCloud<pcl::PointXYZI>::Ptr elevatedCloud,std::vector<std::vector<int>> &cartesianData)
{

    std::vector<std::vector<int>> gridNum(numGrid_x, std::vector<int>(numGrid_y, 0));
    // elevatedCloud 映射到笛卡尔坐标系 //并 统计落在这个grid的有多少个点！！！
    for(int i = 0; i < elevatedCloud->size(); i++){  // 遍历高点数
        float x = elevatedCloud->points[i].x;   // x(-15, -5),y(-50, 50)
        float y = elevatedCloud->points[i].y;
        float xC = x-min_x_map*grid_size;   // float roiM = 50;(0~50)
        float yC = y-min_y_map*grid_size; // (0~50)
        // exclude outside roi points  排除外部roi points  x,y属于(-25, 25)下面才继续执行
        if(xC < 0 || xC >= numGrid_x*grid_size || yC < 0 || yC >=numGrid_y*grid_size) continue; // continue后，下面不执行。gridNum[xI][yI] 值不变 限制范围（0，50）
        // int xI = floor(xC/grid_size);   //  xI .yI    const int numGrid = 250;    floor(x)返回的是小于或等于x的最大整数
        // int yI = floor(yC/grid_size);   // 50x50 映射到→250x250

        int xI = xC/grid_size;
        int yI = yC/grid_size;
        gridNum[xI][yI] +=  1;  // 统计落在这个grid的有多少个点！！！
    }
    //     for (size_t i = 0; i < numGrid_x; ++i) {  
    //     for (size_t j = 0; j < numGrid_y; ++j) {  
    //         // 使用setw设置宽度，并输出元素  
    //         std::cout << std::setw(2) << gridNum[i][j];  
    //     }  
    //     std::cout << std::endl; // 换行  
    // } 
      //   for (size_t i = 0; i < numGrid_x; ++i) {  
      // for (size_t j = 0; j < numGrid_y; ++j) {  
      //     // 使用setw设置宽度，并输出元素  
      //     if (gridNum[i][j])
      //     {
      //       std::cout << RED<< std::setw(2) << gridNum[i][j]<< RESET;  
      //     }else{
      //       std::cout << std::setw(2) << gridNum[i][j];  
      //     }   
      // }  
      // std::cout << std::endl; // 换行  
    // }
    // cout<<numGrid_x<<" "<<numGrid_y<<endl;
// 将x，y位置的单个单元格选作中心单元格，并且clusterID计数器加1。
// 然后所有相邻的相邻像元（即x-1，y  + 1，x，y +1，x +1，y +1 x -1，y，x +1，y，x -1，y -1，x，检查y − 1，x + 1，y +  1）的占用状态，并用当前集群ID标记。
// 对m×n网格中的每个x，y重复此过程，直到为所有非空群集分配了ID。
    for(int xI = 0; xI < numGrid_x; xI++){  //   const int numGrid = 250; 
        for(int yI = 0; yI < numGrid_y; yI++){
            if(gridNum[xI][yI] >1){  // 一个点的直接舍弃？
                cartesianData[xI][yI] = 1;   // 网格分配有2种初始状态，分别为空（0），已占用（-1）和已分配。随后，将x，y位置的单个单元格选作中心单元格，并且clusterID计数器加1
                //下面为设置当前点的周围点数值为-1
                if(xI == 0)
                {
                    if(yI == 0)
                    {
                        cartesianData[xI+1][yI] = 1;  // 角相邻的3个相邻像元
                        cartesianData[xI][yI+1] = 1;
                        cartesianData[xI+1][yI+1] = 1;
                    }
                    else if(yI < numGrid_y - 1)
                    {
                        cartesianData[xI][yI-1] = 1;  // 边有5个相邻点
                        cartesianData[xI][yI+1] = 1;
                        cartesianData[xI+1][yI-1] = 1;
                        cartesianData[xI+1][yI] = 1;
                        cartesianData[xI+1][yI+1] = 1;
                    }
                    else if(yI == numGrid_y - 1)  // 角相邻的3个相邻像元
                    {
                        cartesianData[xI][yI-1] = 1; 
                        cartesianData[xI+1][yI-1] = 1;
                        cartesianData[xI+1][yI] = 1;    
                    }
                }
                else if(xI < numGrid_x - 1)
                {
                    if(yI == 0)
                    {
                        cartesianData[xI-1][yI] = 1;
                        cartesianData[xI-1][yI+1] = 1;
                        cartesianData[xI][yI+1] = 1;
                        cartesianData[xI+1][yI] = 1;
                        cartesianData[xI+1][yI+1] = 1;                
                    }
                    else if(yI < numGrid_y - 1)  // 一般情况四周有8个相邻点
                    {
                        cartesianData[xI-1][yI-1] = 1;
                        cartesianData[xI-1][yI] = 1;
                        cartesianData[xI-1][yI+1] = 1;
                        cartesianData[xI][yI-1] = 1;
                        cartesianData[xI][yI+1] = 1;
                        cartesianData[xI+1][yI-1] = 1;
                        cartesianData[xI+1][yI] = 1;
                        cartesianData[xI+1][yI+1] = 1;                  
                    }
                    else if(yI == numGrid_y - 1)
                    {
                        cartesianData[xI-1][yI-1] = 1;
                        cartesianData[xI-1][yI] = 1;
                        cartesianData[xI][yI-1] = 1;
                        cartesianData[xI+1][yI-1] = 1;
                        cartesianData[xI+1][yI] = 1;                 
                    } 
                }
                else if(xI == numGrid_x - 1)
                {
                    if(yI == 0)
                    {
                        cartesianData[xI-1][yI] = 1;
                        cartesianData[xI-1][yI+1] = 1;
                        cartesianData[xI][yI+1] = 1;
                    }
                    else if(yI < numGrid_y - 1)
                    {
                        cartesianData[xI-1][yI-1] = 1;
                        cartesianData[xI-1][yI] = 1;
                        cartesianData[xI-1][yI+1] = 1;
                        cartesianData[xI][yI-1] = 1;
                        cartesianData[xI][yI+1] = 1;
                    }
                    else if(yI == numGrid_y - 1)
                    {
                        cartesianData[xI-1][yI-1] = 1;
                        cartesianData[xI-1][yI] = 1;
                        cartesianData[xI][yI-1] = 1;    
                    }            
                }

            }      
        }
    }
}
void astar::merge_Clustered(std::vector<std::vector<int>> &cartesianData) 
{
  int rows = cartesianData.size();
    if (rows == 0) return;
    int cols = cartesianData[0].size();
    // 创建一个与原网格大小相同的结果网格，并初始化为0
    std::vector<std::vector<int>> result(rows, std::vector<int>(cols, 0));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (cartesianData[r][c] == 0&& MapData[r][c] == 0) { // 两个网络都没障碍物
                result[r][c] = 0;  
            }else{
                result[r][c] = 1;  
            }
        }
    }
    // 将结果复制回原网格
    cartesianData = result;
}
void astar::expansion_Clustered(std::vector<std::vector<int>> &cartesianData)
{
   int rows = cartesianData.size();
    if (rows == 0) return;
    int cols = cartesianData[0].size();

    // 创建一个与原网格大小相同的结果网格，并初始化为0
    std::vector<std::vector<int>> result(rows, std::vector<int>(cols, 0));

    // 定义相邻单元的坐标偏移量
    std::vector<std::pair<int, int>> directions = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1}, // 上、下、左、右
    };

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (cartesianData[r][c] == 1) { // 遇到障碍物
                // 设置自身为1
                result[r][c] = 1;
                // 设置周围单元为1
                for (const auto& dir : directions) {
                    int nr = r + dir.first;
                    int nc = c + dir.second;
                    if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                        result[nr][nc] = 1;
                    }
                }
            }
        }
    }

    // 将结果复制回原网格
    cartesianData = result;
  
}
void astar::makeClusteredCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr& elevatedCloud,
                        std::vector<std::vector<int>> &cartesianData,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr& clusterCloud){
    for(int i = 0; i < elevatedCloud->size(); i++){
        float x = elevatedCloud->points[i].x;
        float y = elevatedCloud->points[i].y;
        float z = elevatedCloud->points[i].z;
        float xC = x-min_x_map*grid_size;
        float yC = y-min_y_map*grid_size;
        // exclude outside roi points
        if(xC < 0 || xC >= numGrid_x*grid_size || yC < 0 || yC >=numGrid_y*grid_size) continue;
        // int xI = floor(xC/grid_size);   //  xI .yI    const int numGrid = 250;    floor(x)返回的是小于或等于x的最大整数
        // int yI = floor(yC/grid_size); 
        int xI = xC/grid_size;   //  xI .yI    const int numGrid = 250;    floor(x)返回的是小于或等于x的最大整数
        int yI = yC/grid_size; 
        // cout << "xI is "<< xI <<endl;
        // cout << "yI is "<< yI <<endl;
        // cout << "cartesianData is "<< cartesianData[xI][yI]<<endl;  //  (1,2,3,4,...,numCluster)
        int clusterNum = cartesianData[xI][yI]; //  数值  每一点云点对应的栅格聚类数字标签
        if(clusterNum != 0){
            pcl::PointXYZI o;
            o.x = grid_size*xI + min_x_map*grid_size ;  // 网格大小？？ roiM = 50  grid_size = (0.200000003F)
            o.y = grid_size*yI + min_y_map*grid_size ; // 转换成（-25  ~  25）范围
            o.z = 0;  // 高度统一设置为-1
            // o.r = (500*clusterNum)%255;   // 不同类不同颜色   error: ‘struct pcl::PointXYZ’ has no member named ‘r’，用pcl::PointXYZRGBA？
            // o.g = (100*clusterNum)%255;
            // o.b = (150*clusterNum)%255;
            clusterCloud->push_back(o); // 
        }
    }
}

void astar::makeClusteredCloud(std::vector<std::vector<int>> &cartesianData,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr& clusterCloud){
    int rows = cartesianData.size();
    if (rows == 0) return;
    int cols = cartesianData[0].size();
     for (int xI = 0; xI < rows; ++xI) {
        for (int yI = 0; yI < cols; ++yI) {
            if (cartesianData[xI][yI])
            {
              pcl::PointXYZI o;
              o.x = grid_size*xI + min_x_map*grid_size ;  // 网格大小？？ roiM = 50  grid_size = (0.200000003F)
              o.y = grid_size*yI + min_y_map*grid_size ; // 转换成（-25  ~  25）范围
              o.z = -1;  // 高度统一设置为-1
              clusterCloud->push_back(o); // 
            }
            
        }
     }
}

void astar::insertPoints(const pcl::PointXYZI& start, const pcl::PointXYZI& end, float step, pcl::PointCloud<pcl::PointXYZI>::Ptr& points) {
    float distance = std::sqrt(std::pow(end.x - start.x, 2) + std::pow(end.y - start.y, 2));
    int numSteps = static_cast<int>(distance / step);
    
    for (int i = 0; i <= numSteps; ++i) {
        pcl::PointXYZI point;
        point.x = start.x + (end.x - start.x) * (i / static_cast<float>(numSteps));
        point.y = start.y + (end.y - start.y) * (i / static_cast<float>(numSteps));
        points->push_back(point);
    }
}

void astar::Poscallback(const planning_msgs::car_info::ConstPtr &car_pose)
{
  car_position_posture.car_pos.x=car_pose->x;
  car_position_posture.car_pos.y=car_pose->y;
  car_position_posture.car_pos.z = car_pose->yaw;
  
  std::cout<<"x y yaw: "<<car_position_posture.car_pos.x<<" "<<car_position_posture.car_pos.y
           <<" "<<car_position_posture.car_pos.z<<std::endl;
}


void astar::Goalcallback(const geometry_msgs::PoseStamped::ConstPtr& end)
{
  // goal_point.x=floor(end->pose.position.x/grid_size)*grid_size;
  // goal_point.y=floor(end->pose.position.y/grid_size)*grid_size;
  goal_point.x=end->pose.position.x;
  goal_point.y=end->pose.position.y;
  goal_point.z=tf::getYaw(end->pose.orientation);
  cout<<"Goal:x ,y,yaw  "<<goal_point.x<<" "<<goal_point.y
  <<" "<<goal_point.z<<endl;
  Astar(car_position_posture.car_pos, goal_point);
  
}

void astar::reGoalcallback(const planning_msgs::ObstacleList::ConstPtr& long_obstacleList)
{
    car_state = replanning;
    pcl::PointCloud<pcl::PointXYZI>::Ptr newObstacle_to_map(new pcl::PointCloud<pcl::PointXYZI>);
    for (size_t i = 0; i < long_obstacleList->obstacles.size(); i++)
    {
      pcl::PointXYZI point_left_up;
      pcl::PointXYZI point_right_up;
      pcl::PointXYZI point_left_down;
      pcl::PointXYZI point_right_down;
      point_left_up.x = long_obstacleList->obstacles[i].bounding_boxs[0].x; //收集每个障碍物的四个角点
      point_left_up.y = long_obstacleList->obstacles[i].bounding_boxs[0].y;
      point_right_up.x = long_obstacleList->obstacles[i].bounding_boxs[1].x; 
      point_right_up.y = long_obstacleList->obstacles[i].bounding_boxs[1].y;
      point_left_down.x = long_obstacleList->obstacles[i].bounding_boxs[2].x; 
      point_left_down.y = long_obstacleList->obstacles[i].bounding_boxs[2].y;
      point_right_down.x = long_obstacleList->obstacles[i].bounding_boxs[3].x; 
      point_right_down.y = long_obstacleList->obstacles[i].bounding_boxs[3].y;
      insertPoints(point_left_up,point_right_up,0.1,newObstacle_to_map);
      insertPoints(point_right_up,point_right_down,0.1,newObstacle_to_map);
      insertPoints(point_right_down,point_left_down,0.1,newObstacle_to_map);
      insertPoints(point_left_down,point_left_up,0.1,newObstacle_to_map);
    }
    std::vector<std::vector<int>> cartesianData(numGrid_x, std::vector<int>(numGrid_y, 0));
    componentClustering(newObstacle_to_map, cartesianData);
    merge_Clustered(cartesianData);
    InitAstar(cartesianData);//把栅格化给A*算法
    pcl::PointXYZI new_goal_point;
    new_goal_point.x = long_obstacleList->goal_point.x;
    new_goal_point.y = long_obstacleList->goal_point.y;
    Astar(car_position_posture.car_pos, new_goal_point);
    InitAstar(MapData);//把栅格化给A*算法
    car_state = normal;
}

void astar::Startcallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& initial)
{
  // car_position_posture.car_pos.x=floor(initial->pose.pose.position.x/grid_size)*grid_size;
  // car_position_posture.car_pos.y=floor(initial->pose.pose.position.y/grid_size)*grid_size;

  car_position_posture.car_pos.x=initial->pose.pose.position.x;
  car_position_posture.car_pos.y=initial->pose.pose.position.y;
  car_position_posture.car_pos.z = tf::getYaw(initial->pose.pose.orientation);

  // std::cout<<"x y yaw: "<<car_position_posture.car_pos.x<<" "<<car_position_posture.car_pos.y
  //          <<" "<<car_position_posture.car_pos.z<<std::endl;
  
}


void astar::Astar(pcl::PointXYZI startPt, pcl::PointXYZI endPt)
{
  
    pcl::PointCloud<pcl::PointXYZI>::Ptr path_Astar(new pcl::PointCloud<pcl::PointXYZI>);

    int x_num_start=(startPt.x-map_range_min_x)/grid_L;
    int y_num_start=(startPt.y-map_range_min_y)/grid_L;
    int x_num_end=(endPt.x-map_range_min_x)/grid_L;
    int y_num_end=(endPt.y-map_range_min_y)/grid_L;

    if(maze[x_num_start][y_num_start])
    {
      cout<<"Move the car and reset the starting point!"<<endl;
      return;
    }
    if(maze[x_num_end][y_num_end])
    {
      cout<<"Reset the ending point!"<<endl;
      return;
    }
    Point start(x_num_start, y_num_start);
    Point end(x_num_end, y_num_end);
    list<Point *> path = GetPath(start, end, false);

    std::vector<Point *> PathPoints;
    for (auto &p : path)
    {
        PathPoints.push_back(new Point(p->x, p->y));
    }

    std::vector<Point *> PathPoints_node;
    PathPoints_node.push_back(PathPoints[0]); //起点栅格位置
    for(int i=1; i<PathPoints.size()-1; i++)  //根据两点斜率公式，忽略同一条直线上的点
    {
        double line_index=(PathPoints[i+1]->y-PathPoints[i-1]->y)*(PathPoints[i]->x-PathPoints[i-1]->x)
                          -(PathPoints[i]->y-PathPoints[i-1]->y)*(PathPoints[i+1]->x-PathPoints[i-1]->x);  //(y3-y1)/(x3-x1) = (y2-y1)/(x2-x1)
        if(line_index!=0)
        {
          PathPoints_node.push_back(PathPoints[i]);
        }
    }
    PathPoints_node.push_back(PathPoints[PathPoints.size()-1]); //终点栅格位置

    //   line_global->points.clear();
    // for (auto &p : PathPoints_node)
    // {
    //     pcl::PointXYZI point;
    //     point.x = p->x*grid_L+map_range_min_x;
    //     point.y = p->y*grid_L+map_range_min_y;
    //     point.z = 0;
    //     line_global->push_back(point);
    // }

    pcl::PointCloud<pcl::PointXYZI>::Ptr path_Astar_node(new pcl::PointCloud<pcl::PointXYZI>);
    int j, k;
    for(int i=0; i<PathPoints_node.size(); i++)      //再次去点，得到每个相邻的点的连线不经过障碍物
    {
        for(j=PathPoints_node.size()-1; j>i; j--)
        {
            double a_temp = PathPoints_node[i]->y-PathPoints_node[j]->y;
            double b_temp = PathPoints_node[j]->x-PathPoints_node[i]->x;
            double c_temp = PathPoints_node[i]->x*PathPoints_node[j]->y-PathPoints_node[j]->x*PathPoints_node[i]->y;
            double angle_temp = atan2(-a_temp, b_temp);

            double y_diff=PathPoints_node[j]->y-PathPoints_node[i]->y;
            double two_node_dis=sqrt(pow(b_temp, 2)+pow(y_diff, 2));
            double step_len= grid_L/20;
            double dis_temp=0;
            int num=two_node_dis/step_len;
            for( k=0; k<num; k++)
            {
              double path_temp_x=k*step_len*cos(angle_temp);
              double path_temp_y=k*step_len*sin(angle_temp);
              int num_x=path_temp_x;
              int num_y=path_temp_y;
              // if(path_temp_x<0)
              // {
              //   num_x -= 1;
              // }
              // if(path_temp_y<0)
              // {
              //   num_y -= 1;
              // }
              // dis_temp+=step_len;
              
              if(maze[PathPoints_node[i]->x+num_x][PathPoints_node[i]->y+num_y]) break;
            }

            if(k==num) break;
        }

        if(i==0)
        {
          pcl::PointXYZI ptemp;
          ptemp.x=PathPoints_node[i]->x*grid_L+map_range_min_x;
          ptemp.y=PathPoints_node[i]->y*grid_L+map_range_min_y;
          path_Astar_node->points.push_back(ptemp);
        }

        pcl::PointXYZI ptemp1;
        ptemp1.x=PathPoints_node[j]->x*grid_L+map_range_min_x;
        ptemp1.y=PathPoints_node[j]->y*grid_L+map_range_min_y;
        path_Astar_node->points.push_back(ptemp1);
        if(j==i) break;
        i=j-1;
    }

    vector<double> v_x;
    v_x.clear();
    vector<double> v_y;
    v_y.clear();
  //  pcl::copyPointCloud(*path_Astar_node, *line_global_test);

    pcl::PointCloud<pcl::PointXYZI>::Ptr node_temp(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr node(new pcl::PointCloud<pcl::PointXYZI>);

    for(int i=1; i<path_Astar_node->points.size(); i++)
    {
        double a_temp = path_Astar_node->points[i-1].y-path_Astar_node->points[i].y;
        double b_temp = path_Astar_node->points[i].x-path_Astar_node->points[i-1].x;
        double c_temp = path_Astar_node->points[i-1].x*path_Astar_node->points[i].y-path_Astar_node->points[i].x*path_Astar_node->points[i-1].y;
        double angle_temp = atan2(-a_temp, b_temp);

        double y_dis=path_Astar_node->points[i].y-path_Astar_node->points[i-1].y;
        double two_node_dis_temp=sqrt(pow(b_temp, 2)+pow(y_dis, 2));
        double ibdex=two_node_dis_temp-(int)two_node_dis_temp;
        int two_node_dis;
        if(ibdex>0.5) two_node_dis=two_node_dis_temp+1;
        else two_node_dis=two_node_dis_temp;

        for(int var=0; var<two_node_dis; var+=2)  //对于得到的点，每两米中间插一个点
        {
            double path_temp_x=var*cos(angle_temp);
            double path_temp_y=var*sin(angle_temp);

            pcl::PointXYZI line_point;
            line_point.x=path_Astar_node->points[i-1].x+path_temp_x;
            line_point.y=path_Astar_node->points[i-1].y+path_temp_y;            
            node_temp->points.push_back(line_point);
        }
    }
    node_temp->points.push_back(path_Astar_node->points[path_Astar_node->size() - 1]);
    //  pcl::copyPointCloud(*node_temp, *line_global_test);

    static int ind_turn=0;
    for (int i=0; i<node_temp->points.size(); i++)
    {
        if(i<node_temp->points.size()-2&&i>1)
        {
            double a_temp_back = node_temp->points[i-1].y-node_temp->points[i].y;
            double b_temp_back = node_temp->points[i].x-node_temp->points[i-1].x;
            double angle_temp_back = atan2(-a_temp_back, b_temp_back);
            double a_temp_front = node_temp->points[i].y-node_temp->points[i+1].y;
            double b_temp_front = node_temp->points[i+1].x-node_temp->points[i].x;
            double angle_temp_front = atan2(-a_temp_front, b_temp_front);
            if(fabs(angle_temp_front-angle_temp_back)>30 * M_PI / 180){
                pcl::PointXYZI point1;
                point1.x=(node_temp->points[i].x+node_temp->points[i-1].x)/2.0;
                point1.y=(node_temp->points[i].y+node_temp->points[i-1].y)/2.0;
                pcl::PointXYZI point2;
                point2.x=(node_temp->points[i].x+node_temp->points[i+1].x)/2.0;
                point2.y=(node_temp->points[i].y+node_temp->points[i+1].y)/2.0;
                pcl::PointXYZI point3;
                point3.x=(point1.x+point2.x)/2.0;
                point3.y=(point1.y+point2.y)/2.0;
                pcl::PointXYZI center_point1;
                center_point1.x=(node_temp->points[i-1].x+point1.x)/2.0;
                center_point1.y=(node_temp->points[i-1].y+point1.y)/2.0;
                pcl::PointXYZI center_point2;
                center_point2.x=(point3.x+point1.x)/2.0;
                center_point2.y=(point3.y+point1.y)/2.0;
                pcl::PointXYZI center_point3;
                center_point3.x=(point3.x+point2.x)/2.0;
                center_point3.y=(point3.y+point2.y)/2.0;
                pcl::PointXYZI center_point4;
                center_point4.x=(node_temp->points[i+1].x+point2.x)/2.0;
                center_point4.y=(node_temp->points[i+1].y+point2.y)/2.0;

                double a_temp_front_front = node_temp->points[i+1].y-node_temp->points[i+2].y;
                double b_temp_front_front = node_temp->points[i+2].x-node_temp->points[i+1].x;
                double angle_temp_front_front = atan2(-a_temp_front_front, b_temp_front_front);
                if(ind_turn==1)
                {
                    if(fabs(angle_temp_front-angle_temp_front_front)>30 * M_PI / 180)
                    {                    
                        node->points.push_back(center_point3);
                    }
                    else
                    {
                        node->points.push_back(center_point3);
                        node->points.push_back(center_point4);
                    }
                }
                else
                {
                    if(fabs(angle_temp_front-angle_temp_front_front)>30 * M_PI / 180)
                    {
                        node->points.push_back(center_point1);
                        node->points.push_back(center_point2);
                        node->points.push_back(center_point3);
                    }
                    else
                    {
                        node->points.push_back(center_point1);
                        node->points.push_back(center_point2);
                        node->points.push_back(center_point3);
                        node->points.push_back(center_point4);
                    }
                }
                ind_turn=1;
            }
            else {
                ind_turn=0;
                node->points.push_back(node_temp->points[i]);
            }
        }
        else {
            node->points.push_back(node_temp->points[i]);
        }
    }
     pcl::copyPointCloud(*node, *line_global_test);

    s.clear();
    s.push_back(0);
    h.clear();
    h.push_back(0);
    vector<double> s_sequence;
    s_sequence.clear();
    s_sequence.push_back(0);

    double mark_points_x;
    double mark_points_y;
    double distance = 0;
    pcl::PointXYZI point_whole_frist;
    v_x.push_back(node->points[0].x);
    v_y.push_back(node->points[0].y);

    pcl::PointXYZI point_whole;
    for (size_t j = 1; j < node->points.size(); j++)
     {
        mark_points_x = node->points[j].x ;
        mark_points_y = node->points[j].y ;
        v_x.push_back(mark_points_x);
        v_y.push_back(mark_points_y);

        double x_dif=node->points[j].x-node->points[j-1].x;
        double y_dif=node->points[j].y-node->points[j-1].y;
        distance += sqrt(pow(x_dif, 2)+pow(y_dif, 2));
        s.push_back(distance) ;
     }
    s1.set_points(s, v_x); //三次样条插值函数
    s2.set_points(s, v_y); //三次样条插值函数

    for(int i = 0 ; h[i] <= s[s.size()-1] ; i++)
    {
         h.push_back(h[i] + 0.1);
    }

    line->points.clear();

    for(double t= 0 ; t < h.size() ; t++)
    {
        pcl::PointXYZI points;
        points.x  = s1( h[t]) ;
        points.y  = s2( h[t]) ;
        line->points.push_back(points);
    }
    pcl::copyPointCloud(*line, *line_global);
   
}

void astar::InitAstar(std::vector<std::vector<int>> &_maze)//输入栅格
{
  maze = _maze;
}

int astar::calcG(Point *temp_start, Point *point)
{
  int extraG = (abs(point->x - temp_start->x) + abs(point->y - temp_start->y)) == 1 ? kCost1 : kCost2;
  int parentG = point->parent == NULL ? 0 : point->parent->G; //如果是初始节点，则其父节点是空
  return parentG + extraG;
}

int astar::calcH(Point *point, Point *end)
{
  //用简单的欧几里得距离计算H，这个H的计算是关键，还有很多算法
  return sqrt((double)(end->x - point->x)*(double)(end->x - point->x) + (double)(end->y - point->y)*(double)(end->y - point->y))*kCost1;
}

int astar::calcF(Point *point)
{
  return point->G + point->H;
}

Point *astar::getLeastFpoint()
{
  if (!openList.empty())
  {
    auto resPoint = openList.front();
    for (auto &point : openList)
    if ((point->F)<(resPoint->F))
      resPoint = point;
    return resPoint;
  }
  return NULL;
}

Point *astar::findPath(Point &startPoint, Point &endPoint, bool isIgnoreCorner)
{
  openList.push_back(new Point(startPoint.x, startPoint.y)); //置入起点,拷贝开辟一个节点，内外隔离
  while (!openList.empty())
  {
    auto curPoint = getLeastFpoint(); //找到F值最小的点
    openList.remove(curPoint); //从开启列表中删除
    closeList.push_back(curPoint); //放到关闭列表
    //1,找到当前周围八个格中可以通过的格子
    auto surroundPoints = getSurroundPoints(curPoint, isIgnoreCorner);
    for (auto &target : surroundPoints)
    {
      //2,对某一个格子，如果它不在开启列表中，加入到开启列表，设置当前格为其父节点，计算F G H
      if (!isInList(openList, target))
      {
        target->parent = curPoint;

        target->G = calcG(curPoint, target);
        target->H = calcH(target, &endPoint);
        target->F = calcF(target);

        openList.push_back(target);
      }
      //3，对某一个格子，它在开启列表中，计算G值, 如果比原来的大, 就什么都不做, 否则设置它的父节点为当前点,并更新G和F
      else
      {
        int tempG = calcG(curPoint, target);
        if (tempG<target->G)
        {
          target->parent = curPoint;

          target->G = tempG;
          target->F = calcF(target);
        }
      }
      Point *resPoint = isInList(openList, &endPoint);
      if (resPoint)
        return resPoint; //返回列表里的节点指针，不要用原来传入的endpoint指针，因为发生了深拷贝
    }
  }

  return NULL;
}

std::list<Point *> astar::GetPath(Point &startPoint, Point &endPoint, bool isIgnoreCorner)
{
  Point *result = findPath(startPoint, endPoint, isIgnoreCorner);
  std::list<Point *> path;
  //返回路径，如果没找到路径，返回空链表
  while (result)
  {
    path.push_front(result); //在容器开头增加新的元素
    result = result->parent;
  }

  // 清空临时开闭列表，防止重复执行GetPath导致结果异常
  openList.clear();
  closeList.clear();

  return path;
}

Point *astar::isInList(const std::list<Point *> &list, const Point *point) const
{
  //判断某个节点是否在列表中，这里不能比较指针，因为每次加入列表是新开辟的节点，只能比较坐标
  for (auto p : list)
  if (p->x == point->x&&p->y == point->y)
    return p;
  return NULL;
}

bool astar::isCanreach(const Point *point, const Point *target, bool isIgnoreCorner) const
{
  if (target->x<0 || target->x>maze.size() - 1
    || target->y<0 || target->y>maze[0].size() - 1
    || maze[target->x][target->y] == 1
    || target->x == point->x&&target->y == point->y
    || isInList(closeList, target)) //如果点与当前节点重合、超出地图、是障碍物、或者在关闭列表中，返回false
    return false;
  else
  {
    if (abs(point->x - target->x) + abs(point->y - target->y) == 1) //非斜角可以
      return true;
    else
    {
      //斜对角要判断是否绊住
      if (maze[point->x][target->y] == 0 && maze[target->x][point->y] == 0)
        return true;
      else
        return isIgnoreCorner;
    }
  }
}

std::vector<Point *> astar::getSurroundPoints(const Point *point, bool isIgnoreCorner) const
{
  std::vector<Point *> surroundPoints;

  for (int x = point->x - 1; x <= point->x + 1; x++)
  for (int y = point->y - 1; y <= point->y + 1; y++)
  if (isCanreach(point, new Point(x, y), isIgnoreCorner))
    surroundPoints.push_back(new Point(x, y));

  return surroundPoints;
}


void timerCallback(const ros::TimerEvent& event)
{
      
       pcl::toROSMsg(*clusterCloud, clusterCloud_output);
       clusterCloud_output.header.frame_id = "velodyne";
       cluster_map_pub.publish(clusterCloud_output);
      if (car_state == normal)
      {
        pcl::toROSMsg(*line_global, line_global_out);
        line_global_out.header.frame_id = "velodyne";
        line_pub.publish(line_global_out);
        pcl::toROSMsg(*line_global_test, line_global_out_test);
        line_global_out_test.header.frame_id = "velodyne";
        line_pub_test.publish(line_global_out_test);
      }
      
       
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "astar");
  ros::NodeHandle nh;
  //  bound_pub = nh2.advertise<sensor_msgs::PointCloud2> ("bound_output", 1);
  cluster_map_pub = nh.advertise<sensor_msgs::PointCloud2> ("map_output", 1);
  line_pub = nh.advertise<sensor_msgs::PointCloud2> ("line_output", 1);
  line_pub_test = nh.advertise<sensor_msgs::PointCloud2> ("line_output_test", 1);
  ros::Timer timer=nh.createTimer(ros::Duration(0.3), timerCallback);

  astar start_detec(nh);
  ros::MultiThreadedSpinner spinner(4);
  spinner.spin();
  return 0;
}
