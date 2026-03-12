#include "lattice/lattice_planner.h"
using namespace Eigen;
const std::string RED = "\033[31m";//红色
const std::string YELLOW = "\033[33m"; // 黄色
const std::string RESET = "\033[0m";

const std::string SHAN = "\033[5;30;41m";
const std::string GREEN = "\033[0;30;42m";

#define BULE1 "\033[0;30;44m"
#define PURPLE "\033[0;30;45m"
#define BULE2 "\033[0;30;46m"
#define WHITE "\033[0;30;47m"
#define ENDL "\033[0m\n"

lattice_planner::lattice_planner(ros::NodeHandle &nh)
{
    std::string play_path = ros::package::getPath("play_path");
    FILE_NAME = play_path+"/text/trajectory.txt";
    // read_path(FILE_NAME);
    // line_astar_sub = nh.subscribe("line_output", 100, &lattice_planner::callBack_line_astar, this);

    line_astar_sub = nh.subscribe("/run_hybrid_astar/hybrid_astar_path_watch", 100, &lattice_planner::callBack_line_astar, this);
    car_pos_sub = nh.subscribe<planning_msgs::car_info>("/car_pos", 10, &lattice_planner::callBack_Pos, this);
    car_info_sub = nh.subscribe<planning_msgs::car_info>("/car_info", 10, &lattice_planner::callBack_Info, this);
    obstacleList_lidar_sub = nh.subscribe<planning_msgs::ObstacleList>("/obstacleList_lidar", 10, &lattice_planner::callBack_obstacleLidar, this);
    obstacleList_camera_sub = nh.subscribe<obj_msgs::ObstacleList>("/object_detection_local", 10, &lattice_planner::callBack_obstacleCamera, this);
    line_record_pub = nh.advertise<sensor_msgs::PointCloud2>("watch_line_Path_tra", 10);
    local_path_pub = nh.advertise<sensor_msgs::PointCloud2>("local_Path", 10);
    referenceline_pub = nh.advertise<planning_msgs::car_path>("QP_Path", 10);
    long_obstacle_pub = nh.advertise<planning_msgs::ObstacleList>("/long_obstacleList", 10);
    ros::spin();
}

lattice_planner::~lattice_planner()
{
}

void lattice_planner::pub_cloud(pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud, ros::Publisher Publisher)
{
    sensor_msgs::PointCloud2 ros_msgs;
    pcl::toROSMsg(*in_cloud, ros_msgs);
    ros_msgs.header.frame_id = "velodyne";
    ros_msgs.header.stamp = ros::Time::now();
    Publisher.publish(ros_msgs);
}

void lattice_planner::pub_referenceline(planning_msgs::car_path reference, ros::Publisher Publisher)
{
    Publisher.publish(reference);
}

void lattice_planner::pub_goal_point(planning_msgs::ObstacleList ObstacleList_long, ros::Publisher Publisher)
{
    Publisher.publish(ObstacleList_long);
}

void lattice_planner::callBack_obstacleLidar(const planning_msgs::ObstacleList::ConstPtr &ObstacleList_lidar)
{
    obstacleList_lidar.reset(new planning_msgs::ObstacleList);
    *obstacleList_lidar = *ObstacleList_lidar;
    for (size_t i = 0; i < ObstacleList_lidar->obstacles.size(); i++)
    {
        obstacleList_lidar->obstacles[i].isLidarObs = true;
    }
    
}

void lattice_planner::callBack_obstacleCamera(const obj_msgs::ObstacleList::ConstPtr &ObstacleList_camera)
{
    obstacleList_camera.reset(new planning_msgs::ObstacleList);
    for (int i = 0; i < ObstacleList_camera->obstacles.size(); i++)
    {
        planning_msgs::Obstacle obs_vision;
        for (int j = 0; j < obstacleList_camera->obstacles[i].bounding_boxs.size(); j++)
        {
            obs_vision.bounding_boxs[j].x =ObstacleList_camera->obstacles[i].bounding_boxs[j].x;
            obs_vision.bounding_boxs[j].y =ObstacleList_camera->obstacles[i].bounding_boxs[j].y;
        }
        obs_vision.isLidarObs = false;
        obs_vision.x_vel =ObstacleList_camera->obstacles[i].vel.x;

        obs_vision.y_vel =ObstacleList_camera->obstacles[i].vel.y;
        obstacleList_camera->obstacles.push_back(obs_vision);
    }
    // cout<<"obstacleList_camera->obstacles.size():"<<obstacleList_camera->obstacles.size()<<endl;
}
void lattice_planner::callBack_Info(const planning_msgs::car_info::ConstPtr &Car_info)
{
    *car_info = *Car_info;
}

void lattice_planner::callBack_line_astar(const sensor_msgs::PointCloud2 &line_astar)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr line_global_astar_temp(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(line_astar,*line_global_astar_temp);
    if (line_global_astar_temp->size()==line_global_astar->size())
    {
       flag_line_change = false;
    }else{
       flag_line_change = true;
    }
    if (flag_line_change)
    {
        *line_global_astar = *line_global_astar_temp;
        for (size_t i = 0; i < line_global_astar->size()-1; i++)
        {
            
            float dx = line_global_astar->points[i+1].x - line_global_astar->points[i].x;
            float dy = line_global_astar->points[i+1].y - line_global_astar->points[i].y;
            float k = atan2(dy,dx);
            line_global_astar->points[i].z = k;
        }
        line_global_astar->points.back().z = line_global_astar->points[line_global_astar->size()-2].z;
        *line_record_opt = *line_global_astar;
        
        line_record->clear();
        global_path.points.clear();
        Mean_Interpolation(line_record_opt, line_record);
        // 路径参数计算
        Path_Parameter_Calculate(global_path);
    }
    
}

void lattice_planner::callBack_Pos(const planning_msgs::car_info::ConstPtr &car_pos)
{
    auto start = std::chrono::high_resolution_clock::now();

    cout << RED << "回调开始！！！" << RESET << endl;
    Plan(car_pos);
    // pub_cloud(line_record_watch, line_record_pub);
    pub_cloud(LateralTrajectory,local_path_pub);
    pub_referenceline(ReferenceLine,referenceline_pub);

    if (get_newGoalPoint() && car_state == normal)
    {
        car_state = replanning;
        goal_point.x = obstacleList_long->goal_point.x;
        goal_point.y = obstacleList_long->goal_point.y;
        cout << YELLOW << "进入重规划状态！！" << RESET << endl;
        pub_goal_point(*obstacleList_long,long_obstacle_pub);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    cout<<"运行时间："<<elapsed.count()<<endl;
}


void lattice_planner::Plan(planning_msgs::car_info::ConstPtr car_pos)
{
    if(!global_path.points.size()) return;
    LateralTrajectory->clear(); //先清空全局变量
    ReferenceLine.points.clear();
    obstacleList_long->obstacles.clear(); 
    // static int num = 0;
    // num++;

    float dis = sqrt((car_pos->x-goal_point.x)*(car_pos->x-goal_point.x)+(car_pos->y-goal_point.y)*(car_pos->y-goal_point.y));
    if (dis<0.2)
    {
        car_state = normal;
        // num = 0;
        cout << GREEN << "车辆回到正常模式！！" << RESET << endl;
    }
    
    planning_msgs::path_point matched_point;
    car_point.x = car_pos->x;
    car_point.y = car_pos->y;
    car_point.yaw = car_pos->yaw;
    car_point.vel = car_info->speed;
    std::array<double, 2> init_s;
    std::array<double, 2> init_l;
    matched_point = MatchToPath(global_path , car_point.x, car_point.y);
    cartesian_to_frenet(matched_point,car_point,init_s,init_l);
    // cout<<"car_point.absolute_s:"<<car_point.absolute_s<<" car_point.l:"<<car_point.l<<endl;
    planning_msgs::ObstacleList ObstacleList_static;

    ST_graph STGraph = obstacleList_Preprocessing(ObstacleList_static);
    Trajectory1DBundle lon_trajectory1d_bundle;
    Trajectory1DBundle lat_trajectory1d_bundle;
    GenerateTrajectoryBundles(init_s,init_l,STGraph,lon_trajectory1d_bundle,lat_trajectory1d_bundle);
    trajectory_evaluate(lat_trajectory1d_bundle,ObstacleList_static);
}

// 求出投影点（x , y , yaw ,kappa, absolute_s , tor ,nor）
planning_msgs::path_point MatchToPath(planning_msgs::car_path global_path , float x ,float y)
{
    
    float distance_min = cal_distance_square(global_path.points.front(), x, y);
    std::size_t index_min = 0;

    for (std::size_t i = 1; i < global_path.points.size(); ++i) {
        float distance_temp = cal_distance_square(global_path.points[i], x, y);
        if (distance_temp < distance_min) {
        distance_min = distance_temp;
        index_min = i;
        }
    }

    std::size_t index_start = (index_min == 0) ? index_min : index_min - 1;
    std::size_t index_end =
        (index_min + 1 == global_path.points.size()) ? index_min : index_min + 1;

    if (index_start == index_end) {
        return global_path.points[index_start];
    }
    return FindProjectionPoint(global_path.points[index_start],
                            global_path.points[index_end], x, y);
}

planning_msgs::path_point MatchToPath(planning_msgs::car_path global_path , float s)
{
    assert(global_path.points.size() != 0);
    int index = 0;
    for (int i= 0 ; i<global_path.points.size() ; i++)
    {
        planning_msgs::path_point point = global_path.points[i];
        if(point.absolute_s>s)
        {
            break;
        }
        index = i;
        
    }

    std::size_t index_end = (index + 1 == global_path.points.size()) ? index : index + 1;
    if (index == index_end) {
        // return global_path.points[index];
        planning_msgs::path_point back_point =  InterpolateUsingLinearApproximation(global_path.points[index-1], global_path.points[index], s);
        // cout<<"index-1:"<<global_path.points[index-1].absolute_s<<" "<<global_path.points[index-1].yaw<<endl;
        // cout<<"index:"<<global_path.points[index].absolute_s<<" "<<global_path.points[index].yaw<<endl;

        // cout<<"back_point:"<<back_point.absolute_s<<" "<<back_point.x<<" "<<back_point.y<<" "<<back_point.yaw<<endl;
        return back_point;
    }
    return InterpolateUsingLinearApproximation(global_path.points[index], global_path.points[index_end], s);
}


//计算两点之间的距离
float cal_distance_square(planning_msgs::path_point point , float x , float y)
{
    float dx = point.x - x;
    float dy = point.y - y;
    return dx * dx + dy * dy;
}

planning_msgs::path_point FindProjectionPoint(planning_msgs::path_point p0,planning_msgs::path_point p1, float x, float y)
{
    float v0x = x - p0.x;
    float v0y = y - p0.y;

    float v1x = p1.x - p0.x;
    float v1y = p1.y - p0.y;

    float v1_norm = std::sqrt(v1x * v1x + v1y * v1y);
    float dot = v0x * v1x + v0y * v1y;

    float delta_s = dot / v1_norm;
    return InterpolateUsingLinearApproximation(p0, p1, p0.absolute_s + delta_s);
}

planning_msgs::path_point InterpolateUsingLinearApproximation(const planning_msgs::path_point &p0,
                                                              const planning_msgs::path_point &p1,
                                                              const float s) {
    float s0 = p0.absolute_s;
    float s1 = p1.absolute_s;

    planning_msgs::path_point match_point;
    float weight = (s - s0) / (s1 - s0);
    match_point.x = (1 - weight) * p0.x + weight * p1.x;
    match_point.y = (1 - weight) * p0.y + weight * p1.y;
    match_point.yaw = slerp(p0.yaw, p0.absolute_s, p1.yaw, p1.absolute_s, s);
    match_point.kappa = (1 - weight) * p0.kappa + weight * p1.kappa;
    match_point.absolute_s = s;
    match_point.tor.x = cos(match_point.yaw);
    match_point.tor.y = sin(match_point.yaw);
    match_point.nor.x = -sin(match_point.yaw);
    match_point.nor.y = cos(match_point.yaw);
    return match_point;
}

float slerp(const float a0, const float t0, const float a1, const float t1,
             const float t) {

  const float a0_n = NormalizeAngle(a0);
  const float a1_n = NormalizeAngle(a1);
  float d = a1_n - a0_n;
  if (d > M_PI) {
    d = d - 2 * M_PI;
  } else if (d < -M_PI) {
    d = d + 2 * M_PI;
  }

  const float r = (t - t0) / (t1 - t0);
  const float a = a0_n + d * r;
  return NormalizeAngle(a);
}

float NormalizeAngle(const float angle) {
  float a = std::fmod(angle + M_PI, 2.0 * M_PI);
  if (a < 0.0) {
    a += (2.0 * M_PI);
  }
  return a - M_PI;
}

void cartesian_to_frenet(const planning_msgs::path_point matched_point,planning_msgs::path_point &car_point,
                            std::array<double, 2> &init_s,std::array<double, 2> &init_l)
{
    const float dx = car_point.x - matched_point.x;
    const float dy = car_point.y - matched_point.y;
    const float cross_rd_nd = matched_point.tor.x * dy - matched_point.tor.y * dx;
    car_point.absolute_s = matched_point.absolute_s;
    car_point.l = copysign(sqrt(dx * dx + dy * dy), cross_rd_nd);
    float dyaw = car_point.yaw-matched_point.yaw;
    car_ddaw = dyaw;
    init_s[0] = car_point.absolute_s;
    init_s[1] = car_point.vel*cos(dyaw)/(1-matched_point.kappa*matched_point.l);

    init_l[0] = car_point.l;
    init_l[1] = (1-matched_point.kappa*matched_point.l)*tan(dyaw);
}

VectorXd cartesian_to_frenet(const planning_msgs::path_point matched_point,float x,float y)
{
    const float dx = x - matched_point.x;
    const float dy = y - matched_point.y;
    const float cross_rd_nd = matched_point.tor.x * dy - matched_point.tor.y * dx;
    VectorXd sl(2);
    sl<<matched_point.absolute_s,copysign(sqrt(dx * dx + dy * dy), cross_rd_nd);
    return sl;
}

PointXY frenet_to_cartesian(const planning_msgs::path_point matched_point,float l)
{
    PointXY point;
    float xr = matched_point.x;
    float yr = matched_point.y;
    float yawr = matched_point.yaw;
    point.x = xr - l*sin(yawr);
    point.y = yr + l*cos(yawr);
    return point;
}
ST_graph obstacleList_Preprocessing(planning_msgs::ObstacleList &ObstacleList_static)
{
    
    obstacleList_total.reset(new planning_msgs::ObstacleList);
    for (size_t i = 0; i < obstacleList_lidar->obstacles.size(); i++)
    {
        obstacleList_total->obstacles.push_back(obstacleList_lidar->obstacles[i]);
    }
    for (size_t i = 0; i < obstacleList_camera->obstacles.size(); i++)
    {
        obstacleList_total->obstacles.push_back(obstacleList_camera->obstacles[i]);
    }
    
    
    cout<<"obstacleList_lidar："<<obstacleList_lidar->obstacles.size()<<" camera:"<<
    obstacleList_camera->obstacles.size()<<" total:"<<obstacleList_total->obstacles.size()<<endl;
    ST_graph STGraph;
    for (size_t i = 0; i < obstacleList_total->obstacles.size(); i++)
    {
        double vel = sqrt(pow(obstacleList_total->obstacles[i].x_vel,2)+pow(obstacleList_total->obstacles[i].y_vel,2));
        
        if(vel<0.4)
        {
            // cout<<"vel:"<<vel<<endl;
            obstacleList_total->obstacles[i].is_dynamic_obs = false;
            SetStaticObstacle(obstacleList_total->obstacles[i], global_path, STGraph);
            if(obstacleList_total->obstacles[i].max_s>0 && 
               obstacleList_total->obstacles[i].min_s<global_path.points.back().absolute_s) //只保留车前方的障碍物
            {
                ObstacleList_static.obstacles.push_back(obstacleList_total->obstacles[i]);
            }

        }else{
            obstacleList_total->obstacles[i].is_dynamic_obs = true;
             SetDynamicObstacle(obstacleList_total->obstacles[i], global_path,STGraph);
        }
    }
    // cout<<"11"<<endl;
    //静态障碍物由小到大排序
    std::sort(ObstacleList_static.obstacles.begin(), ObstacleList_static.obstacles.end(),
        [](const planning_msgs::Obstacle sl0, const planning_msgs::Obstacle& sl1) {
            return sl0.min_s < sl1.min_s;
        });
    cout<<"排序后"<<endl;
    for (const auto& obstacles : ObstacleList_static.obstacles)
    {
        cout<<"car自身S, 静态障碍物信息mins,maxs,minl,maxl："<<car_point.absolute_s<<" "<<obstacles.min_s<<" "<<obstacles.max_s
            <<" "<<obstacles.min_l<<" "<<obstacles.max_l<<endl;
        if (obstacles.min_l*obstacles.max_l<0&&(abs(obstacles.min_l)>long_flag&&abs(obstacles.max_l)>long_flag))
        {
            obstacleList_long->obstacles.push_back(obstacles);
            cout<<"出现较宽的障碍物！！"<<endl;
        }
        
    }
    // cout<<"个数："<<STGraph.ST_graph_node.size()<<endl;
    return STGraph;
}

//初始化参数，生成ST图
void SetStaticObstacle(planning_msgs::Obstacle &Obstacle, planning_msgs::car_path global_path,ST_graph &STGraph)
{
    // cout<<"静态障碍物ST图"<<endl;
    Obstacle_init(Obstacle,global_path,true);
   
    // float left_width = 1.0;
    // float right_width = 1.0;
     
    // if (Obstacle.min_s>(car_point.absolute_s + front_plan_length) || Obstacle.max_s<(car_point.absolute_s-back_plan_length)
    //     ||  Obstacle.min_l > left_width                         || Obstacle.max_l  <   -right_width)
    // {
    //     return;
    // }

    // ST_graph_single stGraph_node;
    // stGraph_node.isStatic = true;
    // stGraph_node.s_vel = Obstacle.s_vel;
    // stGraph_node.left_bottom_point.s = Obstacle.min_s;
    // stGraph_node.left_bottom_point.t = 0;
    // stGraph_node.right_bottom_point.s = Obstacle.min_s;
    // stGraph_node.right_bottom_point.t = trajectory_time_length;
    // stGraph_node.left_upper_point.s = Obstacle.max_s;
    // stGraph_node.left_upper_point.t = 0;
    // stGraph_node.right_upper_point.s = Obstacle.max_s;
    // stGraph_node.right_upper_point.t = trajectory_time_length;

    // cout<<"left_bottom_point.s t:"<<stGraph_node.left_bottom_point.s<<" "<<stGraph_node.left_bottom_point.t<<endl;
    // cout<<"right_bottom_point.s t:"<<stGraph_node.right_bottom_point.s<<" "<<stGraph_node.right_bottom_point.t<<endl;
    // cout<<"left_upper_point.s t:"<<stGraph_node.left_upper_point.s<<" "<<stGraph_node.left_upper_point.t<<endl;
    // cout<<"right_upper_point.s t:"<<stGraph_node.right_upper_point.s<<" "<<stGraph_node.right_upper_point.t<<endl;
    // for (float i = 0; i < trajectory_time_length; i+=0.1)
    // {
    //     STPoint max_st_point;
    //     STPoint min_st_point;
    //     max_st_point.s = Obstacle.max_s;
    //     max_st_point.t = i;
    //     min_st_point.s = Obstacle.min_s;
    //     max_st_point.t = i;
    //     stGraph_node.Max_s_point.push_back(max_st_point);
    //     stGraph_node.Min_s_point.push_back(min_st_point);
    // }
    
    // STGraph.ST_graph_node.push_back(stGraph_node);
    // // cout<<"静态障碍物ST图完成"<<endl;
}


void SetDynamicObstacle(planning_msgs::Obstacle &Obstacle, planning_msgs::car_path global_path,ST_graph &STGraph)
{
    // cout<<"动态障碍物ST图"<<endl;
    Obstacle_init(Obstacle,global_path,false);
    float left_width = 2.0;
    float right_width = 2.0;
    if (Obstacle.min_s>(car_point.absolute_s + front_plan_length) || Obstacle.max_s<(car_point.absolute_s-back_plan_length)
            ||  Obstacle.min_l > left_width                         || Obstacle.max_l  <   -right_width)
    {
        return;
    }
    float box_x_predict = Obstacle.x+Obstacle.x_vel*trajectory_time_length;
    float box_y_predict = Obstacle.y+Obstacle.y_vel*trajectory_time_length;
    planning_msgs::path_point matched_point_predict;
    matched_point_predict = MatchToPath(global_path , box_x_predict, box_y_predict);
    VectorXd sl = cartesian_to_frenet(matched_point_predict,box_x_predict,box_y_predict);
    float ds = sl(0) - Obstacle.s;
    float dl = sl(1) - Obstacle.l;

    ST_graph_single stGraph_node;
    stGraph_node.isStatic = false;
    stGraph_node.s_vel = Obstacle.s_vel;
    stGraph_node.left_bottom_point.s = Obstacle.min_s;
    stGraph_node.left_bottom_point.t = 0;
    stGraph_node.right_bottom_point.s = Obstacle.min_s + ds;
    stGraph_node.right_bottom_point.t = trajectory_time_length;
    stGraph_node.left_upper_point.s = Obstacle.max_s;
    stGraph_node.left_upper_point.t = 0;
    stGraph_node.right_upper_point.s = Obstacle.max_s + ds;
    stGraph_node.right_upper_point.t = trajectory_time_length;
    // cout<<"left_bottom_point.s t:"<<stGraph_node.left_bottom_point.s<<" "<<stGraph_node.left_bottom_point.t<<endl;
    // cout<<"right_bottom_point.s t:"<<stGraph_node.right_bottom_point.s<<" "<<stGraph_node.right_bottom_point.t<<endl;
    // cout<<"left_upper_point.s t:"<<stGraph_node.left_upper_point.s<<" "<<stGraph_node.left_upper_point.t<<endl;
    // cout<<"right_upper_point.s t:"<<stGraph_node.right_upper_point.s<<" "<<stGraph_node.right_upper_point.t<<endl;
    cout<<"stGraph_node.Min_s_point size: "<<stGraph_node.Min_s_point.size()<<endl;
    int t = 0;
    float i;
    for (i = 0.0; i <trajectory_time_length; i+=0.1)
    {
        t++;
        STPoint max_st_point;
        STPoint min_st_point;
        max_st_point.s = Obstacle.max_s + ds*i/trajectory_time_length;
        max_st_point.t = i;
        min_st_point.s = Obstacle.min_s + ds*i/trajectory_time_length;
        max_st_point.t = i;
        // float min_l_predict = Obstacle.min_l + dl*i/trajectory_time_length;
        // float max_l_predict = Obstacle.max_l + dl*i/trajectory_time_length;
        stGraph_node.Max_s_point.push_back(max_st_point);
        stGraph_node.Min_s_point.push_back(min_st_point);
        cout<<"i:"<<i<<" "<<trajectory_time_length<<endl;
    }
    cout<<"stGraph_node.Min_s_point size: "<<stGraph_node.Min_s_point.size()<<" "<<t<<" "<<i<<endl;
    STGraph.ST_graph_node.push_back(stGraph_node);
    // cout<<"动态障碍物ST图完成"<<endl;
}
void Obstacle_init(planning_msgs::Obstacle &Obstacle, planning_msgs::car_path global_path,bool static_flag)
{
    
    ComputeObstacleBoundary(Obstacle,global_path);
    PointXY cp = getCpFromBbox(Obstacle);
    planning_msgs::path_point matched_point;
    Obstacle.x = cp.x;
    Obstacle.y = cp.y;
    matched_point = MatchToPath(global_path , Obstacle.x, Obstacle.y);
    VectorXd sl = cartesian_to_frenet(matched_point,Obstacle.x,Obstacle.y);
    Obstacle.s = Obstacle.absolute_s = sl(0);
    Obstacle.l = Obstacle.absolute_s = sl(1);
    Obstacle.absolute_s_max = Obstacle.max_s;
    Obstacle.absolute_s_min = Obstacle.min_s;
    if(static_flag){
        Obstacle.s_vel = 0;
        Obstacle.l_vel = 0;
    }else{
        float yaw = atan2(Obstacle.y_vel,Obstacle.x_vel);
        float dyaw = yaw-matched_point.yaw;
        float v = sqrt(pow(Obstacle.x_vel,2)+pow(Obstacle.y_vel,2));
        Obstacle.s_vel = v*cos(dyaw)/(1-matched_point.kappa*matched_point.l);
        Obstacle.l_vel = v*sin(dyaw);
    }

}
// 求出一个障碍物最大和最小的s,l,并给Obstacle.bounding_boxs_SL赋值
void ComputeObstacleBoundary(planning_msgs::Obstacle &Obstacle, planning_msgs::car_path global_path)
{
    double start_s(std::numeric_limits<double>::max());
    double end_s(std::numeric_limits<double>::lowest());
    double start_l(std::numeric_limits<double>::max());
    double end_l(std::numeric_limits<double>::lowest());
    for (size_t i = 0; i < Obstacle.bounding_boxs.size(); i++)
    {
        planning_msgs::path_point matched_point;
        float box_x = Obstacle.bounding_boxs[i].x;
        float box_y = Obstacle.bounding_boxs[i].y;
        matched_point = MatchToPath(global_path , box_x, box_y);
        VectorXd sl = cartesian_to_frenet(matched_point,box_x,box_y);
        Obstacle.bounding_boxs_SL[i].x = sl(0);
        Obstacle.bounding_boxs_SL[i].y = sl(1);
        start_s = std::fmin(start_s, sl(0));
        end_s = std::fmax(end_s, sl(0));
        start_l = std::fmin(start_l, sl(1));
        end_l = std::fmax(end_l, sl(1));
    }
    Obstacle.min_s = start_s;
    Obstacle.min_l = start_l;
    Obstacle.max_s = end_s;
    Obstacle.max_l = end_l;
}

PointXY getCpFromBbox(const planning_msgs::Obstacle Obstacle){
    PointXY p1 , p2 , p3 ,p4 ,cp;
    p1.x = Obstacle.bounding_boxs[0].x;
    p1.y = Obstacle.bounding_boxs[0].y;
    p2.x = Obstacle.bounding_boxs[1].x;
    p2.y = Obstacle.bounding_boxs[1].y;
    p3.x = Obstacle.bounding_boxs[2].x;
    p3.y = Obstacle.bounding_boxs[2].y;
    p4.x = Obstacle.bounding_boxs[3].x;
    p4.y = Obstacle.bounding_boxs[3].y;
    
    float S1 = ((p4.x -p2.x)*(p1.y - p2.y) - (p4.y - p2.y)*(p1.x - p2.x))/2;
    float S2 = ((p4.x -p2.x)*(p2.y - p3.y) - (p4.y - p2.y)*(p2.x - p3.x))/2;
    cp.x = p1.x + (p3.x-p1.x)*S1/(S1+S2);
    cp.y = p1.y + (p3.y-p1.y)*S1/(S1+S2);
    return cp;
}

void GenerateTrajectoryBundles(std::array<double, 2> init_s,std::array<double, 2> init_l,ST_graph STGraph,
                              Trajectory1DBundle &lon_trajectory1d_bundle,Trajectory1DBundle &lat_trajectory1d_bundle)
{
    GenerateLongitudinalTrajectoryBundle(init_s,STGraph,lon_trajectory1d_bundle); 

    GenerateLateralTrajectoryBundle(init_l,lat_trajectory1d_bundle);
}

//生成纵向的线簇
void GenerateLongitudinalTrajectoryBundle(std::array<double, 2> init_s,ST_graph STGraph,Trajectory1DBundle &lon_trajectory1d_bundle)
{
    GenerateSpeedProfilesForCruising(init_s,lon_trajectory1d_bundle);
    GenerateSpeedProfilesForPathTimeObstacles(init_s, STGraph,lon_trajectory1d_bundle);
}

//用于巡航的线簇
void GenerateSpeedProfilesForCruising(std::array<double, 2> init_s,Trajectory1DBundle &lon_trajectory1d_bundle)
{
    trajectory1d trajectory1d_;
    trajectory1d_.start_condition_ = init_s;
    trajectory1d_.target_time = 0.01;
    trajectory1d_.target_vel = trajectory1d_.end_condition_[1] = V_straight;
    trajectory1d_.coef4_ = ComputeCoefficients(init_s[0],init_s[1],trajectory1d_.target_vel,0.01);
    trajectory1d_.end_condition_[0] = ComputeS(trajectory1d_);
    lon_trajectory1d_bundle.push_back(trajectory1d_);
    for (size_t t = 1; t <= trajectory_time_length; t++)
    {
        trajectory1d trajectory1d_;
        trajectory1d_.start_condition_ = init_s;
        trajectory1d_.target_time = t;
        trajectory1d_.target_vel = trajectory1d_.end_condition_[1] = V_straight;
        trajectory1d_.coef4_ = ComputeCoefficients(init_s[0],init_s[1],trajectory1d_.target_vel,t);
        trajectory1d_.end_condition_[0] = ComputeS(trajectory1d_);
        lon_trajectory1d_bundle.push_back(trajectory1d_);
    }
    
}
//用于障碍物的线簇
void GenerateSpeedProfilesForPathTimeObstacles(std::array<double, 2> init_s,ST_graph STGraph,Trajectory1DBundle &lon_trajectory1d_bundle)
{
    // for (size_t i = 0; i < STGraph.ST_graph_node.size(); i++)
    // {
    //     const int timespan = 1.0;
    //     for (size_t time = 0; time < STGraph.ST_graph_node[i].Min_s_point.size(); time+=timespan*10)
    //     {
    //         for (size_t s = STGraph.ST_graph_node[i].left_bottom_point.s - longitudinal_safe_distance;
    //                                  s > STGraph.ST_graph_node[i].left_bottom_point.s-5; s-=2)
    //         {
    //             trajectory1d trajectory1d_;
    //             trajectory1d_.start_condition_ = init_s;
    //             trajectory1d_.target_time = time;
    //             trajectory1d_.end_condition_[0] = s;
    //             trajectory1d_.target_vel = trajectory1d_.end_condition_[1] = STGraph.ST_graph_node[i].s_vel;
    //             trajectory1d_.coef5_ = ComputeCoefficients(init_s[0],init_s[1],trajectory1d_.end_condition_[0],trajectory1d_.end_condition_[1],time);
    //             lon_trajectory1d_bundle.push_back(trajectory1d_);

    //         }
            
    //     }
        
    // } 
    ST_graph_single stGraph_node_safe;

    const float maxFloat = std::numeric_limits<float>::max();
    const float timespan = 0.5;
    for (float time = 0; time <= trajectory_time_length; time+=timespan)
    {
        STPoint point = {maxFloat , time};
        for (size_t i = 0; i < STGraph.ST_graph_node.size(); i++)
        {
            cout<<STGraph.ST_graph_node[i].Min_s_point.size()<<endl;
            if (STGraph.ST_graph_node[i].Min_s_point[time*5].s<point.s)
            {
                point.s = STGraph.ST_graph_node[i].Min_s_point[time*5].s;
            }
            // cout<<"STGraph min_s : "<<point.s - car_point.absolute_s<<endl;
        }
        stGraph_node_safe.Min_s_point.push_back(point);
    }
    STPoint point_min = {maxFloat , 0};
    for (size_t i = 0; i < stGraph_node_safe.Min_s_point.size(); i++)
    {
        if (stGraph_node_safe.Min_s_point[i].s<=point_min.s)
        {
            point_min.s = stGraph_node_safe.Min_s_point[i].s;
            point_min.t = stGraph_node_safe.Min_s_point[i].t;
        }
    }
    float V_s = V_straight*cos(car_ddaw);
    float car_vel_s =  min((point_min.s - car_point.absolute_s)/point_min.t , V_s);
    target_v = car_vel_s/cos(car_ddaw);
    if (obstacleList_long->obstacles.size())
    {
        target_v = 0;
    }
    
    cout<<"target_v car_vel_s: "<<target_v<<" "<<car_vel_s<<" "<<(point_min.s - car_point.absolute_s)/point_min.t<<" "<<point_min.t<<" "<<V_s<<endl;
}


//生成横向的线簇
void GenerateLateralTrajectoryBundle(std::array<double, 2> init_l,Trajectory1DBundle &lat_trajectory1d_bundle)
{
    // std::array<double, 13> end_d_candidates = {0.0, -0.25, 0.25, -0.5, 0.5, -0.75, 0.75, -1.0, 1.0,-1.25,1.25,-1.5,1.5};
    std::array<double,3 > end_s_candidates = {2.0,4.0,6.0};
     for (const auto& s : end_s_candidates) {
    for (float l = -sampling_max_l;l<=sampling_max_l;l+=0.25) {
        trajectory1d trajectory1d_;
        trajectory1d_.start_condition_ = init_l;
        trajectory1d_.end_condition_[0] = l;
        trajectory1d_.end_condition_[1] = 0;
        trajectory1d_.coef4_ = ComputeCoefficients_lateral(init_l[0],init_l[1],l,s);  //终点在线段两侧
        trajectory1d_.target_s = s;
        lat_trajectory1d_bundle.push_back(trajectory1d_);
    }
  }

}

void trajectory_evaluate(Trajectory1DBundle &lat_trajectory1d_bundle , planning_msgs::ObstacleList ObstacleList_static)
{
    cout<<"trajectory_evaluate!!!"<<endl;
    // float maxl = std::numeric_limits<float>::lowest();
    // for (const auto& trajectory1d : lat_trajectory1d_bundle)
    // {
    //     if(trajectory1d_.end_condition_[0]>maxl)
    //     {
    //         maxl = trajectory1d_.end_condition_[0];
    //     }
    // }
    // cout<<"检测前："<<lat_trajectory1d_bundle.size()<<endl;
    // //trajectory_feasibility_test(lat_trajectory1d_bundle);
    // cout<<"检测后："<<lat_trajectory1d_bundle.size()<<endl;
    int count = 0;
    for ( auto& trajectory1d : lat_trajectory1d_bundle)
    {
        count++;
        trajectory1d.cost_lateral_offsets = Calcost_lateral_offsets(trajectory1d);
        trajectory1d.cost_lateral_comfort = Calcost_lateral_comfort(trajectory1d);
        trajectory1d.cost_crash =  Calcost_crash(trajectory1d,ObstacleList_static);
        trajectory1d.cost = K_offsets*trajectory1d.cost_lateral_offsets + K_comfort*trajectory1d.cost_lateral_comfort+trajectory1d.cost_crash;
        // cout<<"cost: "<<trajectory1d.cost<<" "<<trajectory1d.cost_lateral_offsets<<" "<<trajectory1d.cost_lateral_comfort
        //     <<" "<<trajectory1d.cost_crash<<endl;
        //     if (count%17 == 0)
        //     {
        //         cout<<endl;
        //     }
            
    }

    std::sort(lat_trajectory1d_bundle.begin(), lat_trajectory1d_bundle.end(),
        [](const trajectory1d &d0, const trajectory1d& d1) {
            return d0.cost < d1.cost;
        });
    if (lat_trajectory1d_bundle.begin()->cost<100)
    {
        trajectory_accessible = *lat_trajectory1d_bundle.begin();
    }
    if (trajectory_accessible.target_s>0) //有解了
    {
        GenerateReferenceLine(trajectory_accessible,ReferenceLine);
        ShowReferenceLine(ReferenceLine,LateralTrajectory);
    }
    

}

void trajectory_feasibility_test(Trajectory1DBundle &lat_trajectory1d_bundle)
{
    Trajectory1DBundle lat_trajectory1d_bundle_;
    for ( auto& trajectory1d : lat_trajectory1d_bundle)
    {
        float alpha0 = trajectory1d.coef4_[0];
        float alpha1 = trajectory1d.coef4_[1];
        float alpha2 = trajectory1d.coef4_[2];
        float alpha3 = trajectory1d.coef4_[3];
        float S = trajectory1d.target_s;
        float s;
        for (s = 0; s < S; s+=0.2)
        {
            float l = alpha0+alpha1*s+alpha2*s*s+alpha3*s*s*s;
            float dl = alpha1 + 2*alpha2*s + 3*alpha3*s*s;
            float ddl = 2*alpha2 + 6*alpha3*s;
            float curv = abs(ddl)/pow(1+dl*dl,3/2);
            if (curv>curv_max)
            {
                break;
            }
        }
        if (s>=S)
        {
            lat_trajectory1d_bundle_.push_back(trajectory1d);
        }   
    }
    lat_trajectory1d_bundle = lat_trajectory1d_bundle_; 

}
float Calcost_lateral_offsets(trajectory1d trajectory1d)
{
    float alpha0 = trajectory1d.coef4_[0];
    float alpha1 = trajectory1d.coef4_[1];
    float alpha2 = trajectory1d.coef4_[2];
    float alpha3 = trajectory1d.coef4_[3];
    float S = trajectory1d.target_s;
    float l_offset_init = trajectory1d.start_condition_[0];
    double cost_sqr_sum = 0.0;
    double cost_abs_sum = 0.0;
    for (float s = 0; s < S; s+=0.2)
    {
        float l_offset = alpha0+alpha1*s+alpha2*s*s+alpha3*s*s*s;
        if (l_offset * l_offset_init < 0.0) {
        cost_sqr_sum += l_offset * l_offset * 10;
        cost_abs_sum += std::fabs(l_offset) * 10;
        } else {
        cost_sqr_sum += l_offset * l_offset;
        cost_abs_sum += std::fabs(l_offset);
        }
    } 
    float cost_lateral_offsets = cost_sqr_sum / (cost_abs_sum + 1e-6);
    return cost_lateral_offsets;
}

float Calcost_lateral_comfort(trajectory1d trajectory1d)
{
    float alpha0 = trajectory1d.coef4_[0];
    float alpha1 = trajectory1d.coef4_[1];
    float alpha2 = trajectory1d.coef4_[2];
    float alpha3 = trajectory1d.coef4_[3];
    float S = trajectory1d.target_s;
    float cost_lateral_comfort = 0.0;
    for (float s = 0; s < S; s+=0.2)
    {
        float l_primeprime = 2*alpha2 + 6*alpha3*s;
        cost_lateral_comfort = max(cost_lateral_comfort,fabs(l_primeprime));
    } 
    return cost_lateral_comfort;
}

float Calcost_crash(trajectory1d trajectory1d,planning_msgs::ObstacleList ObstacleList_static)
{
    float alpha0 = trajectory1d.coef4_[0];
    float alpha1 = trajectory1d.coef4_[1];
    float alpha2 = trajectory1d.coef4_[2];
    float alpha3 = trajectory1d.coef4_[3];
    float S = trajectory1d.target_s; 
    float extra_cost = 0.0;
    if (S == 4)
    {
        extra_cost = 10;
    }else if (S == 2)
    {
        extra_cost = 20;
    }
    
    
    float cost_crash = 0.0;
    for (const auto& obstacle : ObstacleList_static.obstacles)
    {
        float min_s = obstacle.min_s- car_point.absolute_s - obstacle_safe_range;
        min_s = max(min_s,(float)0.0);
        float max_s = obstacle.max_s - car_point.absolute_s + obstacle_safe_range;
        max_s = min(max_s,S);
        float min_l = obstacle.min_l - obstacle_safe_range;
        float max_l = obstacle.max_l + obstacle_safe_range;
        for (float s = min_s; s < max_s; s+=0.2)
        {
            float l_offset = alpha0+alpha1*s+alpha2*s*s+alpha3*s*s*s;
            // cout<<"s l:"<<s<<" "<<l_offset<<endl;
            if (l_offset>=min_l&&l_offset<=max_l)
            {
                cost_crash = 1e2;
                break;
            }
        } 
        if(cost_crash)
        {
            break;
        }
    }
    
    return cost_crash + extra_cost;
}

void GenerateReferenceLine(trajectory1d trajectory1d,planning_msgs::car_path &ReferenceLine)
{ 
    // planning_msgs::car_info car_info;
    ReferenceLine.car_direct = car_info->speedDrietion;
    ReferenceLine.V_straight = target_v;
    
    ReferenceLine.RunningNormally = true;
    ReferenceLine.car_wait = false;
    if(abs(car_point.absolute_s-global_path.points.back().absolute_s)<0.2)
    {
        ReferenceLine.car_wait = true;
    }
    cout<<"GenerateReferenceLine!!!"<<endl;
    float alpha0 = trajectory1d.coef4_[0];
    float alpha1 = trajectory1d.coef4_[1];
    float alpha2 = trajectory1d.coef4_[2];
    float alpha3 = trajectory1d.coef4_[3];
    float S = trajectory1d.target_s;
    // cout<<"alpha0 ,1 ,2 ,3: "<<alpha0<<" "<<alpha1<<" "<<alpha2<<" "<<alpha3<<endl;
    for (float s = 0; s < S; s+=0.1)
    {
        float l = alpha0+alpha1*s+alpha2*s*s+alpha3*s*s*s;
        planning_msgs::path_point matched_point , referenceLine_point;
        matched_point = MatchToPath(global_path,s + car_point.absolute_s);
        PointXY point = frenet_to_cartesian(matched_point,l);
        referenceLine_point.x = point.x;
        referenceLine_point.y = point.y;
        ReferenceLine.points.push_back(referenceLine_point);
    } 
    for (size_t i = 1; i < ReferenceLine.points.size()-2; i++)
    {
        float x0 = ReferenceLine.points[i].x;
        float y0 = ReferenceLine.points[i].y;
        float x1 = ReferenceLine.points[i+1].x;
        float y1 = ReferenceLine.points[i+1].y;
        float x2 = ReferenceLine.points[i+2].x;
        float y2 = ReferenceLine.points[i+2].y;
        float yaw1 = atan2(y1-y0,x1-x0);
        float yaw2 = atan2(y2-y1,x2-x1);
        float yaw;
        if((abs(yaw1+yaw2)<M_PI) && abs(yaw1-yaw2)>M_PI)
        {
            if(abs(yaw1)>=abs(yaw2)) 
            {
                if(yaw1>=0)
                {
                    yaw1 -= 2*M_PI;
                    yaw = (yaw1 + yaw2)/2;
                }else{
                    yaw1 += 2*M_PI;
                    yaw = (yaw1 + yaw2)/2;
                }
            }else{
                if(yaw2>=0)
                {
                    yaw2 -= 2*M_PI;
                    yaw = (yaw1 + yaw2)/2;
                }else{
                    yaw2 += 2*M_PI;
                    yaw = (yaw1 + yaw2)/2;
                }
            }
        }else{
            yaw = (yaw1 + yaw2)/2;
        }

        ReferenceLine.points[i].yaw = yaw;
    }
    ReferenceLine.points.begin()->yaw = ReferenceLine.points[1].yaw;
    ReferenceLine.points[ReferenceLine.points.size()-2].yaw = ReferenceLine.points.back().yaw 
    =  ReferenceLine.points[ReferenceLine.points.size()-3].yaw;

}

//显示
void ShowReferenceLine(planning_msgs::car_path ReferenceLine,pcl::PointCloud<pcl::PointXYZI>::Ptr LateralTrajectory)
{
    cout<<"ShowReferenceLine!!!"<<endl;
    for (const auto & ReferencePoint : ReferenceLine.points)
    {
        // cout<<"ReferencePoint yaw: "<<ReferencePoint.yaw<<endl;
        pcl::PointXYZI points;
        points.x = ReferencePoint.x;
        points.y = ReferencePoint.y;
        points.z = 0;
        LateralTrajectory->push_back(points);
    }
    
}
void ShowLateralTrajectoryBundle(Trajectory1DBundle lat_trajectory1d_bundle,pcl::PointCloud<pcl::PointXYZI>::Ptr LateralTrajectory)
{
    cout<<"ShowLateralTrajectoryBundle!!!"<<endl;
    for (const auto& trajectory1d : lat_trajectory1d_bundle)
    {
        float alpha0 = trajectory1d.coef4_[0];
        float alpha1 = trajectory1d.coef4_[1];
        float alpha2 = trajectory1d.coef4_[2];
        float alpha3 = trajectory1d.coef4_[3];
        float S = trajectory1d.target_s;
        // cout<<"alpha0 ,1 ,2 ,3: "<<alpha0<<" "<<alpha1<<" "<<alpha2<<" "<<alpha3<<endl;
        for (float s = 0; s < S; s+=0.1)
        {
            float l = alpha0+alpha1*s+alpha2*s*s+alpha3*s*s*s;
            planning_msgs::path_point matched_point;
            matched_point = MatchToPath(global_path,s + car_point.absolute_s);
            PointXY point = frenet_to_cartesian(matched_point,l);
            pcl::PointXYZI points;
            points.x = point.x;
            points.y = point.y;
            points.z = 0;
            LateralTrajectory->push_back(points);
        } 
    }
}

void ShowBestLateralTrajectory(trajectory1d trajectory1d,pcl::PointCloud<pcl::PointXYZI>::Ptr LateralTrajectory)
{
    cout<<"ShowBestLateralTrajectory!!!"<<endl;

    float alpha0 = trajectory1d.coef4_[0];
    float alpha1 = trajectory1d.coef4_[1];
    float alpha2 = trajectory1d.coef4_[2];
    float alpha3 = trajectory1d.coef4_[3];
    float S = trajectory1d.target_s;
    // cout<<"alpha0 ,1 ,2 ,3: "<<alpha0<<" "<<alpha1<<" "<<alpha2<<" "<<alpha3<<endl;
    for (float s = 0; s < S; s+=0.1)
    {
        float l = alpha0+alpha1*s+alpha2*s*s+alpha3*s*s*s;
        planning_msgs::path_point matched_point;
        matched_point = MatchToPath(global_path,s + car_point.absolute_s);
        PointXY point = frenet_to_cartesian(matched_point,l);
        pcl::PointXYZI points;
        points.x = point.x;
        points.y = point.y;
        points.z = 0;
        LateralTrajectory->push_back(points);
    } 
}

bool get_newGoalPoint()
{
    if (!obstacleList_long->obstacles.size()) return false;
    float max_s = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < obstacleList_long->obstacles.size(); i++)
    {
        if (obstacleList_long->obstacles[i].max_s>max_s)
        {
            max_s = obstacleList_long->obstacles[i].max_s;
        }
    }
    planning_msgs::path_point matched_point;
    max_s = max_s + 4;
    max_s = min(max_s,global_path.points[global_path.points.size() - 1].absolute_s);
    matched_point = MatchToPath(global_path,max_s);
    obstacleList_long->goal_point.x = matched_point.x;
    obstacleList_long->goal_point.y = matched_point.y;
    obstacleList_long->goal_point.theta = matched_point.yaw;
    return true;
}

/********************生成参考线***********************/
void lattice_planner::read_path(string file)
{
    string line;
    file_.open(file,ios::in|ios::out);
    int lineCount = 0;
    while (getline(file_, line)) {
            lineCount++;
        } 
        // 重新定位文件流到文件开头
    file_.clear();  // 清除文件流的状态
    file_.seekg(0, std::ios::beg);  // 将文件流定位到文件开头
    int i_num=0; 
    Eigen::VectorXd Solution(lineCount*2);
    while (getline(file_, line))
    {
        std::istringstream iss(line);
        iss >> Solution(i_num) >> Solution(i_num+1);
        i_num=i_num+2;
        // cout<<"lineCount:"<<lineCount<<"   i_num:"<<i_num<<endl;
    }
    total_points_num = lineCount;
    // 路径优化
    // Eigen::VectorXd QPSolution = Smooth_Reference_Line(line_record_);

    // 计算路线航向角(斜率）
    vector<float> temp_phi(total_points_num);
    Calculate_OptimizedPath_Heading(Solution, temp_phi);
    // 进行均值插值
    Mean_Interpolation(line_record_opt, line_record);
    // 路径参数计算
    Path_Parameter_Calculate(global_path);
}

/**
 * @description: 通过优化后的路径来计算航向角
 * @param {VectorXd} QPSolution 优化后路径点
 * @param {vector<float>&} temp_phi 临时存储航向角变量
 * @return {*}
 */
void Calculate_OptimizedPath_Heading(const Eigen::VectorXd &QPSolution, vector<float> &temp_phi)
{

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
 * @description: 均值插值
 * @param {Ptr} line_record_opt 输入路径点
 * @param {Ptr} line_record 输出插值路径点
 * @return {*}
 */
void Mean_Interpolation(pcl::PointCloud<pcl::PointXYZI>::Ptr line_record_opt, pcl::PointCloud<pcl::PointXYZI>::Ptr line_record)
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
    line_record_watch->clear();
    for (size_t i = 0; i < line_record->size(); i++)
    {
        pcl::PointXYZI points = line_record->points[i];
        points.z = 0;
        line_record_watch->points.push_back(points);
    }
    
}

void Path_Parameter_Calculate(planning_msgs::car_path &path)
{
    cout << "Path_Parameter_Calculate" << endl;
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
    cout<<path.points.back().yaw<<endl;
    cout<<path.points[path.points.size()-2].yaw<<endl;
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
    // 基于三点求外接圆的曲率计算方法
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
void get_front_point_and_setVel(planning_msgs::car_path &path_, int begin)
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

std::array<double, 4> ComputeCoefficients(double x0, double dx0, double dx1, double param)
{
    std::array<double, 4> coef_ = {{0.0, 0.0, 0.0, 0.0}};
    coef_[0] = x0;
    coef_[1] = dx0;
    coef_[2] = (dx0 - dx1)/param;
    coef_[3] = (dx1 - dx0)/(3*param*param);
    return coef_;
}

//默认dx1为0
std::array<double, 4> ComputeCoefficients_lateral(double x0, double dx0, double x1, double t) 
{
    std::array<double, 4> coef_ = {{0.0, 0.0, 0.0, 0.0}};
    Eigen::Matrix<double, 4, 4> A;
     A<< 1 , 0 , 0  , 0    ,
         0 , 1 , 0  , 0    ,
         1 , t , t*t, t*t*t,
         0 , 1 , 2*t, 3*t*t;
    Eigen::Matrix<double, 4, 1> B;
    B<<x0,dx0,x1,0;
    Eigen::Matrix<double, 4, 1> result = A.inverse()*B;
    coef_[0] = result(0);
    coef_[1] = result(1);
    coef_[2] = result(2);
    coef_[3] = result(3);
    return coef_;
}

std::array<double, 5> ComputeCoefficients(double x0, double dx0, double x1, double dx1, double t)
{
    std::array<double, 5> coef_ = {{0.0, 0.0, 0.0, 0.0, 0.0}};
    Eigen::Matrix<double, 5, 5> A;
    A<< 1 , 0 , 0  , 0     , 0 ,
        0 , 1 , 0  , 0     , 0 ,
        0 , 0 , 2  , 6*t   , 12*t*t,
        1 , t , t*t, t*t*t , t*t*t*t,
        0 , 1 , 2*t, 3*t*t , 4*t*t*t;
    Eigen::Matrix<double, 5, 1> B;
    B<<x0,dx0,0,x1,dx1;
    Eigen::Matrix<double, 5, 1> result = A.inverse()*B;

    coef_[0] = result(0);
    coef_[1] = result(1);
    coef_[2] = result(2);
    coef_[3] = result(3);
    coef_[4] = result(4);
    return coef_;
}

float ComputeS(trajectory1d trajectory1d)
{
    float t = trajectory1d.target_time;
    float alpha0 = trajectory1d.coef4_[0];
    float alpha1 = trajectory1d.coef4_[1];
    float alpha2 = trajectory1d.coef4_[2];
    float alpha3 = trajectory1d.coef4_[3];
    float s = alpha0 + alpha1*t + alpha2*t*t + alpha3*t*t*t;
    return s;
}