#include"include/rtk_planner.h"

// 构造函数，初始化发布器
RTK_planner::RTK_planner(ros::NodeHandle& nh,const planning_msgs::car_scene &car_scene)
{
    last_task_type=-1;
    car_scene_ = std::shared_ptr<planning_msgs::car_scene>(new planning_msgs::car_scene(car_scene)); 
    rtk_path_watch = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    Car_Pose = boost::make_shared<geometry_msgs::Pose2D>();

    RawRoute_pub_watch = nh.advertise<sensor_msgs::PointCloud2>("rtk_path", 10);;  // 未优化路线发布器
    RawRoute_pub = nh.advertise<planning_msgs::hybrid_astar_paths>("/enter_elevator_path", 10);  // 初始化发布器
    // PatrolRoute_pub = nh.advertise<planning_msgs::car_path>("/route", 10);  // 初始化发布器

    RTK_planner::sub_location = nh.subscribe<planning_msgs::car_info>("/car_pos", 10, &RTK_planner::callBack_location, this); 
    *car_scene_=car_scene;
    loadRoute(*car_scene_);
}

void RTK_planner::Plan(planning_msgs::car_scene &car_scene)
{
    if (car_scene_->floor!=car_scene.floor||car_scene_->task_type!=car_scene.task_type)  //当车辆场景发生改变，重新读取
    {
        bool route_is_load=loadRoute(car_scene);
        *car_scene_=car_scene;
        if (!route_is_load) {
            ROS_WARN("Failed to load route for floor %d, task type %d.", car_scene.floor, car_scene.task_type);
        }
    }
    
    publishRoute();
}

// 从文件加载路线数据
bool RTK_planner::loadRoute(planning_msgs::car_scene car_scene)
{
    if (car_scene.task_type==1 && last_task_type !=1)  //准备进入电梯
    {
        last_task_type=1;
        return loadElevatorPath(car_scene);
    }
    
    return false;
}

bool RTK_planner::SwitchScene(planning_msgs::car_scene &car_scene)
{
    
}

// 发布路线
void RTK_planner::publishRoute()
{
    RawRoute_pub.publish(route);  // 发布包含路线数据的消息
    ss_local_path_watch.header.frame_id = "velodyne";
    RawRoute_pub_watch.publish(ss_local_path_watch);
}

bool RTK_planner::loadElevatorPath(planning_msgs::car_scene car_scene)
{
    route.Path.clear();
    std::stringstream filename;
    filename << base_path <<  std::to_string(car_scene.floor) << "/enter_elevator_path.txt";

    std::ifstream file(filename.str());  // 打开指定的文件
    if (!file.is_open()) {
        ROS_ERROR("Failed to open file: %s", filename.str().c_str());
        return false;
    }


    planning_msgs::hybrid_astar_path path;
    //TODO:倒车时候有bug 起点为倒车的路线可能会有错误
    //down : if (path.single_path.size()>0)  加了判断
            
    int vel_dir=1;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream line_stream(line);
        planning_msgs::hybrid_astar_path_point path_point;
        if (!(line_stream >> path_point.x >> path_point.y >> path_point.yaw >> path_point.theta >> path_point.vel)) {
            ROS_WARN("Malformed line, skipping: %s", line.c_str());
            continue;
        }

        if (vel_dir*path_point.vel>0 &&abs(path_point.vel)>0.05)
        {
            path.direct=vel_dir;

            path.single_path.push_back(path_point);
        }
        else if(path_point.vel==0||abs(path_point.vel)<0.05)  //防止在倒车与前进的状态出现突变
        {
            
        }
        else
        {
            vel_dir=vel_dir*(-1);
            if (path.single_path.size()>0)
            {
                route.Path.push_back(path);
            }

            path.single_path.clear();
            path.single_path.push_back(path_point);
        }
            std::cout<< path_point.x<<","<< path_point.y<<","<< path_point.yaw<<","<< path_point.theta<<","<< path_point.vel<<std::endl;

    }
    route.Path.push_back(path);
    route.path_type=1;

    rtk_path_watch.reset(new pcl::PointCloud<pcl::PointXYZI>);
    for (int i = 0; i < route.Path.size(); i++)
    {
        for (int  j = 0; j < route.Path[i].single_path.size(); j++)
        {
            pcl::PointXYZI local_path_point;
            local_path_point.x = route.Path[i].single_path[j].x;
            local_path_point.y = route.Path[i].single_path[j].y;
            rtk_path_watch->points.push_back(local_path_point);
        }
    }

    pcl::toROSMsg(*rtk_path_watch, ss_local_path_watch);
    ss_local_path_watch.header.frame_id = "velodyne";
    RawRoute_pub_watch.publish(ss_local_path_watch);
    
    file.close();  // 关闭文件
    return true;
}

void RTK_planner::callBack_location(const planning_msgs::car_info::ConstPtr &car_pose)//NDT定位回调
{
    Car_Pose->x = car_pose->x; // 纠正为后车身XY坐标
    Car_Pose->y = car_pose->y; // 纠正为后车身XY坐标
    Car_Pose->theta = car_pose->yaw;

}

// 析构函数
RTK_planner::~RTK_planner() {}