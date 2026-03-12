#include "include/planner.h"

Planner::Planner(ros::NodeHandle& nh)
{
    // TODO: 测试用
    start_floor=1;
    start_task_type=0;
    scene.floor=start_floor;
    scene.task_type=start_task_type;
    std::cout << "Debug: scene.floor = " << static_cast<int>(scene.floor) 
          << ", scene.task_type = " << static_cast<int>(scene.task_type) << std::endl;
    //初始化
    last_task_type=-1;
    car_scene_pub = nh.advertise<planning_msgs::car_scene>("/car_scene", 100);
    scene_chang_task_info_sub = nh.subscribe<planning_msgs::car_scene>("/scene_change_task_info", 100, &Planner::change_scene_task_callBack, this); 
    scene_chang_floor_info_sub = nh.subscribe<planning_msgs::car_scene>("/scene_change_floor_info", 100, &Planner::change_scene_floor_callBack, this); 
}

void Planner::plan(ros::NodeHandle& nh)
{
    std::unique_ptr<RTK_planner> rtkPlanner;
    // rtkPlanner = std::make_unique<RTK_planner>(nh, scene);
    std::unique_ptr<EMPlanner> emPlanner;
    // emPlanner = std::make_unique<EMPlanner>(nh, scene);
    std::unique_ptr<HybridAStarFlow> hybridAStarFlow;

    ros::Rate rate(10);

    while (ros::ok()) {
        cout<<"001"<<endl;
        // 根据 task_type 判断需要构造的规划器
        if (scene.task_type != last_task_type) {
            if (scene.task_type == 1) {
                rtkPlanner = std::make_unique<RTK_planner>(nh, scene);
                emPlanner.reset();
                hybridAStarFlow.reset();
                
            } else if (scene.task_type == 0) {
                emPlanner = std::make_unique<EMPlanner>(nh,scene);
                // rtkPlanner = std::make_unique<RTK_planner>(nh, scene);
                // rtkPlanner.reset();
                 rtkPlanner.reset();
                hybridAStarFlow.reset();
            } 
            else if (scene.task_type == 10) {
                hybridAStarFlow = std::make_unique<HybridAStarFlow>(nh);
                rtkPlanner.reset();
                emPlanner.reset();
            }
            last_task_type = scene.task_type; // 更新任务类型记录
        }
        if (scene.task_type==2&&scene.floor==target_floor_number)
        {
            scene.task_type=3;
        }

        // 根据当前的规划器类型调用 plan()
        if (scene.task_type == 0 && rtkPlanner && emPlanner) {
            // rtkPlanner->Plan(scene);
        
            emPlanner->Plan(scene);
            if(emPlanner->QP_path->ReachTarget)
            {
                scene.task_type = 1;
            }
        } 
        else if (scene.task_type == 1 && rtkPlanner) {
            rtkPlanner->Plan(scene);
            //TODO:1231 完成场景切换
        } else if ((scene.task_type == 10||scene.task_type == 11) && hybridAStarFlow) {
            // std::cout<<"hybridAStarFlow->run"<<std::endl;
            hybridAStarFlow->Run(nh,scene);
        }
        else if (emPlanner)
        {
            cout<<"002"<<endl;
            emPlanner->Plan(scene);
        }
        else
        {
            ROS_WARN_THROTTLE(1.0, "No planner initialized for task_type=%d",
                              static_cast<int>(scene.task_type));
        }

        std::cout<<"scene_:floor task:"<<static_cast<int>(scene.floor)<<","<<static_cast<int>(scene.task_type)<<std::endl;
        car_scene_pub.publish(scene);

        ros::spinOnce();
        rate.sleep();
    }
}

void Planner::change_scene_task_callBack(const planning_msgs::car_scene::ConstPtr &scene_task_change)
{
    scene.task_type=scene_task_change->task_type;
}

void Planner::change_scene_floor_callBack(const planning_msgs::car_scene::ConstPtr &scene_task_change)
{
    scene.floor=scene_task_change->floor;
}
