
#include "hybrid_a_star/init_pose_subscriber.h"

InitPoseSubscriber2D::InitPoseSubscriber2D(ros::NodeHandle &nh,
                                           const std::string &topic_name,
                                           size_t buff_size) {
    subscriber_ = nh.subscribe(
            topic_name, buff_size, &InitPoseSubscriber2D::MessageCallBack, this
    );
}

void InitPoseSubscriber2D::MessageCallBack(
        const geometry_msgs::PoseWithCovarianceStampedPtr &init_pose_ptr
) {
    buff_mutex_.lock();
    init_poses_.emplace_back(init_pose_ptr);
    buff_mutex_.unlock();
}

void InitPoseSubscriber2D::MessageCallBack(
        const planning_msgs::car_info::ConstPtr& car_pose
) {
    std::cout<<"car_pos_start"<<std::endl;
    buff_mutex_.lock();
    car_position_ptr->x=car_pose->x;
    car_position_ptr->y=car_pose->y;
    car_position_ptr->yaw=car_pose->yaw;
    car_position_ptr->is_initialized=true;
    std::cout<<"car_pos:"<<car_position_ptr->x<<","<<car_position_ptr->y<<","<<car_position_ptr->yaw<<std::endl;

    // init_poses_.emplace_back(init_pose_ptr);
    buff_mutex_.unlock();
}

void InitPoseSubscriber2D::ParseData(
        std::deque<geometry_msgs::PoseWithCovarianceStampedPtr> &pose_data_buff
) {
    buff_mutex_.lock();
    if (!init_poses_.empty()) {
        pose_data_buff.insert(pose_data_buff.end(), init_poses_.begin(), init_poses_.end());
        init_poses_.clear();
    }
    buff_mutex_.unlock();
}

void InitPoseSubscriber2D::ParseData(
        geometry_msgs::Pose2D::Ptr& car_pose
) {
    if (!car_pose) {
        ROS_ERROR("Received null car_pose in ParseData");
        return;
    }
    // std::cout<<"car_pose:"<<car_pose->x<<","<<car_pose->y<<","<< car_pose->theta<<std::endl;
    if(car_position_ptr->is_initialized)
    {
        // std::cout<<"777"<<std::endl;
        car_pose->x=car_position_ptr->x;
        car_pose->y=car_position_ptr->y;
        car_pose->theta=car_position_ptr->yaw;
    }

}