#include <cmath>
#include <fstream>
#include <cstdint>
#include <memory>
#include <string>

#include <ros/ros.h>
#include <ros/package.h>

#include <emplanner.hpp>
#include <planning_msgs/Obstacle.h>
#include <planning_msgs/ObstacleList.h>
#include <planning_msgs/car_info.h>
#include <planning_msgs/car_scene.h>

class EmplannerRvizSim
{
public:
  EmplannerRvizSim(ros::NodeHandle& nh, ros::NodeHandle& pnh) : nh_(nh), pnh_(pnh)
  {
    loadParams();

    planner_ = std::make_shared<EMPlanner>(nh_, car_scene_);

    obstacle_pub_ = nh_.advertise<planning_msgs::ObstacleList>("/obstacleList_lidar", 1, true);
    car_pos_pub_ = nh_.advertise<planning_msgs::car_info>("/car_pos", 1);
    car_info_pub_ = nh_.advertise<planning_msgs::car_info>("/car_info", 1);

    publishCarStateNow();

    obstacle_timer_ = nh_.createTimer(ros::Duration(obstacle_period_), &EmplannerRvizSim::publishObstacles, this);
    car_timer_ = nh_.createTimer(ros::Duration(car_state_period_), &EmplannerRvizSim::publishCarState, this);
    plan_timer_ = nh_.createTimer(ros::Duration(plan_period_), &EmplannerRvizSim::runPlanner, this);
  }

private:
  void loadParams()
  {
    pnh_.param<std::string>("frame_id", frame_id_, std::string("map"));
    pnh_.param<double>("plan_period", plan_period_, 0.1);
    pnh_.param<double>("obstacle_period", obstacle_period_, 0.1);
    pnh_.param<double>("car_state_period", car_state_period_, 0.05);
    pnh_.param<double>("car_speed", car_speed_, 0.2);

    int floor_param = 3;
    pnh_.param<int>("floor", floor_param, floor_param);
    int task_type_param = 0;
    pnh_.param<int>("task_type", task_type_param, task_type_param);
    bool is_indoor_param = true;
    pnh_.param<bool>("is_indoor", is_indoor_param, is_indoor_param);

    car_scene_.floor = static_cast<int8_t>(floor_param);
    car_scene_.task_type = static_cast<uint8_t>(task_type_param);
    car_scene_.is_indoor = is_indoor_param;

    double default_x = 0.0;
    double default_y = 0.0;
    double default_yaw = 0.0;
    if (!loadDefaultPoseFromPath(floor_param, default_x, default_y, default_yaw))
    {
      ROS_WARN_STREAM("Falling back to zero start pose for RViz sim.");
    }

    pnh_.param<double>("start_x", car_x_, default_x);
    pnh_.param<double>("start_y", car_y_, default_y);
    pnh_.param<double>("start_yaw", car_yaw_, default_yaw);

    pnh_.param<double>("obstacle_center_x", obs_cx_, car_x_ + 2.0);
    pnh_.param<double>("obstacle_center_y", obs_cy_, car_y_ + 0.5);
    pnh_.param<double>("obstacle_length", obs_length_, 1.0);
    pnh_.param<double>("obstacle_width", obs_width_, 0.8);
  }

  bool loadDefaultPoseFromPath(int floor, double& x, double& y, double& yaw)
  {
    const std::string file_path = ros::package::getPath("planner") + "/text/F" + std::to_string(floor) + "/elevator_path_op.txt";
    std::ifstream path_file(file_path);

    double x1 = 0.0;
    double y1 = 0.0;
    double x2 = 0.0;
    double y2 = 0.0;

    if (!path_file.good() || !(path_file >> x1 >> y1))
    {
      ROS_WARN_STREAM("Unable to read path file: " << file_path);
      return false;
    }

    x = x1;
    y = y1;
    if (path_file >> x2 >> y2)
    {
      yaw = std::atan2(y2 - y1, x2 - x1);
    }
    else
    {
      yaw = 0.0;
    }

    return true;
  }

  planning_msgs::Obstacle createBoxObstacle(int id, double center_x, double center_y, double length, double width)
  {
    planning_msgs::Obstacle obs;
    const double half_l = length * 0.5;
    const double half_w = width * 0.5;

    obs.number = id;
    obs.x = center_x;
    obs.y = center_y;
    obs.max_x = center_x + half_l;
    obs.min_x = center_x - half_l;
    obs.max_y = center_y + half_w;
    obs.min_y = center_y - half_w;
    obs.x_vel = 0.0;
    obs.y_vel = 0.0;
    obs.s_vel = 0.0;
    obs.l_vel = 0.0;

    obs.bounding_boxs[0].x = obs.max_x;
    obs.bounding_boxs[0].y = obs.max_y;
    obs.bounding_boxs[1].x = obs.max_x;
    obs.bounding_boxs[1].y = obs.min_y;
    obs.bounding_boxs[2].x = obs.min_x;
    obs.bounding_boxs[2].y = obs.max_y;
    obs.bounding_boxs[3].x = obs.min_x;
    obs.bounding_boxs[3].y = obs.min_y;

    return obs;
  }

  void publishObstacles(const ros::TimerEvent& /*event*/)
  {
    planning_msgs::ObstacleList list;
    list.header.stamp = ros::Time::now();
    list.header.frame_id = frame_id_;
    list.goal_point.header = list.header;
    list.goal_point.x = 0.0;
    list.goal_point.y = 0.0;
    list.goal_point.theta = 0.0;

    list.obstacles.push_back(createBoxObstacle(0, obs_cx_, obs_cy_, obs_length_, obs_width_));
    obstacle_pub_.publish(list);
  }

  void publishCarStateNow()
  {
    planning_msgs::car_info msg;
    msg.header.stamp = ros::Time::now();
    msg.x = car_x_;
    msg.y = car_y_;
    msg.yaw = car_yaw_;
    msg.speedDrietion = (car_speed_ >= 0.0) ? 1 : -1;
    msg.speed = std::fabs(car_speed_);
    msg.turnAngle = 0.0;
    msg.yawrate = 0.0;

    car_pos_pub_.publish(msg);
    car_info_pub_.publish(msg);
  }

  void publishCarState(const ros::TimerEvent& event)
  {
    const double dt = event.last_real.isZero() ? car_state_period_ : (event.current_real - event.last_real).toSec();

    car_x_ += std::cos(car_yaw_) * car_speed_ * dt;
    car_y_ += std::sin(car_yaw_) * car_speed_ * dt;

    publishCarStateNow();
  }

  void runPlanner(const ros::TimerEvent& /*event*/)
  {
    if (planner_)
    {
      planner_->Plan(car_scene_);
    }
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  std::shared_ptr<EMPlanner> planner_;

  planning_msgs::car_scene car_scene_;

  ros::Publisher obstacle_pub_;
  ros::Publisher car_pos_pub_;
  ros::Publisher car_info_pub_;

  ros::Timer obstacle_timer_;
  ros::Timer car_timer_;
  ros::Timer plan_timer_;

  std::string frame_id_;

  double car_x_ = 0.0;
  double car_y_ = 0.0;
  double car_yaw_ = 0.0;
  double car_speed_ = 0.0;

  double plan_period_ = 0.1;
  double obstacle_period_ = 0.1;
  double car_state_period_ = 0.05;

  double obs_cx_ = 0.0;
  double obs_cy_ = 0.0;
  double obs_length_ = 1.0;
  double obs_width_ = 0.8;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "emplanner_rviz_sim");

  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  EmplannerRvizSim sim(nh, pnh);

  ros::AsyncSpinner spinner(2);
  spinner.start();
  ros::waitForShutdown();
  return 0;
}
