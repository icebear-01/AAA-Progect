#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose2D.h>
#include <planning_msgs/car_info.h>
#include <tf/transform_datatypes.h>

#include <algorithm>
#include <random>
#include <vector>
#include <string>
#include <cmath>

struct ObstacleBox {
    double x{0.0};
    double y{0.0};
    double yaw{0.0};
    double length{1.0};
    double width{1.0};
};

class RandomParkingEnvironment {
public:
    explicit RandomParkingEnvironment(ros::NodeHandle& nh)
        : nh_(nh),
          rng_(std::random_device{}()),
          uniform_zero_one_(0.0, 1.0) {
        LoadParameters();
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/parking_env/markers", 1, true);
        car_pose_pub_ = nh_.advertise<planning_msgs::car_info>("/car_pos", 1, true);
        goal_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 1, true);

        if (auto_refresh_) {
            timer_ = nh_.createTimer(ros::Duration(refresh_period_), &RandomParkingEnvironment::TimerCallback, this);
        }

        RegenerateScenario();
    }

private:
    struct SlotInfo {
        double x{0.0};
        double y{0.0};
        double yaw{0.0};
        bool is_goal{false};
    };

    void LoadParameters() {
        world_frame_ = nh_.param<std::string>("random_env/world_frame", "velodyne");
        refresh_period_ = nh_.param("random_env/refresh_period", 3.0);
        auto_refresh_ = nh_.param("random_env/auto_refresh", true);

        boundary_length_ = nh_.param("random_env/boundary_length", 20.0);
        boundary_width_ = nh_.param("random_env/boundary_width", 16.0);
        start_clearance_ = nh_.param("random_env/start_clearance", 1.5);
        goal_clearance_ = nh_.param("random_env/goal_clearance", 2.0);

        slot_length_ = nh_.param("random_env/slot_length", 5.5);
        slot_width_ = nh_.param("random_env/slot_width", 2.7);
        slot_spacing_ = nh_.param("random_env/slot_spacing", 3.0);
        slot_row_spacing_ = nh_.param("random_env/slot_row_spacing", 6.0);
        slot_columns_ = nh_.param("random_env/slot_columns", 5);
        slot_rows_ = nh_.param("random_env/slot_rows", 3);
        goal_yaw_ = nh_.param("random_env/slot_yaw", M_PI_2);
        goal_center_x_ = nh_.param("random_env/slot_center_x", 0.0);
        goal_center_y_ = nh_.param("random_env/slot_center_y", 0.0);
        goal_radius_ = 0.5 * std::sqrt(slot_length_ * slot_length_ + slot_width_ * slot_width_);
        slot_keepout_extra_ = nh_.param("random_env/slot_keepout", 0.75);

        car_length_ = nh_.param("random_env/car_length", 4.4);
        car_width_ = nh_.param("random_env/car_width", 1.9);
        car_height_ = nh_.param("random_env/car_height", 1.8);
        car_radius_ = 0.5 * std::sqrt(car_length_ * car_length_ + car_width_ * car_width_);

        obstacle_height_ = nh_.param("random_env/obstacle_height", 1.5);
        obstacle_min_length_ = nh_.param("random_env/obstacle_min_length", 1.0);
        obstacle_max_length_ = nh_.param("random_env/obstacle_max_length", 4.0);
        obstacle_min_width_ = nh_.param("random_env/obstacle_min_width", 0.5);
        obstacle_max_width_ = nh_.param("random_env/obstacle_max_width", 3.0);
        min_obstacle_count_ = nh_.param("random_env/min_obstacles", 3);
        max_obstacle_count_ = nh_.param("random_env/max_obstacles", 7);
        obstacle_margin_from_boundary_ = nh_.param("random_env/obstacle_boundary_margin", 0.5);
        max_sampling_attempts_ = nh_.param("random_env/max_sampling_attempts", 200);
    }

    void BuildParkingSlots() {
        slots_.clear();
        goal_slot_valid_ = false;

        slot_columns_ = std::max(1, slot_columns_);
        slot_rows_ = std::max(1, slot_rows_);

        if (slot_rows_ % 2 == 0) {
            ROS_WARN_ONCE("random_env/slot_rows should be odd to center the goal slot; incrementing by one.");
            ++slot_rows_;
        }

        std::vector<double> row_offsets{0.0};
        int row_pair_count = (slot_rows_ - 1) / 2;
        for (int i = 1; i <= row_pair_count; ++i) {
            double offset = i * slot_row_spacing_;
            row_offsets.push_back(offset);
            row_offsets.push_back(-offset);
        }

        double start_x = -0.5 * slot_spacing_ * (slot_columns_ - 1);

        for (double offset : row_offsets) {
            double yaw = goal_yaw_;
            if (offset < 0.0) {
                yaw = -goal_yaw_;
            }

            for (int c = 0; c < slot_columns_; ++c) {
                double x = start_x + c * slot_spacing_;
                SlotInfo slot;
                slot.x = x + goal_center_x_;
                slot.y = offset + goal_center_y_;
                slot.yaw = yaw;
                slot.is_goal = (std::abs(x) < 1e-3 && std::abs(offset) < 1e-3);

                if (slot.is_goal) {
                    goal_slot_ = slot;
                    goal_slot_valid_ = true;
                }
                slots_.emplace_back(slot);
            }
        }

        if (!goal_slot_valid_) {
            SlotInfo goal;
            goal.x = goal_center_x_;
            goal.y = goal_center_y_;
            goal.yaw = goal_yaw_;
            goal.is_goal = true;
            goal_slot_ = goal;
            slots_.push_back(goal);
            goal_slot_valid_ = true;
        }
    }

    void TimerCallback(const ros::TimerEvent&) {
        RegenerateScenario();
    }

    void RegenerateScenario() {
        BuildParkingSlots();
        GenerateGoalPose();
        GenerateObstacles();
        if (!SampleStartPose()) {
            ROS_WARN("Failed to sample a valid start pose; reusing the previous one.");
        }
        PublishScenario();
        ROS_INFO_STREAM("Random environment ready. Start pose: (" << start_pose_.x << ", "
                        << start_pose_.y << ", " << start_pose_.theta << ") with "
                        << obstacles_.size() << " obstacles.");
    }

    void GenerateGoalPose() {
        if (!goal_slot_valid_) {
            ROS_ERROR("No goal slot defined; cannot publish goal pose.");
            return;
        }

        goal_pose_.header.frame_id = world_frame_;
        goal_pose_.pose.position.x = goal_slot_.x;
        goal_pose_.pose.position.y = goal_slot_.y;
        goal_pose_.pose.position.z = 0.0;
        goal_pose_.pose.orientation = tf::createQuaternionMsgFromYaw(goal_slot_.yaw);
    }

    void GenerateObstacles() {
        obstacles_.clear();
        std::uniform_int_distribution<int> count_dist(min_obstacle_count_, max_obstacle_count_);
        std::uniform_real_distribution<double> length_dist(obstacle_min_length_, obstacle_max_length_);
        std::uniform_real_distribution<double> width_dist(obstacle_min_width_, obstacle_max_width_);
        std::uniform_real_distribution<double> yaw_dist(-M_PI, M_PI);
        std::uniform_real_distribution<double> x_dist(-0.5 * boundary_length_, 0.5 * boundary_length_);
        std::uniform_real_distribution<double> y_dist(-0.5 * boundary_width_, 0.5 * boundary_width_);

        const int desired = std::max(0, count_dist(rng_));
        int attempts = 0;
        while (obstacles_.size() < static_cast<size_t>(desired) && attempts < max_sampling_attempts_) {
            ObstacleBox candidate;
            candidate.length = length_dist(rng_);
            candidate.width = width_dist(rng_);
            candidate.yaw = yaw_dist(rng_);
            candidate.x = x_dist(rng_);
            candidate.y = y_dist(rng_);
            const double footprint_radius = 0.5 * std::sqrt(candidate.length * candidate.length +
                                                            candidate.width * candidate.width);

            if (!IsInsideBoundary(candidate.x, candidate.y, footprint_radius)) {
                ++attempts;
                continue;
            }
            if (CloseToSlots(candidate.x, candidate.y, slot_keepout_extra_ + footprint_radius)) {
                ++attempts;
                continue;
            }
            bool overlap = false;
            for (const auto& obstacle : obstacles_) {
                if (BoxesOverlap(candidate, obstacle)) {
                    overlap = true;
                    break;
                }
            }
            if (!overlap) {
                obstacles_.push_back(candidate);
            }
            ++attempts;
        }
        if (obstacles_.size() < static_cast<size_t>(desired)) {
            ROS_WARN_STREAM("Placed " << obstacles_.size() << " / " << desired
                            << " requested obstacle boxes.");
        }
    }

    bool SampleStartPose() {
        std::uniform_real_distribution<double> x_dist(-0.5 * boundary_length_, 0.5 * boundary_length_);
        std::uniform_real_distribution<double> y_dist(-0.5 * boundary_width_, 0.5 * boundary_width_);
        std::uniform_real_distribution<double> yaw_dist(-M_PI, M_PI);

        for (int i = 0; i < max_sampling_attempts_; ++i) {
            geometry_msgs::Pose2D candidate;
            candidate.x = x_dist(rng_);
            candidate.y = y_dist(rng_);
            candidate.theta = yaw_dist(rng_);

            if (!IsInsideBoundary(candidate.x, candidate.y, car_radius_ + start_clearance_)) {
                continue;
            }
            if (CloseToSlots(candidate.x, candidate.y, goal_clearance_ + car_radius_)) {
                continue;
            }
            if (CloseToAnyObstacle(candidate.x, candidate.y, start_clearance_ + car_radius_)) {
                continue;
            }

            start_pose_ = candidate;
            return true;
        }
        return false;
    }

    bool BoxesOverlap(const ObstacleBox& a, const ObstacleBox& b) const {
        const double ax = a.x - b.x;
        const double ay = a.y - b.y;
        const double distance_sq = ax * ax + ay * ay;
        const double ra = 0.5 * std::sqrt(a.length * a.length + a.width * a.width);
        const double rb = 0.5 * std::sqrt(b.length * b.length + b.width * b.width);
        const double min_dist = ra + rb + 0.2;  // small safety margin
        return distance_sq < (min_dist * min_dist);
    }

    bool CloseToAnyObstacle(double x, double y, double clearance) const {
        for (const auto& obstacle : obstacles_) {
            const double dx = x - obstacle.x;
            const double dy = y - obstacle.y;
            const double radius = 0.5 * std::sqrt(obstacle.length * obstacle.length +
                                                  obstacle.width * obstacle.width);
            if ((dx * dx + dy * dy) < std::pow(radius + clearance, 2)) {
                return true;
            }
        }
        return false;
    }

    bool CloseToSlots(double x, double y, double extra_radius) const {
        if (slots_.empty()) {
            return false;
        }

        const double keepout = goal_radius_ + extra_radius;
        const double keepout_sq = keepout * keepout;

        for (const auto& slot : slots_) {
            const double dx = x - slot.x;
            const double dy = y - slot.y;
            if ((dx * dx + dy * dy) < keepout_sq) {
                return true;
            }
        }
        return false;
    }

    bool IsInsideBoundary(double x, double y, double margin) const {
        const double half_len = 0.5 * boundary_length_ - margin - obstacle_margin_from_boundary_;
        const double half_wid = 0.5 * boundary_width_ - margin - obstacle_margin_from_boundary_;
        return std::abs(x) <= half_len && std::abs(y) <= half_wid;
    }

    void PublishScenario() {
        PublishMarkers();
        PublishGoal();
        PublishStartPose();
    }

    void PublishMarkers() {
        visualization_msgs::MarkerArray array_msg;
        {
            visualization_msgs::Marker reset_marker;
            reset_marker.header.frame_id = world_frame_;
            reset_marker.action = visualization_msgs::Marker::DELETEALL;
            reset_marker.id = 0;
            array_msg.markers.push_back(reset_marker);
        }

        ros::Time stamp = ros::Time::now();
        int marker_id = 1;

        // Environment boundary
        visualization_msgs::Marker boundary;
        boundary.header.frame_id = world_frame_;
        boundary.header.stamp = stamp;
        boundary.ns = "parking_boundary";
        boundary.id = marker_id++;
        boundary.type = visualization_msgs::Marker::LINE_STRIP;
        boundary.scale.x = 0.05;
        boundary.color.a = 0.6;
        boundary.color.r = 0.7;
        boundary.color.g = 0.7;
        boundary.color.b = 0.7;
        boundary.pose.orientation.w = 1.0;
        boundary.points.resize(5);
        const double half_len = 0.5 * boundary_length_;
        const double half_wid = 0.5 * boundary_width_;
        boundary.points[0].x = -half_len; boundary.points[0].y = -half_wid;
        boundary.points[1].x = half_len;  boundary.points[1].y = -half_wid;
        boundary.points[2].x = half_len;  boundary.points[2].y = half_wid;
        boundary.points[3].x = -half_len; boundary.points[3].y = half_wid;
        boundary.points[4] = boundary.points[0];
        array_msg.markers.push_back(boundary);

        // Obstacles
        for (const auto& obstacle : obstacles_) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = world_frame_;
            marker.header.stamp = stamp;
            marker.ns = "obstacles";
            marker.id = marker_id++;
            marker.type = visualization_msgs::Marker::CUBE;
            marker.scale.x = obstacle.length;
            marker.scale.y = obstacle.width;
            marker.scale.z = obstacle_height_;
            marker.pose.position.x = obstacle.x;
            marker.pose.position.y = obstacle.y;
            marker.pose.position.z = obstacle_height_ * 0.5;
            marker.pose.orientation = tf::createQuaternionMsgFromYaw(obstacle.yaw);
            marker.color.a = 0.8;
            marker.color.r = 0.3 + 0.7 * uniform_zero_one_(rng_);
            marker.color.g = 0.3 + 0.5 * uniform_zero_one_(rng_);
            marker.color.b = 0.3 + 0.5 * uniform_zero_one_(rng_);
            array_msg.markers.push_back(marker);
        }

        // Parking slots
        for (const auto& slot : slots_) {
            visualization_msgs::Marker slot_marker;
            slot_marker.header.frame_id = world_frame_;
            slot_marker.header.stamp = stamp;
            slot_marker.ns = "parking_slot";
            slot_marker.id = marker_id++;
            slot_marker.type = visualization_msgs::Marker::CUBE;
            slot_marker.scale.x = slot_length_;
            slot_marker.scale.y = slot_width_;
            slot_marker.scale.z = 0.05;
            slot_marker.pose.position.x = slot.x;
            slot_marker.pose.position.y = slot.y;
            slot_marker.pose.position.z = 0.01;
            slot_marker.pose.orientation = tf::createQuaternionMsgFromYaw(slot.yaw);

            if (slot.is_goal) {
                slot_marker.color.a = 0.45;
                slot_marker.color.r = 0.1;
                slot_marker.color.g = 0.9;
                slot_marker.color.b = 0.3;
            } else {
                slot_marker.color.a = 0.2;
                slot_marker.color.r = 0.6;
                slot_marker.color.g = 0.6;
                slot_marker.color.b = 0.6;
            }

            array_msg.markers.push_back(slot_marker);
        }

        // Ego vehicle footprint
        visualization_msgs::Marker car_marker;
        car_marker.header.frame_id = world_frame_;
        car_marker.header.stamp = stamp;
        car_marker.ns = "ego_vehicle";
        car_marker.id = marker_id++;
        car_marker.type = visualization_msgs::Marker::CUBE;
        car_marker.scale.x = car_length_;
        car_marker.scale.y = car_width_;
        car_marker.scale.z = car_height_;
        car_marker.pose.position.x = start_pose_.x;
        car_marker.pose.position.y = start_pose_.y;
        car_marker.pose.position.z = car_height_ * 0.5;
        car_marker.pose.orientation = tf::createQuaternionMsgFromYaw(start_pose_.theta);
        car_marker.color.a = 0.95;
        car_marker.color.r = 0.1;
        car_marker.color.g = 0.3;
        car_marker.color.b = 0.9;
        array_msg.markers.push_back(car_marker);

        marker_pub_.publish(array_msg);
    }

    void PublishGoal() {
        if (!goal_slot_valid_) {
            return;
        }

        goal_pose_.header.stamp = ros::Time::now();
        goal_pose_pub_.publish(goal_pose_);
    }

    void PublishStartPose() {
        planning_msgs::car_info pose_msg;
        pose_msg.header.stamp = ros::Time::now();
        pose_msg.header.frame_id = world_frame_;
        pose_msg.x = start_pose_.x;
        pose_msg.y = start_pose_.y;
        pose_msg.yaw = start_pose_.theta;
        pose_msg.speed = 0.0;
        pose_msg.speedDrietion = 1;
        pose_msg.turnAngle = 0.0;
        pose_msg.yawrate = 0.0;
        car_pose_pub_.publish(pose_msg);
    }

    ros::NodeHandle nh_;
    ros::Publisher marker_pub_;
    ros::Publisher car_pose_pub_;
    ros::Publisher goal_pose_pub_;
    ros::Timer timer_;

    std::string world_frame_;
    double refresh_period_{3.0};
    bool auto_refresh_{true};

    double boundary_length_{20.0};
    double boundary_width_{16.0};
    double start_clearance_{1.5};
    double goal_clearance_{2.0};
    double obstacle_margin_from_boundary_{0.5};

    double slot_length_{5.5};
    double slot_width_{2.7};
    double slot_spacing_{3.0};
    double slot_row_spacing_{6.0};
    int slot_columns_{5};
    int slot_rows_{3};
    double goal_center_x_{0.0};
    double goal_center_y_{0.0};
    double goal_yaw_{M_PI_2};
    double goal_radius_{3.0};
    double slot_keepout_extra_{0.75};

    double car_length_{4.4};
    double car_width_{1.9};
    double car_height_{1.8};
    double car_radius_{2.5};

    double obstacle_height_{1.5};
    double obstacle_min_length_{1.0};
    double obstacle_max_length_{4.0};
    double obstacle_min_width_{0.5};
    double obstacle_max_width_{3.0};
    int min_obstacle_count_{3};
    int max_obstacle_count_{7};
    int max_sampling_attempts_{200};

    geometry_msgs::PoseStamped goal_pose_;
    SlotInfo goal_slot_;
    bool goal_slot_valid_{false};
    geometry_msgs::Pose2D start_pose_;
    std::vector<ObstacleBox> obstacles_;
    std::vector<SlotInfo> slots_;

    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_zero_one_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "random_parking_env");
    ros::NodeHandle nh("~");
    RandomParkingEnvironment env(nh);
    ros::spin();
    return 0;
}
