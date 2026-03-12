#include <ros/ros.h>

#include <geometry_msgs/Point.h>
#include <nav_msgs/OccupancyGrid.h>
#include <planning_msgs/car_info.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <random>
#include <string>
#include <utility>
#include <vector>

class FrontendRandomMapEnvironment {
public:
    explicit FrontendRandomMapEnvironment(ros::NodeHandle& nh)
        : nh_(nh), rng_(std::random_device{}()) {
        LoadParameters();
        map_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>(map_topic_, 1, true);
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(marker_topic_, 1, true);
        car_pose_pub_ = nh_.advertise<planning_msgs::car_info>(car_pose_topic_, 1, true);

        if (auto_refresh_) {
            timer_ = nh_.createTimer(
                ros::Duration(refresh_period_),
                &FrontendRandomMapEnvironment::TimerCallback,
                this);
        }

        RegenerateMap();
    }

private:
    struct GridCell {
        int x{0};
        int y{0};
    };

    void LoadParameters() {
        world_frame_ = nh_.param<std::string>("frontend_random_map/world_frame", "velodyne");
        map_topic_ = nh_.param<std::string>("frontend_random_map/map_topic", "/guided_frontend_random_map");
        marker_topic_ = nh_.param<std::string>("frontend_random_map/marker_topic", "/guided_frontend_random_map/markers");
        car_pose_topic_ = nh_.param<std::string>("frontend_random_map/car_pose_topic", "/car_pos");
        auto_refresh_ = nh_.param("frontend_random_map/auto_refresh", false);
        refresh_period_ = nh_.param("frontend_random_map/refresh_period", 5.0);
        width_ = std::max(8, nh_.param("frontend_random_map/width", 64));
        height_ = std::max(8, nh_.param("frontend_random_map/height", 64));
        resolution_ = nh_.param("frontend_random_map/resolution", 0.25);
        obstacle_prob_ = std::min(0.45, std::max(0.01, nh_.param("frontend_random_map/obstacle_prob", 0.18)));
        free_border_cells_ = std::max(1, nh_.param("frontend_random_map/free_border_cells", 2));
        min_component_ratio_ = std::min(0.95, std::max(0.05, nh_.param("frontend_random_map/min_component_ratio", 0.35)));
        max_generation_attempts_ = std::max(1, nh_.param("frontend_random_map/max_generation_attempts", 100));
        publish_default_start_ = nh_.param("frontend_random_map/publish_default_start", true);
        default_start_yaw_ = nh_.param("frontend_random_map/default_start_yaw", 0.0);
        seed_ = nh_.param("frontend_random_map/seed", -1);

        if (seed_ >= 0) {
            rng_.seed(static_cast<std::mt19937::result_type>(seed_));
        }
    }

    std::size_t Index(int x, int y) const {
        return static_cast<std::size_t>(y) * static_cast<std::size_t>(width_) + static_cast<std::size_t>(x);
    }

    bool IsInside(int x, int y) const {
        return x >= 0 && x < width_ && y >= 0 && y < height_;
    }

    geometry_msgs::Point CellCenterPoint(int x, int y, double z = 0.0) const {
        geometry_msgs::Point p;
        p.x = origin_x_ + (static_cast<double>(x) + 0.5) * resolution_;
        p.y = origin_y_ + (static_cast<double>(y) + 0.5) * resolution_;
        p.z = z;
        return p;
    }

    std::vector<GridCell> LargestFreeComponent(const std::vector<int8_t>& occupancy) const {
        std::vector<uint8_t> visited(static_cast<std::size_t>(width_ * height_), 0);
        std::vector<GridCell> best_component;
        static const int kDx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
        static const int kDy[8] = {0, 0, 1, -1, 1, -1, 1, -1};

        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                const std::size_t idx = Index(x, y);
                if (visited[idx] || occupancy[idx] > 50) {
                    continue;
                }

                std::queue<GridCell> q;
                std::vector<GridCell> component;
                q.push(GridCell{x, y});
                visited[idx] = 1;

                while (!q.empty()) {
                    const GridCell current = q.front();
                    q.pop();
                    component.push_back(current);

                    for (int i = 0; i < 8; ++i) {
                        const int nx = current.x + kDx[i];
                        const int ny = current.y + kDy[i];
                        if (!IsInside(nx, ny)) {
                            continue;
                        }
                        const std::size_t nidx = Index(nx, ny);
                        if (visited[nidx] || occupancy[nidx] > 50) {
                            continue;
                        }
                        visited[nidx] = 1;
                        q.push(GridCell{nx, ny});
                    }
                }

                if (component.size() > best_component.size()) {
                    best_component.swap(component);
                }
            }
        }
        return best_component;
    }

    bool GenerateOccupancy(std::vector<int8_t>& occupancy, GridCell& start_cell) {
        const int total_cells = width_ * height_;
        const int min_component_cells = std::max(
            width_ * height_ / 10,
            static_cast<int>(std::round(static_cast<double>(total_cells) * min_component_ratio_)));
        std::bernoulli_distribution obstacle_dist(obstacle_prob_);

        for (int attempt = 0; attempt < max_generation_attempts_; ++attempt) {
            occupancy.assign(static_cast<std::size_t>(total_cells), 0);
            for (int y = 0; y < height_; ++y) {
                for (int x = 0; x < width_; ++x) {
                    const bool on_border =
                        x < free_border_cells_ || x >= width_ - free_border_cells_ ||
                        y < free_border_cells_ || y >= height_ - free_border_cells_;
                    if (on_border) {
                        occupancy[Index(x, y)] = 0;
                        continue;
                    }
                    occupancy[Index(x, y)] = obstacle_dist(rng_) ? 100 : 0;
                }
            }

            std::vector<GridCell> component = LargestFreeComponent(occupancy);
            if (static_cast<int>(component.size()) < min_component_cells) {
                continue;
            }

            std::uniform_int_distribution<int> pick_dist(0, static_cast<int>(component.size() - 1));
            start_cell = component[static_cast<std::size_t>(pick_dist(rng_))];
            return true;
        }
        return false;
    }

    void PublishMap(const std::vector<int8_t>& occupancy) {
        nav_msgs::OccupancyGrid grid;
        grid.header.stamp = ros::Time::now();
        grid.header.frame_id = world_frame_;
        grid.info.map_load_time = grid.header.stamp;
        grid.info.resolution = resolution_;
        grid.info.width = static_cast<uint32_t>(width_);
        grid.info.height = static_cast<uint32_t>(height_);
        grid.info.origin.position.x = origin_x_;
        grid.info.origin.position.y = origin_y_;
        grid.info.origin.position.z = 0.0;
        grid.info.origin.orientation.w = 1.0;
        grid.data = occupancy;
        map_pub_.publish(grid);
    }

    void PublishMarkers(const std::vector<int8_t>& occupancy) {
        visualization_msgs::MarkerArray array;

        visualization_msgs::Marker obstacles;
        obstacles.header.frame_id = world_frame_;
        obstacles.header.stamp = ros::Time::now();
        obstacles.ns = "frontend_random_map";
        obstacles.id = 0;
        obstacles.type = visualization_msgs::Marker::CUBE_LIST;
        obstacles.action = visualization_msgs::Marker::ADD;
        obstacles.pose.orientation.w = 1.0;
        obstacles.scale.x = resolution_;
        obstacles.scale.y = resolution_;
        obstacles.scale.z = 0.05;
        obstacles.color.r = 0.62f;
        obstacles.color.g = 0.62f;
        obstacles.color.b = 0.62f;
        obstacles.color.a = 0.95f;

        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                if (occupancy[Index(x, y)] <= 50) {
                    continue;
                }
                obstacles.points.push_back(CellCenterPoint(x, y, 0.0));
            }
        }
        array.markers.push_back(obstacles);

        visualization_msgs::Marker boundary;
        boundary.header.frame_id = world_frame_;
        boundary.header.stamp = ros::Time::now();
        boundary.ns = "frontend_random_map";
        boundary.id = 1;
        boundary.type = visualization_msgs::Marker::LINE_STRIP;
        boundary.action = visualization_msgs::Marker::ADD;
        boundary.pose.orientation.w = 1.0;
        boundary.scale.x = 0.08;
        boundary.color.r = 0.35f;
        boundary.color.g = 0.35f;
        boundary.color.b = 0.35f;
        boundary.color.a = 1.0f;
        boundary.points.push_back(CellCenterPoint(0, 0));
        boundary.points.push_back(CellCenterPoint(width_ - 1, 0));
        boundary.points.push_back(CellCenterPoint(width_ - 1, height_ - 1));
        boundary.points.push_back(CellCenterPoint(0, height_ - 1));
        boundary.points.push_back(CellCenterPoint(0, 0));
        array.markers.push_back(boundary);

        marker_pub_.publish(array);
    }

    void PublishDefaultStart(const GridCell& start_cell) {
        if (!publish_default_start_) {
            return;
        }
        planning_msgs::car_info msg;
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = world_frame_;
        const geometry_msgs::Point p = CellCenterPoint(start_cell.x, start_cell.y, 0.0);
        msg.x = p.x;
        msg.y = p.y;
        msg.yaw = default_start_yaw_;
        msg.speedDrietion = 1;
        msg.speed = 0.0;
        msg.turnAngle = 0.0;
        msg.yawrate = 0.0;
        car_pose_pub_.publish(msg);
    }

    void RegenerateMap() {
        origin_x_ = -0.5 * static_cast<double>(width_) * resolution_;
        origin_y_ = -0.5 * static_cast<double>(height_) * resolution_;

        std::vector<int8_t> occupancy;
        GridCell start_cell;
        if (!GenerateOccupancy(occupancy, start_cell)) {
            ROS_ERROR("Failed to generate a valid frontend random map.");
            return;
        }

        PublishMap(occupancy);
        PublishMarkers(occupancy);
        PublishDefaultStart(start_cell);
        ROS_INFO_STREAM("Frontend random map published: " << width_ << "x" << height_
                        << ", resolution=" << resolution_
                        << ", obstacle_prob=" << obstacle_prob_);
    }

    void TimerCallback(const ros::TimerEvent&) {
        RegenerateMap();
    }

private:
    ros::NodeHandle nh_;
    ros::Publisher map_pub_;
    ros::Publisher marker_pub_;
    ros::Publisher car_pose_pub_;
    ros::Timer timer_;

    std::mt19937 rng_;

    std::string world_frame_{"velodyne"};
    std::string map_topic_{"/guided_frontend_random_map"};
    std::string marker_topic_{"/guided_frontend_random_map/markers"};
    std::string car_pose_topic_{"/car_pos"};
    bool auto_refresh_{false};
    double refresh_period_{5.0};
    int width_{64};
    int height_{64};
    double resolution_{0.25};
    double obstacle_prob_{0.18};
    int free_border_cells_{2};
    double min_component_ratio_{0.35};
    int max_generation_attempts_{100};
    bool publish_default_start_{true};
    double default_start_yaw_{0.0};
    int seed_{-1};

    double origin_x_{0.0};
    double origin_y_{0.0};
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "frontend_random_map_env");
    ros::NodeHandle nh("~");
    FrontendRandomMapEnvironment node(nh);
    ros::spin();
    return 0;
}
