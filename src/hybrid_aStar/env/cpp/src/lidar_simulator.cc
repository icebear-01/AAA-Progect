#include "hope/lidar_simulator.h"

#include <cmath>
#include <limits>

namespace hope {

namespace {
constexpr double kPi = 3.14159265358979323846;

double Cross(const Point2& a, const Point2& b) {
    return a.x * b.y - a.y * b.x;
}

}  // namespace

LidarSimulator::LidarSimulator(double lidar_range, int lidar_num)
    : lidar_range_(lidar_range), lidar_num_(lidar_num) {
    angles_.reserve(lidar_num_);
    ray_dirs_.reserve(lidar_num_);
    for (int i = 0; i < lidar_num_; ++i) {
        double angle = static_cast<double>(i) * kPi * 2.0 / static_cast<double>(lidar_num_);
        angles_.push_back(angle);
        ray_dirs_.push_back({std::cos(angle), std::sin(angle)});
    }
    vehicle_boundary_ = ComputeVehicleBoundary();
}

double LidarSimulator::RaySegmentIntersectionDistance(const Point2& dir,
                                                      const Point2& p1,
                                                      const Point2& p2) const {
    const Point2 s{p2.x - p1.x, p2.y - p1.y};
    const double rxs = Cross(dir, s);
    if (std::abs(rxs) < 1e-12) {
        return std::numeric_limits<double>::infinity();
    }
    const double t = Cross(p1, s) / rxs;
    const double u = Cross(p1, dir) / rxs;
    if (t >= 0.0 && u >= 0.0 && u <= 1.0) {
        return t;
    }
    return std::numeric_limits<double>::infinity();
}

std::vector<double> LidarSimulator::ComputeVehicleBoundary() const {
    const auto base = GetVehicleBox();
    Polygon2 vehicle_box;
    vehicle_box.reserve(base.size());
    for (const auto& pt : base) {
        vehicle_box.push_back({pt[0], pt[1]});
    }
    std::vector<double> boundary(lidar_num_, lidar_range_);
    for (int i = 0; i < lidar_num_; ++i) {
        const Point2 dir = ray_dirs_[i];
        double min_dist = lidar_range_;
        for (size_t j = 0; j < vehicle_box.size(); ++j) {
            const Point2 p1 = vehicle_box[j];
            const Point2 p2 = vehicle_box[(j + 1) % vehicle_box.size()];
            const double dist = RaySegmentIntersectionDistance(dir, p1, p2);
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
        boundary[i] = min_dist;
    }
    return boundary;
}

std::vector<double> LidarSimulator::GetObservation(const State& ego_state,
                                                   const std::vector<Polygon2>& obstacles) const {
    const double cos_t = std::cos(ego_state.heading);
    const double sin_t = std::sin(ego_state.heading);

    std::vector<Polygon2> rotated_obstacles;
    rotated_obstacles.reserve(obstacles.size());
    for (const auto& obs : obstacles) {
        Polygon2 rotated;
        rotated.reserve(obs.size());
        for (const auto& pt : obs) {
            const double dx = pt.x - ego_state.loc.x;
            const double dy = pt.y - ego_state.loc.y;
            const double x = cos_t * dx + sin_t * dy;
            const double y = -sin_t * dx + cos_t * dy;
            rotated.push_back({x, y});
        }
        rotated_obstacles.push_back(rotated);
    }

    std::vector<double> lidar_obs(lidar_num_, lidar_range_);
    for (int i = 0; i < lidar_num_; ++i) {
        const Point2 dir = ray_dirs_[i];
        double min_dist = lidar_range_;
        for (const auto& obs : rotated_obstacles) {
            for (size_t j = 0; j < obs.size(); ++j) {
                const Point2 p1 = obs[j];
                const Point2 p2 = obs[(j + 1) % obs.size()];
                const double dist = RaySegmentIntersectionDistance(dir, p1, p2);
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
        }
        lidar_obs[i] = std::min(min_dist, lidar_range_);
    }

    std::vector<double> result(lidar_num_, 0.0);
    for (int i = 0; i < lidar_num_; ++i) {
        result[i] = lidar_obs[i] - vehicle_boundary_[i];
    }
    return result;
}

}  // namespace hope
