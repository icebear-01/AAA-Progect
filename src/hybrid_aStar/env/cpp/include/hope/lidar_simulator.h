#ifndef HOPE_LIDAR_SIMULATOR_H
#define HOPE_LIDAR_SIMULATOR_H

#include <vector>

#include "hope/configs.h"
#include "hope/geometry.h"
#include "hope/vehicle.h"

namespace hope {

class LidarSimulator {
public:
    LidarSimulator(double lidar_range = kLidarRange, int lidar_num = kLidarNum);

    std::vector<double> GetObservation(const State& ego_state, const std::vector<Polygon2>& obstacles) const;
    const std::vector<double>& GetVehicleBoundary() const { return vehicle_boundary_; }

private:
    double lidar_range_;
    int lidar_num_;
    std::vector<double> angles_;
    std::vector<Point2> ray_dirs_;
    std::vector<double> vehicle_boundary_;

    double RaySegmentIntersectionDistance(const Point2& dir, const Point2& p1, const Point2& p2) const;
    std::vector<double> ComputeVehicleBoundary() const;
};

}  // namespace hope

#endif  // HOPE_LIDAR_SIMULATOR_H
