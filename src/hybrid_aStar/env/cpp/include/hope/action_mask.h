#ifndef HOPE_ACTION_MASK_H
#define HOPE_ACTION_MASK_H

#include <vector>

#include "hope/configs.h"
#include "hope/geometry.h"

namespace hope {

class ActionMask {
public:
    explicit ActionMask(int n_iter = 10);

    std::vector<double> GetSteps(const std::vector<double>& raw_lidar_obs) const;

    Action ChooseAction(const std::vector<double>& action_mean,
                        const std::vector<double>& action_std,
                        const std::vector<double>& action_mask) const;

    int NumActions() const { return static_cast<int>(actions_.size()); }

private:
    int n_iter_;
    int up_sample_rate_ = 10;
    std::vector<Action> actions_;
    std::vector<Point2> ray_dirs_;
    std::vector<double> vehicle_lidar_base_;
    std::vector<double> dist_star_;

    std::vector<std::vector<std::vector<Point2>>> InitVehicleBox() const;
    std::vector<double> PrecomputeDistStar() const;

    double RaySegmentIntersectionDistance(const Point2& dir, const Point2& p1, const Point2& p2) const;
    std::vector<double> ComputeVehicleLidarBase() const;
    std::vector<double> LinearInterpolate(const std::vector<double>& x, int upsample_rate) const;
    std::vector<double> MinimumFilter1D(const std::vector<double>& data, int kernel) const;
};

}  // namespace hope

#endif  // HOPE_ACTION_MASK_H
