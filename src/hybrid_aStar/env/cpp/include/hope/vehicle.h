#ifndef HOPE_VEHICLE_H
#define HOPE_VEHICLE_H

#include <vector>

#include "hope/configs.h"
#include "hope/geometry.h"

namespace hope {

enum class Status {
    Continue = 1,
    Arrived = 2,
    Collided = 3,
    Outbound = 4,
    OutTime = 5,
};

struct State {
    Point2 loc;
    double heading = 0.0;
    double speed = 0.0;
    double steering = 0.0;

    State() = default;
    State(double x, double y, double heading_in, double speed_in = 0.0, double steering_in = 0.0)
        : loc{ x, y }, heading(heading_in), speed(speed_in), steering(steering_in) {}

    Polygon2 CreateBox() const;
    std::array<double, 3> GetPos() const;
};

struct VehicleStepInfo {
    State state;
    Polygon2 box;
    double v_max = 0.0;
    double v_min = 0.0;
};

class KSModel {
public:
    KSModel(double wheel_base,
            double step_len,
            int n_step,
            std::array<double, 2> speed_range,
            std::array<double, 2> angle_range);

    State Step(const State& state, const Action& action, int step_time = kNumStep) const;

private:
    double wheel_base_;
    double step_len_;
    int n_step_;
    std::array<double, 2> speed_range_;
    std::array<double, 2> angle_range_;
    int mini_iter_ = 20;
};

class Vehicle {
public:
    Vehicle(double wheel_base = kWheelBase,
            double step_len = kStepLength,
            int n_step = kNumStep,
            std::array<double, 2> speed_range = {kValidSpeedMin, kValidSpeedMax},
            std::array<double, 2> angle_range = {kValidSteerMin, kValidSteerMax});

    void Reset(const State& initial_state);
    VehicleStepInfo Step(const Action& action, int step_time = kNumStep);
    void Retreat(const VehicleStepInfo& prev_info);
    void CompressTrajectory(int steps);

    const State& GetState() const { return state_; }
    const Polygon2& GetBox() const { return box_; }
    const std::vector<State>& GetTrajectory() const { return trajectory_; }

private:
    State state_;
    State initial_state_;
    Polygon2 box_;
    std::vector<State> trajectory_;
    std::vector<State> tmp_trajectory_;
    KSModel kinetic_model_;
    double v_max_ = 0.0;
    double v_min_ = 0.0;
};

}  // namespace hope

#endif  // HOPE_VEHICLE_H
