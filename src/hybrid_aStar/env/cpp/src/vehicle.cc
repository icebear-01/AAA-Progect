#include "hope/vehicle.h"

#include <algorithm>
#include <cmath>

namespace hope {

Polygon2 State::CreateBox() const {
    const auto base = GetVehicleBox();
    Polygon2 poly;
    poly.reserve(base.size());
    for (const auto& pt : base) {
        poly.push_back({pt[0], pt[1]});
    }
    const double cos_t = std::cos(heading);
    const double sin_t = std::sin(heading);
    return TransformPolygon(poly, cos_t, sin_t, loc.x, loc.y);
}

std::array<double, 3> State::GetPos() const {
    return {loc.x, loc.y, heading};
}

KSModel::KSModel(double wheel_base,
                 double step_len,
                 int n_step,
                 std::array<double, 2> speed_range,
                 std::array<double, 2> angle_range)
    : wheel_base_(wheel_base),
      step_len_(step_len),
      n_step_(n_step),
      speed_range_(speed_range),
      angle_range_(angle_range) {}

State KSModel::Step(const State& state, const Action& action, int step_time) const {
    State new_state = state;
    double x = new_state.loc.x;
    double y = new_state.loc.y;
    double steer = action.steer;
    double speed = action.speed;
    new_state.steering = std::clamp(steer, angle_range_[0], angle_range_[1]);
    new_state.speed = std::clamp(speed, speed_range_[0], speed_range_[1]);

    for (int step = 0; step < step_time; ++step) {
        for (int i = 0; i < mini_iter_; ++i) {
            x += new_state.speed * std::cos(new_state.heading) * step_len_ / mini_iter_;
            y += new_state.speed * std::sin(new_state.heading) * step_len_ / mini_iter_;
            new_state.heading +=
                new_state.speed * std::tan(new_state.steering) / wheel_base_ * step_len_ / mini_iter_;
        }
    }

    new_state.loc = {x, y};
    return new_state;
}

Vehicle::Vehicle(double wheel_base,
                 double step_len,
                 int n_step,
                 std::array<double, 2> speed_range,
                 std::array<double, 2> angle_range)
    : kinetic_model_(wheel_base, step_len, n_step, speed_range, angle_range) {}

void Vehicle::Reset(const State& initial_state) {
    initial_state_ = initial_state;
    state_ = initial_state_;
    v_max_ = state_.speed;
    v_min_ = state_.speed;
    box_ = state_.CreateBox();
    trajectory_.clear();
    trajectory_.push_back(state_);
    tmp_trajectory_ = trajectory_;
}

VehicleStepInfo Vehicle::Step(const Action& action, int step_time) {
    VehicleStepInfo prev_info{state_, box_, v_max_, v_min_};
    state_ = kinetic_model_.Step(state_, action, step_time);
    box_ = state_.CreateBox();
    trajectory_.push_back(state_);
    tmp_trajectory_.push_back(state_);
    v_max_ = state_.speed > v_max_ ? state_.speed : v_max_;
    v_min_ = state_.speed < v_min_ ? state_.speed : v_min_;
    return prev_info;
}

void Vehicle::Retreat(const VehicleStepInfo& prev_info) {
    state_ = prev_info.state;
    box_ = prev_info.box;
    v_max_ = prev_info.v_max;
    v_min_ = prev_info.v_min;
    if (!trajectory_.empty()) {
        trajectory_.pop_back();
    }
    if (!tmp_trajectory_.empty()) {
        tmp_trajectory_.pop_back();
    }
}

void Vehicle::CompressTrajectory(int steps) {
    if (steps <= 1) {
        return;
    }
    if (trajectory_.size() >= static_cast<size_t>(steps)) {
        trajectory_.erase(trajectory_.end() - steps, trajectory_.end() - 1);
    }
    if (tmp_trajectory_.size() >= static_cast<size_t>(steps)) {
        tmp_trajectory_.erase(tmp_trajectory_.end() - steps, tmp_trajectory_.end() - 1);
    }
}

}  // namespace hope
