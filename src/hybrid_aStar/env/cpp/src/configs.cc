#include "hope/configs.h"

#include <cmath>

namespace hope {

const RewardWeights& GetRewardWeights() {
    static const RewardWeights weights{};
    return weights;
}

std::vector<Action> BuildDiscreteActions() {
    std::vector<Action> actions;
    const double step = kValidSteerMax / static_cast<double>(kPrecision);
    for (double steer = kValidSteerMax; steer > -(kValidSteerMax + step / 2.0); steer -= step) {
        actions.push_back(Action{steer, static_cast<double>(kStepSpeed)});
    }
    for (double steer = kValidSteerMax; steer > -(kValidSteerMax + step / 2.0); steer -= step) {
        actions.push_back(Action{steer, -static_cast<double>(kStepSpeed)});
    }
    return actions;
}

const std::vector<Action>& GetDiscreteActions() {
    static const std::vector<Action> actions = BuildDiscreteActions();
    return actions;
}

int GetNumDiscreteActions() {
    return static_cast<int>(GetDiscreteActions().size());
}

MapLevel ParseMapLevel(const std::string& level) {
    if (level == "Normal" || level == "normal") {
        return MapLevel::Normal;
    }
    if (level == "Complex" || level == "complex") {
        return MapLevel::Complex;
    }
    if (level == "Extrem" || level == "extrem" || level == "Extreme" || level == "extreme") {
        return MapLevel::Extrem;
    }
    if (level == "dlp" || level == "DLP") {
        return MapLevel::Dlp;
    }
    return MapLevel::Normal;
}

std::string MapLevelToString(MapLevel level) {
    switch (level) {
        case MapLevel::Normal:
            return "Normal";
        case MapLevel::Complex:
            return "Complex";
        case MapLevel::Extrem:
            return "Extrem";
        case MapLevel::Dlp:
            return "dlp";
    }
    return "Normal";
}

std::vector<std::array<double, 2>> GetVehicleBox() {
    return {
        {-kRearHang, -kVehicleWidth / 2.0},
        {kFrontHang + kWheelBase, -kVehicleWidth / 2.0},
        {kFrontHang + kWheelBase, kVehicleWidth / 2.0},
        {-kRearHang, kVehicleWidth / 2.0},
    };
}

LevelParams GetLevelParams(MapLevel level) {
    LevelParams params{};
    switch (level) {
        case MapLevel::Extrem:
            params.min_park_lot_len = kVehicleLength + 0.6;
            params.max_park_lot_len = kVehicleLength + 0.9;
            params.min_park_lot_width = kVehicleWidth + 0.4;
            params.max_park_lot_width = kVehicleWidth + 0.85;
            params.para_park_wall_dist = 3.5;
            params.bay_park_wall_dist = 6.0;
            params.n_obstacle = 8;
            break;
        case MapLevel::Complex:
            params.min_park_lot_len = kVehicleLength + 0.9;
            params.max_park_lot_len = kVehicleLength * 1.25;
            params.min_park_lot_width = kVehicleWidth + 0.4;
            params.max_park_lot_width = kVehicleWidth + 0.85;
            params.para_park_wall_dist = 4.0;
            params.bay_park_wall_dist = 6.0;
            params.n_obstacle = 5;
            break;
        case MapLevel::Normal:
        case MapLevel::Dlp:
        default:
            params.min_park_lot_len = kVehicleLength * 1.25;
            params.max_park_lot_len = kVehicleLength * 1.25 + 0.5;
            params.min_park_lot_width = kVehicleWidth + 0.85;
            params.max_park_lot_width = kVehicleWidth + 1.2;
            params.para_park_wall_dist = 4.5;
            params.bay_park_wall_dist = 7.0;
            params.n_obstacle = 3;
            break;
    }
    return params;
}

}  // namespace hope
