#ifndef HOPE_CONFIGS_H
#define HOPE_CONFIGS_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace hope {

struct Color {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

struct Action {
    double steer;
    double speed;
};

constexpr double kWheelBase = 2.8;
constexpr double kFrontHang = 0.96;
constexpr double kRearHang = 0.93;
constexpr double kVehicleLength = kWheelBase + kFrontHang + kRearHang;
constexpr double kVehicleWidth = 1.94;

constexpr double kValidSpeedMin = -2.5;
constexpr double kValidSpeedMax = 2.5;
constexpr double kValidSteerMin = -0.75;
constexpr double kValidSteerMax = 0.75;
constexpr double kValidAccelMin = -1.0;
constexpr double kValidAccelMax = 1.0;
constexpr double kValidAngularSpeedMin = -0.5;
constexpr double kValidAngularSpeedMax = 0.5;

constexpr int kNumStep = 10;
constexpr double kStepLength = 5e-2;

constexpr const char* kDefaultMapLevel = "Normal";

constexpr double kMinDistToObst = 0.1;
constexpr double kMaxDriveDistance = 15.0;
constexpr double kDropOutObst = 0.0;

constexpr bool kEnvCollide = false;

constexpr Color kBgColor{220, 220, 220, 255};
constexpr Color kStartColor{255, 130, 0, 255};
constexpr Color kStartBorderColor{180, 70, 0, 255};
constexpr Color kDestColor{0, 160, 170, 255};
constexpr Color kDestBorderColor{0, 100, 110, 255};
constexpr Color kObstacleColor{70, 70, 70, 255};
constexpr Color kVehicleBorderColor{30, 30, 30, 255};

constexpr int kSlotBorderWidth = 3;
constexpr int kVehicleBorderWidth = 2;

constexpr Color kTrajColorHigh{10, 10, 200, 255};
constexpr Color kTrajColorLow{10, 10, 10, 255};
constexpr int kTrajRenderLen = 20;

constexpr int kObsW = 256;
constexpr int kObsH = 256;
constexpr int kVideoW = 600;
constexpr int kVideoH = 400;
constexpr int kWinW = 500;
constexpr int kWinH = 500;

constexpr double kLidarRange = 10.0;
constexpr int kLidarNum = 120;

constexpr int kFps = 100;
constexpr double kTolerantTime = 200;

constexpr bool kUseLidar = true;
constexpr bool kUseImg = true;
constexpr bool kUseActionMask = true;

constexpr double kMaxDistToDest = 20.0;
constexpr double kRenderScaleK = 12.0;
constexpr double kRsMaxDist = 10.0;
constexpr double kAngleRewardDist = kVehicleLength;
constexpr bool kRenderTraj = true;

constexpr int kPrecision = 10;
constexpr int kStepSpeed = 1;

constexpr double kRewardRatio = 0.1;

constexpr double kGearSwitchPenalty = 5.0;

enum class MapLevel {
    Normal,
    Complex,
    Extrem,
    Dlp
};

struct RewardWeights {
    double time_cost = 1.0;
    double rs_dist_reward = 0.0;
    double dist_reward = 5.0;
    double angle_reward = 0.0;
    double box_union_reward = 10.0;
    double gear_switch_penalty = 1.0;
};

const RewardWeights& GetRewardWeights();

std::vector<Action> BuildDiscreteActions();
const std::vector<Action>& GetDiscreteActions();
int GetNumDiscreteActions();

MapLevel ParseMapLevel(const std::string& level);
std::string MapLevelToString(MapLevel level);

std::vector<std::array<double, 2>> GetVehicleBox();

struct LevelParams {
    double min_park_lot_len;
    double max_park_lot_len;
    double min_park_lot_width;
    double max_park_lot_width;
    double para_park_wall_dist;
    double bay_park_wall_dist;
    int n_obstacle;
};

LevelParams GetLevelParams(MapLevel level);

}  // namespace hope

#endif  // HOPE_CONFIGS_H
