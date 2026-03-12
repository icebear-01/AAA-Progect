#ifndef HOPE_ENV_H
#define HOPE_ENV_H

#include <array>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "hope/action_mask.h"
#include "hope/configs.h"
#include "hope/lidar_simulator.h"
#include "hope/map_types.h"
#include "hope/parking_map_dlp.h"
#include "hope/parking_map_normal.h"
#include "hope/reeds_shepp.h"
#include "hope/vehicle.h"

namespace hope {

enum class RenderMode {
    Human,
    RgbArray,
};

struct Observation {
    cv::Mat img;
    std::vector<double> lidar;
    std::array<double, 5> target{};
    std::vector<double> action_mask;
};

struct RewardInfo {
    double time_cost = 0.0;
    double rs_dist_reward = 0.0;
    double dist_reward = 0.0;
    double angle_reward = 0.0;
    double box_union_reward = 0.0;
    double gear_switch_penalty = 0.0;
};

struct StepInfo {
    RewardInfo reward_info;
    std::optional<ReedsSheppPath> path_to_dest;
    int case_id = 0;
    MapLevel level = MapLevel::Normal;
};

struct StepResult {
    Observation observation;
    RewardInfo reward_info;
    Status status = Status::Continue;
    StepInfo info;
};

class CarParking {
public:
    CarParking(RenderMode render_mode = RenderMode::Human,
               int fps = kFps,
               bool verbose = true,
               bool use_lidar_observation = kUseLidar,
               bool use_img_observation = kUseImg,
               bool use_action_mask = kUseActionMask,
               bool limit_fps = true);

    Observation Reset(int case_id = -1,
                      const std::string& data_dir = std::string(),
                      const std::string& level = std::string());

    StepResult Step(const std::optional<Action>& action = std::nullopt);

    void SetLevel(const std::string& level);
    void Close();

private:
    std::array<double, 6> CoordTransformMatrix() const;
    Polygon2 CoordTransform(const Polygon2& poly) const;

    bool DetectCollision() const;
    bool DetectOutbound() const;
    bool CheckArrived() const;
    bool CheckTimeExceeded() const;
    Status CheckStatus() const;

    RewardInfo GetRewardInfo(Status status, const State& prev_state);
    RewardInfo ComputeReward(const State& prev_state, const State& curr_state);

    Observation Render(RenderMode mode);
    void DrawMap(cv::Mat& canvas) const;
    cv::Mat GetImgObservation(const cv::Mat& canvas) const;
    cv::Mat ProcessImgObservation(const cv::Mat& img) const;
    std::array<double, 5> GetTargetRepresentation() const;

    std::optional<ReedsSheppPath> FindRsPath(Status status) const;
    bool IsTrajValid(const std::vector<std::array<double, 3>>& traj) const;

    RenderMode render_mode_ = RenderMode::Human;
    int fps_ = kFps;
    bool verbose_ = true;
    bool use_lidar_observation_ = kUseLidar;
    bool use_img_observation_ = kUseImg;
    bool use_action_mask_ = kUseActionMask;
    bool limit_fps_ = true;

    double t_ = 0.0;
    double accum_arrive_reward_ = 0.0;
    int prev_motion_dir_ = 0;
    double gear_switch_penalty_ = 0.0;

    MapLevel level_ = MapLevel::Normal;

    ParkingMapNormal map_normal_;
    ParkingMapDlp map_dlp_;
    const ParkingMapState* map_state_ = nullptr;

    Vehicle vehicle_;
    LidarSimulator lidar_;
    std::optional<ActionMask> action_mask_;

    cv::Mat screen_;
};

}  // namespace hope

#endif  // HOPE_ENV_H
