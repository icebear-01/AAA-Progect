#include "hope/env.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "hope/geometry.h"

namespace hope {

namespace {
constexpr double kPi = 3.14159265358979323846;

cv::Scalar ToBgr(const Color& color) {
    return cv::Scalar(color.b, color.g, color.r);
}

Color LerpColor(const Color& a, const Color& b, double t) {
    auto lerp = [t](uint8_t v0, uint8_t v1) {
        return static_cast<uint8_t>(std::round((1.0 - t) * v0 + t * v1));
    };
    return Color{lerp(a.r, b.r), lerp(a.g, b.g), lerp(a.b, b.b), 255};
}

}  // namespace

CarParking::CarParking(RenderMode render_mode,
                       int fps,
                       bool verbose,
                       bool use_lidar_observation,
                       bool use_img_observation,
                       bool use_action_mask,
                       bool limit_fps)
    : render_mode_(render_mode),
      fps_(fps),
      verbose_(verbose),
      use_lidar_observation_(use_lidar_observation),
      use_img_observation_(use_img_observation),
      use_action_mask_(use_action_mask),
      limit_fps_(limit_fps),
      level_(ParseMapLevel(kDefaultMapLevel)),
      map_normal_(level_),
      map_dlp_() {
    if (use_action_mask_) {
        action_mask_.emplace();
    }
}

void CarParking::SetLevel(const std::string& level) {
    if (level.empty()) {
        return;
    }
    level_ = ParseMapLevel(level);
    if (level_ == MapLevel::Dlp) {
        map_state_ = &map_dlp_.GetState();
    } else {
        map_normal_ = ParkingMapNormal(level_);
        map_state_ = &map_normal_.GetState();
    }
}

Observation CarParking::Reset(int case_id, const std::string& data_dir, const std::string& level) {
    accum_arrive_reward_ = 0.0;
    t_ = 0.0;
    prev_motion_dir_ = 0;
    gear_switch_penalty_ = 0.0;

    if (!level.empty()) {
        SetLevel(level);
    }

    if (level_ == MapLevel::Dlp) {
        map_dlp_.Reset(case_id, data_dir);
        map_state_ = &map_dlp_.GetState();
    } else {
        map_normal_.Reset(case_id);
        map_state_ = &map_normal_.GetState();
    }

    vehicle_.Reset(map_state_->start);
    return Step(std::nullopt).observation;
}

std::array<double, 6> CarParking::CoordTransformMatrix() const {
    const double k = kRenderScaleK;
    const double bx = 0.5 * (kWinW - k * (map_state_->boundary.xmax + map_state_->boundary.xmin));
    const double by = 0.5 * (kWinH - k * (map_state_->boundary.ymax + map_state_->boundary.ymin));
    return {k, 0.0, 0.0, k, bx, by};
}

Polygon2 CarParking::CoordTransform(const Polygon2& poly) const {
    const auto mat = CoordTransformMatrix();
    return AffineTransformPolygon(poly, mat[0], mat[1], mat[2], mat[3], mat[4], mat[5]);
}

bool CarParking::DetectCollision() const {
    const auto& box = vehicle_.GetBox();
    for (const auto& obstacle : map_state_->obstacles) {
        if (PolygonIntersects(box, obstacle.shape)) {
            return true;
        }
    }
    return false;
}

bool CarParking::DetectOutbound() const {
    const auto& pos = vehicle_.GetState().loc;
    return pos.x > map_state_->boundary.xmax || pos.x < map_state_->boundary.xmin ||
           pos.y > map_state_->boundary.ymax || pos.y < map_state_->boundary.ymin;
}

bool CarParking::CheckArrived() const {
    const double union_area = PolygonIntersectionArea(vehicle_.GetBox(), map_state_->dest_box);
    const double dest_area = PolygonArea(map_state_->dest_box);
    return dest_area > 0.0 && (union_area / dest_area) > 0.9;
}

bool CarParking::CheckTimeExceeded() const {
    return t_ > kTolerantTime;
}

Status CarParking::CheckStatus() const {
    if (DetectCollision()) {
        return Status::Collided;
    }
    if (DetectOutbound()) {
        return Status::Outbound;
    }
    if (CheckArrived()) {
        return Status::Arrived;
    }
    if (CheckTimeExceeded()) {
        return Status::OutTime;
    }
    return Status::Continue;
}

RewardInfo CarParking::ComputeReward(const State& prev_state, const State& curr_state) {
    RewardInfo reward;
    reward.time_cost = -std::tanh(t_ / (10.0 * kTolerantTime));

    if (GetRewardWeights().rs_dist_reward != 0.0) {
        const double maxc = std::tan(kValidSteerMax) / kWheelBase;
        const auto curr_path = CalcOptimalPath(curr_state.loc.x, curr_state.loc.y, curr_state.heading,
                                               map_state_->dest.loc.x, map_state_->dest.loc.y, map_state_->dest.heading,
                                               maxc, 0.1);
        const auto prev_path = CalcOptimalPath(prev_state.loc.x, prev_state.loc.y, prev_state.heading,
                                               map_state_->dest.loc.x, map_state_->dest.loc.y, map_state_->dest.heading,
                                               maxc, 0.1);
        const auto norm_path = CalcOptimalPath(map_state_->start.loc.x, map_state_->start.loc.y, map_state_->start.heading,
                                               map_state_->dest.loc.x, map_state_->dest.loc.y, map_state_->dest.heading,
                                               maxc, 0.1);
        const double norm_ratio = norm_path.total_length;
        reward.rs_dist_reward = std::exp(-curr_path.total_length / norm_ratio) -
                                std::exp(-prev_path.total_length / norm_ratio);
    }

    const auto dist_diff = std::hypot(curr_state.loc.x - map_state_->dest.loc.x,
                                      curr_state.loc.y - map_state_->dest.loc.y);
    const auto prev_dist_diff = std::hypot(prev_state.loc.x - map_state_->dest.loc.x,
                                           prev_state.loc.y - map_state_->dest.loc.y);
    const double dist_norm_ratio = std::max(std::hypot(map_state_->dest.loc.x - map_state_->start.loc.x,
                                                       map_state_->dest.loc.y - map_state_->start.loc.y),
                                            10.0);
    reward.dist_reward = prev_dist_diff / dist_norm_ratio - dist_diff / dist_norm_ratio;

    auto angle_diff = [](double a1, double a2) {
        const double diff = std::acos(std::cos(a1 - a2));
        return diff < kPi / 2.0 ? diff : kPi - diff;
    };

    const double prev_angle_diff = angle_diff(prev_state.heading, map_state_->dest.heading);
    const double curr_angle_diff = angle_diff(curr_state.heading, map_state_->dest.heading);
    const double angle_norm_ratio = kPi;
    if (std::min(prev_dist_diff, dist_diff) <= kAngleRewardDist) {
        reward.angle_reward = prev_angle_diff / angle_norm_ratio - curr_angle_diff / angle_norm_ratio;
    }

    const double union_area = PolygonIntersectionArea(vehicle_.GetBox(), map_state_->dest_box);
    const double dest_area = PolygonArea(map_state_->dest_box);
    double box_union_reward = 0.0;
    if (dest_area > 0.0) {
        box_union_reward = union_area / (2.0 * dest_area - union_area);
        if (box_union_reward < accum_arrive_reward_) {
            box_union_reward = 0.0;
        } else {
            const double prev_arrive = accum_arrive_reward_;
            accum_arrive_reward_ = box_union_reward;
            box_union_reward -= prev_arrive;
        }
    }
    reward.box_union_reward = box_union_reward;
    reward.gear_switch_penalty = gear_switch_penalty_;

    return reward;
}

RewardInfo CarParking::GetRewardInfo(Status status, const State& prev_state) {
    if (status == Status::Continue) {
        return ComputeReward(prev_state, vehicle_.GetState());
    }
    RewardInfo reward;
    reward.gear_switch_penalty = gear_switch_penalty_;
    return reward;
}

Observation CarParking::Render(RenderMode mode) {
    if (screen_.empty()) {
        screen_ = cv::Mat(kWinH, kWinW, CV_8UC3, ToBgr(kBgColor));
    }

    DrawMap(screen_);

    Observation obs;
    if (use_img_observation_) {
        const cv::Mat img = GetImgObservation(screen_);
        obs.img = ProcessImgObservation(img);
    }
    if (use_lidar_observation_) {
        std::vector<Polygon2> obs_shapes;
        obs_shapes.reserve(map_state_->obstacles.size());
        for (const auto& obs_area : map_state_->obstacles) {
            obs_shapes.push_back(obs_area.shape);
        }
        obs.lidar = lidar_.GetObservation(vehicle_.GetState(), obs_shapes);
    }
    if (use_action_mask_ && action_mask_) {
        if (!obs.lidar.empty()) {
            obs.action_mask = action_mask_->GetSteps(obs.lidar);
        } else {
            obs.action_mask.assign(GetNumDiscreteActions(), 0.0);
        }
    }
    obs.target = GetTargetRepresentation();

    if (mode == RenderMode::Human) {
        cv::imshow("CarParking", screen_);
        if (limit_fps_ && fps_ > 0) {
            cv::waitKey(static_cast<int>(1000.0 / fps_));
        } else {
            cv::waitKey(1);
        }
    }

    return obs;
}

void CarParking::DrawMap(cv::Mat& canvas) const {
    canvas.setTo(ToBgr(kBgColor));

    for (const auto& obstacle : map_state_->obstacles) {
        const auto poly = CoordTransform(obstacle.shape);
        std::vector<cv::Point> pts;
        pts.reserve(poly.size());
        for (const auto& pt : poly) {
            pts.emplace_back(static_cast<int>(std::round(pt.x)), static_cast<int>(std::round(pt.y)));
        }
        const std::vector<std::vector<cv::Point>> ppt = {pts};
        cv::fillPoly(canvas, ppt, ToBgr(kObstacleColor));
    }

    const auto start_poly = CoordTransform(map_state_->start_box);
    std::vector<cv::Point> start_pts;
    for (const auto& pt : start_poly) {
        start_pts.emplace_back(static_cast<int>(std::round(pt.x)), static_cast<int>(std::round(pt.y)));
    }
    cv::fillPoly(canvas, std::vector<std::vector<cv::Point>>{start_pts}, ToBgr(kStartColor));
    cv::polylines(canvas, start_pts, true, ToBgr(kStartBorderColor), kSlotBorderWidth);

    const auto dest_poly = CoordTransform(map_state_->dest_box);
    std::vector<cv::Point> dest_pts;
    for (const auto& pt : dest_poly) {
        dest_pts.emplace_back(static_cast<int>(std::round(pt.x)), static_cast<int>(std::round(pt.y)));
    }
    cv::fillPoly(canvas, std::vector<std::vector<cv::Point>>{dest_pts}, ToBgr(kDestColor));
    cv::polylines(canvas, dest_pts, true, ToBgr(kDestBorderColor), kSlotBorderWidth);

    const auto vehicle_poly = CoordTransform(vehicle_.GetBox());
    std::vector<cv::Point> vehicle_pts;
    for (const auto& pt : vehicle_poly) {
        vehicle_pts.emplace_back(static_cast<int>(std::round(pt.x)), static_cast<int>(std::round(pt.y)));
    }
    const Color vehicle_color{255, 94, 19, 255};
    cv::fillPoly(canvas, std::vector<std::vector<cv::Point>>{vehicle_pts}, ToBgr(vehicle_color));
    cv::polylines(canvas, vehicle_pts, true, ToBgr(kVehicleBorderColor), kVehicleBorderWidth);

    if (kRenderTraj) {
        const auto& traj = vehicle_.GetTrajectory();
        if (traj.size() > 1) {
            const int render_len = std::min(static_cast<int>(traj.size()), kTrajRenderLen);
            for (int i = 0; i < render_len; ++i) {
                const int idx = static_cast<int>(traj.size()) - render_len + i;
                const auto traj_box = CoordTransform(traj[idx].CreateBox());
                std::vector<cv::Point> traj_pts;
                for (const auto& pt : traj_box) {
                    traj_pts.emplace_back(static_cast<int>(std::round(pt.x)), static_cast<int>(std::round(pt.y)));
                }
                const double t = static_cast<double>(i) / static_cast<double>(render_len - 1);
                const Color c = LerpColor(kTrajColorLow, kTrajColorHigh, t);
                cv::fillPoly(canvas, std::vector<std::vector<cv::Point>>{traj_pts}, ToBgr(c));
            }
        }
    }
}

cv::Mat CarParking::GetImgObservation(const cv::Mat& canvas) const {
    const cv::Point2f center(kWinW / 2.0f, kWinH / 2.0f);
    const double angle_deg = vehicle_.GetState().heading * 180.0 / kPi;
    cv::Mat rot = cv::getRotationMatrix2D(center, angle_deg, 1.0);

    const Point2 vehicle_center_world = PolygonCentroid(vehicle_.GetBox());
    const auto mat = CoordTransformMatrix();
    const double vc_x = mat[0] * vehicle_center_world.x + mat[1] * vehicle_center_world.y + mat[4];
    const double vc_y = mat[2] * vehicle_center_world.x + mat[3] * vehicle_center_world.y + mat[5];

    const double angle = vehicle_.GetState().heading;
    const double dx = (vc_x - center.x) * std::cos(angle) + (vc_y - center.y) * std::sin(angle);
    const double dy = -(vc_x - center.x) * std::sin(angle) + (vc_y - center.y) * std::cos(angle);

    rot.at<double>(0, 2) += -dx;
    rot.at<double>(1, 2) += -dy;

    cv::Mat rotated;
    cv::warpAffine(canvas, rotated, rot, canvas.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, ToBgr(kBgColor));

    const int x0 = (kWinW - kObsW) / 2;
    const int y0 = (kWinH - kObsH) / 2;
    cv::Rect roi(x0, y0, kObsW, kObsH);
    cv::Mat cropped = rotated(roi).clone();

    cv::Mat rgb;
    cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

cv::Mat CarParking::ProcessImgObservation(const cv::Mat& img) const {
    cv::Mat processed = img.clone();
    const cv::Scalar bg_rgb(kBgColor.r, kBgColor.g, kBgColor.b);
    cv::Mat mask;
    cv::inRange(processed, bg_rgb, bg_rgb, mask);
    processed.setTo(cv::Scalar(0, 0, 0), mask);

    cv::Mat resized;
    cv::resize(processed, resized, cv::Size(kObsW / 4, kObsH / 4));
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
    return resized;
}

std::array<double, 5> CarParking::GetTargetRepresentation() const {
    const auto& dest = map_state_->dest;
    const auto& ego = vehicle_.GetState();
    const double dx = dest.loc.x - ego.loc.x;
    const double dy = dest.loc.y - ego.loc.y;
    const double rel_distance = std::hypot(dx, dy);
    const double rel_angle = std::atan2(dy, dx) - ego.heading;
    const double rel_heading = dest.heading - ego.heading;
    return {rel_distance, std::cos(rel_angle), std::sin(rel_angle), std::cos(rel_heading), std::sin(rel_heading)};
}

std::optional<ReedsSheppPath> CarParking::FindRsPath(Status status) const {
    if (status != Status::Continue) {
        return std::nullopt;
    }

    const double maxc = std::tan(kValidSteerMax) / kWheelBase;
    auto paths = CalcAllPaths(vehicle_.GetState().loc.x,
                              vehicle_.GetState().loc.y,
                              vehicle_.GetState().heading,
                              map_state_->dest.loc.x,
                              map_state_->dest.loc.y,
                              map_state_->dest.heading,
                              maxc,
                              0.1);
    if (paths.empty()) {
        return std::nullopt;
    }

    std::sort(paths.begin(), paths.end(), [](const ReedsSheppPath& a, const ReedsSheppPath& b) {
        return a.total_length < b.total_length;
    });

    double min_len = -1.0;
    int idx = 0;
    for (const auto& path : paths) {
        ++idx;
        if (min_len < 0.0) {
            min_len = path.total_length;
        }
        if (path.total_length > 1.6 * min_len && idx > 2) {
            break;
        }
        std::vector<std::array<double, 3>> traj;
        traj.reserve(path.x.size());
        for (size_t i = 0; i < path.x.size(); ++i) {
            traj.push_back({path.x[i], path.y[i], path.yaw[i]});
        }
        if (IsTrajValid(traj)) {
            return path;
        }
    }

    return std::nullopt;
}

bool CarParking::IsTrajValid(const std::vector<std::array<double, 3>>& traj) const {
    for (const auto& pose : traj) {
        const double x = pose[0];
        const double y = pose[1];
        if (x < map_state_->boundary.xmin || x > map_state_->boundary.xmax ||
            y < map_state_->boundary.ymin || y > map_state_->boundary.ymax) {
            return false;
        }
        State state(x, y, pose[2], 0.0, 0.0);
        const auto box = state.CreateBox();
        for (const auto& obs : map_state_->obstacles) {
            if (PolygonIntersects(box, obs.shape)) {
                return false;
            }
        }
    }
    return true;
}

StepResult CarParking::Step(const std::optional<Action>& action) {
    const State prev_state = vehicle_.GetState();
    bool collide = false;
    bool arrive = false;
    gear_switch_penalty_ = 0.0;

    if (action.has_value()) {
        int curr_dir = 0;
        if (action->speed > 1e-3) {
            curr_dir = 1;
        } else if (action->speed < -1e-3) {
            curr_dir = -1;
        }
        if (curr_dir != 0 && prev_motion_dir_ != 0 && curr_dir != prev_motion_dir_) {
            gear_switch_penalty_ = -kGearSwitchPenalty;
        }
        if (curr_dir != 0) {
            prev_motion_dir_ = curr_dir;
        }

        int simu_step_num = 0;
        for (; simu_step_num < kNumStep; ++simu_step_num) {
            const auto prev_info = vehicle_.Step(*action, 1);
            if (CheckArrived()) {
                arrive = true;
                break;
            }
            if (DetectCollision()) {
                if (simu_step_num == 0) {
                    collide = kEnvCollide;
                    vehicle_.Retreat(prev_info);
                } else {
                    vehicle_.Retreat(prev_info);
                }
                break;
            }
        }

        simu_step_num += 1;
        if (simu_step_num > 1) {
            vehicle_.CompressTrajectory(simu_step_num);
        }
    }

    t_ += 1.0;
    Observation obs = Render(render_mode_);

    Status status = Status::Continue;
    if (arrive) {
        status = Status::Arrived;
    } else {
        status = collide ? Status::Collided : CheckStatus();
    }

    RewardInfo reward_info = GetRewardInfo(status, prev_state);

    StepInfo info;
    info.reward_info = reward_info;
    info.case_id = map_state_->case_id;
    info.level = level_;

    const double dist_to_dest = std::hypot(vehicle_.GetState().loc.x - map_state_->dest.loc.x,
                                           vehicle_.GetState().loc.y - map_state_->dest.loc.y);
    if (t_ > 1.0 && status == Status::Continue && dist_to_dest < kRsMaxDist) {
        info.path_to_dest = FindRsPath(status);
    }

    StepResult result;
    result.observation = std::move(obs);
    result.reward_info = reward_info;
    result.status = status;
    result.info = std::move(info);
    return result;
}

void CarParking::Close() {
    if (render_mode_ == RenderMode::Human) {
        cv::destroyAllWindows();
    }
}

}  // namespace hope
