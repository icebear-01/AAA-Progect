#include "hope/action_mask.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "hope/random_utils.h"

namespace hope {

namespace {
constexpr double kPi = 3.14159265358979323846;

double Cross(const Point2& a, const Point2& b) {
    return a.x * b.y - a.y * b.x;
}

int ReflectIndex(int idx, int n) {
    if (n <= 1) {
        return 0;
    }
    if (idx < 0) {
        return -idx - 1;
    }
    if (idx >= n) {
        return 2 * n - idx - 1;
    }
    return idx;
}

}  // namespace

ActionMask::ActionMask(int n_iter)
    : n_iter_(n_iter), actions_(GetDiscreteActions()) {
    ray_dirs_.reserve(kLidarNum);
    for (int i = 0; i < kLidarNum; ++i) {
        const double angle = static_cast<double>(i) * kPi * 2.0 / static_cast<double>(kLidarNum);
        ray_dirs_.push_back({std::cos(angle), std::sin(angle)});
    }
    vehicle_lidar_base_ = ComputeVehicleLidarBase();
    dist_star_ = PrecomputeDistStar();
}

double ActionMask::RaySegmentIntersectionDistance(const Point2& dir,
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

std::vector<double> ActionMask::ComputeVehicleLidarBase() const {
    const auto base = GetVehicleBox();
    Polygon2 vehicle_box;
    vehicle_box.reserve(base.size());
    for (const auto& pt : base) {
        vehicle_box.push_back({pt[0], pt[1]});
    }

    std::vector<double> base_dist(kLidarNum, kLidarRange);
    for (int i = 0; i < kLidarNum; ++i) {
        const Point2 dir = ray_dirs_[i];
        double min_dist = kLidarRange;
        for (size_t j = 0; j < vehicle_box.size(); ++j) {
            const Point2 p1 = vehicle_box[j];
            const Point2 p2 = vehicle_box[(j + 1) % vehicle_box.size()];
            const double dist = RaySegmentIntersectionDistance(dir, p1, p2);
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
        base_dist[i] = min_dist;
    }
    return base_dist;
}

std::vector<std::vector<std::vector<Point2>>> ActionMask::InitVehicleBox() const {
    const auto base = GetVehicleBox();
    std::vector<Point2> car_coords;
    car_coords.reserve(base.size());
    for (const auto& pt : base) {
        car_coords.push_back({pt[0], pt[1]});
    }

    std::vector<std::vector<std::vector<Point2>>> boxes;
    boxes.resize(actions_.size());
    for (size_t a = 0; a < actions_.size(); ++a) {
        boxes[a].resize(n_iter_);
        const double steer = actions_[a].steer;
        const double speed = actions_[a].speed;
        double radius;
        const double tan_val = std::tan(steer);
        if (std::abs(tan_val) < 1e-6) {
            radius = 1e9;
        } else {
            radius = 1.0 / (tan_val / kWheelBase);
        }
        const double delta_phi = 0.5 * speed / 10.0 / radius;
        double ptheta = 0.0;
        double px = 0.0;
        double py = 0.0;
        const double Ox = -radius * std::sin(0.0);
        const double Oy = radius * std::cos(0.0);

        for (int t = 0; t < n_iter_; ++t) {
            ptheta += delta_phi;
            px = Ox + radius * std::sin(ptheta);
            py = Oy - radius * std::cos(ptheta);
            const double cos_t = std::cos(ptheta);
            const double sin_t = std::sin(ptheta);
            std::vector<Point2> coords;
            coords.reserve(car_coords.size());
            for (const auto& c : car_coords) {
                const double x = cos_t * c.x - sin_t * c.y + px;
                const double y = sin_t * c.x + cos_t * c.y + py;
                coords.push_back({x, y});
            }
            boxes[a][t] = coords;
        }
    }
    return boxes;
}

std::vector<double> ActionMask::PrecomputeDistStar() const {
    const auto vehicle_boxes = InitVehicleBox();
    const int n_action = static_cast<int>(actions_.size());
    const int lidar_num = kLidarNum;
    const int upsampled = lidar_num * up_sample_rate_;

    std::vector<double> base_dist(lidar_num * n_action * n_iter_, 0.0);

    for (int lidar_idx = 0; lidar_idx < lidar_num; ++lidar_idx) {
        const Point2 dir = ray_dirs_[lidar_idx];
        for (int action_idx = 0; action_idx < n_action; ++action_idx) {
            for (int iter = 0; iter < n_iter_; ++iter) {
                double max_dist = 0.0;
                const auto& box = vehicle_boxes[action_idx][iter];
                for (size_t e = 0; e < box.size(); ++e) {
                    const Point2 p1 = box[e];
                    const Point2 p2 = box[(e + 1) % box.size()];
                    double dist = RaySegmentIntersectionDistance(dir, p1, p2);
                    if (!std::isfinite(dist)) {
                        dist = 0.0;
                    }
                    if (dist > max_dist) {
                        max_dist = dist;
                    }
                }
                const size_t idx = (static_cast<size_t>(lidar_idx) * n_action + action_idx) * n_iter_ + iter;
                base_dist[idx] = max_dist;
            }
        }
    }

    std::vector<double> upsampled_dist(upsampled * n_action * n_iter_, 0.0);
    for (int lidar_idx = 0; lidar_idx < lidar_num; ++lidar_idx) {
        const int next_idx = (lidar_idx + 1) % lidar_num;
        for (int u = 0; u < up_sample_rate_; ++u) {
            const double ratio = static_cast<double>(u) / static_cast<double>(up_sample_rate_);
            const int out_idx = lidar_idx * up_sample_rate_ + u;
            for (int action_idx = 0; action_idx < n_action; ++action_idx) {
                for (int iter = 0; iter < n_iter_; ++iter) {
                    const size_t base_idx = (static_cast<size_t>(lidar_idx) * n_action + action_idx) * n_iter_ + iter;
                    const size_t next_base_idx = (static_cast<size_t>(next_idx) * n_action + action_idx) * n_iter_ + iter;
                    const size_t out_offset = (static_cast<size_t>(out_idx) * n_action + action_idx) * n_iter_ + iter;
                    const double value = base_dist[base_idx] * (1.0 - ratio) + base_dist[next_base_idx] * ratio;
                    upsampled_dist[out_offset] = value;
                }
            }
        }
    }

    return upsampled_dist;
}

std::vector<double> ActionMask::LinearInterpolate(const std::vector<double>& x, int upsample_rate) const {
    const int n = static_cast<int>(x.size());
    const int out_n = n * upsample_rate;
    std::vector<double> y(out_n, 0.0);
    for (int j = 0; j < out_n; ++j) {
        const int base = j / upsample_rate;
        const int next = (base + 1) % n;
        const double frac = static_cast<double>(j % upsample_rate) / static_cast<double>(upsample_rate);
        y[j] = x[base] * (1.0 - frac) + x[next] * frac;
    }
    return y;
}

std::vector<double> ActionMask::MinimumFilter1D(const std::vector<double>& data, int kernel) const {
    if (data.empty()) {
        return {};
    }
    std::vector<double> output(data.size(), 0.0);
    const int radius = kernel / 2;
    const int n = static_cast<int>(data.size());
    for (int i = 0; i < n; ++i) {
        double min_val = std::numeric_limits<double>::infinity();
        for (int k = -radius; k <= radius; ++k) {
            const int idx = ReflectIndex(i + k, n);
            if (data[idx] < min_val) {
                min_val = data[idx];
            }
        }
        output[i] = min_val;
    }
    return output;
}

std::vector<double> ActionMask::GetSteps(const std::vector<double>& raw_lidar_obs) const {
    std::vector<double> lidar_obs(kLidarNum, 0.0);
    for (int i = 0; i < kLidarNum; ++i) {
        const double clipped = std::clamp(raw_lidar_obs[i], 0.0, 10.0);
        lidar_obs[i] = clipped + vehicle_lidar_base_[i];
    }

    const std::vector<double> dist_obs = LinearInterpolate(lidar_obs, up_sample_rate_);
    const int n_action = static_cast<int>(actions_.size());
    const int lidar_up = kLidarNum * up_sample_rate_;

    std::vector<double> step_len(n_action, static_cast<double>(n_iter_));
    for (int lidar_idx = 0; lidar_idx < lidar_up; ++lidar_idx) {
        const double obs_dist = dist_obs[lidar_idx];
        for (int action_idx = 0; action_idx < n_action; ++action_idx) {
            int max_step = n_iter_;
            for (int iter = 0; iter < n_iter_; ++iter) {
                const size_t offset = (static_cast<size_t>(lidar_idx) * n_action + action_idx) * n_iter_ + iter;
                if (dist_star_[offset] > obs_dist) {
                    max_step = iter;
                    break;
                }
            }
            if (max_step < step_len[action_idx]) {
                step_len[action_idx] = static_cast<double>(max_step);
            }
        }
    }

    if (step_len.empty()) {
        return step_len;
    }

    const int half = n_action / 2;
    std::vector<double> forward(step_len.begin(), step_len.begin() + half);
    std::vector<double> backward(step_len.begin() + half, step_len.end());
    if (!forward.empty()) {
        forward.front() -= 1.0;
        forward.back() -= 1.0;
    }
    if (!backward.empty()) {
        backward.front() -= 1.0;
        backward.back() -= 1.0;
    }

    const std::vector<double> forward_min = MinimumFilter1D(forward, 5);
    const std::vector<double> backward_min = MinimumFilter1D(backward, 5);

    std::vector<double> processed;
    processed.reserve(n_action);
    processed.insert(processed.end(), forward_min.begin(), forward_min.end());
    processed.insert(processed.end(), backward_min.begin(), backward_min.end());

    double sum = 0.0;
    for (auto& v : processed) {
        v = std::clamp(v, 0.0, static_cast<double>(n_iter_)) / static_cast<double>(n_iter_);
        sum += v;
    }

    if (sum == 0.0) {
        for (auto& v : processed) {
            v = std::clamp(v, 0.01, 1.0);
        }
    }

    return processed;
}

Action ActionMask::ChooseAction(const std::vector<double>& action_mean,
                                const std::vector<double>& action_std,
                                const std::vector<double>& action_mask) const {
    const double scale_steer = kValidSteerMax;
    const double scale_speed = 1.0;

    const int n_action = static_cast<int>(actions_.size());
    std::vector<double> scores(n_action, -std::numeric_limits<double>::infinity());

    for (int i = 0; i < n_action; ++i) {
        const double steer = actions_[i].steer / scale_steer;
        const double speed = actions_[i].speed / scale_speed;
        const double z0 = (steer - action_mean[0]) / action_std[0];
        const double z1 = (speed - action_mean[1]) / action_std[1];
        double log_prob = -0.5 * z0 * z0 - std::log(std::sqrt(2.0 * kPi) * action_std[0]);
        log_prob += -0.5 * z1 * z1 - std::log(std::sqrt(2.0 * kPi) * action_std[1]);
        log_prob = std::clamp(log_prob, -10.0, 10.0);
        scores[i] = std::exp(log_prob) * action_mask[i];
    }

    double total = 0.0;
    for (double score : scores) {
        total += score;
    }
    if (total <= 0.0) {
        return actions_[0];
    }

    for (double& score : scores) {
        score /= total;
    }

    RandomGenerator rng;
    const double pick = rng.Uniform(0.0, 1.0);
    double acc = 0.0;
    for (int i = 0; i < n_action; ++i) {
        acc += scores[i];
        if (pick <= acc) {
            return actions_[i];
        }
    }

    return actions_.back();
}

}  // namespace hope
