#include "dp_planner.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
bool ObbIntersectsObb(float c0_s,
                      float c0_l,
                      float half_len0,
                      float half_wid0,
                      float cos0,
                      float sin0,
                      float c1_s,
                      float c1_l,
                      float half_len1,
                      float half_wid1,
                      float cos1,
                      float sin1) {
    float u0x = cos0;
    float u0y = sin0;
    float v0x = -sin0;
    float v0y = cos0;
    float u1x = cos1;
    float u1y = sin1;
    float v1x = -sin1;
    float v1y = cos1;

    float t_x = c1_s - c0_s;
    float t_y = c1_l - c0_l;
    const float eps = 1e-9f;

    auto axis_separates = [&](float ax, float ay) {
        float ra = half_len0 * std::abs(u0x * ax + u0y * ay)
                   + half_wid0 * std::abs(v0x * ax + v0y * ay);
        float rb = half_len1 * std::abs(u1x * ax + u1y * ay)
                   + half_wid1 * std::abs(v1x * ax + v1y * ay);
        float dist = std::abs(t_x * ax + t_y * ay);
        return dist > (ra + rb + eps);
    };

    if (axis_separates(u0x, u0y)) return false;
    if (axis_separates(v0x, v0y)) return false;
    if (axis_separates(u1x, u1y)) return false;
    if (axis_separates(v1x, v1y)) return false;
    return true;
}

bool PointHitsObstacle(float s_value, float l_value, const OBB& obstacle, float fine_inflation) {
    float ds = s_value - obstacle.cx;
    float dl = l_value - obstacle.cy;
    float local_x = obstacle.cos_yaw * ds + obstacle.sin_yaw * dl;
    float local_y = -obstacle.sin_yaw * ds + obstacle.cos_yaw * dl;

    float scale = 1.0f + fine_inflation;
    float half_len = std::max(obstacle.half_len * scale, 1e-6f);
    float half_wid = std::max(obstacle.half_wid * scale, 1e-6f);
    return std::abs(local_x) <= half_len && std::abs(local_y) <= half_wid;
}

bool FootprintHitsObstacle(float s_value,
                           float l_value,
                           float heading,
                           const OBB& obstacle,
                           float fine_inflation,
                           float vehicle_length,
                           float vehicle_width) {
    if (vehicle_length <= 0.0f || vehicle_width <= 0.0f) {
        return PointHitsObstacle(s_value, l_value, obstacle, fine_inflation);
    }

    float scale = 1.0f + fine_inflation;
    float obs_half_len = std::max(obstacle.half_len * scale, 1e-6f);
    float obs_half_wid = std::max(obstacle.half_wid * scale, 1e-6f);
    float veh_half_len = std::max(0.5f * vehicle_length, 1e-6f);
    float veh_half_wid = std::max(0.5f * vehicle_width, 1e-6f);
    float cos0 = std::cos(heading);
    float sin0 = std::sin(heading);
    return ObbIntersectsObb(
        s_value,
        l_value,
        veh_half_len,
        veh_half_wid,
        cos0,
        sin0,
        obstacle.cx,
        obstacle.cy,
        obs_half_len,
        obs_half_wid,
        obstacle.cos_yaw,
        obstacle.sin_yaw);
}

bool FootprintHitsAnyObstacle(float s_value,
                              float l_value,
                              float heading,
                              const std::vector<OBB>& obstacles,
                              float fine_inflation,
                              float vehicle_length,
                              float vehicle_width) {
    for (const auto& obstacle : obstacles) {
        if (FootprintHitsObstacle(
                s_value, l_value, heading, obstacle, fine_inflation, vehicle_length, vehicle_width)) {
            return true;
        }
    }
    return false;
}

bool InterpolatedHitsAnyObstacle(float start_s,
                                 float start_l,
                                 float end_s,
                                 float end_l,
                                 const std::vector<OBB>& obstacles,
                                 int interpolation_points,
                                 float fine_inflation,
                                 float vehicle_length,
                                 float vehicle_width) {
    if (obstacles.empty()) {
        return false;
    }
    float delta_s = end_s - start_s;
    float delta_l = end_l - start_l;
    float heading = (delta_s != 0.0f || delta_l != 0.0f) ? std::atan2(delta_l, delta_s) : 0.0f;
    if (FootprintHitsAnyObstacle(end_s, end_l, heading, obstacles,
                                 fine_inflation, vehicle_length, vehicle_width)) {
        return true;
    }
    if (interpolation_points <= 0) {
        return false;
    }
    for (int idx = 0; idx < interpolation_points; ++idx) {
        float t = static_cast<float>(idx + 1) / static_cast<float>(interpolation_points + 1);
        float s_mid = start_s + (end_s - start_s) * t;
        float l_mid = start_l + (end_l - start_l) * t;
        if (FootprintHitsAnyObstacle(s_mid, l_mid, heading, obstacles,
                                     fine_inflation, vehicle_length, vehicle_width)) {
            return true;
        }
    }
    return false;
}
}  // namespace

DPPlanner::DPPlanner(DPPolicy& policy,
                     int s_samples,
                     int l_samples,
                     int lateral_move_limit,
                     float s_min,
                     float s_max,
                     float l_min,
                     float l_max,
                     int interpolation_points,
                     float fine_collision_inflation,
                     float vehicle_length,
                     float vehicle_width)
    : policy_(policy),
      s_samples_(s_samples),
      l_samples_(l_samples),
      lateral_move_limit_(lateral_move_limit),
      s_min_(s_min),
      s_max_(s_max),
      l_min_(l_min),
      l_max_(l_max),
      grid_size_(s_samples * l_samples),
      interpolation_points_(std::max(0, interpolation_points)),
      fine_collision_inflation_(fine_collision_inflation),
      vehicle_length_(vehicle_length),
      vehicle_width_(vehicle_width) {}

std::vector<int> DPPlanner::plan(const std::vector<float>& occupancy,
                                 int start_l_index,
                                 const std::vector<OBB>& obstacles,
                                 float start_l_value) {
    if (static_cast<int>(occupancy.size()) != grid_size_) {
        throw std::runtime_error("occupancy size mismatch");
    }
    float l_step = (l_max_ - l_min_) / std::max(1, l_samples_ - 1);
    auto clamp_l = [&](float v) {
        return std::min(std::max(v, l_min_), l_max_);
    };
    auto index_from_value = [&](float v) {
        float clamped = clamp_l(v);
        int idx = static_cast<int>(std::round((clamped - l_min_) / l_step));
        return std::max(0, std::min(l_samples_ - 1, idx));
    };
    std::vector<int> path(static_cast<size_t>(s_samples_), 0);
    int start_l = start_l_index;
    if (std::isfinite(start_l_value)) {
        start_l = index_from_value(start_l_value);
    } else if (start_l < 0 || start_l >= l_samples_) {
        start_l = 0;
    }
    int last_l = std::max(0, std::min(l_samples_ - 1, start_l));
    float start_l_value_used = start_l_value;
    if (!std::isfinite(start_l_value_used)) {
        start_l_value_used = l_min_ + l_step * static_cast<float>(last_l);
    }
    start_l_value_used = clamp_l(start_l_value_used);
    path[0] = last_l;

    for (int s_idx = 1; s_idx < s_samples_; ++s_idx) {
        std::vector<float> mask_row(static_cast<size_t>(l_samples_), 1.0f);
        // 横向跳变限制
        if (lateral_move_limit_ >= 0) {
            for (int l = 0; l < l_samples_; ++l) {
                if (std::abs(l - last_l) > lateral_move_limit_) {
                    mask_row[static_cast<size_t>(l)] = 0.0f;
                }
            }
        }
        // 插值点判碰：若车身在插值点与障碍相交，则屏蔽该动作。
        if (!obstacles.empty()) {
            float prev_s = s_min_ + (s_max_ - s_min_) * static_cast<float>(s_idx - 1)
                           / std::max(1, s_samples_ - 1);
            float prev_l = l_min_ + (l_max_ - l_min_) * static_cast<float>(last_l)
                           / std::max(1, l_samples_ - 1);
            float curr_s = s_min_ + (s_max_ - s_min_) * static_cast<float>(s_idx)
                           / std::max(1, s_samples_ - 1);
            for (int l = 0; l < l_samples_; ++l) {
                if (mask_row[static_cast<size_t>(l)] <= 0.0f) continue;
                float curr_l = l_min_ + (l_max_ - l_min_) * static_cast<float>(l)
                               / std::max(1, l_samples_ - 1);
                if (InterpolatedHitsAnyObstacle(prev_s, prev_l, curr_s, curr_l,
                                                obstacles, interpolation_points_,
                                                fine_collision_inflation_,
                                                vehicle_length_, vehicle_width_)) {
                    mask_row[static_cast<size_t>(l)] = 0.0f;
                }
            }
        }
        // 如果掩码全 False，兜底忽略掩码直接选 logits 最大值
        bool any = false;
        for (float v : mask_row) {
            if (v > 0.0f) { any = true; break; }
        }
        int action = last_l;
        if (!any) {
            std::cerr << "DPPlanner: no valid action at s_idx=" << s_idx
                      << ", last_l=" << last_l
                      << ", fallback=unmasked_argmax" << std::endl;
            action = policy_.select_action(occupancy, mask_row, s_idx, last_l, obstacles,
                                           start_l, start_l_value_used, true);
        } else {
            action = policy_.select_action(occupancy, mask_row, s_idx, last_l, obstacles,
                                           start_l, start_l_value_used);
        }
        path[static_cast<size_t>(s_idx)] = action;
        last_l = action;
    }
    return path;
}
