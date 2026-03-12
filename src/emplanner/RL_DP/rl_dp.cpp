#include "rl_dp.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {
float Distance(const SLCorner& a, const SLCorner& b) {
    float ds = b.s - a.s;
    float dl = b.l - a.l;
    return std::hypot(ds, dl);
}

std::vector<double> Linspace(double start, double end, int num) {
    std::vector<double> vals;
    if (num <= 1) {
        vals.push_back(start);
        return vals;
    }
    double step = (end - start) / static_cast<double>(num - 1);
    vals.reserve(static_cast<size_t>(num));
    for (int i = 0; i < num; ++i) {
        vals.push_back(start + step * i);
    }
    return vals;
}

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

std::vector<float> RasterizeOccupancy(const std::vector<OBB>& obstacles,
                                      int s_samples,
                                      int l_samples,
                                      float s_min,
                                      float s_max,
                                      float l_min,
                                      float l_max,
                                      float coarse_inflation,
                                      float fine_inflation,
                                      float vehicle_length,
                                      float vehicle_width) {
    std::vector<float> occ(static_cast<size_t>(s_samples * l_samples), 0.0f);
    if (obstacles.empty()) {
        return occ;
    }

    std::vector<double> s_coords = Linspace(s_min, s_max, s_samples);
    std::vector<double> l_coords = Linspace(l_min, l_max, l_samples);
    const float eps = 1e-6f;
    const float coarse_scale = 1.0f + coarse_inflation;
    const float fine_scale = 1.0f + fine_inflation;
    const bool use_vehicle_footprint = vehicle_length > 0.0f && vehicle_width > 0.0f;
    float vehicle_radius = 0.0f;
    if (use_vehicle_footprint) {
        float veh_half_len = std::max(0.5f * vehicle_length, eps);
        float veh_half_wid = std::max(0.5f * vehicle_width, eps);
        vehicle_radius = std::hypot(veh_half_len, veh_half_wid);
    }

    for (const auto& o : obstacles) {
        float coarse_half_len = std::max(o.half_len * coarse_scale, eps);
        float coarse_half_wid = std::max(o.half_wid * coarse_scale, eps);
        float fine_half_len = std::max(o.half_len * fine_scale, eps);
        float fine_half_wid = std::max(o.half_wid * fine_scale, eps);

        float s_min_c = std::numeric_limits<float>::max();
        float s_max_c = -std::numeric_limits<float>::max();
        float l_min_c = std::numeric_limits<float>::max();
        float l_max_c = -std::numeric_limits<float>::max();
        float local_x[4] = {coarse_half_len, coarse_half_len, -coarse_half_len, -coarse_half_len};
        float local_y[4] = {coarse_half_wid, -coarse_half_wid, -coarse_half_wid, coarse_half_wid};
        for (int i = 0; i < 4; ++i) {
            float gx = o.cx + o.cos_yaw * local_x[i] - o.sin_yaw * local_y[i];
            float gy = o.cy + o.sin_yaw * local_x[i] + o.cos_yaw * local_y[i];
            s_min_c = std::min(s_min_c, gx);
            s_max_c = std::max(s_max_c, gx);
            l_min_c = std::min(l_min_c, gy);
            l_max_c = std::max(l_max_c, gy);
        }

        s_min_c = std::max(s_min_c, s_min);
        s_max_c = std::min(s_max_c, s_max);
        l_min_c = std::max(l_min_c, l_min);
        l_max_c = std::min(l_max_c, l_max);
        if (use_vehicle_footprint) {
            s_min_c = std::max(s_min_c - vehicle_radius, s_min);
            s_max_c = std::min(s_max_c + vehicle_radius, s_max);
            l_min_c = std::max(l_min_c - vehicle_radius, l_min);
            l_max_c = std::min(l_max_c + vehicle_radius, l_max);
        }

        std::vector<int> s_idx;
        std::vector<int> l_idx;
        for (int si = 0; si < s_samples; ++si) {
            float s_val = static_cast<float>(s_coords[si]);
            if (s_val >= s_min_c && s_val <= s_max_c) {
                s_idx.push_back(si);
            }
        }
        for (int li = 0; li < l_samples; ++li) {
            float l_val = static_cast<float>(l_coords[li]);
            if (l_val >= l_min_c && l_val <= l_max_c) {
                l_idx.push_back(li);
            }
        }
        if (s_idx.empty() || l_idx.empty()) {
            continue;
        }

        if (use_vehicle_footprint) {
            const float heading = 0.0f;
            for (int si : s_idx) {
                float s_val = static_cast<float>(s_coords[si]);
                for (int li : l_idx) {
                    if (occ[static_cast<size_t>(si * l_samples + li)] > 0.5f) {
                        continue;
                    }
                    float l_val = static_cast<float>(l_coords[li]);
                    if (FootprintHitsObstacle(s_val, l_val, heading, o,
                                              fine_inflation, vehicle_length, vehicle_width)) {
                        occ[static_cast<size_t>(si * l_samples + li)] = 1.0f;
                    }
                }
            }
            continue;
        }

        for (int si : s_idx) {
            float s_val = static_cast<float>(s_coords[si]);
            for (int li : l_idx) {
                float l_val = static_cast<float>(l_coords[li]);
                float ds = s_val - o.cx;
                float dl = l_val - o.cy;
                float local_x2 = o.cos_yaw * ds + o.sin_yaw * dl;
                float local_y2 = -o.sin_yaw * ds + o.cos_yaw * dl;
                if (std::abs(local_x2) <= fine_half_len + eps && std::abs(local_y2) <= fine_half_wid + eps) {
                    occ[static_cast<size_t>(si * l_samples + li)] = 1.0f;
                }
            }
        }
    }
    return occ;
}
}  // namespace

RL_DP::RL_DP(const std::string& model_path,
             int s_samples,
             int l_samples,
             float s_min,
             float s_max,
             float l_min,
             float l_max,
             int lateral_move_limit,
             int interpolation_points,
             float coarse_inflation,
             float fine_inflation,
             float vehicle_length,
             float vehicle_width)
    : s_samples_(s_samples),
      l_samples_(l_samples),
      s_min_(s_min),
      s_max_(s_max),
      l_min_(l_min),
      l_max_(l_max),
      coarse_inflation_(coarse_inflation),
      fine_inflation_(fine_inflation),
      vehicle_length_(vehicle_length),
      vehicle_width_(vehicle_width),
      policy_(model_path, s_samples, l_samples, s_min, s_max, l_min, l_max),
      planner_(policy_, s_samples, l_samples, lateral_move_limit,
               s_min, s_max, l_min, l_max,
               interpolation_points, fine_inflation,
               vehicle_length, vehicle_width) {
    if ((vehicle_length_ > 0.0f) != (vehicle_width_ > 0.0f)) {
        throw std::runtime_error("vehicle_length/vehicle_width must both be zero or positive.");
    }
    if (s_samples_ <= 0 || l_samples_ <= 0) {
        throw std::runtime_error("s_samples/l_samples must be positive.");
    }
}

OBB RL_DP::ObbFromCorners(const ObstacleCorners& corners) const {
    float center_s = 0.0f;
    float center_l = 0.0f;
    for (const auto& c : corners) {
        center_s += c.s;
        center_l += c.l;
    }
    center_s /= 4.0f;
    center_l /= 4.0f;

    std::vector<SLCorner> ordered(corners.begin(), corners.end());
    std::sort(ordered.begin(), ordered.end(), [&](const SLCorner& a, const SLCorner& b) {
        float ang_a = std::atan2(a.l - center_l, a.s - center_s);
        float ang_b = std::atan2(b.l - center_l, b.s - center_s);
        return ang_a < ang_b;
    });

    SLCorner e01{ordered[1].s - ordered[0].s, ordered[1].l - ordered[0].l};
    SLCorner e12{ordered[2].s - ordered[1].s, ordered[2].l - ordered[1].l};
    float len01 = std::hypot(e01.s, e01.l);
    float len12 = std::hypot(e12.s, e12.l);

    float length = std::max(len01, len12);
    float width = std::min(len01, len12);
    float dir_s = (len01 >= len12) ? e01.s : e12.s;
    float dir_l = (len01 >= len12) ? e01.l : e12.l;

    const float eps = 1e-4f;
    if (length < eps || width < eps) {
        float s_min = corners[0].s;
        float s_max = corners[0].s;
        float l_min = corners[0].l;
        float l_max = corners[0].l;
        for (const auto& c : corners) {
            s_min = std::min(s_min, c.s);
            s_max = std::max(s_max, c.s);
            l_min = std::min(l_min, c.l);
            l_max = std::max(l_max, c.l);
        }
        center_s = 0.5f * (s_min + s_max);
        center_l = 0.5f * (l_min + l_max);
        length = std::max(s_max - s_min, eps);
        width = std::max(l_max - l_min, eps);
        return MakeOBB(center_s, center_l, length, width, 0.0f);
    }

    float yaw = std::atan2(dir_l, dir_s);
    return MakeOBB(center_s, center_l, length, width, yaw);
}

std::vector<float> RL_DP::BuildOccupancy(const std::vector<OBB>& obstacles) const {
    return RasterizeOccupancy(obstacles,
                              s_samples_,
                              l_samples_,
                              s_min_,
                              s_max_,
                              l_min_,
                              l_max_,
                              coarse_inflation_,
                              fine_inflation_,
                              vehicle_length_,
                              vehicle_width_);
}

std::vector<int> RL_DP::Plan(const std::vector<ObstacleCorners>& obstacles, double start_l_value) {
    return Plan(obstacles, static_cast<float>(start_l_value));
}

std::vector<int> RL_DP::Plan(const std::vector<ObstacleCorners>& obstacles, float start_l_value) {
    float l_step = (l_max_ - l_min_) / std::max(1, l_samples_ - 1);
    float start_l_safe = start_l_value;
    if (!std::isfinite(start_l_safe)) {
        start_l_safe = l_min_ + l_step * static_cast<float>(l_samples_ - 1) * 0.5f;
    }
    float clamped = std::min(std::max(start_l_safe, l_min_), l_max_);
    int start_l_index = static_cast<int>(std::round((clamped - l_min_) / l_step));
    start_l_index = std::max(0, std::min(l_samples_ - 1, start_l_index));

    std::vector<OBB> obb_list;
    obb_list.reserve(obstacles.size());
    for (const auto& obs : obstacles) {
        obb_list.push_back(ObbFromCorners(obs));
    }
    auto occupancy = BuildOccupancy(obb_list);
    return planner_.plan(occupancy, start_l_index, obb_list, start_l_safe);
}

std::vector<int> RL_DP::Plan(const std::vector<ObstacleCorners>& obstacles, int start_l_index) {
    int clamped_index = std::max(0, std::min(l_samples_ - 1, start_l_index));
    float l_step = (l_max_ - l_min_) / std::max(1, l_samples_ - 1);
    float start_l_value = l_min_ + l_step * static_cast<float>(clamped_index);

    std::vector<OBB> obb_list;
    obb_list.reserve(obstacles.size());
    for (const auto& obs : obstacles) {
        obb_list.push_back(ObbFromCorners(obs));
    }
    auto occupancy = BuildOccupancy(obb_list);
    return planner_.plan(occupancy, start_l_index, obb_list, start_l_value);
}
