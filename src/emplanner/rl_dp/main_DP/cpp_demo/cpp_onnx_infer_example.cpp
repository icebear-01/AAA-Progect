#include <filesystem>
#include <fstream>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "dp_policy.h"
#include "dp_infer.h"
#include "dp_planner.h"

#ifdef USE_MATPLOTLIBCPP
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

static std::vector<double> linspace(double start, double end, int num) {
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

static int count_gt(const std::vector<float>& data, float threshold) {
    int count = 0;
    for (float v : data) {
        if (v > threshold) {
            ++count;
        }
    }
    return count;
}

static void log_action_mask_summary(const std::vector<float>& mask) {
    int count = 0;
    int min_idx = -1;
    int max_idx = -1;
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] > 0.0f) {
            ++count;
            if (min_idx < 0) {
                min_idx = static_cast<int>(i);
            }
            max_idx = static_cast<int>(i);
        }
    }
    std::cout << "Action mask: valid=" << count << "/" << mask.size();
    if (count > 0) {
        std::cout << ", idx_range=[" << min_idx << "," << max_idx << "]";
    }
    std::cout << std::endl;
}

static bool footprint_hits_obstacle(float s_value,
                                    float l_value,
                                    float heading,
                                    const OBB& obstacle,
                                    float fine_inflation,
                                    float vehicle_length,
                                    float vehicle_width);

static void log_section_stats(const char* name,
                              const std::vector<float>& data,
                              size_t start,
                              size_t length) {
    if (length == 0 || start >= data.size()) {
        std::cout << "Input section " << name << ": len=0" << std::endl;
        return;
    }
    size_t end = std::min(start + length, data.size());
    float sum = 0.0f;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    size_t nonzero = 0;
    for (size_t i = start; i < end; ++i) {
        float v = data[i];
        sum += v;
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        if (std::abs(v) > 1e-6f) {
            ++nonzero;
        }
    }
    float mean = sum / static_cast<float>(end - start);
    std::cout << "Input section " << name
              << ": len=" << (end - start)
              << ", sum=" << sum
              << ", mean=" << mean
              << ", min=" << min_v
              << ", max=" << max_v
              << ", nonzero=" << nonzero
              << std::endl;
}

static void log_input_overview(const std::vector<float>& data) {
    if (data.empty()) {
        std::cout << "Input overview: empty" << std::endl;
        return;
    }
    float sum = 0.0f;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    double hash = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        float v = data[i];
        sum += v;
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        hash += static_cast<double>(v) * static_cast<double>(i + 1);
    }
    std::cout << "Input overview: len=" << data.size()
              << ", sum=" << sum
              << ", min=" << min_v
              << ", max=" << max_v
              << ", hash=" << hash
              << std::endl;
}

static bool dump_input_binary(const std::string& path,
                              const std::vector<float>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(float)));
    return out.good();
}

static std::vector<float> build_corner_plane(const std::vector<OBB>& obstacles,
                                             int s_samples,
                                             int l_samples,
                                             float s_min,
                                             float s_max,
                                             float l_min,
                                             float l_max,
                                             int s_index,
                                             int last_l_index) {
    std::vector<float> plane(static_cast<size_t>(s_samples * l_samples), 0.0f);
    if (obstacles.empty()) {
        return plane;
    }
    float s_span = std::max(s_max - s_min, 1e-6f);
    float l_span = std::max(l_max - l_min, 1e-6f);
    float current_s = s_min + (s_max - s_min) * static_cast<float>(s_index) / std::max(1, s_samples - 1);
    float l_step = (l_max - l_min) / std::max(1, l_samples - 1);
    float current_l = l_min + l_step * static_cast<float>(last_l_index);

    std::vector<int> indices(obstacles.size());
    std::iota(indices.begin(), indices.end(), 0);
    auto dist_sq = [&](int idx) {
        const auto& o = obstacles[static_cast<size_t>(idx)];
        float ds = o.cx - current_s;
        float dl = o.cy - current_l;
        return ds * ds + dl * dl;
    };
    std::sort(indices.begin(), indices.end(), [&](int a, int b) { return dist_sq(a) < dist_sq(b); });
    int take = std::min<int>(10, static_cast<int>(indices.size()));

    for (int i = 0; i < take; ++i) {
        const auto& o = obstacles[static_cast<size_t>(indices[static_cast<size_t>(i)])];
        std::vector<std::pair<float, float>> local = {
            { o.half_len,  o.half_wid},
            { o.half_len, -o.half_wid},
            {-o.half_len, -o.half_wid},
            {-o.half_len,  o.half_wid},
        };
        for (const auto& p : local) {
            float gx = o.cx + o.cos_yaw * p.first - o.sin_yaw * p.second;
            float gy = o.cy + o.sin_yaw * p.first + o.cos_yaw * p.second;
            float s_norm = (gx - s_min) / s_span;
            float l_norm = (gy - l_min) / l_span;
            int s_idx = static_cast<int>(std::round(s_norm * static_cast<float>(s_samples - 1)));
            int l_idx = static_cast<int>(std::round(l_norm * static_cast<float>(l_samples - 1)));
            if (s_idx < 0 || s_idx >= s_samples || l_idx < 0 || l_idx >= l_samples) {
                continue;
            }
            plane[static_cast<size_t>(s_idx * l_samples + l_idx)] = 1.0f;
        }
    }

    return plane;
}

static float clamp01(float v) {
    return std::max(0.0f, std::min(1.0f, v));
}

static std::vector<float> build_corner_flat(const std::vector<OBB>& obstacles,
                                            int max_obstacles,
                                            float s_min,
                                            float s_max,
                                            float l_min,
                                            float l_max,
                                            float current_s,
                                            float current_l) {
    std::vector<float> flat(static_cast<size_t>(max_obstacles * 8), 0.0f);
    if (max_obstacles <= 0) {
        return flat;
    }

    float s_span = std::max(s_max - s_min, 1e-6f);
    float l_span = std::max(l_max - l_min, 1e-6f);
    float pad_s = clamp01((0.0f - s_min) / s_span);
    float pad_l = clamp01((0.0f - l_min) / l_span);
    for (size_t i = 0; i + 1 < flat.size(); i += 2) {
        flat[i] = pad_s;
        flat[i + 1] = pad_l;
    }
    if (obstacles.empty()) {
        return flat;
    }

    std::vector<int> indices(obstacles.size());
    std::iota(indices.begin(), indices.end(), 0);
    auto dist_sq = [&](int idx) {
        const auto& o = obstacles[static_cast<size_t>(idx)];
        float ds = o.cx - current_s;
        float dl = o.cy - current_l;
        return ds * ds + dl * dl;
    };
    std::sort(indices.begin(), indices.end(), [&](int a, int b) { return dist_sq(a) < dist_sq(b); });

    int take = std::min<int>(max_obstacles, static_cast<int>(indices.size()));
    size_t offset = 0;
    for (int i = 0; i < take; ++i) {
        const auto& o = obstacles[static_cast<size_t>(indices[static_cast<size_t>(i)])];
        std::vector<std::pair<float, float>> local = {
            { o.half_len,  o.half_wid},
            { o.half_len, -o.half_wid},
            {-o.half_len, -o.half_wid},
            {-o.half_len,  o.half_wid},
        };
        for (const auto& p : local) {
            float gx = o.cx + o.cos_yaw * p.first - o.sin_yaw * p.second;
            float gy = o.cy + o.sin_yaw * p.first + o.cos_yaw * p.second;
            float s_norm = clamp01((gx - s_min) / s_span);
            float l_norm = clamp01((gy - l_min) / l_span);
            if (offset + 1 < flat.size()) {
                flat[offset++] = s_norm;
                flat[offset++] = l_norm;
            }
        }
    }
    return flat;
}

static std::vector<float> rasterize_occupancy_from_obstacles(const std::vector<OBB>& obstacles,
                                                             int s_samples,
                                                             int l_samples,
                                                             float s_min,
                                                             float s_max,
                                                             float l_min,
                                                             float l_max,
                                                             float coarse_inflation = 0.2f,
                                                             float fine_inflation = 0.2f,
                                                             float vehicle_length = 0.0f,
                                                             float vehicle_width = 0.0f) {
    std::vector<float> occ(static_cast<size_t>(s_samples * l_samples), 0.0f);
    if (obstacles.empty()) return occ;

    std::vector<double> s_coords = linspace(s_min, s_max, s_samples);
    std::vector<double> l_coords = linspace(l_min, l_max, l_samples);
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
                    if (footprint_hits_obstacle(
                            s_val, l_val, heading, o, fine_inflation, vehicle_length, vehicle_width)) {
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
                float local_x = o.cos_yaw * ds + o.sin_yaw * dl;
                float local_y = -o.sin_yaw * ds + o.cos_yaw * dl;
                if (std::abs(local_x) <= fine_half_len + eps && std::abs(local_y) <= fine_half_wid + eps) {
                    occ[static_cast<size_t>(si * l_samples + li)] = 1.0f;
                }
            }
        }
    }
    return occ;
}

static bool obb_intersects_obb(float c0_s,
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

static bool point_hits_obstacle(float s_value, float l_value, const OBB& obstacle, float fine_inflation) {
    float ds = s_value - obstacle.cx;
    float dl = l_value - obstacle.cy;
    float local_x = obstacle.cos_yaw * ds + obstacle.sin_yaw * dl;
    float local_y = -obstacle.sin_yaw * ds + obstacle.cos_yaw * dl;

    float scale = 1.0f + fine_inflation;
    float half_len = std::max(obstacle.half_len * scale, 1e-6f);
    float half_wid = std::max(obstacle.half_wid * scale, 1e-6f);
    return std::abs(local_x) <= half_len && std::abs(local_y) <= half_wid;
}

static bool footprint_hits_obstacle(float s_value,
                                    float l_value,
                                    float heading,
                                    const OBB& obstacle,
                                    float fine_inflation,
                                    float vehicle_length,
                                    float vehicle_width) {
    if (vehicle_length <= 0.0f || vehicle_width <= 0.0f) {
        return point_hits_obstacle(s_value, l_value, obstacle, fine_inflation);
    }

    float scale = 1.0f + fine_inflation;
    float obs_half_len = std::max(obstacle.half_len * scale, 1e-6f);
    float obs_half_wid = std::max(obstacle.half_wid * scale, 1e-6f);
    float veh_half_len = std::max(0.5f * vehicle_length, 1e-6f);
    float veh_half_wid = std::max(0.5f * vehicle_width, 1e-6f);
    float cos0 = std::cos(heading);
    float sin0 = std::sin(heading);

    return obb_intersects_obb(
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

static bool footprint_hits_any_obstacle(float s_value,
                                        float l_value,
                                        float heading,
                                        const std::vector<OBB>& obstacles,
                                        float fine_inflation,
                                        float vehicle_length,
                                        float vehicle_width) {
    for (const auto& obstacle : obstacles) {
        if (footprint_hits_obstacle(
                s_value, l_value, heading, obstacle,
                fine_inflation, vehicle_length, vehicle_width)) {
            return true;
        }
    }
    return false;
}

static bool interpolated_hits_any_obstacle(float start_s,
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

    if (footprint_hits_any_obstacle(end_s, end_l, heading, obstacles,
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
        if (footprint_hits_any_obstacle(s_mid, l_mid, heading, obstacles,
                                        fine_inflation, vehicle_length, vehicle_width)) {
            return true;
        }
    }
    return false;
}

static std::vector<float> build_action_mask_row(int s_index,
                                                int last_l_index,
                                                int s_samples,
                                                int l_samples,
                                                float s_min,
                                                float s_max,
                                                float l_min,
                                                float l_max,
                                                int lateral_move_limit,
                                                const std::vector<OBB>& obstacles,
                                                int interpolation_points,
                                                float fine_inflation,
                                                float vehicle_length,
                                                float vehicle_width) {
    std::vector<float> mask(static_cast<size_t>(l_samples), 1.0f);
    if (s_index >= s_samples) {
        std::fill(mask.begin(), mask.end(), 0.0f);
        return mask;
    }
    if (lateral_move_limit >= 0) {
        for (int l = 0; l < l_samples; ++l) {
            if (std::abs(l - last_l_index) > lateral_move_limit) {
                mask[static_cast<size_t>(l)] = 0.0f;
            }
        }
    }
    if (!obstacles.empty() && s_index > 0) {
        float prev_s = s_min + (s_max - s_min) * static_cast<float>(s_index - 1)
                       / std::max(1, s_samples - 1);
        float prev_l = l_min + (l_max - l_min) * static_cast<float>(last_l_index)
                       / std::max(1, l_samples - 1);
        float curr_s = s_min + (s_max - s_min) * static_cast<float>(s_index)
                       / std::max(1, s_samples - 1);
        for (int l = 0; l < l_samples; ++l) {
            if (mask[static_cast<size_t>(l)] <= 0.0f) continue;
            float curr_l = l_min + (l_max - l_min) * static_cast<float>(l)
                           / std::max(1, l_samples - 1);
            if (interpolated_hits_any_obstacle(prev_s, prev_l, curr_s, curr_l,
                                               obstacles, interpolation_points,
                                               fine_inflation, vehicle_length, vehicle_width)) {
                mask[static_cast<size_t>(l)] = 0.0f;
            }
        }
    }
    return mask;
}

static std::vector<OBB> sample_random_obstacles(int max_count,
                                                float s_min,
                                                float s_max,
                                                float l_min,
                                                float l_max,
                                                float len_min,
                                                float len_max,
                                                float wid_min,
                                                float wid_max,
                                                unsigned int seed) {
    std::vector<OBB> obstacles;
    if (max_count <= 0) {
        return obstacles;
    }
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> count_dist(0, max_count);
    std::uniform_real_distribution<float> len_dist(len_min, len_max);
    std::uniform_real_distribution<float> wid_dist(wid_min, wid_max);
    std::uniform_real_distribution<float> yaw_dist(0.0f, 3.1415926f);

    auto sample_in_range = [&](float low, float high) {
        if (low > high) {
            return 0.5f * (low + high);
        }
        std::uniform_real_distribution<float> dist(low, high);
        return dist(rng);
    };

    int count = count_dist(rng);
    obstacles.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        float len = std::max(1e-3f, len_dist(rng));
        float wid = std::max(1e-3f, wid_dist(rng));
        float yaw = yaw_dist(rng);
        float margin = 0.5f * std::hypot(len, wid);
        float s_low = s_min + margin;
        float s_high = s_max - margin;
        float l_low = l_min + margin;
        float l_high = l_max - margin;
        float cx = sample_in_range(s_low, s_high);
        float cy = sample_in_range(l_low, l_high);
        obstacles.push_back(MakeOBB(cx, cy, len, wid, yaw));
    }
    return obstacles;
}

static void plot_scene(const std::vector<float>& occupancy,
                       const std::vector<int>& path,
                       const std::vector<OBB>& obstacles,
                       int s_samples,
                       int l_samples,
                       float s_min,
                       float s_max,
                       float l_min,
                       float l_max) {
    const int grid_size = s_samples * l_samples;
    if (static_cast<int>(occupancy.size()) != grid_size) {
        std::cerr << "plot_scene: occupancy size mismatch\n";
        return;
    }
    std::vector<double> s_coords = linspace(s_min, s_max, s_samples);
    std::vector<double> l_coords = linspace(l_min, l_max, l_samples);

    std::vector<double> free_x, free_y, occ_x, occ_y;
    free_x.reserve(grid_size);
    free_y.reserve(grid_size);
    occ_x.reserve(grid_size);
    occ_y.reserve(grid_size);
    for (int s_idx = 0; s_idx < s_samples; ++s_idx) {
        for (int l_idx = 0; l_idx < l_samples; ++l_idx) {
            double sx = s_coords[s_idx];
            double ly = l_coords[l_idx];
            if (occupancy[s_idx * l_samples + l_idx] > 0.5f) {
                occ_x.push_back(sx);
                occ_y.push_back(ly);
            } else {
                free_x.push_back(sx);
                free_y.push_back(ly);
            }
        }
    }

    plt::figure_size(800, 800);
    plt::scatter(free_x, free_y, 15.0, {{"marker", "s"}, {"color", "#1f77b4"}});
    if (!occ_x.empty()) {
        plt::scatter(occ_x, occ_y, 25.0, {{"marker", "s"}, {"color", "#d62728"}});
    }

    // 绘制障碍物轮廓
    for (const auto& o : obstacles) {
        double hx = o.half_len;
        double hy = o.half_wid;
        std::vector<std::pair<double, double>> local = {{hx, hy}, {hx, -hy}, {-hx, -hy}, {-hx, hy}, {hx, hy}};
        std::vector<double> xs, ys;
        xs.reserve(local.size());
        ys.reserve(local.size());
        for (auto& p : local) {
            double lx = p.first, ly = p.second;
            double gx = o.cx + o.cos_yaw * lx - o.sin_yaw * ly;
            double gy = o.cy + o.sin_yaw * lx + o.cos_yaw * ly;
            xs.push_back(gx);
            ys.push_back(gy);
        }
        plt::plot(xs, ys, {{"color", "orange"}});
    }

    // 绘制路径
    if (!path.empty()) {
        std::vector<double> path_x, path_y;
        path_x.reserve(path.size());
        path_y.reserve(path.size());
        for (size_t i = 0; i < path.size(); ++i) {
            double sx = s_coords[std::min<int>(i, s_samples - 1)];
            int l_idx = std::max(0, std::min(l_samples - 1, path[i]));
            double ly = l_coords[l_idx];
            path_x.push_back(sx);
            path_y.push_back(ly);
        }
        plt::plot(path_x, path_y, {{"color", "red"}, {"marker", "o"}});
    }

    plt::title("s-l Grid with Obstacles and Path");
    plt::xlabel("s (longitudinal)");
    plt::ylabel("l (lateral)");
    plt::axis("equal");
    plt::grid(true);
    plt::show();
}
#else
static void plot_scene(const std::vector<float>&,
                       const std::vector<int>&,
                       const std::vector<OBB>&,
                       int,
                       int,
                       float,
                       float,
                       float,
                       float) {
    std::cerr << "matplotlibcpp disabled (build with -DUSE_MATPLOTLIBCPP=ON and Python dev headers)" << std::endl;
}
#endif
int main(int argc, char* argv[]) {
    // TODO: set your ONNX model path here.
    const char* model_path = "/home/wmd/rl_dp/main_DP/main/checkpoints/ppo_policy_20251225_101829_update_12450.onnx";
    
    auto total_start = std::chrono::high_resolution_clock::now();
    // 检查模型路径是否存在
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "Model not found: " << model_path << std::endl;
        return 1;
    }

    auto load_start = std::chrono::high_resolution_clock::now();
    DPInference infer(model_path);
    auto load_end = std::chrono::high_resolution_clock::now();

    auto prep_start = std::chrono::high_resolution_clock::now();
    int64_t feature_dim = infer.feature_dim();

    // 构造包含通道数据的示例输入：3 张平面（occupancy/action_mask/obstacle_corners）+ 2 个标量。
    const int S_SAMPLES = 9;
    const int L_SAMPLES = 23;
    const float S_MIN = 0.0f;
    const float S_MAX = 8.0f;
    const float L_MIN = -3.85f;
    const float L_MAX = 3.85f;
    const int GRID_SIZE = S_SAMPLES * L_SAMPLES;
    const int EXTRA_COUNT_V1 = 2;
    const int EXTRA_COUNT_V2 = 3;
    const int PLANE_FEATURES_V1 = 3 * GRID_SIZE + EXTRA_COUNT_V1;
    const int PLANE_FEATURES_V2 = 3 * GRID_SIZE + EXTRA_COUNT_V2;
    const float S_STEP = (S_MAX - S_MIN) / std::max(1, S_SAMPLES - 1);
    const float L_STEP = (L_MAX - L_MIN) / std::max(1, L_SAMPLES - 1);
    const float COARSE_INFLATION = 0.2f;
    const float FINE_INFLATION = 0.2f;
    const int LATERAL_MOVE_LIMIT = 3;
    const int INTERPOLATION_POINTS = 3;
    float vehicle_length = 0.0f;
    float vehicle_width = 0.0f;

    bool use_random = true;
    bool has_seed = false;
    unsigned int seed = 0;
    int random_max_obstacles = 6;
    bool log_inputs = false;
    bool no_obstacles = false;
    float start_l_value = std::numeric_limits<float>::quiet_NaN();
    std::string dump_input_path;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--random") {
            use_random = true;
        } else if (arg == "--log-input") {
            log_inputs = true;
        } else if (arg == "--no-obstacles") {
            no_obstacles = true;
            use_random = false;
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<unsigned int>(std::stoul(argv[++i]));
            use_random = true;
            has_seed = true;
        } else if (arg == "--max-obstacles" && i + 1 < argc) {
            random_max_obstacles = std::max(0, std::stoi(argv[++i]));
            use_random = true;
        } else if (arg == "--start-l" && i + 1 < argc) {
            start_l_value = std::stof(argv[++i]);
        } else if (arg == "--dump-input" && i + 1 < argc) {
            dump_input_path = argv[++i];
        } else if (arg == "--vehicle-length" && i + 1 < argc) {
            vehicle_length = std::stof(argv[++i]);
        } else if (arg == "--vehicle-width" && i + 1 < argc) {
            vehicle_width = std::stof(argv[++i]);
        }
    }
    if (vehicle_length < 0.0f || vehicle_width < 0.0f) {
        std::cerr << "vehicle_length/vehicle_width must be non-negative." << std::endl;
        return 1;
    }
    if ((vehicle_length > 0.0f) != (vehicle_width > 0.0f)) {
        std::cerr << "vehicle_length/vehicle_width must both be zero or positive." << std::endl;
        return 1;
    }

    std::vector<OBB> obstacles;
    if (no_obstacles) {
        std::cout << "No obstacles: using empty obstacle set." << std::endl;
    } else if (use_random) {
        if (!has_seed) {
            seed = std::random_device{}();
        }
        obstacles = sample_random_obstacles(
            random_max_obstacles,
            S_MIN,
            S_MAX,
            L_MIN,
            L_MAX,
            0.6f,
            1.8f,
            0.4f,
            1.4f,
            seed);
        std::cout << "Random obstacles: seed=" << seed
                  << ", count=" << obstacles.size() << std::endl;
    } else {
        // 示例障碍物（供角点通道与 DPPlanner 测试共用）
        obstacles.push_back(MakeOBB(2.0f, -1.0f, 1.0f, 0.8f, 0.2f));
        obstacles.push_back(MakeOBB(4.2f, 2.0f, 0.9f, 0.7f, -0.3f));
        obstacles.push_back(MakeOBB(5.5f, 0.0f, 1.2f, 0.6f, 0.5f));
        obstacles.push_back(MakeOBB(6.5f, -2.0f, 1.1f, 0.9f, -0.4f));
    }

    std::vector<float> input_data(static_cast<size_t>(feature_dim), 0.0f);
    int extra_count = 0;
    bool use_plane = false;
    bool use_flat = false;
    int max_obstacles_feature = 0;
    if (feature_dim == PLANE_FEATURES_V2) {
        use_plane = true;
        extra_count = EXTRA_COUNT_V2;
    } else if (feature_dim == PLANE_FEATURES_V1) {
        use_plane = true;
        extra_count = EXTRA_COUNT_V1;
    } else {
        int obstacle_dim = static_cast<int>(feature_dim) - GRID_SIZE - L_SAMPLES - EXTRA_COUNT_V2;
        if (obstacle_dim >= 0 && obstacle_dim % 8 == 0) {
            use_flat = true;
            max_obstacles_feature = obstacle_dim / 8;
            extra_count = EXTRA_COUNT_V2;
        } else {
            obstacle_dim = static_cast<int>(feature_dim) - GRID_SIZE - L_SAMPLES - EXTRA_COUNT_V1;
            if (obstacle_dim >= 0 && obstacle_dim % 8 == 0) {
                use_flat = true;
                max_obstacles_feature = obstacle_dim / 8;
                extra_count = EXTRA_COUNT_V1;
            }
        }
    }
    std::cout << "Input layout: "
              << (use_plane ? "plane" : (use_flat ? "flat" : "unknown"))
              << ", feature_dim=" << feature_dim
              << ", grid_size=" << GRID_SIZE
              << ", extra_count=" << extra_count;
    if (use_flat) {
        std::cout << ", max_obstacles_feature=" << max_obstacles_feature;
    }
    std::cout << std::endl;
    if (log_inputs) {
        std::cout << "Grid: s_range=[" << S_MIN << "," << S_MAX << "], s_samples=" << S_SAMPLES
                  << ", s_step=" << S_STEP
                  << " | l_range=[" << L_MIN << "," << L_MAX << "], l_samples=" << L_SAMPLES
                  << ", l_step=" << L_STEP
                  << std::endl;
        std::cout << "Vehicle footprint: length=" << vehicle_length
                  << ", width=" << vehicle_width << std::endl;
    }

    if (use_plane) {
        // 按障碍物栅格化占用
        int s_index = 4;
        int last_l_index = 9;
        float start_l_coord = std::isfinite(start_l_value)
                                  ? std::min(std::max(start_l_value, L_MIN), L_MAX)
                                  : (L_MIN + L_STEP * static_cast<float>(last_l_index));
        int start_l_index = static_cast<int>(std::round((start_l_coord - L_MIN) / L_STEP));
        start_l_index = std::max(0, std::min(L_SAMPLES - 1, start_l_index));

        std::vector<float> occupancy = rasterize_occupancy_from_obstacles(
            obstacles, S_SAMPLES, L_SAMPLES, S_MIN, S_MAX, L_MIN, L_MAX,
            COARSE_INFLATION, FINE_INFLATION, vehicle_length, vehicle_width);

        std::vector<float> action_mask_row = build_action_mask_row(
            s_index,
            last_l_index,
            S_SAMPLES,
            L_SAMPLES,
            S_MIN,
            S_MAX,
            L_MIN,
            L_MAX,
            LATERAL_MOVE_LIMIT,
            obstacles,
            INTERPOLATION_POINTS,
            FINE_INFLATION,
            vehicle_length,
            vehicle_width);

        if (log_inputs) {
            int occ_count = count_gt(occupancy, 0.5f);
            int mask_count = count_gt(action_mask_row, 0.0f);
            float s_coord = S_MIN + S_STEP * static_cast<float>(s_index);
            float l_coord = L_MIN + L_STEP * static_cast<float>(last_l_index);
            float s_norm_dbg = static_cast<float>(s_index) / std::max(1, S_SAMPLES - 1);
            float l_norm_dbg = (l_coord - L_MIN) / (L_MAX - L_MIN);
            float start_l_norm_dbg = (start_l_coord - L_MIN) / (L_MAX - L_MIN);
            std::cout << "Scene: obstacles=" << obstacles.size()
                      << ", occupancy_cells=" << occ_count
                      << ", action_mask_valid=" << mask_count << "/" << L_SAMPLES
                      << ", s_index=" << s_index
                      << ", last_l_index=" << last_l_index
                      << ", s_coord=" << s_coord
                      << ", l_coord=" << l_coord
                      << ", start_l=" << start_l_coord
                      << ", s_norm=" << s_norm_dbg
                      << ", l_norm=" << l_norm_dbg
                      << ", start_l_norm=" << start_l_norm_dbg
                      << std::endl;
            if (mask_count == 0) {
                std::cerr << "Warning: action_mask_row is empty at s_index=" << s_index << std::endl;
            }
            log_action_mask_summary(action_mask_row);
        }

        // 当前列 s_index 的动作掩码，其他列为 0。
        std::vector<float> mask_plane(GRID_SIZE, 0.0f);
        if (s_index >= 0 && s_index < S_SAMPLES) {
            for (int l = 0; l < L_SAMPLES; ++l) {
                mask_plane[s_index * L_SAMPLES + l] = action_mask_row[static_cast<size_t>(l)];
            }
        }

        std::vector<float> corner_plane = build_corner_plane(
            obstacles, S_SAMPLES, L_SAMPLES, S_MIN, S_MAX, L_MIN, L_MAX, s_index, last_l_index);

        size_t offset = 0;
        for (float v : occupancy)   input_data[offset++] = v;
        for (float v : mask_plane)  input_data[offset++] = v;
        for (float v : corner_plane)input_data[offset++] = v;

        float s_norm = static_cast<float>(s_index) / std::max(1, S_SAMPLES - 1);
        float l_coord = L_MIN + L_STEP * static_cast<float>(last_l_index);
        float l_norm = (l_coord - L_MIN) / (L_MAX - L_MIN);
        float start_l_norm = (start_l_coord - L_MIN) / (L_MAX - L_MIN);
        input_data[offset++] = s_norm;
        input_data[offset++] = l_norm;
        if (extra_count >= 3) {
            input_data[offset++] = start_l_norm;
        }
        assert(static_cast<int64_t>(offset) == feature_dim);
    } else if (use_flat) {
        int s_index = 4;
        int last_l_index = 9;
        float start_l_coord = std::isfinite(start_l_value)
                                  ? std::min(std::max(start_l_value, L_MIN), L_MAX)
                                  : (L_MIN + L_STEP * static_cast<float>(last_l_index));
        int start_l_index = static_cast<int>(std::round((start_l_coord - L_MIN) / L_STEP));
        start_l_index = std::max(0, std::min(L_SAMPLES - 1, start_l_index));

        std::vector<float> occupancy = rasterize_occupancy_from_obstacles(
            obstacles, S_SAMPLES, L_SAMPLES, S_MIN, S_MAX, L_MIN, L_MAX,
            COARSE_INFLATION, FINE_INFLATION, vehicle_length, vehicle_width);

        std::vector<float> action_mask_row = build_action_mask_row(
            s_index,
            last_l_index,
            S_SAMPLES,
            L_SAMPLES,
            S_MIN,
            S_MAX,
            L_MIN,
            L_MAX,
            LATERAL_MOVE_LIMIT,
            obstacles,
            INTERPOLATION_POINTS,
            FINE_INFLATION,
            vehicle_length,
            vehicle_width);

        if (log_inputs) {
            int occ_count = count_gt(occupancy, 0.5f);
            int mask_count = count_gt(action_mask_row, 0.0f);
            float s_coord = S_MIN + S_STEP * static_cast<float>(s_index);
            float l_coord = L_MIN + L_STEP * static_cast<float>(last_l_index);
            float s_norm_dbg = static_cast<float>(s_index) / std::max(1, S_SAMPLES - 1);
            float l_norm_dbg = (l_coord - L_MIN) / (L_MAX - L_MIN);
            float start_l_norm_dbg = (start_l_coord - L_MIN) / (L_MAX - L_MIN);
            std::cout << "Scene: obstacles=" << obstacles.size()
                      << ", occupancy_cells=" << occ_count
                      << ", action_mask_valid=" << mask_count << "/" << L_SAMPLES
                      << ", s_index=" << s_index
                      << ", last_l_index=" << last_l_index
                      << ", s_coord=" << s_coord
                      << ", l_coord=" << l_coord
                      << ", start_l=" << start_l_coord
                      << ", s_norm=" << s_norm_dbg
                      << ", l_norm=" << l_norm_dbg
                      << ", start_l_norm=" << start_l_norm_dbg
                      << std::endl;
            if (mask_count == 0) {
                std::cerr << "Warning: action_mask_row is empty at s_index=" << s_index << std::endl;
            }
            log_action_mask_summary(action_mask_row);
        }

        float s_coord = S_MIN + (S_MAX - S_MIN) * static_cast<float>(s_index) / std::max(1, S_SAMPLES - 1);
        float l_coord = L_MIN + L_STEP * static_cast<float>(last_l_index);
        std::vector<float> corner_flat = build_corner_flat(
            obstacles, max_obstacles_feature, S_MIN, S_MAX, L_MIN, L_MAX, s_coord, l_coord);

        size_t offset = 0;
        for (float v : occupancy) input_data[offset++] = v;
        for (float v : corner_flat) input_data[offset++] = v;
        for (float v : action_mask_row) input_data[offset++] = v;

        float s_norm = static_cast<float>(s_index) / std::max(1, S_SAMPLES - 1);
        float l_norm = (l_coord - L_MIN) / (L_MAX - L_MIN);
        float start_l_norm = (start_l_coord - L_MIN) / (L_MAX - L_MIN);
        input_data[offset++] = s_norm;
        input_data[offset++] = l_norm;
        if (extra_count >= 3) {
            input_data[offset++] = start_l_norm;
        }
        assert(static_cast<int64_t>(offset) == feature_dim);
    } else {
        std::cerr << "Warning: feature_dim(" << feature_dim
                  << ") does not match known layouts, using zero input." << std::endl;
    }
    if (log_inputs) {
        log_input_overview(input_data);
        if (use_flat) {
            int obstacle_dim = max_obstacles_feature * 8;
            size_t offset = 0;
            log_section_stats("occupancy", input_data, offset, static_cast<size_t>(GRID_SIZE));
            offset += static_cast<size_t>(GRID_SIZE);
            log_section_stats("obstacle_flat", input_data, offset, static_cast<size_t>(obstacle_dim));
            offset += static_cast<size_t>(obstacle_dim);
            log_section_stats("action_mask", input_data, offset, static_cast<size_t>(L_SAMPLES));
            offset += static_cast<size_t>(L_SAMPLES);
            log_section_stats("extras", input_data, offset, static_cast<size_t>(extra_count));
        }
        if (use_plane) {
            size_t offset = 0;
            log_section_stats("occupancy", input_data, offset, static_cast<size_t>(GRID_SIZE));
            offset += static_cast<size_t>(GRID_SIZE);
            log_section_stats("mask_plane", input_data, offset, static_cast<size_t>(GRID_SIZE));
            offset += static_cast<size_t>(GRID_SIZE);
            log_section_stats("corner_plane", input_data, offset, static_cast<size_t>(GRID_SIZE));
            offset += static_cast<size_t>(GRID_SIZE);
            log_section_stats("extras", input_data, offset, static_cast<size_t>(extra_count));
        }
    }
    if (!dump_input_path.empty()) {
        if (dump_input_binary(dump_input_path, input_data)) {
            std::cout << "Dumped input to " << dump_input_path << std::endl;
        } else {
            std::cerr << "Failed to dump input to " << dump_input_path << std::endl;
        }
    }
    if (log_inputs) {
        std::cout << "Output names: [0]=" << infer.logits_name()
                  << ", [1]=" << infer.value_name() << std::endl;
    }

    // 运行模型并获取输出，记录推理耗时
    auto prep_end = std::chrono::high_resolution_clock::now();
    InferenceOutput output;

    // 按 s_samples 次数循环推理（默认 9 次），便于对齐环境 rollout
    const int s_iters = 9;
    auto t_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < s_iters; ++i) {
        output = infer.Run(input_data);
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms_runs = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double infer_ms = total_ms_runs / s_iters;

    const int64_t action_dim = static_cast<int64_t>(output.logits.size());
    if (log_inputs) {
        std::cout << "All logits (action 0.." << action_dim - 1 << "):" << std::endl;
        for (int64_t i = 0; i < action_dim; ++i) {
            std::cout << "  action " << i << " logit=" << output.logits[static_cast<size_t>(i)] << std::endl;
        }
    }

    std::cout << "Value: " << output.value << std::endl;
    std::cout << "9-run total inference time: " << total_ms_runs << " ms, avg: "
              << infer_ms << " ms/run" << std::endl;

    auto load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    auto prep_ms = std::chrono::duration<double, std::milli>(prep_end - prep_start).count();
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    std::cout << "Load session time: " << load_ms << " ms" << std::endl;
    std::cout << "Input prep time: " << prep_ms << " ms" << std::endl;
    std::cout << "Total forward time: " << total_ms_runs << " ms" << std::endl;
    std::cout << "Program total time: " << total_ms << " ms" << std::endl;

    // 额外示例：使用 DPPlanner 滚动 9 列
    try {
        // 与 test_dp_policy 中相同的占用/掩码示例
        int s_samples = 9;
        int l_samples = 23;
        std::vector<float> occupancy(static_cast<size_t>(s_samples * l_samples), 0.0f);
        occupancy = rasterize_occupancy_from_obstacles(obstacles, s_samples, l_samples,
                                                       S_MIN, S_MAX, L_MIN, L_MAX,
                                                       COARSE_INFLATION, FINE_INFLATION,
                                                       vehicle_length, vehicle_width);

        // 初始化 policy 与 planner（含几何判碰）
        DPPolicy policy(model_path);
        DPPlanner planner(policy, s_samples, l_samples, LATERAL_MOVE_LIMIT,
                          S_MIN, S_MAX, L_MIN, L_MAX,
                          INTERPOLATION_POINTS, FINE_INFLATION,
                          vehicle_length, vehicle_width);
        int start_l_index = l_samples / 2;
        float start_l_coord = std::isfinite(start_l_value)
                                  ? std::min(std::max(start_l_value, L_MIN), L_MAX)
                                  : (L_MIN + L_STEP * static_cast<float>(start_l_index));
        start_l_index = static_cast<int>(std::round((start_l_coord - L_MIN) / L_STEP));
        start_l_index = std::max(0, std::min(l_samples - 1, start_l_index));
        auto path = planner.plan(occupancy, start_l_index, obstacles, start_l_coord);
        std::cout << "Planner path indices:";
        for (int idx : path) std::cout << " " << idx;
        std::cout << std::endl;

        if (log_inputs) {
            std::cout << "Planner path coords:";
            for (size_t i = 0; i < path.size(); ++i) {
                float s_coord = S_MIN + S_STEP * static_cast<float>(std::min<int>(i, s_samples - 1));
                float l_coord = L_MIN + L_STEP * static_cast<float>(path[i]);
                std::cout << " (" << s_coord << "," << l_coord << ")";
            }
            std::cout << std::endl;
        }

        // 可视化
        plot_scene(occupancy, path, obstacles, s_samples, l_samples, S_MIN, S_MAX, L_MIN, L_MAX);
    } catch (const std::exception& e) {
        std::cerr << "DPPlanner test failed: " << e.what() << std::endl;
    }

    return 0;
}
