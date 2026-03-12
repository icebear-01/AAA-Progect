#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "dp_planner.h"
#include "dp_policy.h"

#ifdef USE_MATPLOTLIBCPP
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

struct ObstacleSpec {
    float cx;
    float cy;
    float len;
    float wid;
    float yaw;
};

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

static bool footprint_hits_obstacle(float s_value,
                                    float l_value,
                                    float heading,
                                    const OBB& obstacle,
                                    float fine_inflation,
                                    float vehicle_length,
                                    float vehicle_width) {
    float ds = s_value - obstacle.cx;
    float dl = l_value - obstacle.cy;
    float local_x = obstacle.cos_yaw * ds + obstacle.sin_yaw * dl;
    float local_y = -obstacle.sin_yaw * ds + obstacle.cos_yaw * dl;

    float scale = 1.0f + fine_inflation;
    float obs_half_len = std::max(obstacle.half_len * scale, 1e-6f);
    float obs_half_wid = std::max(obstacle.half_wid * scale, 1e-6f);
    float veh_half_len = std::max(0.5f * vehicle_length, 1e-6f);
    float veh_half_wid = std::max(0.5f * vehicle_width, 1e-6f);
    float cos0 = std::cos(heading);
    float sin0 = std::sin(heading);

    float u0x = cos0;
    float u0y = sin0;
    float v0x = -sin0;
    float v0y = cos0;
    float u1x = obstacle.cos_yaw;
    float u1y = obstacle.sin_yaw;
    float v1x = -obstacle.sin_yaw;
    float v1y = obstacle.cos_yaw;

    float t_x = obstacle.cx - s_value;
    float t_y = obstacle.cy - l_value;
    const float eps = 1e-9f;

    auto axis_separates = [&](float ax, float ay) {
        float ra = veh_half_len * std::abs(u0x * ax + u0y * ay)
                   + veh_half_wid * std::abs(v0x * ax + v0y * ay);
        float rb = obs_half_len * std::abs(u1x * ax + u1y * ay)
                   + obs_half_wid * std::abs(v1x * ax + v1y * ay);
        float dist = std::abs(t_x * ax + t_y * ay);
        return dist > (ra + rb + eps);
    };

    if (axis_separates(u0x, u0y)) return false;
    if (axis_separates(v0x, v0y)) return false;
    if (axis_separates(u1x, u1y)) return false;
    if (axis_separates(v1x, v1y)) return false;
    return true;
}

static std::vector<float> rasterize_occupancy_from_obstacles(const std::vector<OBB>& obstacles,
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

static std::vector<OBB> build_shifted_obstacles(const std::vector<ObstacleSpec>& specs,
                                                float s_shift) {
    std::vector<OBB> obstacles;
    obstacles.reserve(specs.size());
    for (const auto& spec : specs) {
        obstacles.push_back(MakeOBB(spec.cx + s_shift, spec.cy, spec.len, spec.wid, spec.yaw));
    }
    return obstacles;
}

#ifdef USE_MATPLOTLIBCPP
static void plot_scene(const std::vector<float>& occupancy,
                       const std::vector<int>& path,
                       const std::vector<OBB>& obstacles,
                       int s_samples,
                       int l_samples,
                       float s_min,
                       float s_max,
                       float l_min,
                       float l_max,
                       const std::string& title,
                       const std::string& save_path) {
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

    plt::figure();
    plt::figure_size(800, 800);
    plt::scatter(free_x, free_y, 15.0, {{"marker", "s"}, {"color", "#1f77b4"}});
    if (!occ_x.empty()) {
        plt::scatter(occ_x, occ_y, 25.0, {{"marker", "s"}, {"color", "#d62728"}});
    }

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

    if (!path.empty()) {
        std::vector<double> path_x, path_y;
        path_x.reserve(path.size());
        path_y.reserve(path.size());
        for (size_t i = 0; i < path.size(); ++i) {
            double sx = s_coords[std::min<int>(static_cast<int>(i), s_samples - 1)];
            int l_idx = std::max(0, std::min(l_samples - 1, path[i]));
            double ly = l_coords[l_idx];
            path_x.push_back(sx);
            path_y.push_back(ly);
        }
        plt::plot(path_x, path_y, {{"color", "red"}, {"marker", "o"}});
    }

    plt::title(title);
    plt::xlabel("s (longitudinal)");
    plt::ylabel("l (lateral)");
    plt::axis("equal");
    plt::grid(true);
    plt::save(save_path);
    plt::close();
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
                       float,
                       const std::string&,
                       const std::string&) {
    std::cerr << "matplotlibcpp disabled (build with -DUSE_MATPLOTLIBCPP=ON)\n";
}
#endif

int main(int argc, char* argv[]) {
    const char* model_path = "/home/wmd/rl_dp/main_DP/main/checkpoints/ppo_policy_20251223_113744_update_31400.onnx";
    int steps = 9;
    float start_l_value = 0.0f;
    float vehicle_length = 0.0f;
    float vehicle_width = 0.0f;
    std::string out_dir = "rollout_plots";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--steps" && i + 1 < argc) {
            steps = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--start-l" && i + 1 < argc) {
            start_l_value = std::stof(argv[++i]);
        } else if (arg == "--vehicle-length" && i + 1 < argc) {
            vehicle_length = std::stof(argv[++i]);
        } else if (arg == "--vehicle-width" && i + 1 < argc) {
            vehicle_width = std::stof(argv[++i]);
        } else if (arg == "--out-dir" && i + 1 < argc) {
            out_dir = argv[++i];
        }
    }
    if ((vehicle_length > 0.0f) != (vehicle_width > 0.0f)) {
        std::cerr << "vehicle_length/vehicle_width must both be zero or positive.\n";
        return 1;
    }
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "Model not found: " << model_path << "\n";
        return 1;
    }
    std::filesystem::create_directories(out_dir);

    const int S_SAMPLES = 9;
    const int L_SAMPLES = 19;
    const float S_MIN = 0.0f;
    const float S_MAX = 8.0f;
    const float L_MIN = -4.0f;
    const float L_MAX = 4.0f;
    const float COARSE_INFLATION = 0.2f;
    const float FINE_INFLATION = 0.2f;
    const int LATERAL_MOVE_LIMIT = 3;
    const int INTERPOLATION_POINTS = 3;

    std::vector<ObstacleSpec> base_obstacles = {
        {2.0f, -1.0f, 1.0f, 0.8f, 0.2f},
        {4.2f,  2.0f, 0.9f, 0.7f, -0.3f},
        {5.5f,  0.0f, 1.2f, 0.6f, 0.5f},
        {6.5f, -2.0f, 1.1f, 0.9f, -0.4f},
    };

    DPPolicy policy(model_path, S_SAMPLES, L_SAMPLES, S_MIN, S_MAX, L_MIN, L_MAX);
    DPPlanner planner(policy, S_SAMPLES, L_SAMPLES, LATERAL_MOVE_LIMIT,
                      S_MIN, S_MAX, L_MIN, L_MAX,
                      INTERPOLATION_POINTS, FINE_INFLATION,
                      vehicle_length, vehicle_width);

    float l_step = (L_MAX - L_MIN) / std::max(1, L_SAMPLES - 1);
    float start_l_coord = std::isfinite(start_l_value)
                              ? std::min(std::max(start_l_value, L_MIN), L_MAX)
                              : (L_MIN + l_step * static_cast<float>(L_SAMPLES - 1) * 0.5f);
    int current_start = static_cast<int>(std::round((start_l_coord - L_MIN) / l_step));
    current_start = std::max(0, std::min(L_SAMPLES - 1, current_start));
    float current_start_value = start_l_coord;
    for (int step = 0; step < steps; ++step) {
        float s_shift = -static_cast<float>(step);
        auto obstacles = build_shifted_obstacles(base_obstacles, s_shift);
        std::vector<OBB> visible_obstacles;
        visible_obstacles.reserve(obstacles.size());
        for (const auto& o : obstacles) {
            if (o.aabb_max_s >= S_MIN) {
                visible_obstacles.push_back(o);
            }
        }
        auto occupancy = rasterize_occupancy_from_obstacles(
            visible_obstacles, S_SAMPLES, L_SAMPLES, S_MIN, S_MAX, L_MIN, L_MAX,
            COARSE_INFLATION, FINE_INFLATION, vehicle_length, vehicle_width);

        auto path = planner.plan(occupancy, current_start, visible_obstacles, current_start_value);
        std::string title = "step " + std::to_string(step) + ", start_l=" + std::to_string(current_start_value);
        std::string save_path = out_dir + "/rollout_step_" + std::to_string(step) + ".png";
        plot_scene(occupancy, path, visible_obstacles, S_SAMPLES, L_SAMPLES,
                   S_MIN, S_MAX, L_MIN, L_MAX, title, save_path);
        std::cout << "Saved: " << save_path << "\n";

        if (path.size() > 1) {
            current_start = path[1];
        } else if (!path.empty()) {
            current_start = path[0];
        }
        current_start_value = L_MIN + l_step * static_cast<float>(current_start);
    }

    return 0;
}
