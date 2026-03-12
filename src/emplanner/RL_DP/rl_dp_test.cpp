#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "rl_dp.h"

#ifdef USE_MATPLOTLIBCPP
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

static ObstacleCorners MakeCorners(float cx, float cy, float len, float wid, float yaw) {
    float half_len = 0.5f * len;
    float half_wid = 0.5f * wid;
    float c = std::cos(yaw);
    float s = std::sin(yaw);
    std::array<std::pair<float, float>, 4> local = {{
        {half_len, half_wid},
        {half_len, -half_wid},
        {-half_len, -half_wid},
        {-half_len, half_wid},
    }};
    ObstacleCorners corners;
    for (size_t i = 0; i < local.size(); ++i) {
        float lx = local[i].first;
        float ly = local[i].second;
        float gx = cx + c * lx - s * ly;
        float gy = cy + s * lx + c * ly;
        corners[i] = {gx, gy};
    }
    return corners;
}

#ifdef USE_MATPLOTLIBCPP
static std::vector<double> Linspace(double start, double end, int num) {
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

static void PlotResult(const std::vector<ObstacleCorners>& obstacles,
                       const std::vector<int>& path,
                       int s_samples,
                       int l_samples,
                       float s_min,
                       float s_max,
                       float l_min,
                       float l_max,
                       const std::string& save_path) {
    std::vector<double> s_coords = Linspace(s_min, s_max, s_samples);
    std::vector<double> l_coords = Linspace(l_min, l_max, l_samples);

    std::vector<double> grid_s;
    std::vector<double> grid_l;
    grid_s.reserve(static_cast<size_t>(s_samples * l_samples));
    grid_l.reserve(static_cast<size_t>(s_samples * l_samples));
    for (int s_idx = 0; s_idx < s_samples; ++s_idx) {
        for (int l_idx = 0; l_idx < l_samples; ++l_idx) {
            grid_s.push_back(s_coords[s_idx]);
            grid_l.push_back(l_coords[l_idx]);
        }
    }

    plt::figure();
    plt::figure_size(800, 800);
    plt::scatter(grid_s, grid_l, 15.0, {{"marker", "s"}, {"color", "#1f77b4"}});

    for (const auto& obs : obstacles) {
        std::vector<double> xs;
        std::vector<double> ys;
        xs.reserve(obs.size() + 1);
        ys.reserve(obs.size() + 1);
        for (const auto& pt : obs) {
            xs.push_back(pt.s);
            ys.push_back(pt.l);
        }
        xs.push_back(obs[0].s);
        ys.push_back(obs[0].l);
        plt::plot(xs, ys, {{"color", "orange"}});
    }

    if (!path.empty()) {
        std::vector<double> path_s;
        std::vector<double> path_l;
        path_s.reserve(path.size());
        path_l.reserve(path.size());
        for (size_t i = 0; i < path.size(); ++i) {
            int s_idx = std::min<int>(static_cast<int>(i), s_samples - 1);
            int l_idx = std::max(0, std::min(l_samples - 1, path[i]));
            path_s.push_back(s_coords[s_idx]);
            path_l.push_back(l_coords[l_idx]);
        }
        plt::plot(path_s, path_l, {{"color", "red"}, {"marker", "o"}});
    }

    plt::title("RL_DP Test Path");
    plt::xlabel("s (longitudinal)");
    plt::ylabel("l (lateral)");
    plt::axis("equal");
    plt::grid(true);
    plt::save(save_path);
    plt::close();
}
#endif

int main(int argc, char* argv[]) {
    const char* model_path = "/home/wmd/rl_dp/main_DP/main/checkpoints/ppo_policy_20251225_101829_update_12450.onnx";
    float start_l_value = 0.0f;
    bool no_obstacles = false;
    float vehicle_length = 0.0f;
    float vehicle_width = 0.0f; 
    std::string plot_out = "rl_dp_test.png";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--start-l" && i + 1 < argc) {
            start_l_value = std::stof(argv[++i]);
        } else if (arg == "--no-obstacles") {
            no_obstacles = true;
        } else if (arg == "--vehicle-length" && i + 1 < argc) {
            vehicle_length = std::stof(argv[++i]);
        } else if (arg == "--vehicle-width" && i + 1 < argc) {
            vehicle_width = std::stof(argv[++i]);
        } else if (arg == "--plot-out" && i + 1 < argc) {
            plot_out = argv[++i];
        }
    }

    if (!std::filesystem::exists(model_path)) {
        std::cerr << "Model not found: " << model_path << std::endl;
        return 1;
    }
    if ((vehicle_length > 0.0f) != (vehicle_width > 0.0f)) {
        std::cerr << "vehicle_length/vehicle_width must both be zero or positive." << std::endl;
        return 1;
    }

    RL_DP rl(model_path,
             9,
             19,
             0.0f,
             8.0f,
             -4.0f,
             4.0f,
             3,
             3,
             0.2f,
             0.2f,
             vehicle_length,
             vehicle_width);

    std::vector<ObstacleCorners> obstacles;
    if (!no_obstacles) {
        obstacles.push_back(MakeCorners(2.0f, -1.0f, 1.0f, 0.8f, 0.2f));
        obstacles.push_back(MakeCorners(4.2f, 2.0f, 0.9f, 0.7f, -0.3f));
        obstacles.push_back(MakeCorners(5.5f, 0.0f, 1.2f, 0.6f, 0.5f));
        obstacles.push_back(MakeCorners(6.5f, -2.0f, 1.1f, 0.9f, -0.4f));
    }

    auto path = rl.Plan(obstacles, start_l_value);
    std::cout << "Start l (continuous): " << start_l_value << std::endl;
    std::cout << "Path size: " << path.size() << std::endl;
    std::cout << "Path indices:";
    for (int idx : path) {
        std::cout << " " << idx;
    }
    std::cout << std::endl;

#ifdef USE_MATPLOTLIBCPP
    PlotResult(obstacles, path, 9, 19, 0.0f, 8.0f, -4.0f, 4.0f, plot_out);
    std::cout << "Saved plot: " << plot_out << std::endl;
#else
    std::cout << "matplotlibcpp disabled; no plot generated." << std::endl;
#endif

    return 0;
}
