#include <chrono>
#include <iostream>
#include <random>
#include <string>

#include "hope/env.h"

namespace {

double ToDouble(const std::string& s, double fallback) {
    try {
        return std::stod(s);
    } catch (...) {
        return fallback;
    }
}

int ToInt(const std::string& s, int fallback) {
    try {
        return std::stoi(s);
    } catch (...) {
        return fallback;
    }
}

}  // namespace

int main(int argc, char** argv) {
    int episodes = 10;
    int steps = 120;
    std::string level = hope::kDefaultMapLevel;
    int fps = 30;
    bool random_action = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--episodes" && i + 1 < argc) {
            episodes = ToInt(argv[++i], episodes);
        } else if (arg == "--steps" && i + 1 < argc) {
            steps = ToInt(argv[++i], steps);
        } else if (arg == "--level" && i + 1 < argc) {
            level = argv[++i];
        } else if (arg == "--fps" && i + 1 < argc) {
            fps = ToInt(argv[++i], fps);
        } else if (arg == "--random-action") {
            random_action = true;
        }
    }

    hope::CarParking env(hope::RenderMode::Human, fps, false, true, true, true, true);

    std::mt19937 rng(static_cast<unsigned int>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::uniform_real_distribution<double> steer_dist(hope::kValidSteerMin, hope::kValidSteerMax);
    std::uniform_real_distribution<double> speed_dist(hope::kValidSpeedMin, hope::kValidSpeedMax);

    for (int ep = 0; ep < episodes; ++ep) {
        env.Reset(-1, std::string(), level);
        for (int st = 0; st < steps; ++st) {
            if (random_action) {
                hope::Action action{steer_dist(rng), speed_dist(rng)};
                env.Step(action);
            } else {
                env.Step(std::nullopt);
            }
        }
    }

    env.Close();
    return 0;
}
