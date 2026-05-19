#include "rl_dp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct ObstacleSpec {
    float center_s = 0.0f;
    float center_l = 0.0f;
    float length = 1.0f;
    float width = 0.8f;
    float yaw = 0.0f;
};

struct Scenario {
    int scenario_id = -1;
    int obstacle_count = 0;
    float start_l = 0.0f;
    std::vector<ObstacleSpec> obstacles;
};

struct RunRecord {
    int scenario_id = -1;
    int obstacle_count = 0;
    double runtime_ms = 0.0;
    int path_size = 0;
};

struct Stats {
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double max_ms = 0.0;
};

std::string Trim(const std::string& value) {
    const size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return std::string();
    }
    const size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> SplitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        fields.push_back(Trim(item));
    }
    return fields;
}

int ParseInt(const std::string& value) {
    return std::stoi(value);
}

float ParseFloat(const std::string& value) {
    return std::stof(value);
}

ObstacleCorners MakeCorners(const ObstacleSpec& obstacle) {
    const float half_len = 0.5f * obstacle.length;
    const float half_wid = 0.5f * obstacle.width;
    const float c = std::cos(obstacle.yaw);
    const float s = std::sin(obstacle.yaw);
    const std::vector<std::pair<float, float>> local = {
        {half_len, half_wid},
        {half_len, -half_wid},
        {-half_len, -half_wid},
        {-half_len, half_wid},
    };

    ObstacleCorners corners{};
    for (size_t idx = 0; idx < local.size(); ++idx) {
        const float lx = local[idx].first;
        const float ly = local[idx].second;
        const float gx = obstacle.center_s + c * lx - s * ly;
        const float gy = obstacle.center_l + s * lx + c * ly;
        corners[idx] = {gx, gy};
    }
    return corners;
}

std::vector<Scenario> LoadScenarios(const std::string& csv_path) {
    std::ifstream ifs(csv_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open scenario CSV: " + csv_path);
    }

    std::string line;
    if (!std::getline(ifs, line)) {
        return {};
    }

    std::vector<Scenario> scenarios;
    std::unordered_map<int, size_t> scenario_index;
    while (std::getline(ifs, line)) {
        line = Trim(line);
        if (line.empty()) {
            continue;
        }
        const auto fields = SplitCsvLine(line);
        if (fields.size() != 8) {
            throw std::runtime_error("Expected 8 CSV columns, got " + std::to_string(fields.size()));
        }

        const int scenario_id = ParseInt(fields[0]);
        const int obstacle_count = ParseInt(fields[1]);
        const float start_l = ParseFloat(fields[2]);
        const ObstacleSpec obstacle{
            ParseFloat(fields[3]),
            ParseFloat(fields[4]),
            ParseFloat(fields[5]),
            ParseFloat(fields[6]),
            ParseFloat(fields[7]),
        };

        auto it = scenario_index.find(scenario_id);
        if (it == scenario_index.end()) {
            Scenario scenario;
            scenario.scenario_id = scenario_id;
            scenario.obstacle_count = obstacle_count;
            scenario.start_l = start_l;
            scenarios.push_back(scenario);
            scenario_index[scenario_id] = scenarios.size() - 1;
            it = scenario_index.find(scenario_id);
        }

        Scenario& scenario = scenarios[it->second];
        scenario.obstacles.push_back(obstacle);
    }

    std::sort(
        scenarios.begin(),
        scenarios.end(),
        [](const Scenario& lhs, const Scenario& rhs) { return lhs.scenario_id < rhs.scenario_id; });
    return scenarios;
}

Stats ComputeStats(std::vector<double> values) {
    Stats stats;
    if (values.empty()) {
        return stats;
    }
    std::sort(values.begin(), values.end());
    const auto percentile = [&](double q) {
        if (values.size() == 1) {
            return values.front();
        }
        const double pos = q * static_cast<double>(values.size() - 1);
        const size_t lo = static_cast<size_t>(std::floor(pos));
        const size_t hi = static_cast<size_t>(std::ceil(pos));
        const double frac = pos - static_cast<double>(lo);
        return values[lo] * (1.0 - frac) + values[hi] * frac;
    };

    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    stats.mean_ms = sum / static_cast<double>(values.size());
    stats.p50_ms = percentile(0.50);
    stats.p95_ms = percentile(0.95);
    stats.max_ms = values.back();
    return stats;
}

void WriteJson(
    const std::string& output_path,
    const std::string& model_path,
    int s_samples,
    int l_samples,
    float s_min,
    float s_max,
    float l_min,
    float l_max,
    int lateral_move_limit,
    int interpolation_points,
    const std::vector<RunRecord>& records) {
    std::ofstream ofs(output_path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to write JSON: " + output_path);
    }

    std::map<int, std::vector<double>> grouped;
    for (const auto& record : records) {
        grouped[record.obstacle_count].push_back(record.runtime_ms);
    }

    ofs << "{\n";
    ofs << "  \"config\": {\n";
    ofs << "    \"model_path\": \"" << model_path << "\",\n";
    ofs << "    \"s_samples\": " << s_samples << ",\n";
    ofs << "    \"l_samples\": " << l_samples << ",\n";
    ofs << "    \"s_range\": [" << s_min << ", " << s_max << "],\n";
    ofs << "    \"l_range\": [" << l_min << ", " << l_max << "],\n";
    ofs << "    \"lateral_move_limit\": " << lateral_move_limit << ",\n";
    ofs << "    \"interpolation_points\": " << interpolation_points << "\n";
    ofs << "  },\n";

    ofs << "  \"records\": [\n";
    for (size_t idx = 0; idx < records.size(); ++idx) {
        const auto& r = records[idx];
        ofs << "    {\"scenario_id\": " << r.scenario_id
            << ", \"obstacle_count\": " << r.obstacle_count
            << ", \"runtime_ms\": " << std::fixed << std::setprecision(6) << r.runtime_ms
            << ", \"path_size\": " << r.path_size << "}";
        if (idx + 1 != records.size()) {
            ofs << ",";
        }
        ofs << "\n";
    }
    ofs << "  ],\n";

    ofs << "  \"summary\": [\n";
    size_t emitted = 0;
    for (const auto& item : grouped) {
        const Stats stats = ComputeStats(item.second);
        ofs << "    {\"obstacle_count\": " << item.first
            << ", \"mean_ms\": " << std::fixed << std::setprecision(6) << stats.mean_ms
            << ", \"p50_ms\": " << stats.p50_ms
            << ", \"p95_ms\": " << stats.p95_ms
            << ", \"max_ms\": " << stats.max_ms << "}";
        if (++emitted != grouped.size()) {
            ofs << ",";
        }
        ofs << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string scenario_csv;
    std::string output_json;
    int s_samples = 9;
    int l_samples = 23;
    float s_min = 0.0f;
    float s_max = 8.0f;
    float l_min = -3.85f;
    float l_max = 3.85f;
    int lateral_move_limit = 3;
    int interpolation_points = 3;
    float coarse_inflation = 0.2f;
    float fine_inflation = 0.2f;
    float vehicle_length = 0.0f;
    float vehicle_width = 0.0f;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_next = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };
        if (arg == "--model") {
            model_path = require_next(arg);
        } else if (arg == "--scenario-csv") {
            scenario_csv = require_next(arg);
        } else if (arg == "--output-json") {
            output_json = require_next(arg);
        } else if (arg == "--s-samples") {
            s_samples = ParseInt(require_next(arg));
        } else if (arg == "--l-samples") {
            l_samples = ParseInt(require_next(arg));
        } else if (arg == "--s-min") {
            s_min = ParseFloat(require_next(arg));
        } else if (arg == "--s-max") {
            s_max = ParseFloat(require_next(arg));
        } else if (arg == "--l-min") {
            l_min = ParseFloat(require_next(arg));
        } else if (arg == "--l-max") {
            l_max = ParseFloat(require_next(arg));
        } else if (arg == "--lateral-move-limit") {
            lateral_move_limit = ParseInt(require_next(arg));
        } else if (arg == "--interpolation-points") {
            interpolation_points = ParseInt(require_next(arg));
        } else if (arg == "--coarse-inflation") {
            coarse_inflation = ParseFloat(require_next(arg));
        } else if (arg == "--fine-inflation") {
            fine_inflation = ParseFloat(require_next(arg));
        } else if (arg == "--vehicle-length") {
            vehicle_length = ParseFloat(require_next(arg));
        } else if (arg == "--vehicle-width") {
            vehicle_width = ParseFloat(require_next(arg));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (model_path.empty() || scenario_csv.empty() || output_json.empty()) {
        throw std::runtime_error(
            "Usage: benchmark_rl_dp_runtime --model model.onnx --scenario-csv scenarios.csv --output-json result.json");
    }

    const auto scenarios = LoadScenarios(scenario_csv);
    if (scenarios.empty()) {
        throw std::runtime_error("No scenarios loaded from CSV.");
    }

    RL_DP planner(
        model_path,
        s_samples,
        l_samples,
        s_min,
        s_max,
        l_min,
        l_max,
        lateral_move_limit,
        interpolation_points,
        coarse_inflation,
        fine_inflation,
        vehicle_length,
        vehicle_width);

    std::vector<RunRecord> records;
    records.reserve(scenarios.size());

    const size_t warmup_runs = std::min<size_t>(3, scenarios.size());
    for (size_t i = 0; i < warmup_runs; ++i) {
        std::vector<ObstacleCorners> corners;
        corners.reserve(scenarios[i].obstacles.size());
        for (const auto& obstacle : scenarios[i].obstacles) {
            corners.push_back(MakeCorners(obstacle));
        }
        static_cast<void>(planner.Plan(corners, scenarios[i].start_l));
    }

    std::map<int, std::vector<double>> grouped;
    for (const auto& scenario : scenarios) {
        std::vector<ObstacleCorners> corners;
        corners.reserve(scenario.obstacles.size());
        for (const auto& obstacle : scenario.obstacles) {
            corners.push_back(MakeCorners(obstacle));
        }

        const auto t0 = std::chrono::high_resolution_clock::now();
        const auto path = planner.Plan(corners, scenario.start_l);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double runtime_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        records.push_back(RunRecord{
            scenario.scenario_id,
            scenario.obstacle_count,
            runtime_ms,
            static_cast<int>(path.size()),
        });
        grouped[scenario.obstacle_count].push_back(runtime_ms);
    }

    for (const auto& item : grouped) {
        const Stats stats = ComputeStats(item.second);
        std::cout << "obs=" << item.first
                  << " | cpp_rl_mean=" << std::fixed << std::setprecision(3) << stats.mean_ms
                  << " ms | p50=" << stats.p50_ms
                  << " ms | p95=" << stats.p95_ms << " ms" << std::endl;
    }

    WriteJson(
        output_json,
        model_path,
        s_samples,
        l_samples,
        s_min,
        s_max,
        l_min,
        l_max,
        lateral_move_limit,
        interpolation_points,
        records);

    std::cout << "Saved JSON to " << output_json << std::endl;
    return 0;
}
