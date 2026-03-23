#include "hybrid_a_star/hybrid_a_star.h"

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string input_yaml;
    std::string output_csv;
    std::string seed_csv;
    std::string split_points_csv;
    std::string metrics_json;
};

Args ParseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string token(argv[i]);
        if (token == "--input-yaml" && i + 1 < argc) {
            args.input_yaml = argv[++i];
        } else if (token == "--output-csv" && i + 1 < argc) {
            args.output_csv = argv[++i];
        } else if (token == "--seed-csv" && i + 1 < argc) {
            args.seed_csv = argv[++i];
        } else if (token == "--split-points-csv" && i + 1 < argc) {
            args.split_points_csv = argv[++i];
        } else if (token == "--metrics-json" && i + 1 < argc) {
            args.metrics_json = argv[++i];
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + token);
        }
    }
    if (args.input_yaml.empty() || args.output_csv.empty()) {
        throw std::runtime_error(
            "Usage: smooth_path_cli --input-yaml <yaml> --output-csv <csv> [--seed-csv <csv>] [--split-points-csv <csv>] [--metrics-json <json>]");
    }
    return args;
}

double ReadDouble(const YAML::Node& node, const std::string& key, double default_value) {
    if (!node[key]) {
        return default_value;
    }
    return node[key].as<double>();
}

int ReadInt(const YAML::Node& node, const std::string& key, int default_value) {
    if (!node[key]) {
        return default_value;
    }
    return node[key].as<int>();
}

VectorVec4d LoadRawPath(const YAML::Node& root) {
    const YAML::Node raw_path = root["raw_path"];
    if (!raw_path || !raw_path.IsSequence() || raw_path.size() < 2) {
        throw std::runtime_error("raw_path must be a sequence with at least 2 points");
    }

    VectorVec4d path;
    path.reserve(raw_path.size());
    for (std::size_t i = 0; i < raw_path.size(); ++i) {
        const YAML::Node point = raw_path[i];
        if (!point.IsSequence() || point.size() < 2) {
            throw std::runtime_error("raw_path point must be [x, y] or [x, y, yaw, dir]");
        }
        Vec4d pose = Vec4d::Zero();
        pose.x() = point[0].as<double>();
        pose.y() = point[1].as<double>();
        pose.z() = point.size() >= 3 ? point[2].as<double>() : 0.0;
        pose.w() = point.size() >= 4 ? point[3].as<double>() : 1.0;
        path.emplace_back(pose);
    }
    return path;
}

void WritePathCsv(const std::string& output_csv, const VectorVec4d& path) {
    std::ofstream out(output_csv);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output csv: " + output_csv);
    }
    out << "x,y,yaw,dir\n";
    for (const auto& pose : path) {
        out << pose.x() << "," << pose.y() << "," << pose.z() << "," << pose.w() << "\n";
    }
}

void WriteSplitPointsCsv(const std::string& output_csv, const VectorVec4d& points) {
    std::ofstream out(output_csv);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open split points csv: " + output_csv);
    }
    out << "index,x,y,yaw,dir\n";
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto& pose = points[i];
        out << i << "," << pose.x() << "," << pose.y() << "," << pose.z() << "," << pose.w() << "\n";
    }
}

double ComputePathLength(const VectorVec4d& path) {
    double length = 0.0;
    for (std::size_t i = 1; i < path.size(); ++i) {
        const double dx = path[i].x() - path[i - 1].x();
        const double dy = path[i].y() - path[i - 1].y();
        length += std::hypot(dx, dy);
    }
    return length;
}

struct SeedOptimizationResult {
    VectorVec4d path;
    bool qp_init_failed = false;
    bool qp_solve_failed = false;
};

void WriteMetricsJson(const std::string& output_json,
                      double seed_stage_ms,
                      double smooth_stage_ms,
                      double total_cli_ms,
                      const VectorVec4d& raw_path,
                      const VectorVec4d& seed_path,
                      const VectorVec4d& smoothed_path,
                      std::size_t split_points_count,
                      bool seed_qp_init_fallback_raw,
                      bool seed_qp_solve_fallback_raw,
                      bool seed_collision_fallback_raw,
                      bool resampled_seed_collision_fallback_raw,
                      bool smooth_collision_fallback_seed) {
    const bool seed_success = !(seed_qp_init_fallback_raw || seed_qp_solve_fallback_raw ||
                                seed_collision_fallback_raw || resampled_seed_collision_fallback_raw);
    const bool smooth_success = !smooth_collision_fallback_seed;
    const bool backend_success = seed_success && smooth_success;
    std::ofstream out(output_json);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open metrics json: " + output_json);
    }
    out << "{\n"
        << "  \"seed_stage_ms\": " << seed_stage_ms << ",\n"
        << "  \"smooth_stage_ms\": " << smooth_stage_ms << ",\n"
        << "  \"total_cli_ms\": " << total_cli_ms << ",\n"
        << "  \"raw_points\": " << raw_path.size() << ",\n"
        << "  \"seed_points\": " << seed_path.size() << ",\n"
        << "  \"smoothed_points\": " << smoothed_path.size() << ",\n"
        << "  \"raw_length_m\": " << ComputePathLength(raw_path) << ",\n"
        << "  \"seed_length_m\": " << ComputePathLength(seed_path) << ",\n"
        << "  \"smoothed_length_m\": " << ComputePathLength(smoothed_path) << ",\n"
        << "  \"segment_split_points\": " << split_points_count << ",\n"
        << "  \"seed_qp_init_fallback_raw\": " << (seed_qp_init_fallback_raw ? "true" : "false") << ",\n"
        << "  \"seed_qp_solve_fallback_raw\": " << (seed_qp_solve_fallback_raw ? "true" : "false") << ",\n"
        << "  \"seed_collision_fallback_raw\": " << (seed_collision_fallback_raw ? "true" : "false") << ",\n"
        << "  \"resampled_seed_collision_fallback_raw\": "
        << (resampled_seed_collision_fallback_raw ? "true" : "false") << ",\n"
        << "  \"smooth_collision_fallback_seed\": " << (smooth_collision_fallback_seed ? "true" : "false") << ",\n"
        << "  \"seed_success\": " << (seed_success ? "true" : "false") << ",\n"
        << "  \"smooth_success\": " << (smooth_success ? "true" : "false") << ",\n"
        << "  \"backend_success\": " << (backend_success ? "true" : "false") << "\n"
        << "}\n";
}

VectorVec4d BuildPathWithYaw(const std::vector<Vec2d>& xy, const VectorVec4d& ref_path) {
    VectorVec4d path;
    path.reserve(xy.size());
    for (std::size_t i = 0; i < xy.size(); ++i) {
        Vec4d pose = Vec4d::Zero();
        pose.x() = xy[i].x();
        pose.y() = xy[i].y();
        pose.w() = i < ref_path.size() ? ref_path[i].w() : 1.0;
        path.emplace_back(pose);
    }
    if (path.size() == 1) {
        path.front().z() = ref_path.front().z();
        return path;
    }
    for (std::size_t i = 0; i + 1 < path.size(); ++i) {
        const double dx = path[i + 1].x() - path[i].x();
        const double dy = path[i + 1].y() - path[i].y();
        path[i].z() = std::atan2(dy, dx);
    }
    path.back().z() = path[path.size() - 2].z();
    return path;
}

SeedOptimizationResult OptimizeSeedPathXY(const VectorVec4d& raw_path, double xy_box_half_extent) {
    if (raw_path.size() < 3) {
        return SeedOptimizationResult{raw_path, false, false};
    }

    const int n_total = static_cast<int>(raw_path.size());
    const double box_half_extent = std::max(0.0, xy_box_half_extent);
    const double x_lb_default = -box_half_extent;
    const double x_ub_default = box_half_extent;
    const double y_lb_default = -box_half_extent;
    const double y_ub_default = box_half_extent;
    constexpr double w_smooth = 50000.0;
    constexpr double w_length = 20000.0;
    constexpr double w_ref = 0.1;

    Eigen::VectorXd reference = Eigen::VectorXd::Zero(2 * n_total);
    Eigen::VectorXd lower_bound = Eigen::VectorXd::Zero(2 * n_total);
    Eigen::VectorXd upper_bound = Eigen::VectorXd::Zero(2 * n_total);

    for (int i = 0; i < n_total; ++i) {
        reference(2 * i) = raw_path[i].x();
        reference(2 * i + 1) = raw_path[i].y();
        lower_bound(2 * i) = reference(2 * i) + x_lb_default;
        upper_bound(2 * i) = reference(2 * i) + x_ub_default;
        lower_bound(2 * i + 1) = reference(2 * i + 1) + y_lb_default;
        upper_bound(2 * i + 1) = reference(2 * i + 1) + y_ub_default;
    }

    // Keep endpoint positions fixed while leaving endpoint heading free.
    lower_bound(0) = upper_bound(0) = reference(0);
    lower_bound(1) = upper_bound(1) = reference(1);
    lower_bound(2 * (n_total - 1)) = upper_bound(2 * (n_total - 1)) = reference(2 * (n_total - 1));
    lower_bound(2 * (n_total - 1) + 1) = upper_bound(2 * (n_total - 1) + 1) = reference(2 * (n_total - 1) + 1);

    Eigen::SparseMatrix<double> A1(2 * n_total - 4, 2 * n_total);
    Eigen::SparseMatrix<double> A2(2 * n_total - 2, 2 * n_total);
    Eigen::SparseMatrix<double> A3(2 * n_total, 2 * n_total);
    for (int i = 0; i < 2 * n_total; ++i) {
        A3.insert(i, i) = 1.0;
    }
    for (int i = 0; i < 2 * n_total - 4; ++i) {
        A1.insert(i, i) = 1.0;
        A1.insert(i, i + 2) = -2.0;
        A1.insert(i, i + 4) = 1.0;
    }
    for (int i = 0; i < 2 * n_total - 2; ++i) {
        A2.insert(i, i) = 1.0;
        A2.insert(i, i + 2) = -1.0;
    }
    A1.makeCompressed();
    A2.makeCompressed();
    A3.makeCompressed();

    const Eigen::MatrixXd h_dense =
        2.0 * (w_smooth * (A1.transpose() * A1) + w_length * (A2.transpose() * A2) + w_ref * A3);
    Eigen::VectorXd gradient = -2.0 * w_ref * reference;
    Eigen::SparseMatrix<double> hessian = h_dense.sparseView();
    hessian.makeCompressed();

    Eigen::SparseMatrix<double> constraint_matrix(2 * n_total, 2 * n_total);
    constraint_matrix.setIdentity();
    constraint_matrix.makeCompressed();

    OsqpEigen::Solver solver;
    solver.settings()->setVerbosity(false);
    solver.settings()->setWarmStart(true);
    solver.data()->setNumberOfVariables(2 * n_total);
    solver.data()->setNumberOfConstraints(2 * n_total);
    if (!solver.data()->setHessianMatrix(hessian) || !solver.data()->setGradient(gradient) ||
        !solver.data()->setLinearConstraintsMatrix(constraint_matrix) ||
        !solver.data()->setLowerBound(lower_bound) || !solver.data()->setUpperBound(upper_bound) ||
        !solver.initSolver()) {
        std::cerr << "seed QP init failed, fallback to raw path" << std::endl;
        return SeedOptimizationResult{raw_path, true, false};
    }

    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
        std::cerr << "seed QP solve failed, fallback to raw path" << std::endl;
        return SeedOptimizationResult{raw_path, false, true};
    }

    const Eigen::VectorXd solution = solver.getSolution();
    std::vector<Vec2d> seed_xy;
    seed_xy.reserve(raw_path.size());
    for (int i = 0; i < n_total; ++i) {
        seed_xy.emplace_back(solution(2 * i), solution(2 * i + 1));
    }
    return SeedOptimizationResult{BuildPathWithYaw(seed_xy, raw_path), false, false};
}

VectorVec4d ResamplePathUniform(const VectorVec4d& path, double step) {
    if (path.size() < 2 || step <= 1e-6) {
        return path;
    }

    VectorVec4d resampled;
    resampled.reserve(path.size() * 2);
    resampled.push_back(path.front());

    for (std::size_t i = 0; i + 1 < path.size(); ++i) {
        const double ax = path[i].x();
        const double ay = path[i].y();
        const double bx = path[i + 1].x();
        const double by = path[i + 1].y();
        const double dx = bx - ax;
        const double dy = by - ay;
        const double dist = std::hypot(dx, dy);
        const int segments = std::max(1, static_cast<int>(std::ceil(dist / step)));
        for (int seg = 1; seg <= segments; ++seg) {
            const double ratio = static_cast<double>(seg) / static_cast<double>(segments);
            Vec4d pose = Vec4d::Zero();
            pose.x() = ax + dx * ratio;
            pose.y() = ay + dy * ratio;
            pose.w() = path[i].w();
            resampled.emplace_back(pose);
        }
    }

    if (resampled.size() == 1) {
        resampled.front().z() = path.front().z();
        resampled.front().w() = path.front().w();
        return resampled;
    }
    for (std::size_t i = 0; i + 1 < resampled.size(); ++i) {
        const double dx = resampled[i + 1].x() - resampled[i].x();
        const double dy = resampled[i + 1].y() - resampled[i].y();
        resampled[i].z() = std::atan2(dy, dx);
    }
    resampled.back().z() = resampled[resampled.size() - 2].z();
    resampled.back().w() = path.back().w();
    return resampled;
}

bool PointCollisionFree(
    const std::vector<uint8_t>& occupancy,
    int width,
    int height,
    double occupancy_resolution,
    double origin_x,
    double origin_y,
    double x,
    double y) {
    const int gx = static_cast<int>(std::round((x - origin_x) / occupancy_resolution - 0.5));
    const int gy = static_cast<int>(std::round((y - origin_y) / occupancy_resolution - 0.5));
    if (gx < 0 || gx >= width || gy < 0 || gy >= height) {
        return false;
    }
    return occupancy[gy * width + gx] == 0;
}

bool SegmentCollisionFree(
    const std::vector<uint8_t>& occupancy,
    int width,
    int height,
    double occupancy_resolution,
    double collision_grid_resolution,
    double origin_x,
    double origin_y,
    const Vec2d& a,
    const Vec2d& b) {
    const double dx = b.x() - a.x();
    const double dy = b.y() - a.y();
    const double distance = std::hypot(dx, dy);
    const int steps = std::max(
        2,
        static_cast<int>(std::ceil(distance / std::max(0.03, collision_grid_resolution * 0.5))));
    for (int i = 0; i <= steps; ++i) {
        const double ratio = static_cast<double>(i) / static_cast<double>(steps);
        const double x = a.x() + dx * ratio;
        const double y = a.y() + dy * ratio;
        if (!PointCollisionFree(occupancy, width, height, occupancy_resolution, origin_x, origin_y, x, y)) {
            return false;
        }
    }
    return true;
}

bool PathCollisionFree(
    const std::vector<uint8_t>& occupancy,
    int width,
    int height,
    double occupancy_resolution,
    double collision_grid_resolution,
    double origin_x,
    double origin_y,
    const VectorVec4d& path) {
    if (path.empty()) {
        return false;
    }
    for (const auto& pose : path) {
        if (!PointCollisionFree(
                occupancy, width, height, occupancy_resolution, origin_x, origin_y, pose.x(), pose.y())) {
            return false;
        }
    }
    for (std::size_t i = 0; i + 1 < path.size(); ++i) {
        if (!SegmentCollisionFree(
                occupancy,
                width,
                height,
                occupancy_resolution,
                collision_grid_resolution,
                origin_x,
                origin_y,
                Vec2d(path[i].x(), path[i].y()),
                Vec2d(path[i + 1].x(), path[i + 1].y()))) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto cli_begin = std::chrono::steady_clock::now();
        const Args args = ParseArgs(argc, argv);
        const YAML::Node root = YAML::LoadFile(args.input_yaml);
        const YAML::Node map = root["map"];
        if (!map) {
            throw std::runtime_error("Missing 'map' node in input yaml");
        }

        const int width = map["width"].as<int>();
        const int height = map["height"].as<int>();
        const double resolution = map["resolution"].as<double>();
        const double collision_grid_resolution = ReadDouble(map, "collision_grid_resolution", resolution);
        const double origin_x = map["origin_x"].as<double>();
        const double origin_y = map["origin_y"].as<double>();
        const double state_grid_resolution = ReadDouble(map, "state_grid_resolution", 1.0);
        const double steering_angle = ReadDouble(map, "steering_angle", 10.0);
        const int steering_angle_discrete_num = ReadInt(map, "steering_angle_discrete_num", 1);
        const double wheel_base = ReadDouble(map, "wheel_base", 0.8);
        const double segment_length = ReadDouble(map, "segment_length", 1.6);
        const int segment_length_discrete_num = ReadInt(map, "segment_length_discrete_num", 8);
        const double steering_penalty = ReadDouble(map, "steering_penalty", 1.05);
        const double reversing_penalty = ReadDouble(map, "reversing_penalty", 2.0);
        const double steering_change_penalty = ReadDouble(map, "steering_change_penalty", 1.5);
        const double shot_distance = ReadDouble(map, "shot_distance", 5.0);
        const double seed_resample_step = ReadDouble(map, "seed_resample_step", 0.10);
        const double seed_xy_box_half_extent = ReadDouble(map, "seed_xy_box_half_extent", 0.10);
        const bool skip_seed_collision_check =
            map["skip_seed_collision_check"] ? map["skip_seed_collision_check"].as<bool>() : false;
        const bool simplified_collision_check =
            map["simplified_collision_check"] ? map["simplified_collision_check"].as<bool>() : false;
        const bool fix_endpoint_heading =
            map["fix_endpoint_heading"] ? map["fix_endpoint_heading"].as<bool>() : true;

        const YAML::Node occupancy = map["occupancy"];
        if (!occupancy || !occupancy.IsSequence() || static_cast<int>(occupancy.size()) != height) {
            throw std::runtime_error("map.occupancy must be a sequence with height rows");
        }
        std::vector<uint8_t> occupancy_flat(static_cast<std::size_t>(width * height), 0);
        const int collision_sub_div =
            std::max(1, static_cast<int>(std::ceil(resolution / collision_grid_resolution)));
        const int collision_width = width * collision_sub_div;
        const int collision_height = height * collision_sub_div;
        std::vector<uint8_t> occupancy_collision_flat(
            static_cast<std::size_t>(collision_width * collision_height), 0);

        HybridAStar smoother(
            steering_angle,
            steering_angle_discrete_num,
            segment_length,
            segment_length_discrete_num,
            wheel_base,
            steering_penalty,
            reversing_penalty,
            steering_change_penalty,
            shot_distance);
        smoother.SetSimplifiedCollisionCheck(simplified_collision_check);
        smoother.SetFixEndpointHeading(fix_endpoint_heading);

        const double map_max_x = origin_x + static_cast<double>(width) * resolution;
        const double map_max_y = origin_y + static_cast<double>(height) * resolution;
        smoother.Init(origin_x, map_max_x, origin_y, map_max_y, state_grid_resolution, collision_grid_resolution);

        for (int y = 0; y < height; ++y) {
            const YAML::Node row = occupancy[y];
            if (!row.IsSequence() || static_cast<int>(row.size()) != width) {
                throw std::runtime_error("Each occupancy row must have width entries");
            }
            for (int x = 0; x < width; ++x) {
                const int occ_value = row[x].as<int>();
                occupancy_flat[static_cast<std::size_t>(y * width + x)] = static_cast<uint8_t>(occ_value != 0);
                if (occ_value != 0) {
                    const double sub_step = resolution / static_cast<double>(collision_sub_div);
                    for (int sy = 0; sy < collision_sub_div; ++sy) {
                        for (int sx = 0; sx < collision_sub_div; ++sx) {
                            occupancy_collision_flat[static_cast<std::size_t>(
                                (y * collision_sub_div + sy) * collision_width + (x * collision_sub_div + sx))] = 1u;
                            const double wx = origin_x + static_cast<double>(x) * resolution +
                                              (static_cast<double>(sx) + 0.5) * sub_step;
                            const double wy = origin_y + static_cast<double>(y) * resolution +
                                              (static_cast<double>(sy) + 0.5) * sub_step;
                            smoother.SetObstacle(wx, wy);
                        }
                    }
                }
            }
        }

        VectorVec4d raw_path = LoadRawPath(root);
        const auto seed_stage_begin = std::chrono::steady_clock::now();
        const auto seed_result = OptimizeSeedPathXY(raw_path, seed_xy_box_half_extent);
        VectorVec4d seed_path = seed_result.path;
        bool seed_qp_init_fallback_raw = seed_result.qp_init_failed;
        bool seed_qp_solve_fallback_raw = seed_result.qp_solve_failed;
        bool seed_collision_fallback_raw = false;
        bool resampled_seed_collision_fallback_raw = false;
        bool smooth_collision_fallback_seed = false;
        if (!skip_seed_collision_check) {
            if (!PathCollisionFree(
                    occupancy_collision_flat,
                    collision_width,
                    collision_height,
                    collision_grid_resolution,
                    collision_grid_resolution,
                    origin_x,
                    origin_y,
                    seed_path)) {
                std::cerr << "seed path collides with occupancy, fallback to raw path" << std::endl;
                seed_collision_fallback_raw = true;
                seed_path = raw_path;
            }
        }
        seed_path = ResamplePathUniform(seed_path, seed_resample_step);
        const auto split_points = smoother.GetSmoothSegmentSplitPoints(seed_path);
        if (!args.split_points_csv.empty()) {
            WriteSplitPointsCsv(args.split_points_csv, split_points);
        }
        if (!skip_seed_collision_check) {
            if (!PathCollisionFree(
                    occupancy_collision_flat,
                    collision_width,
                    collision_height,
                    collision_grid_resolution,
                    collision_grid_resolution,
                    origin_x,
                    origin_y,
                    seed_path)) {
                std::cerr << "resampled seed path collides with occupancy, fallback to raw path" << std::endl;
                resampled_seed_collision_fallback_raw = true;
                seed_path = raw_path;
            }
        }
        if (!args.seed_csv.empty()) {
            WritePathCsv(args.seed_csv, seed_path);
        }
        const auto seed_stage_end = std::chrono::steady_clock::now();
        const auto smooth_stage_begin = std::chrono::steady_clock::now();
        VectorVec4d smoothed_path = smoother.SmoothPath(seed_path);
        if (smoothed_path.empty()) {
            throw std::runtime_error("Smoother returned empty path");
        }
        if (!PathCollisionFree(
                occupancy_collision_flat,
                collision_width,
                collision_height,
                collision_grid_resolution,
                collision_grid_resolution,
                origin_x,
                origin_y,
                smoothed_path)) {
            std::cerr << "smoothed path collides with occupancy, fallback to seed path" << std::endl;
            smooth_collision_fallback_seed = true;
            smoothed_path = seed_path;
        }
        const auto smooth_stage_end = std::chrono::steady_clock::now();
        WritePathCsv(args.output_csv, smoothed_path);
        const double seed_stage_ms =
            std::chrono::duration<double, std::milli>(seed_stage_end - seed_stage_begin).count();
        const double smooth_stage_ms =
            std::chrono::duration<double, std::milli>(smooth_stage_end - smooth_stage_begin).count();
        const double total_cli_ms =
            std::chrono::duration<double, std::milli>(smooth_stage_end - cli_begin).count();
        if (!args.metrics_json.empty()) {
            WriteMetricsJson(
                args.metrics_json,
                seed_stage_ms,
                smooth_stage_ms,
                total_cli_ms,
                raw_path,
                seed_path,
                smoothed_path,
                split_points.size(),
                seed_qp_init_fallback_raw,
                seed_qp_solve_fallback_raw,
                seed_collision_fallback_raw,
                resampled_seed_collision_fallback_raw,
                smooth_collision_fallback_seed);
        }
        if (!args.seed_csv.empty()) {
            std::cout << "saved_seed_csv=" << args.seed_csv << std::endl;
        }
        if (!args.split_points_csv.empty()) {
            std::cout << "saved_split_points_csv=" << args.split_points_csv << std::endl;
        }
        std::cout << "saved_smoothed_csv=" << args.output_csv << std::endl;
        std::cout << "raw_points=" << raw_path.size() << std::endl;
        std::cout << "seed_points=" << seed_path.size() << std::endl;
        std::cout << "smoothed_points=" << smoothed_path.size() << std::endl;
        std::cout << "seed_stage_ms=" << seed_stage_ms << std::endl;
        std::cout << "smooth_stage_ms=" << smooth_stage_ms << std::endl;
        std::cout << "total_cli_ms=" << total_cli_ms << std::endl;
        if (!args.metrics_json.empty()) {
            std::cout << "saved_metrics_json=" << args.metrics_json << std::endl;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "smooth_path_cli error: " << e.what() << std::endl;
        return 1;
    }
}
