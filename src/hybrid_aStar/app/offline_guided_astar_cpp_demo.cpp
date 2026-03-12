#include "hybrid_a_star/guided_frontend_onnx.h"

#include <yaml-cpp/yaml.h>
#include <zip.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using guided_frontend::GridAstarOptions;
using guided_frontend::GridAstarResult;
using guided_frontend::GuidanceCostMapOnnx;
using guided_frontend::RunGuidedGridAstar;

struct Args {
    std::string dataset_path;
    std::string onnx_path;
    std::string smoother_cli_path;
    std::string python_exec{"python3"};
    std::string plot_script_path;
    std::string split{"train"};
    int map_index{-1};
    int seed{123};
    double resolution{0.25};
    double origin_x{0.0};
    double origin_y{0.0};
    double lambda_guidance{1.0};
    std::string heuristic_mode{"octile"};
    double heuristic_weight{1.0};
    std::string guidance_integration_mode{"g_cost"};
    double guidance_bonus_threshold{0.5};
    bool allow_corner_cut{false};
    bool skip_plot{false};
    double min_start_goal_dist{22.0};
    std::string output_dir;
    std::string case_name{"street_demo_random"};
    int onnx_intra_threads{1};
    int onnx_inter_threads{1};
};

struct StreetMaps {
    std::vector<float> data;
    int map_count{0};
    int height{0};
    int width{0};
};

struct Problem {
    Vec2i start_xy{0, 0};
    Vec2i goal_xy{0, 0};
};

std::string DefaultDatasetPath() {
    return std::string(HYBRID_ASTAR_SOURCE_DIR) +
           "/model_base_astar/neural-astar/planning-datasets/data/street/mixed_064_moore_c16.npz";
}

std::string DefaultOnnxPath() {
    return std::string(HYBRID_ASTAR_SOURCE_DIR) +
           "/model_base_astar/neural-astar/outputs/model_guidance_street/best_cost_map.onnx";
}

std::string DefaultOutputDir() {
    return std::string(HYBRID_ASTAR_SOURCE_DIR) + "/offline_results/street_guided_demo";
}

std::string DefaultPlotScriptPath() {
    return std::string(HYBRID_ASTAR_SOURCE_DIR) + "/scripts/offline_street_guided_astar_demo.py";
}

std::string DefaultSmootherCliPath(char** argv) {
    std::string exe_path(argv[0]);
    const std::size_t slash = exe_path.find_last_of('/');
    if (slash == std::string::npos) {
        return "smooth_path_cli";
    }
    return exe_path.substr(0, slash + 1) + "smooth_path_cli";
}

void EnsureFileExists(const std::string& path, const std::string& label) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        throw std::runtime_error(label + " not found: " + path);
    }
}

Args ParseArgs(int argc, char** argv) {
    Args args;
    args.dataset_path = DefaultDatasetPath();
    args.onnx_path = DefaultOnnxPath();
    args.output_dir = DefaultOutputDir();
    args.plot_script_path = DefaultPlotScriptPath();
    args.smoother_cli_path = DefaultSmootherCliPath(argv);

    for (int i = 1; i < argc; ++i) {
        const std::string token(argv[i]);
        if (token == "--dataset" && i + 1 < argc) {
            args.dataset_path = argv[++i];
        } else if (token == "--onnx" && i + 1 < argc) {
            args.onnx_path = argv[++i];
        } else if (token == "--smoother-cli" && i + 1 < argc) {
            args.smoother_cli_path = argv[++i];
        } else if (token == "--python-exec" && i + 1 < argc) {
            args.python_exec = argv[++i];
        } else if (token == "--plot-script" && i + 1 < argc) {
            args.plot_script_path = argv[++i];
        } else if (token == "--split" && i + 1 < argc) {
            args.split = argv[++i];
        } else if (token == "--map-index" && i + 1 < argc) {
            args.map_index = std::stoi(argv[++i]);
        } else if (token == "--seed" && i + 1 < argc) {
            args.seed = std::stoi(argv[++i]);
        } else if (token == "--resolution" && i + 1 < argc) {
            args.resolution = std::stod(argv[++i]);
        } else if (token == "--origin-x" && i + 1 < argc) {
            args.origin_x = std::stod(argv[++i]);
        } else if (token == "--origin-y" && i + 1 < argc) {
            args.origin_y = std::stod(argv[++i]);
        } else if (token == "--lambda-guidance" && i + 1 < argc) {
            args.lambda_guidance = std::stod(argv[++i]);
        } else if (token == "--heuristic-mode" && i + 1 < argc) {
            args.heuristic_mode = argv[++i];
        } else if (token == "--heuristic-weight" && i + 1 < argc) {
            args.heuristic_weight = std::stod(argv[++i]);
        } else if (token == "--guidance-integration-mode" && i + 1 < argc) {
            args.guidance_integration_mode = argv[++i];
        } else if (token == "--guidance-bonus-threshold" && i + 1 < argc) {
            args.guidance_bonus_threshold = std::stod(argv[++i]);
        } else if (token == "--allow-corner-cut") {
            args.allow_corner_cut = true;
        } else if (token == "--skip-plot") {
            args.skip_plot = true;
        } else if (token == "--min-start-goal-dist" && i + 1 < argc) {
            args.min_start_goal_dist = std::stod(argv[++i]);
        } else if (token == "--output-dir" && i + 1 < argc) {
            args.output_dir = argv[++i];
        } else if (token == "--case-name" && i + 1 < argc) {
            args.case_name = argv[++i];
        } else if (token == "--onnx-intra-threads" && i + 1 < argc) {
            args.onnx_intra_threads = std::stoi(argv[++i]);
        } else if (token == "--onnx-inter-threads" && i + 1 < argc) {
            args.onnx_inter_threads = std::stoi(argv[++i]);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + token);
        }
    }

    if (args.split != "train" && args.split != "valid" && args.split != "test") {
        throw std::runtime_error("--split must be one of train|valid|test");
    }

    EnsureFileExists(args.dataset_path, "dataset");
    EnsureFileExists(args.onnx_path, "onnx model");
    EnsureFileExists(args.smoother_cli_path, "smooth_path_cli");
    if (!args.skip_plot) {
        EnsureFileExists(args.plot_script_path, "plot script");
    }

    return args;
}

std::string SplitArrayName(const std::string& split) {
    if (split == "train") {
        return "arr_0.npy";
    }
    if (split == "valid") {
        return "arr_4.npy";
    }
    return "arr_8.npy";
}

std::vector<char> ReadZipEntry(zip_t* archive, const std::string& entry_name) {
    zip_stat_t stat;
    if (zip_stat(archive, entry_name.c_str(), ZIP_FL_ENC_GUESS, &stat) != 0) {
        throw std::runtime_error("zip entry not found: " + entry_name);
    }
    zip_file_t* file = zip_fopen(archive, entry_name.c_str(), ZIP_FL_ENC_GUESS);
    if (file == nullptr) {
        throw std::runtime_error("failed to open zip entry: " + entry_name);
    }
    std::vector<char> buffer(static_cast<std::size_t>(stat.size));
    zip_int64_t total = 0;
    while (total < stat.size) {
        const zip_int64_t read_size = zip_fread(file, buffer.data() + total, stat.size - total);
        if (read_size < 0) {
            zip_fclose(file);
            throw std::runtime_error("failed reading zip entry: " + entry_name);
        }
        if (read_size == 0) {
            break;
        }
        total += read_size;
    }
    zip_fclose(file);
    if (total != stat.size) {
        throw std::runtime_error("short read for zip entry: " + entry_name);
    }
    return buffer;
}

uint16_t ReadLe16(const unsigned char* ptr) {
    return static_cast<uint16_t>(ptr[0]) | (static_cast<uint16_t>(ptr[1]) << 8);
}

uint32_t ReadLe32(const unsigned char* ptr) {
    return static_cast<uint32_t>(ptr[0]) | (static_cast<uint32_t>(ptr[1]) << 8) |
           (static_cast<uint32_t>(ptr[2]) << 16) | (static_cast<uint32_t>(ptr[3]) << 24);
}

StreetMaps ParseNpyFloat32Maps(const std::vector<char>& npy_bytes) {
    const unsigned char* raw = reinterpret_cast<const unsigned char*>(npy_bytes.data());
    const std::size_t size = npy_bytes.size();
    if (size < 16 || std::memcmp(raw, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("invalid npy header");
    }
    const uint8_t major = raw[6];
    std::size_t header_len = 0;
    std::size_t data_offset = 0;
    if (major == 1) {
        header_len = ReadLe16(raw + 8);
        data_offset = 10;
    } else if (major == 2) {
        header_len = ReadLe32(raw + 8);
        data_offset = 12;
    } else {
        throw std::runtime_error("unsupported npy version");
    }
    if (data_offset + header_len > size) {
        throw std::runtime_error("corrupted npy header length");
    }

    const std::string header(reinterpret_cast<const char*>(raw + data_offset), header_len);
    if (header.find("'descr': '<f4'") == std::string::npos &&
        header.find("\"descr\": \"<f4\"") == std::string::npos) {
        throw std::runtime_error("expected float32 npy tensor");
    }
    if (header.find("False") == std::string::npos) {
        throw std::runtime_error("fortran-order npy is unsupported");
    }

    const std::size_t shape_pos = header.find("shape");
    const std::size_t lparen = header.find('(', shape_pos);
    const std::size_t rparen = header.find(')', lparen);
    if (shape_pos == std::string::npos || lparen == std::string::npos || rparen == std::string::npos) {
        throw std::runtime_error("failed to parse npy shape");
    }
    std::string shape_text = header.substr(lparen + 1, rparen - lparen - 1);
    std::replace(shape_text.begin(), shape_text.end(), ',', ' ');
    std::stringstream ss(shape_text);
    StreetMaps maps;
    if (!(ss >> maps.map_count >> maps.height >> maps.width)) {
        throw std::runtime_error("expected shape [N,H,W] in npy");
    }
    const std::size_t payload_offset = data_offset + header_len;
    const std::size_t expected_values =
        static_cast<std::size_t>(maps.map_count) * static_cast<std::size_t>(maps.height) * static_cast<std::size_t>(maps.width);
    const std::size_t expected_bytes = expected_values * sizeof(float);
    if (payload_offset + expected_bytes > size) {
        throw std::runtime_error("npy payload truncated");
    }
    maps.data.resize(expected_values);
    std::memcpy(maps.data.data(), raw + payload_offset, expected_bytes);
    return maps;
}

StreetMaps LoadStreetMaps(const std::string& dataset_path, const std::string& split) {
    int err = 0;
    zip_t* archive = zip_open(dataset_path.c_str(), ZIP_RDONLY, &err);
    if (archive == nullptr) {
        throw std::runtime_error("failed to open npz dataset: " + dataset_path);
    }
    try {
        const std::vector<char> bytes = ReadZipEntry(archive, SplitArrayName(split));
        zip_close(archive);
        return ParseNpyFloat32Maps(bytes);
    } catch (...) {
        zip_close(archive);
        throw;
    }
}

inline std::size_t MapFlatIndex(int x, int y, int width) {
    return static_cast<std::size_t>(y * width + x);
}

Vec2d GridToWorld(int gx, int gy, double origin_x, double origin_y, double resolution) {
    return Vec2d(origin_x + (static_cast<double>(gx) + 0.5) * resolution,
                 origin_y + (static_cast<double>(gy) + 0.5) * resolution);
}

bool PointCollisionFreeOnOcc(const std::vector<int>& occupancy,
                             int width,
                             int height,
                             double origin_x,
                             double origin_y,
                             double resolution,
                             double x,
                             double y) {
    const double gx_f = (x - origin_x) / resolution - 0.5;
    const double gy_f = (y - origin_y) / resolution - 0.5;
    const int gx = static_cast<int>(std::round(gx_f));
    const int gy = static_cast<int>(std::round(gy_f));
    if (gx < 0 || gx >= width || gy < 0 || gy >= height) {
        return false;
    }
    return occupancy[MapFlatIndex(gx, gy, width)] == 0;
}

bool Footprint16CollisionFreeOnOcc(const std::vector<int>& occupancy,
                                   int width,
                                   int height,
                                   double origin_x,
                                   double origin_y,
                                   double map_resolution,
                                   double collision_grid_resolution,
                                   const Vec2d& world_xy) {
    static constexpr std::array<double, 4> kOffsets{{-1.5, -0.5, 0.5, 1.5}};
    for (double dx : kOffsets) {
        for (double dy : kOffsets) {
            if (!PointCollisionFreeOnOcc(
                    occupancy,
                    width,
                    height,
                    origin_x,
                    origin_y,
                    map_resolution,
                    world_xy.x() + dx * collision_grid_resolution,
                    world_xy.y() + dy * collision_grid_resolution)) {
                return false;
            }
        }
    }
    return true;
}

Problem SampleProblem(const std::vector<int>& occupancy,
                      int width,
                      int height,
                      const Args& args,
                      std::mt19937& rng) {
    std::uniform_int_distribution<int> x_dis(0, width - 1);
    std::uniform_int_distribution<int> y_dis(0, height - 1);
    auto sample_free = [&]() -> Vec2i {
        for (int iter = 0; iter < 10000; ++iter) {
            const Vec2i xy(x_dis(rng), y_dis(rng));
            if (occupancy[MapFlatIndex(xy.x(), xy.y(), width)] == 0) {
                return xy;
            }
        }
        throw std::runtime_error("failed to sample a free cell");
    };

    GridAstarOptions options;
    options.lambda_guidance = 0.0;
    options.allow_corner_cut = args.allow_corner_cut;
    options.heuristic_mode = args.heuristic_mode;
    options.heuristic_weight = args.heuristic_weight;
    options.guidance_integration_mode = "g_cost";
    options.guidance_bonus_threshold = args.guidance_bonus_threshold;
    const std::vector<float> zero_guidance(static_cast<std::size_t>(width * height), 0.0f);
    const double collision_grid_resolution = 0.125;

    for (int iter = 0; iter < 5000; ++iter) {
        const Vec2i start_xy = sample_free();
        const Vec2i goal_xy = sample_free();
        if (start_xy == goal_xy) {
            continue;
        }
        const double dx = static_cast<double>(goal_xy.x() - start_xy.x());
        const double dy = static_cast<double>(goal_xy.y() - start_xy.y());
        if (std::hypot(dx, dy) < args.min_start_goal_dist) {
            continue;
        }
        const Vec2d start_world = GridToWorld(start_xy.x(), start_xy.y(), args.origin_x, args.origin_y, args.resolution);
        const Vec2d goal_world = GridToWorld(goal_xy.x(), goal_xy.y(), args.origin_x, args.origin_y, args.resolution);
        if (!Footprint16CollisionFreeOnOcc(
                occupancy,
                width,
                height,
                args.origin_x,
                args.origin_y,
                args.resolution,
                collision_grid_resolution,
                start_world)) {
            continue;
        }
        if (!Footprint16CollisionFreeOnOcc(
                occupancy,
                width,
                height,
                args.origin_x,
                args.origin_y,
                args.resolution,
                collision_grid_resolution,
                goal_world)) {
            continue;
        }
        const GridAstarResult solvability =
            RunGuidedGridAstar(occupancy, width, height, start_xy, goal_xy, zero_guidance, options);
        if (!solvability.success || solvability.path_xy.size() < 2) {
            continue;
        }
        return Problem{start_xy, goal_xy};
    }
    throw std::runtime_error("failed to sample a solvable start/goal pair");
}

void EnsureDir(const std::string& path) {
    const std::string cmd = "mkdir -p \"" + path + "\"";
    if (std::system(cmd.c_str()) != 0) {
        throw std::runtime_error("failed to create directory: " + path);
    }
}

void WriteRawPathCsv(const std::string& path, const VectorVec4d& world_path) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open raw path csv: " + path);
    }
    out << "x,y\n";
    for (const auto& pose : world_path) {
        out << pose.x() << "," << pose.y() << "\n";
    }
}

void WriteMatrixCsv(const std::string& path,
                    const std::vector<float>& data,
                    int width,
                    int height) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open csv: " + path);
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (x > 0) {
                out << ",";
            }
            out << data[MapFlatIndex(x, y, width)];
        }
        out << "\n";
    }
}

void WriteMatrixCsvInt(const std::string& path,
                       const std::vector<int>& data,
                       int width,
                       int height) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open csv: " + path);
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (x > 0) {
                out << ",";
            }
            out << data[MapFlatIndex(x, y, width)];
        }
        out << "\n";
    }
}

void WriteSmootherYaml(const std::string& path,
                       const std::vector<int>& occupancy,
                       int width,
                       int height,
                       const VectorVec4d& raw_world_path,
                       const Args& args) {
    YAML::Node root;
    YAML::Node map;
    map["width"] = width;
    map["height"] = height;
    map["resolution"] = args.resolution;
    map["collision_grid_resolution"] = 0.125;
    map["origin_x"] = args.origin_x;
    map["origin_y"] = args.origin_y;
    map["state_grid_resolution"] = 1.0;
    map["steering_angle"] = 10.0;
    map["steering_angle_discrete_num"] = 1;
    map["wheel_base"] = 0.8;
    map["segment_length"] = 1.6;
    map["segment_length_discrete_num"] = 8;
    map["steering_penalty"] = 1.05;
    map["reversing_penalty"] = 2.0;
    map["steering_change_penalty"] = 1.5;
    map["shot_distance"] = 5.0;
    map["seed_resample_step"] = 0.10;
    map["simplified_collision_check"] = true;
    map["fix_endpoint_heading"] = false;
    YAML::Node occ_node(YAML::NodeType::Sequence);
    for (int y = 0; y < height; ++y) {
        YAML::Node row(YAML::NodeType::Sequence);
        for (int x = 0; x < width; ++x) {
            row.push_back(occupancy[MapFlatIndex(x, y, width)]);
        }
        occ_node.push_back(row);
    }
    map["occupancy"] = occ_node;
    root["map"] = map;
    YAML::Node raw_path(YAML::NodeType::Sequence);
    for (const auto& pose : raw_world_path) {
        YAML::Node pt(YAML::NodeType::Sequence);
        pt.push_back(pose.x());
        pt.push_back(pose.y());
        raw_path.push_back(pt);
    }
    root["raw_path"] = raw_path;
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open smoother yaml: " + path);
    }
    out << root;
}

std::string JsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '\\' || c == '"') {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    return out;
}

void WriteMetaJson(const std::string& path,
                   const Args& args,
                   int map_index,
                   int width,
                   int height,
                   const Problem& problem,
                   std::size_t raw_points) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open meta json: " + path);
    }
    out << "{\n"
        << "  \"dataset\": \"" << JsonEscape(args.dataset_path) << "\",\n"
        << "  \"split\": \"" << args.split << "\",\n"
        << "  \"map_index\": " << map_index << ",\n"
        << "  \"seed\": " << args.seed << ",\n"
        << "  \"width\": " << width << ",\n"
        << "  \"height\": " << height << ",\n"
        << "  \"start_xy\": [" << problem.start_xy.x() << ", " << problem.start_xy.y() << "],\n"
        << "  \"goal_xy\": [" << problem.goal_xy.x() << ", " << problem.goal_xy.y() << "],\n"
        << "  \"lambda_guidance\": " << args.lambda_guidance << ",\n"
        << "  \"heuristic_mode\": \"" << args.heuristic_mode << "\",\n"
        << "  \"heuristic_weight\": " << args.heuristic_weight << ",\n"
        << "  \"guidance_integration_mode\": \"" << args.guidance_integration_mode << "\",\n"
        << "  \"guidance_bonus_threshold\": " << args.guidance_bonus_threshold << ",\n"
        << "  \"resolution\": " << args.resolution << ",\n"
        << "  \"collision_grid_resolution\": 0.125,\n"
        << "  \"origin_x\": " << args.origin_x << ",\n"
        << "  \"origin_y\": " << args.origin_y << ",\n"
        << "  \"raw_path_points\": " << raw_points << "\n"
        << "}\n";
}

std::string Quote(const std::string& s) {
    return "\"" + s + "\"";
}

int RunCommand(const std::string& cmd) {
    std::cerr << "[offline_cpp_demo] " << cmd << std::endl;
    return std::system(cmd.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = ParseArgs(argc, argv);
        std::mt19937 rng(args.seed);

        const StreetMaps maps = LoadStreetMaps(args.dataset_path, args.split);
        int map_index = args.map_index;
        if (map_index < 0) {
            std::uniform_int_distribution<int> map_dis(0, maps.map_count - 1);
            map_index = map_dis(rng);
        }
        if (map_index < 0 || map_index >= maps.map_count) {
            throw std::runtime_error("map_index out of range");
        }

        std::vector<int> occupancy(static_cast<std::size_t>(maps.width * maps.height), 0);
        const std::size_t offset = static_cast<std::size_t>(map_index) * static_cast<std::size_t>(maps.width * maps.height);
        for (int y = 0; y < maps.height; ++y) {
            for (int x = 0; x < maps.width; ++x) {
                const float free_value = maps.data[offset + MapFlatIndex(x, y, maps.width)];
                occupancy[MapFlatIndex(x, y, maps.width)] = free_value > 0.5f ? 0 : 1;
            }
        }

        const Problem problem = SampleProblem(occupancy, maps.width, maps.height, args, rng);

        GuidanceCostMapOnnx guidance_model(args.onnx_path, args.onnx_intra_threads, args.onnx_inter_threads);
        const std::vector<float> guidance_cost =
            guidance_model.Infer(occupancy, maps.width, maps.height, problem.start_xy, problem.goal_xy, 0.0f, 0.0f);

        GridAstarOptions options;
        options.lambda_guidance = args.lambda_guidance;
        options.heuristic_mode = args.heuristic_mode;
        options.heuristic_weight = args.heuristic_weight;
        options.guidance_integration_mode = args.guidance_integration_mode;
        options.guidance_bonus_threshold = args.guidance_bonus_threshold;
        options.allow_corner_cut = args.allow_corner_cut;
        const GridAstarResult search_result =
            RunGuidedGridAstar(occupancy, maps.width, maps.height, problem.start_xy, problem.goal_xy, guidance_cost, options);
        if (!search_result.success || search_result.path_xy.size() < 2) {
            throw std::runtime_error("guided grid A* failed");
        }

        VectorVec4d raw_world_path;
        raw_world_path.reserve(search_result.path_xy.size());
        for (const auto& xy : search_result.path_xy) {
            const Vec2d world_xy = GridToWorld(xy.x(), xy.y(), args.origin_x, args.origin_y, args.resolution);
            Vec4d pose = Vec4d::Zero();
            pose.x() = world_xy.x();
            pose.y() = world_xy.y();
            pose.w() = 1.0;
            raw_world_path.emplace_back(pose);
        }

        const std::string case_dir = args.output_dir + "/" + args.case_name;
        EnsureDir(case_dir);
        const std::string raw_csv = case_dir + "/frontend_raw_path.csv";
        const std::string seed_csv = case_dir + "/frontend_seed_path.csv";
        const std::string split_csv = case_dir + "/segment_split_points.csv";
        const std::string smooth_csv = case_dir + "/smoothed_path.csv";
        const std::string smoother_yaml = case_dir + "/smoother_request.yaml";
        const std::string occupancy_csv = case_dir + "/occupancy.csv";
        const std::string guidance_csv = case_dir + "/guidance_cost.csv";
        const std::string meta_json = case_dir + "/meta.json";

        WriteRawPathCsv(raw_csv, raw_world_path);
        WriteMatrixCsvInt(occupancy_csv, occupancy, maps.width, maps.height);
        WriteMatrixCsv(guidance_csv, guidance_cost, maps.width, maps.height);
        WriteSmootherYaml(smoother_yaml, occupancy, maps.width, maps.height, raw_world_path, args);
        WriteMetaJson(meta_json, args, map_index, maps.width, maps.height, problem, raw_world_path.size());

        {
            std::stringstream cmd;
            cmd << Quote(args.smoother_cli_path)
                << " --input-yaml " << Quote(smoother_yaml)
                << " --seed-csv " << Quote(seed_csv)
                << " --split-points-csv " << Quote(split_csv)
                << " --output-csv " << Quote(smooth_csv);
            if (RunCommand(cmd.str()) != 0) {
                throw std::runtime_error("smooth_path_cli failed");
            }
        }

        if (!args.skip_plot) {
            std::stringstream cmd;
            cmd << Quote(args.python_exec)
                << " " << Quote(args.plot_script_path)
                << " --plot-only"
                << " --input-dir " << Quote(case_dir);
            if (RunCommand(cmd.str()) != 0) {
                throw std::runtime_error("plot script failed");
            }
        }

        std::cout << "saved_case_dir=" << case_dir << std::endl;
        std::cout << "saved_raw_csv=" << raw_csv << std::endl;
        std::cout << "saved_seed_csv=" << seed_csv << std::endl;
        std::cout << "saved_smoothed_csv=" << smooth_csv << std::endl;
        std::cout << "saved_occupancy_csv=" << occupancy_csv << std::endl;
        std::cout << "saved_guidance_csv=" << guidance_csv << std::endl;
        std::cout << "saved_meta=" << meta_json << std::endl;
        std::cout << "map_index=" << map_index << std::endl;
        std::cout << "start_xy=(" << problem.start_xy.x() << "," << problem.start_xy.y() << ")" << std::endl;
        std::cout << "goal_xy=(" << problem.goal_xy.x() << "," << problem.goal_xy.y() << ")" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "offline_guided_astar_cpp_demo error: " << e.what() << std::endl;
        return 1;
    }
}
