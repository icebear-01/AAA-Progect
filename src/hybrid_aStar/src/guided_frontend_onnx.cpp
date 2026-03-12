#include "hybrid_a_star/guided_frontend_onnx.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace guided_frontend {
namespace {

struct QueueNode {
    double f_cost{0.0};
    Vec2i xy{0, 0};

    bool operator>(const QueueNode& other) const { return f_cost > other.f_cost; }
};

inline bool IsInside(int x, int y, int width, int height) {
    return x >= 0 && x < width && y >= 0 && y < height;
}

inline int FlatIndex(int x, int y, int width) {
    return y * width + x;
}

int ScaleIndex(int src_index, int src_extent, int dst_extent) {
    if (src_extent <= 1 || dst_extent <= 1) {
        return 0;
    }
    const double src_center = (static_cast<double>(src_index) + 0.5) / static_cast<double>(src_extent);
    const int dst_index = static_cast<int>(std::floor(src_center * static_cast<double>(dst_extent)));
    return std::min(std::max(dst_index, 0), dst_extent - 1);
}

double HeuristicCost(int x,
                     int y,
                     int gx,
                     int gy,
                     double diagonal_cost,
                     const std::string& mode) {
    const int dx = std::abs(gx - x);
    const int dy = std::abs(gy - y);
    if (mode == "euclidean") {
        return std::hypot(static_cast<double>(dx), static_cast<double>(dy));
    }
    if (mode == "manhattan") {
        return static_cast<double>(dx + dy);
    }
    if (mode == "chebyshev") {
        return static_cast<double>(std::max(dx, dy));
    }
    if (mode == "octile") {
        const double d_min = static_cast<double>(std::min(dx, dy));
        const double d_max = static_cast<double>(std::max(dx, dy));
        return d_max + (diagonal_cost - 1.0) * d_min;
    }
    throw std::runtime_error("Unknown heuristic_mode: " + mode);
}

double GuidancePriorityBias(double guidance_value,
                            const std::string& integration_mode,
                            double bonus_threshold) {
    if (integration_mode == "heuristic_bias") {
        return guidance_value;
    }
    if (integration_mode == "heuristic_bonus") {
        const double scaled_bonus = (guidance_value - bonus_threshold) / std::max(bonus_threshold, 1e-6);
        return std::min(0.0, scaled_bonus);
    }
    if (integration_mode == "g_cost") {
        return 0.0;
    }
    throw std::runtime_error("Unknown guidance_integration_mode: " + integration_mode);
}

std::vector<Vec2i> ReconstructPath(
    const std::vector<int>& parent,
    int width,
    Vec2i current) {
    std::vector<Vec2i> path;
    path.push_back(current);
    while (true) {
        const int current_index = FlatIndex(current.x(), current.y(), width);
        const int parent_index = parent[static_cast<std::size_t>(current_index)];
        if (parent_index < 0) {
            break;
        }
        current = Vec2i(parent_index % width, parent_index / width);
        path.push_back(current);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

}  // namespace

GuidanceCostMapOnnx::GuidanceCostMapOnnx(const std::string& model_path,
                                         int intra_threads,
                                         int inter_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "guided_frontend_onnx"),
      session_([&]() {
          Ort::SessionOptions options;
          options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
          options.SetIntraOpNumThreads(std::max(1, intra_threads));
          options.SetInterOpNumThreads(std::max(1, inter_threads));
          return Ort::Session(env_, model_path.c_str(), options);
      }()) {
    const std::size_t input_count = session_.GetInputCount();
    input_names_storage_.reserve(input_count);
    input_names_.reserve(input_count);
    for (std::size_t i = 0; i < input_count; ++i) {
        Ort::AllocatedStringPtr name = session_.GetInputNameAllocated(i, allocator_);
        input_names_storage_.emplace_back(name.get());
        input_names_.push_back(input_names_storage_.back().c_str());
    }

    const std::size_t output_count = session_.GetOutputCount();
    output_names_storage_.reserve(output_count);
    output_names_.reserve(output_count);
    for (std::size_t i = 0; i < output_count; ++i) {
        Ort::AllocatedStringPtr name = session_.GetOutputNameAllocated(i, allocator_);
        output_names_storage_.emplace_back(name.get());
        output_names_.push_back(output_names_storage_.back().c_str());
    }

    auto input_shape =
        session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (input_shape.size() == 4 && input_shape[2] > 0 && input_shape[3] > 0) {
        model_height_ = static_cast<int>(input_shape[2]);
        model_width_ = static_cast<int>(input_shape[3]);
    }
}

std::vector<float> GuidanceCostMapOnnx::MakeOneHotMap(int width, int height, const Vec2i& xy) const {
    std::vector<float> one_hot(static_cast<std::size_t>(width * height), 0.0f);
    if (IsInside(xy.x(), xy.y(), width, height)) {
        one_hot[static_cast<std::size_t>(FlatIndex(xy.x(), xy.y(), width))] = 1.0f;
    }
    return one_hot;
}

Vec2i GuidanceCostMapOnnx::ScaleGridCoord(const Vec2i& xy,
                                          int src_width,
                                          int src_height,
                                          int dst_width,
                                          int dst_height) const {
    return Vec2i(
        ScaleIndex(xy.x(), src_width, dst_width),
        ScaleIndex(xy.y(), src_height, dst_height));
}

std::vector<float> GuidanceCostMapOnnx::ResizeBinaryMap(const std::vector<int>& src,
                                                        int src_width,
                                                        int src_height,
                                                        int dst_width,
                                                        int dst_height) const {
    std::vector<float> dst(static_cast<std::size_t>(dst_width * dst_height), 0.0f);
    for (int y = 0; y < dst_height; ++y) {
        const int src_y = ScaleIndex(y, dst_height, src_height);
        for (int x = 0; x < dst_width; ++x) {
            const int src_x = ScaleIndex(x, dst_width, src_width);
            dst[static_cast<std::size_t>(FlatIndex(x, y, dst_width))] =
                src[static_cast<std::size_t>(FlatIndex(src_x, src_y, src_width))] > 0 ? 1.0f : 0.0f;
        }
    }
    return dst;
}

std::vector<float> GuidanceCostMapOnnx::ResizeFloatMap(const std::vector<float>& src,
                                                       int src_width,
                                                       int src_height,
                                                       int dst_width,
                                                       int dst_height) const {
    std::vector<float> dst(static_cast<std::size_t>(dst_width * dst_height), 0.0f);
    if (src_width == dst_width && src_height == dst_height) {
        return src;
    }
    for (int y = 0; y < dst_height; ++y) {
        const double src_y = (static_cast<double>(y) + 0.5) * static_cast<double>(src_height) /
                                 static_cast<double>(dst_height) -
                             0.5;
        const int y0 = std::max(0, std::min(src_height - 1, static_cast<int>(std::floor(src_y))));
        const int y1 = std::max(0, std::min(src_height - 1, y0 + 1));
        const double wy = std::min(1.0, std::max(0.0, src_y - static_cast<double>(y0)));
        for (int x = 0; x < dst_width; ++x) {
            const double src_x = (static_cast<double>(x) + 0.5) * static_cast<double>(src_width) /
                                     static_cast<double>(dst_width) -
                                 0.5;
            const int x0 = std::max(0, std::min(src_width - 1, static_cast<int>(std::floor(src_x))));
            const int x1 = std::max(0, std::min(src_width - 1, x0 + 1));
            const double wx = std::min(1.0, std::max(0.0, src_x - static_cast<double>(x0)));

            const double v00 = src[static_cast<std::size_t>(FlatIndex(x0, y0, src_width))];
            const double v10 = src[static_cast<std::size_t>(FlatIndex(x1, y0, src_width))];
            const double v01 = src[static_cast<std::size_t>(FlatIndex(x0, y1, src_width))];
            const double v11 = src[static_cast<std::size_t>(FlatIndex(x1, y1, src_width))];

            const double top = v00 * (1.0 - wx) + v10 * wx;
            const double bottom = v01 * (1.0 - wx) + v11 * wx;
            dst[static_cast<std::size_t>(FlatIndex(x, y, dst_width))] =
                static_cast<float>(top * (1.0 - wy) + bottom * wy);
        }
    }
    return dst;
}

std::vector<float> GuidanceCostMapOnnx::Infer(const std::vector<int>& occupancy,
                                              int width,
                                              int height,
                                              const Vec2i& start_xy,
                                              const Vec2i& goal_xy,
                                              float start_yaw,
                                              float goal_yaw) {
    if (static_cast<int>(occupancy.size()) != width * height) {
        throw std::runtime_error("Occupancy size mismatch for ONNX frontend inference.");
    }

    const int model_input_width = model_width_ > 0 ? model_width_ : width;
    const int model_input_height = model_height_ > 0 ? model_height_ : height;
    const Vec2i start_xy_model =
        ScaleGridCoord(start_xy, width, height, model_input_width, model_input_height);
    const Vec2i goal_xy_model =
        ScaleGridCoord(goal_xy, width, height, model_input_width, model_input_height);

    std::vector<float> occ_tensor =
        ResizeBinaryMap(occupancy, width, height, model_input_width, model_input_height);
    std::vector<float> start_tensor =
        MakeOneHotMap(model_input_width, model_input_height, start_xy_model);
    std::vector<float> goal_tensor =
        MakeOneHotMap(model_input_width, model_input_height, goal_xy_model);
    const std::array<int64_t, 4> map_shape{1, 1, model_input_height, model_input_width};
    const std::array<int64_t, 1> yaw_shape{1};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<float> start_yaw_tensor{start_yaw};
    std::vector<float> goal_yaw_tensor{goal_yaw};
    std::vector<Ort::Value> inputs;
    inputs.reserve(input_names_.size());

    for (const std::string& name : input_names_storage_) {
        if (name == "occ_map") {
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info, occ_tensor.data(), occ_tensor.size(), map_shape.data(), map_shape.size()));
        } else if (name == "start_map") {
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info, start_tensor.data(), start_tensor.size(), map_shape.data(), map_shape.size()));
        } else if (name == "goal_map") {
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info, goal_tensor.data(), goal_tensor.size(), map_shape.data(), map_shape.size()));
        } else if (name == "start_yaw") {
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info, start_yaw_tensor.data(), start_yaw_tensor.size(), yaw_shape.data(), yaw_shape.size()));
        } else if (name == "goal_yaw") {
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info, goal_yaw_tensor.data(), goal_yaw_tensor.size(), yaw_shape.data(), yaw_shape.size()));
        } else {
            throw std::runtime_error("Unsupported ONNX frontend input name: " + name);
        }
    }

    auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                input_names_.data(),
                                inputs.data(),
                                inputs.size(),
                                output_names_.data(),
                                output_names_.size());
    if (outputs.empty()) {
        throw std::runtime_error("ONNX frontend inference returned no outputs.");
    }

    const Ort::Value& output = outputs.front();
    auto tensor_info = output.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = tensor_info.GetShape();
    const float* data = output.GetTensorData<float>();

    std::size_t value_count = 1;
    for (int64_t dim : shape) {
        value_count *= static_cast<std::size_t>(std::max<int64_t>(dim, 1));
    }

    std::vector<float> model_cost_map;
    if (value_count == static_cast<std::size_t>(model_input_width * model_input_height)) {
        model_cost_map.assign(data, data + value_count);
    } else if (shape.size() == 3 &&
               shape[0] == 1 &&
               shape[1] == model_input_height &&
               shape[2] == model_input_width) {
        model_cost_map.assign(data, data + (model_input_width * model_input_height));
    } else {
        throw std::runtime_error("Unexpected ONNX frontend output shape.");
    }

    if (model_input_width == width && model_input_height == height) {
        return model_cost_map;
    }
    return ResizeFloatMap(model_cost_map, model_input_width, model_input_height, width, height);
}

GridAstarResult RunGuidedGridAstar(const std::vector<int>& occupancy,
                                   int width,
                                   int height,
                                   const Vec2i& start_xy,
                                   const Vec2i& goal_xy,
                                   const std::vector<float>& guidance_cost,
                                   const GridAstarOptions& options) {
    GridAstarResult result;
    if (static_cast<int>(occupancy.size()) != width * height ||
        static_cast<int>(guidance_cost.size()) != width * height) {
        return result;
    }
    if (!IsInside(start_xy.x(), start_xy.y(), width, height) ||
        !IsInside(goal_xy.x(), goal_xy.y(), width, height)) {
        return result;
    }
    if (occupancy[FlatIndex(start_xy.x(), start_xy.y(), width)] > 0 ||
        occupancy[FlatIndex(goal_xy.x(), goal_xy.y(), width)] > 0) {
        return result;
    }

    std::priority_queue<QueueNode, std::vector<QueueNode>, std::greater<QueueNode>> open_heap;
    std::vector<double> g_score(static_cast<std::size_t>(width * height),
                                std::numeric_limits<double>::infinity());
    std::vector<int> parent(static_cast<std::size_t>(width * height), -1);
    std::vector<char> closed(static_cast<std::size_t>(width * height), 0);

    const double start_h = options.heuristic_weight *
        HeuristicCost(start_xy.x(), start_xy.y(), goal_xy.x(), goal_xy.y(),
                      options.diagonal_cost, options.heuristic_mode);
    open_heap.push({start_h, start_xy});
    g_score[static_cast<std::size_t>(FlatIndex(start_xy.x(), start_xy.y(), width))] = 0.0;

    const int neighbor_dx[8] = {-1, 1, 0, 0, -1, 1, -1, 1};
    const int neighbor_dy[8] = {0, 0, -1, 1, -1, -1, 1, 1};

    while (!open_heap.empty()) {
        const QueueNode current = open_heap.top();
        open_heap.pop();
        const int current_index = FlatIndex(current.xy.x(), current.xy.y(), width);
        if (closed[static_cast<std::size_t>(current_index)] != 0) {
            continue;
        }
        closed[static_cast<std::size_t>(current_index)] = 1;
        result.expanded_xy.push_back(current.xy);
        result.expanded_nodes += 1;

        if (current.xy == goal_xy) {
            result.success = true;
            result.path_xy = ReconstructPath(parent, width, current.xy);
            return result;
        }

        for (int dir = 0; dir < 8; ++dir) {
            const int nx = current.xy.x() + neighbor_dx[dir];
            const int ny = current.xy.y() + neighbor_dy[dir];
            if (!IsInside(nx, ny, width, height)) {
                continue;
            }
            if (occupancy[FlatIndex(nx, ny, width)] > 0) {
                continue;
            }

            const bool is_diagonal = (neighbor_dx[dir] != 0 && neighbor_dy[dir] != 0);
            if (is_diagonal && !options.allow_corner_cut) {
                if (occupancy[FlatIndex(nx, current.xy.y(), width)] > 0 ||
                    occupancy[FlatIndex(current.xy.x(), ny, width)] > 0) {
                    continue;
                }
            }

            const double move_cost = is_diagonal ? options.diagonal_cost : 1.0;
            const float guidance_value = guidance_cost[FlatIndex(nx, ny, width)];

            double tentative_g = g_score[static_cast<std::size_t>(current_index)] + move_cost;
            if (options.guidance_integration_mode == "g_cost") {
                tentative_g += options.lambda_guidance * static_cast<double>(guidance_value);
            }

            const Vec2i neighbor(nx, ny);
            const int neighbor_index = FlatIndex(nx, ny, width);
            if (tentative_g < g_score[static_cast<std::size_t>(neighbor_index)]) {
                g_score[static_cast<std::size_t>(neighbor_index)] = tentative_g;
                parent[static_cast<std::size_t>(neighbor_index)] = current_index;
                double f_cost = tentative_g + options.heuristic_weight *
                    HeuristicCost(nx, ny, goal_xy.x(), goal_xy.y(),
                                  options.diagonal_cost, options.heuristic_mode);
                if (options.guidance_integration_mode != "g_cost") {
                    f_cost += options.lambda_guidance *
                        GuidancePriorityBias(static_cast<double>(guidance_value),
                                             options.guidance_integration_mode,
                                             options.guidance_bonus_threshold);
                }
                open_heap.push({f_cost, neighbor});
            }
        }
    }

    return result;
}

}  // namespace guided_frontend
