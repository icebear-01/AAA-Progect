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

enum class HeuristicModeId {
    kEuclidean,
    kManhattan,
    kChebyshev,
    kOctile,
};

enum class GuidanceIntegrationModeId {
    kHeuristicBias,
    kHeuristicBonus,
    kGCost,
};

enum class ClearanceIntegrationModeId {
    kGCost,
    kHeuristicBias,
    kPriorityTieBreak,
};

struct QueueNode {
    double f_cost{0.0};
    double clearance_key{0.0};
    Vec2i xy{0, 0};

    bool operator>(const QueueNode& other) const {
        if (f_cost != other.f_cost) {
            return f_cost > other.f_cost;
        }
        return clearance_key > other.clearance_key;
    }
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

HeuristicModeId ParseHeuristicMode(const std::string& mode) {
    if (mode == "euclidean") {
        return HeuristicModeId::kEuclidean;
    }
    if (mode == "manhattan") {
        return HeuristicModeId::kManhattan;
    }
    if (mode == "chebyshev") {
        return HeuristicModeId::kChebyshev;
    }
    if (mode == "octile") {
        return HeuristicModeId::kOctile;
    }
    throw std::runtime_error("Unknown heuristic_mode: " + mode);
}

double HeuristicCost(int x,
                     int y,
                     int gx,
                     int gy,
                     double diagonal_cost,
                     HeuristicModeId mode) {
    const int dx = std::abs(gx - x);
    const int dy = std::abs(gy - y);
    switch (mode) {
    case HeuristicModeId::kEuclidean:
        return std::hypot(static_cast<double>(dx), static_cast<double>(dy));
    case HeuristicModeId::kManhattan:
        return static_cast<double>(dx + dy);
    case HeuristicModeId::kChebyshev:
        return static_cast<double>(std::max(dx, dy));
    case HeuristicModeId::kOctile: {
        const double d_min = static_cast<double>(std::min(dx, dy));
        const double d_max = static_cast<double>(std::max(dx, dy));
        return d_max + (diagonal_cost - 1.0) * d_min;
    }
    }
    throw std::runtime_error("Unknown heuristic mode id.");
}

GuidanceIntegrationModeId ParseGuidanceIntegrationMode(const std::string& integration_mode) {
    if (integration_mode == "heuristic_bias") {
        return GuidanceIntegrationModeId::kHeuristicBias;
    }
    if (integration_mode == "heuristic_bonus") {
        return GuidanceIntegrationModeId::kHeuristicBonus;
    }
    if (integration_mode == "g_cost") {
        return GuidanceIntegrationModeId::kGCost;
    }
    throw std::runtime_error("Unknown guidance_integration_mode: " + integration_mode);
}

double GuidancePriorityBias(double guidance_value,
                            GuidanceIntegrationModeId integration_mode,
                            double bonus_threshold) {
    if (integration_mode == GuidanceIntegrationModeId::kHeuristicBias) {
        return guidance_value;
    }
    if (integration_mode == GuidanceIntegrationModeId::kHeuristicBonus) {
        const double scaled_bonus = (guidance_value - bonus_threshold) / std::max(bonus_threshold, 1e-6);
        return std::min(0.0, scaled_bonus);
    }
    if (integration_mode == GuidanceIntegrationModeId::kGCost) {
        return 0.0;
    }
    throw std::runtime_error("Unknown guidance integration mode id.");
}

ClearanceIntegrationModeId ParseClearanceIntegrationMode(const std::string& integration_mode) {
    if (integration_mode == "g_cost") {
        return ClearanceIntegrationModeId::kGCost;
    }
    if (integration_mode == "heuristic_bias") {
        return ClearanceIntegrationModeId::kHeuristicBias;
    }
    if (integration_mode == "priority_tie_break") {
        return ClearanceIntegrationModeId::kPriorityTieBreak;
    }
    throw std::runtime_error("Unknown clearance_integration_mode: " + integration_mode);
}

double ClearancePriorityBias(double clearance_value,
                             ClearanceIntegrationModeId integration_mode) {
    if (integration_mode == ClearanceIntegrationModeId::kGCost) {
        return 0.0;
    }
    if (integration_mode == ClearanceIntegrationModeId::kHeuristicBias ||
        integration_mode == ClearanceIntegrationModeId::kPriorityTieBreak) {
        return clearance_value;
    }
    throw std::runtime_error("Unknown clearance integration mode id.");
}

std::vector<double> BuildClearancePenaltyMap(const std::vector<int>& occupancy,
                                             int width,
                                             int height,
                                             double diagonal_cost,
                                             double safe_distance,
                                             double power) {
    std::vector<double> penalty(static_cast<std::size_t>(width * height), 0.0);
    if (safe_distance <= 0.0) {
        return penalty;
    }

    std::vector<double> dist(static_cast<std::size_t>(width * height),
                             std::numeric_limits<double>::infinity());
    std::priority_queue<QueueNode, std::vector<QueueNode>, std::greater<QueueNode>> heap;
    const int neighbor_dx[8] = {-1, 1, 0, 0, -1, 1, -1, 1};
    const int neighbor_dy[8] = {0, 0, -1, 1, -1, -1, 1, 1};

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int idx = FlatIndex(x, y, width);
            if (occupancy[static_cast<std::size_t>(idx)] > 0) {
                dist[static_cast<std::size_t>(idx)] = 0.0;
                heap.push({0.0, 0.0, Vec2i(x, y)});
            }
        }
    }

    while (!heap.empty()) {
        const QueueNode current = heap.top();
        heap.pop();
        const int current_index = FlatIndex(current.xy.x(), current.xy.y(), width);
        if (current.f_cost > dist[static_cast<std::size_t>(current_index)] + 1e-9) {
            continue;
        }
        for (int dir = 0; dir < 8; ++dir) {
            const int nx = current.xy.x() + neighbor_dx[dir];
            const int ny = current.xy.y() + neighbor_dy[dir];
            if (!IsInside(nx, ny, width, height)) {
                continue;
            }
            const bool is_diagonal = (neighbor_dx[dir] != 0 && neighbor_dy[dir] != 0);
            const double step = is_diagonal ? diagonal_cost : 1.0;
            const int neighbor_index = FlatIndex(nx, ny, width);
            const double nd = current.f_cost + step;
            if (nd + 1e-9 < dist[static_cast<std::size_t>(neighbor_index)]) {
                dist[static_cast<std::size_t>(neighbor_index)] = nd;
                heap.push({nd, 0.0, Vec2i(nx, ny)});
            }
        }
    }

    const double scale = std::max(safe_distance, 1e-6);
    const double effective_power = std::max(power, 1e-6);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int idx = FlatIndex(x, y, width);
            if (occupancy[static_cast<std::size_t>(idx)] > 0) {
                penalty[static_cast<std::size_t>(idx)] = 1.0;
                continue;
            }
            const double raw = std::min(1.0, std::max(0.0, (scale - dist[static_cast<std::size_t>(idx)]) / scale));
            penalty[static_cast<std::size_t>(idx)] =
                effective_power == 1.0 ? raw : std::pow(raw, effective_power);
        }
    }
    return penalty;
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
    memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

void GuidanceCostMapOnnx::InvalidateOccupancyCache() {
    occupancy_cache_valid_ = false;
    cached_occupancy_ptr_ = nullptr;
    cached_occupancy_size_ = 0;
    cached_occupancy_width_ = -1;
    cached_occupancy_height_ = -1;
    cached_model_width_ = -1;
    cached_model_height_ = -1;
}

void GuidanceCostMapOnnx::EnsureInputBuffers(int width, int height) {
    const std::size_t value_count = static_cast<std::size_t>(width * height);
    if (occ_tensor_buffer_.size() != value_count) {
        occ_tensor_buffer_.assign(value_count, 0.0f);
        InvalidateOccupancyCache();
    }
    if (start_tensor_buffer_.size() != value_count) {
        start_tensor_buffer_.assign(value_count, 0.0f);
        last_start_hot_index_ = -1;
    }
    if (goal_tensor_buffer_.size() != value_count) {
        goal_tensor_buffer_.assign(value_count, 0.0f);
        last_goal_hot_index_ = -1;
    }
    input_values_.clear();
    input_values_.reserve(input_names_.size());
}

void GuidanceCostMapOnnx::FillOneHotMap(std::vector<float>& one_hot,
                                        int width,
                                        int height,
                                        const Vec2i& xy,
                                        int& last_hot_index) const {
    if (last_hot_index >= 0 &&
        last_hot_index < static_cast<int>(one_hot.size())) {
        one_hot[static_cast<std::size_t>(last_hot_index)] = 0.0f;
    }
    last_hot_index = -1;
    if (IsInside(xy.x(), xy.y(), width, height)) {
        last_hot_index = FlatIndex(xy.x(), xy.y(), width);
        one_hot[static_cast<std::size_t>(last_hot_index)] = 1.0f;
    }
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

void GuidanceCostMapOnnx::FillBinaryMap(const std::vector<int>& src,
                                        int src_width,
                                        int src_height,
                                        int dst_width,
                                        int dst_height,
                                        std::vector<float>& dst) const {
    dst.resize(static_cast<std::size_t>(dst_width * dst_height));
    if (src_width == dst_width && src_height == dst_height) {
        for (std::size_t i = 0; i < src.size(); ++i) {
            dst[i] = src[i] > 0 ? 1.0f : 0.0f;
        }
        return;
    }
    for (int y = 0; y < dst_height; ++y) {
        const int src_y = ScaleIndex(y, dst_height, src_height);
        for (int x = 0; x < dst_width; ++x) {
            const int src_x = ScaleIndex(x, dst_width, src_width);
            dst[static_cast<std::size_t>(FlatIndex(x, y, dst_width))] =
                src[static_cast<std::size_t>(FlatIndex(src_x, src_y, src_width))] > 0 ? 1.0f : 0.0f;
        }
    }
}

const std::vector<float>& GuidanceCostMapOnnx::GetOccupancyTensor(const std::vector<int>& src,
                                                                  int src_width,
                                                                  int src_height,
                                                                  int dst_width,
                                                                  int dst_height) {
    const int* src_ptr = src.empty() ? nullptr : src.data();
    if (occupancy_cache_valid_ &&
        cached_occupancy_ptr_ == src_ptr &&
        cached_occupancy_size_ == src.size() &&
        cached_occupancy_width_ == src_width &&
        cached_occupancy_height_ == src_height &&
        cached_model_width_ == dst_width &&
        cached_model_height_ == dst_height) {
        return occ_tensor_buffer_;
    }

    FillBinaryMap(src, src_width, src_height, dst_width, dst_height, occ_tensor_buffer_);
    cached_occupancy_ptr_ = src_ptr;
    cached_occupancy_size_ = src.size();
    cached_occupancy_width_ = src_width;
    cached_occupancy_height_ = src_height;
    cached_model_width_ = dst_width;
    cached_model_height_ = dst_height;
    occupancy_cache_valid_ = true;
    return occ_tensor_buffer_;
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

    EnsureInputBuffers(model_input_width, model_input_height);
    GetOccupancyTensor(occupancy, width, height, model_input_width, model_input_height);
    FillOneHotMap(start_tensor_buffer_, model_input_width, model_input_height,
                  start_xy_model, last_start_hot_index_);
    FillOneHotMap(goal_tensor_buffer_, model_input_width, model_input_height,
                  goal_xy_model, last_goal_hot_index_);
    const std::array<int64_t, 4> map_shape{1, 1, model_input_height, model_input_width};
    const std::array<int64_t, 1> yaw_shape{1};

    std::vector<float> start_yaw_tensor{start_yaw};
    std::vector<float> goal_yaw_tensor{goal_yaw};
    for (const std::string& name : input_names_storage_) {
        if (name == "occ_map") {
            input_values_.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info_, occ_tensor_buffer_.data(), occ_tensor_buffer_.size(),
                map_shape.data(), map_shape.size()));
        } else if (name == "start_map") {
            input_values_.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info_, start_tensor_buffer_.data(), start_tensor_buffer_.size(),
                map_shape.data(), map_shape.size()));
        } else if (name == "goal_map") {
            input_values_.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info_, goal_tensor_buffer_.data(), goal_tensor_buffer_.size(),
                map_shape.data(), map_shape.size()));
        } else if (name == "start_yaw") {
            input_values_.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info_, start_yaw_tensor.data(), start_yaw_tensor.size(),
                yaw_shape.data(), yaw_shape.size()));
        } else if (name == "goal_yaw") {
            input_values_.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info_, goal_yaw_tensor.data(), goal_yaw_tensor.size(),
                yaw_shape.data(), yaw_shape.size()));
        } else {
            throw std::runtime_error("Unsupported ONNX frontend input name: " + name);
        }
    }

    auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                input_names_.data(),
                                input_values_.data(),
                                input_values_.size(),
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
    const HeuristicModeId heuristic_mode = ParseHeuristicMode(options.heuristic_mode);
    const GuidanceIntegrationModeId guidance_mode =
        ParseGuidanceIntegrationMode(options.guidance_integration_mode);
    const ClearanceIntegrationModeId clearance_mode =
        ParseClearanceIntegrationMode(options.clearance_integration_mode);
    std::vector<double> heuristic_cache(static_cast<std::size_t>(width * height), 0.0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int index = FlatIndex(x, y, width);
            heuristic_cache[static_cast<std::size_t>(index)] =
                options.heuristic_weight *
                HeuristicCost(x, y, goal_xy.x(), goal_xy.y(),
                              options.diagonal_cost, heuristic_mode);
        }
    }
    const std::vector<double> clearance_penalty =
        BuildClearancePenaltyMap(occupancy,
                                 width,
                                 height,
                                 options.diagonal_cost,
                                 options.clearance_safe_distance,
                                 options.clearance_power);

    const int start_index = FlatIndex(start_xy.x(), start_xy.y(), width);
    const double start_h = heuristic_cache[static_cast<std::size_t>(start_index)];
    const double start_clearance_bias =
        clearance_mode == ClearanceIntegrationModeId::kPriorityTieBreak
            ? options.clearance_weight * clearance_penalty[static_cast<std::size_t>(start_index)]
            : 0.0;
    open_heap.push({start_h, start_clearance_bias, start_xy});
    g_score[static_cast<std::size_t>(start_index)] = 0.0;

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
        if (options.record_expanded_xy) {
            result.expanded_xy.push_back(current.xy);
        }
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

            const int neighbor_index = FlatIndex(nx, ny, width);
            if (closed[static_cast<std::size_t>(neighbor_index)] != 0) {
                continue;
            }

            const bool is_diagonal = (neighbor_dx[dir] != 0 && neighbor_dy[dir] != 0);
            if (is_diagonal) {
                const bool side_block_x = occupancy[FlatIndex(nx, current.xy.y(), width)] > 0;
                const bool side_block_y = occupancy[FlatIndex(current.xy.x(), ny, width)] > 0;
                if (side_block_x && side_block_y) {
                    continue;
                }
                if (!options.allow_corner_cut && (side_block_x || side_block_y)) {
                    continue;
                }
            }

            const double move_cost = is_diagonal ? options.diagonal_cost : 1.0;
            const float guidance_value = guidance_cost[FlatIndex(nx, ny, width)];

            double tentative_g = g_score[static_cast<std::size_t>(current_index)] + move_cost;
            if (guidance_mode == GuidanceIntegrationModeId::kGCost) {
                tentative_g += options.lambda_guidance * static_cast<double>(guidance_value);
            }
            if (options.clearance_weight > 0.0 &&
                clearance_mode == ClearanceIntegrationModeId::kGCost) {
                tentative_g += options.clearance_weight *
                    clearance_penalty[static_cast<std::size_t>(neighbor_index)];
            }

            const Vec2i neighbor(nx, ny);
            if (tentative_g < g_score[static_cast<std::size_t>(neighbor_index)]) {
                g_score[static_cast<std::size_t>(neighbor_index)] = tentative_g;
                parent[static_cast<std::size_t>(neighbor_index)] = current_index;
                double f_cost = tentative_g + heuristic_cache[static_cast<std::size_t>(neighbor_index)];
                if (guidance_mode != GuidanceIntegrationModeId::kGCost) {
                    f_cost += options.lambda_guidance *
                        GuidancePriorityBias(static_cast<double>(guidance_value),
                                             guidance_mode,
                                             options.guidance_bonus_threshold);
                }
                double clearance_bias = 0.0;
                if (options.clearance_weight > 0.0 &&
                    clearance_mode != ClearanceIntegrationModeId::kGCost) {
                    clearance_bias = options.clearance_weight * ClearancePriorityBias(
                        clearance_penalty[static_cast<std::size_t>(neighbor_index)],
                        clearance_mode);
                    if (clearance_mode == ClearanceIntegrationModeId::kHeuristicBias) {
                        f_cost += clearance_bias;
                        clearance_bias = 0.0;
                    }
                }
                open_heap.push({f_cost, clearance_bias, neighbor});
            }
        }
    }

    return result;
}

}  // namespace guided_frontend
