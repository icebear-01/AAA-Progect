#include "dp_policy.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <numeric>

namespace {
// 简单的 argmax（忽略 NaN）
int ArgMaxMasked(const std::vector<float>& logits, const std::vector<float>& mask) {  // 选择有效掩码
    int best = 0;
    float best_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < mask.size(); ++i) {
        float m = mask[i];
        if (m <= 0.0f) continue;  // 忽略掩码为0的位置
        float v = logits[i];  // 使用 std::vector 访问元素
        if (v > best_val) {
            best_val = v;
            best = static_cast<int>(i);
        }
    }
    return best;
}

int ArgMaxAll(const std::vector<float>& logits) {
    int best = 0;
    float best_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < logits.size(); ++i) {
        float v = logits[i];
        if (v > best_val) {
            best_val = v;
            best = static_cast<int>(i);
        }
    }
    return best;
}

float LateralDistanceToCenter(const OBB& o) {
    if (o.aabb_min_l > 0.0f) {
        return o.aabb_min_l;
    }
    if (o.aabb_max_l < 0.0f) {
        return -o.aabb_max_l;
    }
    return 0.0f;
}

// 将障碍物角点投影到栅格，返回平面向量。
std::vector<float> CornerPlane(const std::vector<OBB>& obstacles,
                               int s_samples,
                               int l_samples,
                               float s_min,
                               float s_max,
                               float l_min,
                               float l_max,
                               float current_s,
                               float current_l) {
    std::vector<float> plane(static_cast<size_t>(s_samples * l_samples), 0.0f);
    if (obstacles.empty()) {
        return plane;
    }
    static_cast<void>(current_s);
    static_cast<void>(current_l);

    const int max_obs = 10;
    std::vector<int> indices(obstacles.size());
    std::iota(indices.begin(), indices.end(), 0);
    float s_min_filter = s_min - 1.0f;
    float s_max_filter = s_max;
    indices.erase(std::remove_if(indices.begin(), indices.end(), [&](int idx) {
        const auto& o = obstacles[idx];
        return o.aabb_max_s < s_min_filter || o.aabb_min_s > s_max_filter;
    }), indices.end());
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        float la = LateralDistanceToCenter(obstacles[a]);
        float lb = LateralDistanceToCenter(obstacles[b]);
        return la < lb;
    });

    float s_span = std::max(s_max - s_min, 1e-6f);
    float l_span = std::max(l_max - l_min, 1e-6f);

    int take = std::min<int>(max_obs, static_cast<int>(indices.size()));
    for (int i = 0; i < take; ++i) {
        const auto& o = obstacles[indices[static_cast<size_t>(i)]];
        std::vector<std::pair<float, float>> local = {
            { o.half_len,  o.half_wid},
            { o.half_len, -o.half_wid},
            {-o.half_len, -o.half_wid},
            {-o.half_len,  o.half_wid},
        };
        for (const auto& p : local) {
            float lx = p.first;
            float ly = p.second;
            float gx = o.cx + o.cos_yaw * lx - o.sin_yaw * ly;
            float gy = o.cy + o.sin_yaw * lx + o.cos_yaw * ly;
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

float Clamp01(float v) {
    return std::max(0.0f, std::min(1.0f, v));
}

std::vector<float> CornerFlat(const std::vector<OBB>& obstacles,
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
    static_cast<void>(current_s);
    static_cast<void>(current_l);

    float s_span = std::max(s_max - s_min, 1e-6f);
    float l_span = std::max(l_max - l_min, 1e-6f);
    float pad_s = Clamp01((0.0f - s_min) / s_span);
    float pad_l = Clamp01((0.0f - l_min) / l_span);
    for (size_t i = 0; i + 1 < flat.size(); i += 2) {
        flat[i] = pad_s;
        flat[i + 1] = pad_l;
    }
    if (obstacles.empty()) {
        return flat;
    }

    std::vector<int> indices(obstacles.size());
    std::iota(indices.begin(), indices.end(), 0);
    float s_min_filter = s_min - 1.0f;
    float s_max_filter = s_max;
    indices.erase(std::remove_if(indices.begin(), indices.end(), [&](int idx) {
        const auto& o = obstacles[static_cast<size_t>(idx)];
        return o.aabb_max_s < s_min_filter || o.aabb_min_s > s_max_filter;
    }), indices.end());
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        float la = LateralDistanceToCenter(obstacles[static_cast<size_t>(a)]);
        float lb = LateralDistanceToCenter(obstacles[static_cast<size_t>(b)]);
        return la < lb;
    });
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
            float s_norm = Clamp01((gx - s_min) / s_span);
            float l_norm = Clamp01((gy - l_min) / l_span);
            if (offset + 1 < flat.size()) {
                flat[offset++] = s_norm;
                flat[offset++] = l_norm;
            }
        }
    }
    return flat;
}

}  // namespace

// 构造 OBB 预计算粗判 AABB
OBB MakeOBB(float cx, float cy, float len, float wid, float yaw) {
    OBB o{};
    o.cx = cx; o.cy = cy;
    o.half_len = 0.5f * len;
    o.half_wid = 0.5f * wid;
    o.cos_yaw = std::cos(yaw);
    o.sin_yaw = std::sin(yaw);
    float abs_c = std::abs(o.cos_yaw);
    float abs_s = std::abs(o.sin_yaw);
    o.aabb_min_s = cx - abs_c * o.half_len - abs_s * o.half_wid;
    o.aabb_max_s = cx + abs_c * o.half_len + abs_s * o.half_wid;
    o.aabb_min_l = cy - abs_s * o.half_len - abs_c * o.half_wid;
    o.aabb_max_l = cy + abs_s * o.half_len + abs_c * o.half_wid;
    return o;
}

// 点在 OBB 判定：AABB 粗判 + 旋转回局部坐标
bool PointInOBB(const OBB& o, float s, float l) {
    if (s < o.aabb_min_s || s > o.aabb_max_s || l < o.aabb_min_l || l > o.aabb_max_l) {
        return false;
    }
    float ds = s - o.cx;
    float dl = l - o.cy;
    float local_x =  o.cos_yaw * ds + o.sin_yaw * dl;
    float local_y = -o.sin_yaw * ds + o.cos_yaw * dl;
    constexpr float inflate = 1.1f;  // 10% 膨胀
    return std::abs(local_x) <= o.half_len * inflate && std::abs(local_y) <= o.half_wid * inflate;
}

DPPolicy::DPPolicy(const std::string& model_path,
                   int s_samples,
                   int l_samples,
                   float s_min,
                   float s_max,
                   float l_min,
                   float l_max)
    : env_(ORT_LOGGING_LEVEL_WARNING, "dp_policy"),
      session_(env_, model_path.c_str(), Ort::SessionOptions{nullptr}),
      s_samples_(s_samples),
      l_samples_(l_samples),
      s_min_(s_min),
      s_max_(s_max),
      l_min_(l_min),
      l_max_(l_max) {
    // 输入/输出名
    Ort::AllocatedStringPtr in = session_.GetInputNameAllocated(0, allocator_);
    input_name_ = in.get();
    Ort::AllocatedStringPtr out0 = session_.GetOutputNameAllocated(0, allocator_);
    Ort::AllocatedStringPtr out1 = session_.GetOutputNameAllocated(1, allocator_);
    logits_name_ = out0.get();
    value_name_ = out1.get();

    // 推断特征维度
    auto input_type_info = session_.GetInputTypeInfo(0);
    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    auto input_shape = tensor_info.GetShape();
    if (input_shape.size() < 2) {
        throw std::runtime_error("Unexpected input rank");
    }
    feature_dim_ = static_cast<int>(input_shape[1]);
    grid_size_ = s_samples_ * l_samples_;
    const int plane_expected_v1 = 3 * grid_size_ + 2;
    const int plane_expected_v2 = 3 * grid_size_ + 3;
    if (feature_dim_ == plane_expected_v2) {
        input_layout_ = InputLayout::kGridPlanes;
        max_obstacles_ = 0;
        extra_count_ = 3;
    } else if (feature_dim_ == plane_expected_v1) {
        input_layout_ = InputLayout::kGridPlanes;
        max_obstacles_ = 0;
        extra_count_ = 2;
    } else {
        int obstacle_dim = feature_dim_ - grid_size_ - l_samples_ - 3;
        if (obstacle_dim >= 0 && obstacle_dim % 8 == 0) {
            input_layout_ = InputLayout::kFlatFeatures;
            max_obstacles_ = obstacle_dim / 8;
            extra_count_ = 3;
        } else {
            obstacle_dim = feature_dim_ - grid_size_ - l_samples_ - 2;
            if (obstacle_dim >= 0 && obstacle_dim % 8 == 0) {
                input_layout_ = InputLayout::kFlatFeatures;
                max_obstacles_ = obstacle_dim / 8;
                extra_count_ = 2;
            } else {
                throw std::runtime_error(
                    "Unsupported feature dim: " + std::to_string(feature_dim_) +
                    " (expected " + std::to_string(plane_expected_v2) +
                    " or " + std::to_string(plane_expected_v1) +
                    " for 3-plane input, or grid+obstacles+mask+extras for flat input)");
            }
        }
    }
}

int DPPolicy::select_action(const std::vector<float>& occupancy,
                            const std::vector<float>& action_mask_row,
                            int s_index,
                            int last_l_index,
                            const std::vector<OBB>& obstacles,
                            int start_l_index,
                            float start_l_value,
                            bool ignore_action_mask) {
    if (static_cast<int>(occupancy.size()) != grid_size_) {
        throw std::runtime_error("occupancy size mismatch");
    }
    if (static_cast<int>(action_mask_row.size()) != l_samples_) {
        throw std::runtime_error("action_mask size mismatch");
    }

    // 当前位置的连续坐标，用于障碍物排序/投影。
    float s_coord = s_min_ + (s_max_ - s_min_) * static_cast<float>(s_index) / std::max(1, s_samples_ - 1);
    float l_step = (l_max_ - l_min_) / std::max(1, l_samples_ - 1);
    float l_coord = l_min_ + l_step * static_cast<float>(last_l_index);
    float start_l_coord = start_l_value;
    if (!std::isfinite(start_l_coord)) {
        int start_l = start_l_index;
        if (start_l < 0 || start_l >= l_samples_) {
            start_l = last_l_index;
        }
        start_l_coord = l_min_ + l_step * static_cast<float>(start_l);
    }
    start_l_coord = std::min(std::max(start_l_coord, l_min_), l_max_);

    // 创建 mask_plane 数组，初始化为 0.0f
    std::vector<float> mask_plane(grid_size_, 0.0f);
    if (0 <= s_index && s_index < s_samples_) {
        for (int l = 0; l < l_samples_; ++l) {
            mask_plane[s_index * l_samples_ + l] = action_mask_row[l];  // 填充当前行的动作掩码
        }
    }

    // 最近障碍物角点平面
    // 创建输入数组
    std::vector<float> input(feature_dim_, 0.0f);
    size_t offset = 0;
    for (float v : occupancy) input[offset++] = v;      // occupancy 平面
    if (input_layout_ == InputLayout::kGridPlanes) {
        std::vector<float> corner_plane = CornerPlane(obstacles, s_samples_, l_samples_,
                                                      s_min_, s_max_, l_min_, l_max_,
                                                      s_coord, l_coord);
        for (float v : mask_plane) input[offset++] = v;     // action_mask 平面
        for (float v : corner_plane) input[offset++] = v;   // 障碍角点平面
    } else {
        std::vector<float> corner_flat = CornerFlat(obstacles, max_obstacles_,
                                                    s_min_, s_max_, l_min_, l_max_,
                                                    s_coord, l_coord);
        for (float v : corner_flat) input[offset++] = v;    // 障碍角点特征
        for (float v : action_mask_row) input[offset++] = v;  // action_mask 向量
    }

    // 归一化 s_index 和 l_index
    float s_norm = static_cast<float>(s_index) / std::max(1, s_samples_ - 1);
    float l_norm = (l_coord - l_min_) / (l_max_ - l_min_);
    float start_l_norm = (start_l_coord - l_min_) / (l_max_ - l_min_);
    input[offset++] = s_norm;  // 添加 s_index 归一化后的值
    input[offset++] = l_norm;  // 添加 l_index 归一化后的值
    if (extra_count_ >= 3) {
        input[offset++] = start_l_norm;  // 添加起始 l 归一化
    }
    if (static_cast<int>(offset) != feature_dim_) {
        throw std::runtime_error("Input feature size mismatch when packing features.");
    }

    // 创建输入张量
    std::array<int64_t, 2> dims{1, feature_dim_};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input.data(), input.size(), dims.data(), dims.size());

    const char* in_names[] = {input_name_.c_str()};
    const char* out_names[] = {logits_name_.c_str(), value_name_.c_str()};
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names, &input_tensor, 1, out_names, 2);

    const Ort::Value& logits = outputs[0];
    auto logits_info = logits.GetTensorTypeAndShapeInfo();
    auto logits_shape = logits_info.GetShape();
    if (logits_shape.size() != 2 || logits_shape[0] != 1 || logits_shape[1] != l_samples_) {
        throw std::runtime_error("Unexpected logits shape");
    }
    const float* logits_data = logits.GetTensorData<float>();

    // 将 logits_data 转换为 std::vector<float>
    std::vector<float> logits_vector(logits_data, logits_data + logits_shape[1]);

    // 应用动作掩码后 argmax（或忽略掩码）
    int action = ignore_action_mask
                     ? ArgMaxAll(logits_vector)
                     : ArgMaxMasked(logits_vector, action_mask_row);
    return action;
}

// 简单测试函数：构造一个占用/掩码示例并打印选择的动作。
void test_dp_policy() {
    // 路径按需修改为你的三通道 ONNX。
    DPPolicy policy("/home/wmd/rl_dp/main_DP/main/checkpoints/ppo_policy_20251208_095929_update_10268.onnx");
    int s_samples = 9;
    int l_samples = 19;
    std::vector<float> occupancy(static_cast<size_t>(s_samples * l_samples), 0.0f);
    // 随机生成的 4 个障碍物，用中心+长宽+yaw 近似栅格占用
    struct Obs { float cx, cy, len, wid, yaw; };
    std::vector<Obs> obs_list = {
        {2.0f, -1.0f, 1.0f, 0.8f, 0.2f},
        {4.2f,  2.0f, 0.9f, 0.7f, -0.3f},
        {5.5f,  0.0f, 1.2f, 0.6f, 0.5f},
        {6.5f, -2.0f, 1.1f, 0.9f, -0.4f},
    };
    std::vector<OBB> obb_list;
    obb_list.reserve(obs_list.size());
    auto mark_occ = [&](float s, float l) {
        // 将连续坐标映射到最近栅格
        float s_norm = (s - 0.0f) / (8.0f - 0.0f);
        float l_norm = (l - (-4.0f)) / (4.0f - (-4.0f));
        int s_idx = static_cast<int>(std::round(s_norm * (s_samples - 1)));
        int l_idx = static_cast<int>(std::round(l_norm * (l_samples - 1)));
        s_idx = std::max(0, std::min(s_samples - 1, s_idx));
        l_idx = std::max(0, std::min(l_samples - 1, l_idx));
        occupancy[static_cast<size_t>(s_idx * l_samples + l_idx)] = 1.0f;
    };
    auto rotate_point = [](float x, float y, float yaw) {
        float c = std::cos(yaw), s = std::sin(yaw);
        return std::make_pair(c * x - s * y, s * x + c * y);
    };
    for (const auto& o : obs_list) {
        obb_list.push_back(MakeOBB(o.cx, o.cy, o.len, o.wid, o.yaw));
        float hx = o.len * 0.5f;
        float hy = o.wid * 0.5f;
        std::vector<std::pair<float, float>> local = {{hx, hy}, {hx, -hy}, {-hx, -hy}, {-hx, hy}};
        for (auto& p : local) {
            auto r = rotate_point(p.first, p.second, o.yaw);
            float gs = o.cx + r.first;
            float gl = o.cy + r.second;
            mark_occ(gs, gl);
        }
    }

    // 当前列 s_index 及上一列索引
    int s_index = 4;
    int last_l_index = 8;

    // 当前列的动作掩码：示例屏蔽 l=10,11
    std::vector<float> action_mask_row(static_cast<size_t>(l_samples), 1.0f);
    action_mask_row[10] = 0.0f;
    action_mask_row[11] = 0.0f;

    int action = policy.select_action(occupancy, action_mask_row, s_index, last_l_index, obb_list, last_l_index);
    std::cout << "[DPPolicy test] selected action index = " << action << std::endl;
}
