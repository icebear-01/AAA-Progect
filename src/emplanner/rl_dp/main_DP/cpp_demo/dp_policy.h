#pragma once

#include <onnxruntime_cxx_api.h>

#include <limits>
#include <string>
#include <vector>

struct OBB {
    float cx, cy;
    float half_len, half_wid;
    float cos_yaw, sin_yaw;
    float aabb_min_s, aabb_max_s;
    float aabb_min_l, aabb_max_l;
};

enum class InputLayout {
    kGridPlanes,
    kFlatFeatures,
};

// 构造 OBB、判点工具
OBB MakeOBB(float cx, float cy, float len, float wid, float yaw);
bool PointInOBB(const OBB& o, float s, float l);

// 简单封装：加载 ONNX 策略模型，给定占用栅格与动作掩码，输出 argmax 动作。
class DPPolicy {
public:
    // model_path: ONNX 模型路径
    // s_samples/l_samples: 栅格尺寸（默认 9×23，对应 0.35 m 的横向间隔）
    // s_range/l_range: 物理范围，用于 l_norm 归一化
    DPPolicy(const std::string& model_path,
             int s_samples = 9,
             int l_samples = 23,
             float s_min = 0.0f,
             float s_max = 8.0f,
             float l_min = -3.85f,
             float l_max = 3.85f);

    // 选择动作：
    // occupancy: 长度 s_samples*l_samples，0/1 占用
    // action_mask_row: 长度 l_samples，仅当前列的掩码（1 可选，0 禁用）
    // obstacles: 环境障碍物（用于角点特征，数量由模型输入布局决定）
    // s_index: 当前列索引
    // last_l_index: 上一列的 l 索引
    // start_l_index: 起始列的 l 索引（用于路径起点）
    // start_l_value: 起始列的连续 l 坐标（用于模型输入的 start_l_norm）
    // ignore_action_mask: true 时忽略动作掩码，直接按 logits 取最大值
    // 返回 argmax 动作索引；若掩码全 False，则返回 0
    int select_action(const std::vector<float>& occupancy,
                            const std::vector<float>& action_mask_row,
                            int s_index,
                            int last_l_index,
                            const std::vector<OBB>& obstacles,
                            int start_l_index = -1,
                            float start_l_value = std::numeric_limits<float>::quiet_NaN(),
                            bool ignore_action_mask = false);

private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string input_name_;
    std::string logits_name_;
    std::string value_name_;

    int s_samples_;
    int l_samples_;
    float s_min_;
    float s_max_;
    float l_min_;
    float l_max_;
    int feature_dim_;
    int grid_size_;
    InputLayout input_layout_;
    int max_obstacles_;
    int extra_count_;
};

// 简单测试函数，构造示例输入并打印选择动作。
void test_dp_policy();
