#pragma once

#include "dp_policy.h"

#include <limits>
#include <vector>

// 简单规划器：沿 s 轴逐列调用 RL policy，输出一条长度为 s_samples 的离散 l 索引路径。
class DPPlanner {
public:
    // lateral_move_limit < 0 表示不限制跳变
    DPPlanner(DPPolicy& policy,
              int s_samples,
              int l_samples,
              int lateral_move_limit = 3,
              float s_min = 0.0f,
              float s_max = 8.0f,
              float l_min = -4.0f,
              float l_max = 4.0f,
              int interpolation_points = 3,
              float fine_collision_inflation = 0.2f,
              float vehicle_length = 0.0f,
              float vehicle_width = 0.0f);

    // occupancy: 长度 s_samples*l_samples，0/1
    // start_l_index: 起始列（s=0）的 l 索引
    // start_l_value: 起始列的连续 l 坐标（用于模型输入的 start_l_norm）
    // obstacles: 可选 OBB 列表，若提供则使用插值点判碰屏蔽动作
    // 返回路径（长度 s_samples），第 0 个元素为 start_l_index
    std::vector<int> plan(const std::vector<float>& occupancy,
                          int start_l_index,
                          const std::vector<OBB>& obstacles = {},
                          float start_l_value = std::numeric_limits<float>::quiet_NaN());

private:
    DPPolicy& policy_;
    int s_samples_;
    int l_samples_;
    int lateral_move_limit_;
    float s_min_, s_max_, l_min_, l_max_;
    int grid_size_;
    int interpolation_points_;
    float fine_collision_inflation_;
    float vehicle_length_;
    float vehicle_width_;
};
