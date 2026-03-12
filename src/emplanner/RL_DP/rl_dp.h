#pragma once

#include <array>
#include <string>
#include <vector>

#include "dp_planner.h"
#include "dp_policy.h"

struct SLCorner {
    float s = 0.0f;
    float l = 0.0f;
};

using ObstacleCorners = std::array<SLCorner, 4>;

// RL_DP: 输入障碍物四角点（s,l），输出 DP 路径各列的 l 索引。
class RL_DP {
public:
    RL_DP(const std::string& model_path,
          int s_samples = 9,
          int l_samples = 19,
          float s_min = 0.0f,
          float s_max = 8.0f,
          float l_min = -4.0f,
          float l_max = 4.0f,
          int lateral_move_limit = 3,
          int interpolation_points = 3,
          float coarse_inflation = 0.2f,
          float fine_inflation = 0.2f,
          float vehicle_length = 0.0f,
          float vehicle_width = 0.0f);

    // 连续起点版本：start_l_value 为连续 l 坐标，内部映射到最近的栅格索引。
    std::vector<int> Plan(const std::vector<ObstacleCorners>& obstacles, float start_l_value);
    std::vector<int> Plan(const std::vector<ObstacleCorners>& obstacles, double start_l_value);
    // 兼容旧用法：仅提供离散起点索引。
    std::vector<int> Plan(const std::vector<ObstacleCorners>& obstacles, int start_l_index);

private:
    OBB ObbFromCorners(const ObstacleCorners& corners) const;
    std::vector<float> BuildOccupancy(const std::vector<OBB>& obstacles) const;

    int s_samples_;
    int l_samples_;
    float s_min_;
    float s_max_;
    float l_min_;
    float l_max_;
    float coarse_inflation_;
    float fine_inflation_;
    float vehicle_length_;
    float vehicle_width_;

    DPPolicy policy_;
    DPPlanner planner_;
};
