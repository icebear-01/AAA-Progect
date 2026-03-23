#include "hybrid_a_star/hybrid_a_star.h"
#include "hybrid_a_star/display_tools.h"
#include "hybrid_a_star/timer.h"
#include "hybrid_a_star/trajectory_optimizer.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>

namespace {

bool IsFinitePose(const Vec4d &pose) {
    return std::isfinite(pose.x()) && std::isfinite(pose.y()) &&
           std::isfinite(pose.z()) && std::isfinite(pose.w());
}

double NormalizeAngleDiff(double angle) {
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

std::vector<Vec2d> ReadOverrideSplitPointsCsv(const std::string& csv_path) {
    std::vector<Vec2d> points;
    std::ifstream in(csv_path);
    if (!in.is_open()) {
        std::cerr << "[split-override] failed to open csv: " << csv_path << std::endl;
        return points;
    }
    std::string line;
    bool first_line = true;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (first_line) {
            first_line = false;
            if (line.find("x") != std::string::npos) {
                continue;
            }
        }
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> fields;
        while (std::getline(ss, token, ',')) {
            fields.push_back(token);
        }
        if (fields.size() < 3) {
            continue;
        }
        try {
            points.emplace_back(std::stod(fields[1]), std::stod(fields[2]));
        } catch (const std::exception&) {
            continue;
        }
    }
    return points;
}

std::vector<int> MatchOverrideSplitIndices(const VectorVec4d& path, const std::vector<Vec2d>& override_points,
                                           double max_match_dist) {
    std::vector<int> matched;
    if (path.size() < 3 || override_points.empty()) {
        return matched;
    }

    for (const auto& override_pt : override_points) {
        int best_idx = -1;
        double best_dist = std::numeric_limits<double>::max();
        for (int i = 0; i < static_cast<int>(path.size()); ++i) {
            const double dist = (path[i].head(2) - override_pt).norm();
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = i;
            }
        }
        if (best_idx <= 0 || best_idx >= static_cast<int>(path.size()) - 1) {
            continue;
        }
        if (best_dist > max_match_dist) {
            continue;
        }
        matched.push_back(best_idx);
    }

    std::sort(matched.begin(), matched.end());
    matched.erase(std::unique(matched.begin(), matched.end()), matched.end());
    return matched;
}

}  // namespace

HybridAStar::HybridAStar(double steering_angle, int steering_angle_discrete_num, double segment_length,
                         int segment_length_discrete_num, double wheel_base, double steering_penalty,
                         double reversing_penalty, double steering_change_penalty, double shot_distance,
                         int grid_size_phi) : gen(std::random_device{}()),dis(0.0, 1.0){
    wheel_base_ = wheel_base;
    segment_length_ = segment_length;  //<!-- 搜索路径长度 -->
    steering_radian_ = steering_angle * M_PI / 180.0; 
    steering_discrete_num_ = steering_angle_discrete_num;   
    steering_radian_step_size_ = steering_radian_ / steering_discrete_num_;  //角度步进长度 
    move_step_size_ = segment_length / segment_length_discrete_num;
    segment_length_discrete_num_ = static_cast<int>(segment_length_discrete_num);    
    steering_penalty_ = steering_penalty;
    steering_change_penalty_ = steering_change_penalty;
    reversing_penalty_ = reversing_penalty;
    shot_distance_ = shot_distance;
 
    CHECK_EQ(static_cast<float>(segment_length_discrete_num_ * move_step_size_), static_cast<float>(segment_length_))
        << "The segment length must be divisible by the step size. segment_length: "
        << segment_length_ << " | step_size: " << move_step_size_;

    rs_path_ptr_ = std::make_shared<RSPath>(wheel_base_ / std::tan(steering_radian_));
    tie_breaker_ = 1.0 + 1e-3;

    STATE_GRID_SIZE_PHI_ = grid_size_phi;   //72
    ANGULAR_RESOLUTION_ = 360.0 / STATE_GRID_SIZE_PHI_ * M_PI / 180.0;
}

HybridAStar::~HybridAStar() {
    ReleaseMemory();
}

//参数初始化，设置车辆离散，节点地图指针初始化
void HybridAStar::Init(double x_lower, double x_upper, double y_lower, double y_upper,
                       double state_grid_resolution, double map_grid_resolution) {
    SetVehicleShape(1.0, 0.8, 0.5);    //此处源代码是车辆到后轴的距离，但小车是中心定位而非后轴定位

    map_x_lower_ = x_lower;
    map_x_upper_ = x_upper;
    map_y_lower_ = y_lower;
    map_y_upper_ = y_upper;
    STATE_GRID_RESOLUTION_ = state_grid_resolution;
    MAP_GRID_RESOLUTION_ = map_grid_resolution;

    STATE_GRID_SIZE_X_ = std::floor((map_x_upper_ - map_x_lower_) / STATE_GRID_RESOLUTION_);  //状态地图栅格
    STATE_GRID_SIZE_Y_ = std::floor((map_y_upper_ - map_y_lower_) / STATE_GRID_RESOLUTION_);
    // STATE_GRID_SIZE_X_ = 91;  //状态地图栅格
    // STATE_GRID_SIZE_Y_ = 45;

    MAP_GRID_SIZE_X_ = std::floor((map_x_upper_ - map_x_lower_) / MAP_GRID_RESOLUTION_);  //像素地图栅格
    MAP_GRID_SIZE_Y_ = std::floor((map_y_upper_ - map_y_lower_) / MAP_GRID_RESOLUTION_);

    // MAP_GRID_SIZE_X_ = 365;  //像素地图栅格
    // MAP_GRID_SIZE_Y_ = 181;
    std::cout<<"STATE_GRID_SIZE_X_y:"<<STATE_GRID_SIZE_X_<<","<<STATE_GRID_SIZE_Y_<<"  MAP_GRID_SIZE_X_:"<<MAP_GRID_SIZE_X_<<","<<MAP_GRID_SIZE_Y_<<std::endl;

    if (map_data_) {
        delete[] map_data_;
        map_data_ = nullptr;
    }

    map_data_ = new uint8_t[MAP_GRID_SIZE_X_ * MAP_GRID_SIZE_Y_];
    std::fill_n(map_data_, MAP_GRID_SIZE_X_ * MAP_GRID_SIZE_Y_, static_cast<uint8_t>(0));

    if (state_node_map_) {
        for (int i = 0; i < STATE_GRID_SIZE_X_; ++i) {   //判断是否为空

            if (state_node_map_[i] == nullptr)
                continue;

            for (int j = 0; j < STATE_GRID_SIZE_Y_; ++j) {
                if (state_node_map_[i][j] == nullptr)
                    continue;

                for (int k = 0; k < STATE_GRID_SIZE_PHI_; ++k) {   //第三层才是存储数据的地方
                    if (state_node_map_[i][j][k] != nullptr) {
                        delete state_node_map_[i][j][k];
                        state_node_map_[i][j][k] = nullptr;
                    }
                }
                delete[] state_node_map_[i][j];
                state_node_map_[i][j] = nullptr;
            }
            delete[] state_node_map_[i];
            state_node_map_[i] = nullptr;
        }

        delete[] state_node_map_;
        state_node_map_ = nullptr;
    }

    state_node_map_ = new StateNode::Ptr **[STATE_GRID_SIZE_X_];
    for (int i = 0; i < STATE_GRID_SIZE_X_; ++i) {
        state_node_map_[i] = new StateNode::Ptr *[STATE_GRID_SIZE_Y_];
        for (int j = 0; j < STATE_GRID_SIZE_Y_; ++j) {
            state_node_map_[i][j] = new StateNode::Ptr[STATE_GRID_SIZE_PHI_];
            for (int k = 0; k < STATE_GRID_SIZE_PHI_; ++k) {
                state_node_map_[i][j][k] = nullptr;
            }
        }
    }

}


//Bresenham直线算法 
inline bool HybridAStar::LineCheck(double x0, double y0, double x1, double y1) {  //这里的xy都是栅格点的坐标
    bool steep = (std::abs(y1 - y0) > std::abs(x1 - x0));

    if (steep) {
        std::swap(x0, y0);
        std::swap(y1, x1);
    }

    if (x0 > x1) {
        std::swap(x0, x1);
        std::swap(y0, y1);
    }

    auto delta_x = x1 - x0;
    auto delta_y = std::abs(y1 - y0);
    auto delta_error = delta_y / delta_x;
    decltype(delta_x) error = 0;   //表示与delta_x相同的数据类型
    decltype(delta_x) y_step; //表示与delta_x相同的数据类型
    auto yk = y0;

    if (y0 < y1) {
        y_step = 1;
    } else {
        y_step = -1;
    }

    auto N = static_cast<unsigned int>(x1 - x0);  //3 到 4
    // std::cout<<"x1 x0:"<< x1<<","<<x0<<" N:"<<N<<std::endl;
    for (unsigned int i = 0; i < N; ++i) {
        if (steep) {
            if (HasObstacle(Vec2i(yk, x0 + i * 1.0))
                || BeyondBoundary(Vec2d(yk * MAP_GRID_RESOLUTION_+map_x_lower_,
                                        (x0 + i) * MAP_GRID_RESOLUTION_+map_y_lower_))
                    ) {
                return false;
            }
        } else {
            if (HasObstacle(Vec2i(x0 + i * 1.0, yk))
                || BeyondBoundary(Vec2d((x0 + i) * MAP_GRID_RESOLUTION_+map_x_lower_,
                                        yk * MAP_GRID_RESOLUTION_+map_y_lower_))
                    ) {

                return false;
            }
        }

        error += delta_error;  //k斜率
        if (error >= 0.5) {
            yk += y_step;
            error = error - 1.0;
        }
    }

    return true;
}

//判断是否发生碰撞，先将车辆转移到全局坐标系
bool HybridAStar::CheckCollision(const double &x, const double &y, const double &theta) {
    Timer timer;
    if (use_simplified_collision_check_) {
        if (BeyondBoundary(Vec2d(x, y))) {
            return false;
        }
        if (HasObstacle(Coordinate2MapGridIndex(Vec2d(x, y)))) {
            return false;
        }
        check_collision_use_time += timer.End();
        num_check_collision++;
        return true;
    }

    Mat2d R;
    R << std::cos(theta), -std::sin(theta),
            std::sin(theta), std::cos(theta);

    MatXd transformed_vehicle_shape;
    transformed_vehicle_shape.resize(8, 1);
    for (unsigned int i = 0; i < 4u; ++i) {
        transformed_vehicle_shape.block<2, 1>(i * 2, 0)
                = R * vehicle_shape_.block<2, 1>(i * 2, 0) + Vec2d(x, y);
    }

    Vec2i transformed_pt_index_0 = Coordinate2MapGridIndex(
            transformed_vehicle_shape.block<2, 1>(0, 0)
    );
    Vec2i transformed_pt_index_1 = Coordinate2MapGridIndex(
            transformed_vehicle_shape.block<2, 1>(2, 0)
    );

    Vec2i transformed_pt_index_2 = Coordinate2MapGridIndex(
            transformed_vehicle_shape.block<2, 1>(4, 0)
    );

    Vec2i transformed_pt_index_3 = Coordinate2MapGridIndex(
            transformed_vehicle_shape.block<2, 1>(6, 0)
    );

    double y1, y0, x1, x0;
    // pt1 -> pt0
    x0 = static_cast<double>(transformed_pt_index_0.x());
    y0 = static_cast<double>(transformed_pt_index_0.y());
    x1 = static_cast<double>(transformed_pt_index_1.x());
    y1 = static_cast<double>(transformed_pt_index_1.y());

    if (!LineCheck(x1, y1, x0, y0)) {
        return false;
    }

    // pt2 -> pt1
    x0 = static_cast<double>(transformed_pt_index_1.x());
    y0 = static_cast<double>(transformed_pt_index_1.y());
    x1 = static_cast<double>(transformed_pt_index_2.x());
    y1 = static_cast<double>(transformed_pt_index_2.y());

    if (!LineCheck(x1, y1, x0, y0)) {
        return false;
    }

    // pt3 -> pt2
    x0 = static_cast<double>(transformed_pt_index_2.x());
    y0 = static_cast<double>(transformed_pt_index_2.y());
    x1 = static_cast<double>(transformed_pt_index_3.x());
    y1 = static_cast<double>(transformed_pt_index_3.y());

    if (!LineCheck(x1, y1, x0, y0)) {
        return false;
    }

    // pt0 -> pt3
    x0 = static_cast<double>(transformed_pt_index_0.x());  //将int转化为double  小数位后添加.0
    y0 = static_cast<double>(transformed_pt_index_0.y());
    x1 = static_cast<double>(transformed_pt_index_3.x());
    y1 = static_cast<double>(transformed_pt_index_3.y());

    if (!LineCheck(x0, y0, x1, y1)) {
        return false;
    }

    check_collision_use_time += timer.End();
    num_check_collision++;
    return true;
}

bool HybridAStar::HasObstacle(const int grid_index_x, const int grid_index_y) const {
    return (grid_index_x >= 0 && grid_index_x < MAP_GRID_SIZE_X_
            && grid_index_y >= 0 && grid_index_y < MAP_GRID_SIZE_Y_
            && (map_data_[grid_index_y * MAP_GRID_SIZE_X_ + grid_index_x] == 1));
}

bool HybridAStar::HasObstacle(const Vec2i &grid_index) const {
    int grid_index_x = grid_index[0];
    int grid_index_y = grid_index[1];

    return (grid_index_x >= 0 && grid_index_x < MAP_GRID_SIZE_X_
            && grid_index_y >= 0 && grid_index_y < MAP_GRID_SIZE_Y_
            && (map_data_[grid_index_y * MAP_GRID_SIZE_X_ + grid_index_x] == 1));
}

//设置障碍物
void HybridAStar::SetObstacle(unsigned int x, unsigned int y) {
    if (x > static_cast<unsigned int>(MAP_GRID_SIZE_X_)
        || y > static_cast<unsigned int>(MAP_GRID_SIZE_Y_)) {
        return;
    }
    map_data_[x + y * MAP_GRID_SIZE_X_] = 1;
}

void HybridAStar::SetObstacle(const double pt_x, const double pt_y) {

    if (pt_x < map_x_lower_ || pt_x > map_x_upper_ ||
        pt_y < map_y_lower_ || pt_y > map_y_upper_) {
        return;
    }

    int grid_index_x = static_cast<int>((pt_x - map_x_lower_) / MAP_GRID_RESOLUTION_);
    int grid_index_y = static_cast<int>((pt_y - map_y_lower_) / MAP_GRID_RESOLUTION_);

    map_data_[grid_index_x + grid_index_y * MAP_GRID_SIZE_X_] = 1;
}

//离散化求出车辆轮廓点，采用向量
void HybridAStar::SetVehicleShape(double length, double width, double rear_axle_dist) { //rear_axle_dist 
    vehicle_shape_.resize(8);
    vehicle_shape_.block<2, 1>(0, 0) = Vec2d(-rear_axle_dist, width / 2);           //右后
    vehicle_shape_.block<2, 1>(2, 0) = Vec2d(length - rear_axle_dist, width / 2);   //右前
    vehicle_shape_.block<2, 1>(4, 0) = Vec2d(length - rear_axle_dist, -width / 2);  //左前
    vehicle_shape_.block<2, 1>(6, 0) = Vec2d(-rear_axle_dist, -width / 2);          //左后

    const double step_size = move_step_size_;
    const auto N_length = static_cast<unsigned int>(length / step_size);  //1.0/0.2=5
    const auto N_width = static_cast<unsigned int> (width / step_size);   //0.8/0.2=4
    vehicle_shape_discrete_.resize(2, (N_length + N_width) * 2u);         //2*18

    const Vec2d edge_0_normalized = (vehicle_shape_.block<2, 1>(2, 0)
                                     - vehicle_shape_.block<2, 1>(0, 0)).normalized();  //车长方向单位向量
    for (unsigned int i = 0; i < N_length; ++i) {
        vehicle_shape_discrete_.block<2, 1>(0, i + N_length)
                = vehicle_shape_.block<2, 1>(4, 0) - edge_0_normalized * i * step_size;
        vehicle_shape_discrete_.block<2, 1>(0, i)
                = vehicle_shape_.block<2, 1>(0, 0) + edge_0_normalized * i * step_size;
    }

    const Vec2d edge_1_normalized = (vehicle_shape_.block<2, 1>(4, 0)
                                     - vehicle_shape_.block<2, 1>(2, 0)).normalized();  //车宽方向单位向量
    for (unsigned int i = 0; i < N_width; ++i) {
        vehicle_shape_discrete_.block<2, 1>(0, (2 * N_length) + i)
                = vehicle_shape_.block<2, 1>(2, 0) + edge_1_normalized * i * step_size;
        vehicle_shape_discrete_.block<2, 1>(0, (2 * N_length) + i + N_width)
                = vehicle_shape_.block<2, 1>(6, 0) - edge_1_normalized * i * step_size;
    }
}

__attribute__((unused)) Vec2d HybridAStar::CoordinateRounding(const Vec2d &pt) const {
    return MapGridIndex2Coordinate(Coordinate2MapGridIndex(pt));
}

Vec2d HybridAStar::MapGridIndex2Coordinate(const Vec2i &grid_index) const {
    Vec2d pt;
    pt.x() = ((double) grid_index[0] + 0.5) * MAP_GRID_RESOLUTION_ + map_x_lower_;
    pt.y() = ((double) grid_index[1] + 0.5) * MAP_GRID_RESOLUTION_ + map_y_lower_;

    return pt;
}

//
Vec3i HybridAStar::State2Index(const Vec3d &state) const {
    Vec3i index;

    index[0] = std::min(std::max(int((state[0] - map_x_lower_) / STATE_GRID_RESOLUTION_), 0), STATE_GRID_SIZE_X_ - 1);
    index[1] = std::min(std::max(int((state[1] - map_y_lower_) / STATE_GRID_RESOLUTION_), 0), STATE_GRID_SIZE_Y_ - 1);
    index[2] = std::min(std::max(int((state[2] - (-M_PI)) / ANGULAR_RESOLUTION_), 0), STATE_GRID_SIZE_PHI_ - 1);

    return index;
}

Vec3i HybridAStar::State2Index(const Vec4d &state) const {
    Vec3i index;

    index[0] = std::min(std::max(int((state[0] - map_x_lower_) / STATE_GRID_RESOLUTION_), 0), STATE_GRID_SIZE_X_ - 1);
    index[1] = std::min(std::max(int((state[1] - map_y_lower_) / STATE_GRID_RESOLUTION_), 0), STATE_GRID_SIZE_Y_ - 1);
    index[2] = std::min(std::max(int((state[2] - (-M_PI)) / ANGULAR_RESOLUTION_), 0), STATE_GRID_SIZE_PHI_ - 1);

    return index;
}

Vec2i HybridAStar::Coordinate2MapGridIndex(const Vec2d &pt) const {
    Vec2i grid_index;

    grid_index[0] = int((pt[0] - map_x_lower_) / MAP_GRID_RESOLUTION_);
    grid_index[1] = int((pt[1] - map_y_lower_) / MAP_GRID_RESOLUTION_);
    return grid_index;
}

//主要计算先是前向计算角度搜索，再离散计算车辆轨迹，加入邻居节点；后向同理
void HybridAStar::GetNeighborNodes(const StateNode::Ptr &curr_node_ptr,
                                   std::vector<StateNode::Ptr> &neighbor_nodes) {
    neighbor_nodes.clear();

    //角度分支搜索 比如 -3 至 3 共 6 个
    for (int i = -steering_discrete_num_; i <= steering_discrete_num_; ++i) {
        VectorVec4d intermediate_state;
        bool has_obstacle = false;

        double x = curr_node_ptr->state_.x();
        double y = curr_node_ptr->state_.y();
        double theta = curr_node_ptr->state_.z();

        const double phi = i * steering_radian_step_size_;  

        // forward
        //此循环是为了离散化车辆运动学采样，越小则越准确，但时间消耗较大
        for (int j = 1; j <= segment_length_discrete_num_; j++) {
            DynamicModel(move_step_size_, phi, x, y, theta);  //车辆运动学模型
            intermediate_state.emplace_back(Vec4d(x, y, theta,1));

            if (!CheckCollision(x, y, theta)) {
                has_obstacle = true;
                break;
            }
        }

        Vec3i grid_index = State2Index(intermediate_state.back());
        if (!BeyondBoundary(intermediate_state.back().head(2)) && !has_obstacle) {
            auto neighbor_forward_node_ptr = new StateNode(grid_index);
            neighbor_forward_node_ptr->intermediate_states_ = intermediate_state;
            neighbor_forward_node_ptr->state_ = intermediate_state.back();
            neighbor_forward_node_ptr->steering_grade_ = i;
            neighbor_forward_node_ptr->direction_ = StateNode::FORWARD;
            neighbor_nodes.push_back(neighbor_forward_node_ptr);
        }

        // backward
        //此循环是为了离散化车辆运动学采样，越小则越准确，但时间消耗较大
        has_obstacle = false;
        intermediate_state.clear();
        x = curr_node_ptr->state_.x();
        y = curr_node_ptr->state_.y();
        theta = curr_node_ptr->state_.z();
        for (int j = 1; j <= segment_length_discrete_num_; j++) {
            DynamicModel(-move_step_size_, phi, x, y, theta);
            intermediate_state.emplace_back(Vec4d(x, y, theta,-1));

            if (!CheckCollision(x, y, theta)) {
                has_obstacle = true;
                break;
            }
        }

        if (!BeyondBoundary(intermediate_state.back().head(2)) && !has_obstacle) {
            grid_index = State2Index(intermediate_state.back());
            auto neighbor_backward_node_ptr = new StateNode(grid_index);
            neighbor_backward_node_ptr->intermediate_states_ = intermediate_state;
            neighbor_backward_node_ptr->state_ = intermediate_state.back();
            neighbor_backward_node_ptr->steering_grade_ = i;
            neighbor_backward_node_ptr->direction_ = StateNode::BACKWARD;
            neighbor_nodes.push_back(neighbor_backward_node_ptr);
        }
    }
}

//运动学模型
void HybridAStar::DynamicModel(const double &step_size, const double &phi,
                               double &x, double &y, double &theta) const {
    x = x + step_size * std::cos(theta);
    y = y + step_size * std::sin(theta);
    theta = Mod2Pi(theta + step_size / wheel_base_ * std::tan(phi));  //角度变换，theta+变化量
}

double HybridAStar::Mod2Pi(const double &x) {
    double v = fmod(x, 2 * M_PI);

    if (v < -M_PI) {
        v += 2.0 * M_PI;
    } else if (v > M_PI) {
        v -= 2.0 * M_PI;
    }

    return v;
}

bool HybridAStar::BeyondBoundary(const Vec2d &pt) const {
    return pt.x() < map_x_lower_ || pt.x() > map_x_upper_ || pt.y() < map_y_lower_ || pt.y() > map_y_upper_;
}

double HybridAStar::ComputeH(const StateNode::Ptr &current_node_ptr,
                             const StateNode::Ptr &terminal_node_ptr) {
    double h;
    // L2
    h = (current_node_ptr->state_.head(2) - terminal_node_ptr->state_.head(2)).norm();  //head(2)是指前两位  .norm二范数

    // L1
    // h = (current_node_ptr->state_.head(2) - terminal_node_ptr->state_.head(2)).lpNorm<1>();   //.lpnorm 一范数

    if (h < 3.0 * shot_distance_) {
        h = rs_path_ptr_->Distance(current_node_ptr->state_.x(), current_node_ptr->state_.y(),
                                   current_node_ptr->state_.z(),
                                   terminal_node_ptr->state_.x(), terminal_node_ptr->state_.y(),
                                   terminal_node_ptr->state_.z());
    }

    return h;
}

//计算路径的G值，
double HybridAStar::ComputeG(const StateNode::Ptr &current_node_ptr,
                             const StateNode::Ptr &neighbor_node_ptr) const {
    double g;
    if (neighbor_node_ptr->direction_ == StateNode::FORWARD) {  //前向
        if (neighbor_node_ptr->steering_grade_ != current_node_ptr->steering_grade_) {
            if (neighbor_node_ptr->steering_grade_ == 0) {
                g = segment_length_ * steering_change_penalty_;
            } else {
                g = segment_length_ * steering_change_penalty_ * steering_penalty_;
            }
        } else {
            if (neighbor_node_ptr->steering_grade_ == 0) {
                g = segment_length_;
            } else {
                g = segment_length_ * steering_penalty_;
            }
        }
    } else {
        if (neighbor_node_ptr->steering_grade_ != current_node_ptr->steering_grade_) {
            if (neighbor_node_ptr->steering_grade_ == 0) {
                g = segment_length_ * steering_change_penalty_ * reversing_penalty_;
            } else {
                g = segment_length_ * steering_change_penalty_ * steering_penalty_ * reversing_penalty_;
            }
        } else {
            if (neighbor_node_ptr->steering_grade_ == 0) {
                g = segment_length_ * reversing_penalty_;
            } else {
                g = segment_length_ * steering_penalty_ * reversing_penalty_;
            }
        }
    }

    return g;
}

//搜索路径
bool HybridAStar::Search(const Vec4d &start_state, const Vec4d &goal_state) {
    Timer search_used_time;  //计时开始

    double neighbor_time = 0.0, compute_h_time = 0.0, compute_g_time = 0.0;

    const Vec3i start_grid_index = State2Index(start_state);
    const Vec3i goal_grid_index = State2Index(goal_state);
    if (!CheckCollision(goal_state[0],goal_state[1],goal_state[2])
        || BeyondBoundary(Vec2d(goal_state(0),goal_state(1)))
            ) {
        std::cout<<"错误，终点设置有误！！！"<<std::endl;
        return false;
    }

    auto goal_node_ptr = new StateNode(goal_grid_index);
    goal_node_ptr->state_ = goal_state;
    goal_node_ptr->direction_ = StateNode::NO;
    goal_node_ptr->steering_grade_ = 0;

    auto start_node_ptr = new StateNode(start_grid_index);
    start_node_ptr->state_ = start_state;
    start_node_ptr->steering_grade_ = 0;
    start_node_ptr->direction_ = StateNode::NO;
    start_node_ptr->node_status_ = StateNode::IN_OPENSET;
    Vec4d start_state_4v;
    start_state_4v << start_state,0.0;
    start_node_ptr->intermediate_states_.emplace_back(start_state_4v);  //中间节点，暂时不清楚原理
    start_node_ptr->g_cost_ = 0.0;
    start_node_ptr->f_cost_ = ComputeH(start_node_ptr, goal_node_ptr);

    state_node_map_[start_grid_index.x()][start_grid_index.y()][start_grid_index.z()] = start_node_ptr;  //写入开始目标点的节点
    state_node_map_[goal_grid_index.x()][goal_grid_index.y()][goal_grid_index.z()] = goal_node_ptr;

    openset_.clear();
    openset_.insert(std::make_pair(0, start_node_ptr)); 

    std::vector<StateNode::Ptr> neighbor_nodes_ptr;
    StateNode::Ptr current_node_ptr;
    StateNode::Ptr neighbor_node_ptr;

    int count = 0;
    int rscount =0;
    double time_total=0.0;
    while (!openset_.empty()) {
        std::cout<<"openlist_start!!!"<<std::endl;
        current_node_ptr = openset_.begin()->second; 
        current_node_ptr->node_status_ = StateNode::IN_CLOSESET; 
        openset_.erase(openset_.begin());  //移除第一个点，加入close列表
        double DistanceToEnd=(current_node_ptr->state_.head(2) - goal_node_ptr->state_.head(2)).norm();
        double rs_shot_p;
        double random_value = dis(gen);  //随机数
        if (DistanceToEnd <= min_distance_rs_shot) 
            rs_shot_p = 1.0;
        else if (DistanceToEnd > min_distance_rs_shot && DistanceToEnd < max_distance_rs_shot) 
        {
            double normalized_distance = (DistanceToEnd - min_distance_rs_shot) / (max_distance_rs_shot - min_distance_rs_shot);
            // 归一化的指数衰减
            double exp_min = exp(-lambda);
            rs_shot_p = min_probability_rs_shot + 
                    (max_probability_rs_shot - min_probability_rs_shot) * 
                    (exp(-lambda * normalized_distance) - exp_min) / (1 - exp_min); 
        }
        else 
            rs_shot_p = 0.02;
        
        // std::cout<<"random_value:"<<random_value<<" rs_shot_p:"<<rs_shot_p<<" DistanceToEnd:"<<DistanceToEnd<<std::endl;
        if ( random_value<= rs_shot_p) {   //距离小于shot 则尝试使用rs链接
        // if ( DistanceToEnd<= 1) {
            std::cout<<"rs_连接!!!"<<std::endl;
            clock_t time_start = clock();

            rscount++;
            double rs_length = 0.0;
            if (AnalyticExpansions(current_node_ptr, goal_node_ptr, rs_length)) {
                terminal_node_ptr_ = goal_node_ptr;

                StateNode::Ptr grid_node_ptr = terminal_node_ptr_->parent_node_;
                while (grid_node_ptr != nullptr) {
                    grid_node_ptr = grid_node_ptr->parent_node_;
                    path_length_ = path_length_ + segment_length_;
                }
                path_length_ = path_length_ - segment_length_ + rs_length;

                std::cout << "ComputeH use time(ms): " << compute_h_time << std::endl;
                std::cout << "check collision use time(ms): " << check_collision_use_time << std::endl;
                std::cout << "GetNeighborNodes use time(ms): " << neighbor_time << std::endl;
                std::cout << "average time of check collision(ms): "
                          << check_collision_use_time / num_check_collision
                          << std::endl;
                ROS_INFO("\033[1;32m --> Time in Hybrid A star is %f ms, path length: %f  \033[0m\n",
                         search_used_time.End(), path_length_);

                check_collision_use_time = 0.0;
                num_check_collision = 0.0;
                return true;
            }
                clock_t time_end2 = clock();
                double time_diff2 = static_cast<double>(time_end2 - time_start) / CLOCKS_PER_SEC;
                time_total+=time_diff2;
            //  std::cout<<"DistanceToEnd："<<DistanceToEnd<<" 尝试rs连接次数："<<rscount<<" 耗时ms："<<time_diff2<<"/"<<time_total<<std::endl;
        }

        Timer timer_get_neighbor;
        GetNeighborNodes(current_node_ptr, neighbor_nodes_ptr);
        neighbor_time = neighbor_time + timer_get_neighbor.End();
        std::cout<<"得到neighbor!!!"<<std::endl;
        for (unsigned int i = 0; i < neighbor_nodes_ptr.size(); ++i) {
            neighbor_node_ptr = neighbor_nodes_ptr[i];

            //计算G值，仅仅计算长度，加上一些转向和倒车的惩罚
            Timer timer_compute_g;
            const double neighbor_edge_cost = ComputeG(current_node_ptr, neighbor_node_ptr);
            compute_g_time = compute_g_time + timer_get_neighbor.End();

            Timer timer_compute_h;
            //计算H值，使用的是二范数,后期可以改进
            const double current_h = ComputeH(current_node_ptr, goal_node_ptr) * tie_breaker_;
            compute_h_time = compute_h_time + timer_compute_h.End();

            const Vec3i &index = neighbor_node_ptr->grid_index_;
            if (state_node_map_[index.x()][index.y()][index.z()] == nullptr) {  //未探索节点，加入list中
                neighbor_node_ptr->g_cost_ = current_node_ptr->g_cost_ + neighbor_edge_cost;
                neighbor_node_ptr->parent_node_ = current_node_ptr;
                neighbor_node_ptr->node_status_ = StateNode::IN_OPENSET;
                neighbor_node_ptr->f_cost_ = neighbor_node_ptr->g_cost_ + current_h;
                openset_.insert(std::make_pair(neighbor_node_ptr->f_cost_, neighbor_node_ptr));
                state_node_map_[index.x()][index.y()][index.z()] = neighbor_node_ptr;
                continue;
            } else if (state_node_map_[index.x()][index.y()][index.z()]->node_status_ == StateNode::IN_OPENSET) {  //开放列表节点，若代价较小则替换，反之则删除
                double g_cost_temp = current_node_ptr->g_cost_ + neighbor_edge_cost;

                if (state_node_map_[index.x()][index.y()][index.z()]->g_cost_ > g_cost_temp) {
                    neighbor_node_ptr->g_cost_ = g_cost_temp;
                    neighbor_node_ptr->f_cost_ = g_cost_temp + current_h;
                    neighbor_node_ptr->parent_node_ = current_node_ptr;
                    neighbor_node_ptr->node_status_ = StateNode::IN_OPENSET;

                    /// TODO: This will cause a memory leak
                    //delete state_node_map_[index.x()][index.y()][index.z()];
                    state_node_map_[index.x()][index.y()][index.z()] = neighbor_node_ptr;
                } else {
                    delete neighbor_node_ptr;
                }
                continue;
            } else if (state_node_map_[index.x()][index.y()][index.z()]->node_status_ == StateNode::IN_CLOSESET) {  //在关闭列表节点，则直接删除
                delete neighbor_node_ptr;
                continue;
            }
        }

        count++;
        if (count > 100000) {  //最大迭代次数，超过最大迭代次数直接结束
            ROS_WARN("Exceeded the number of iterations, the search failed");
            return false;
        }
    }

    return false;
}

VectorVec4d HybridAStar::GetSearchedTree() {
    VectorVec4d tree;
    Vec4d point_pair;

    visited_node_number_ = 0;
    for (int i = 0; i < STATE_GRID_SIZE_X_; ++i) {
        for (int j = 0; j < STATE_GRID_SIZE_Y_; ++j) {
            for (int k = 0; k < STATE_GRID_SIZE_PHI_; ++k) {
                if (state_node_map_[i][j][k] == nullptr || state_node_map_[i][j][k]->parent_node_ == nullptr) {
                    continue;
                }

                const unsigned int number_states = state_node_map_[i][j][k]->intermediate_states_.size() - 1;
                for (unsigned int l = 0; l < number_states; ++l) {
                    point_pair.head(2) = state_node_map_[i][j][k]->intermediate_states_[l].head(2);
                    point_pair.tail(2) = state_node_map_[i][j][k]->intermediate_states_[l + 1].head(2);

                    tree.emplace_back(point_pair);
                }

                point_pair.head(2) = state_node_map_[i][j][k]->intermediate_states_[0].head(2);
                point_pair.tail(2) = state_node_map_[i][j][k]->parent_node_->state_.head(2);
                tree.emplace_back(point_pair);
                visited_node_number_++;
            }
        }
    }

    return tree;
}

void HybridAStar::ReleaseMemory() {
    if (map_data_ != nullptr) {
        delete[] map_data_;
        map_data_ = nullptr;
    }

    if (state_node_map_) {
        for (int i = 0; i < STATE_GRID_SIZE_X_; ++i) {
            if (state_node_map_[i] == nullptr)
                continue;

            for (int j = 0; j < STATE_GRID_SIZE_Y_; ++j) {
                if (state_node_map_[i][j] == nullptr)
                    continue;

                for (int k = 0; k < STATE_GRID_SIZE_PHI_; ++k) {
                    if (state_node_map_[i][j][k] != nullptr) {
                        delete state_node_map_[i][j][k];
                        state_node_map_[i][j][k] = nullptr;
                    }
                }

                delete[] state_node_map_[i][j];
                state_node_map_[i][j] = nullptr;
            }

            delete[] state_node_map_[i];
            state_node_map_[i] = nullptr;
        }

        delete[] state_node_map_;
        state_node_map_ = nullptr;
    }

    terminal_node_ptr_ = nullptr;
}

__attribute__((unused)) double HybridAStar::GetPathLength() const {
    return path_length_;
}

VectorVec4d HybridAStar::GetPath(VectorVec4d &path_original)  {

    std::vector<StateNode::Ptr> temp_nodes;

    StateNode::Ptr state_grid_node_ptr = terminal_node_ptr_;
    while (state_grid_node_ptr != nullptr) {  //回溯节点
        temp_nodes.emplace_back(state_grid_node_ptr);
        state_grid_node_ptr = state_grid_node_ptr->parent_node_;
    }

    std::reverse(temp_nodes.begin(), temp_nodes.end());

    for (const auto &node: temp_nodes) {  //插入中间节点
        pcl::PointXYZI node_point_watch;
        node_point_watch.x=node->state_.x();
        node_point_watch.y=node->state_.y();
        // std::cout<<"node_x_y:"<<node_point_watch.x<<","<<node_point_watch.y<<","<<node->state_.z()<<","<<node->state_.w()<<std::endl;
        node_points_watch->points.push_back(node_point_watch);

        path_original.insert(path_original.end(), node->intermediate_states_.begin(),
                    node->intermediate_states_.end());
    }
    if (path_original.size()>1)  //第一个点方向赋值
    {
        path_original[0].w()=path_original[1].w();
    }

    VectorVec4d path_smoothed=Smooth(path_original); //后端平滑优化

    return path_smoothed;
    // return path_original;
}

VectorVec4d HybridAStar::SmoothPath(VectorVec4d path_raw) {
    if (path_raw.size() < 2) {
        return path_raw;
    }

    for (auto &point : path_raw) {
        if (std::abs(point[3]) < 1e-6) {
            point[3] = 1.0;
        }
    }
    GetPathYaw(path_raw);
    auto path_is_valid = [&](const VectorVec4d &path) {
        if (path.size() < 2) {
            return false;
        }
        for (const auto &pose : path) {
            if (!IsFinitePose(pose)) {
                return false;
            }
            if (std::abs(pose.x()) > 1e6 || std::abs(pose.y()) > 1e6) {
                return false;
            }
            if (BeyondBoundary(pose.head(2))) {
                return false;
            }
        }
        return true;
    };

    auto segment_collision_free = [&](const Vec4d &a, const Vec4d &b) {
        const double dx = b.x() - a.x();
        const double dy = b.y() - a.y();
        const double heading = std::atan2(dy, dx);
        const double distance = std::hypot(dx, dy);
        const int steps = std::max(2, static_cast<int>(std::ceil(distance / std::max(0.1, MAP_GRID_RESOLUTION_ * 0.5))));
        for (int i = 0; i <= steps; ++i) {
            const double ratio = static_cast<double>(i) / static_cast<double>(steps);
            const double x = a.x() + dx * ratio;
            const double y = a.y() + dy * ratio;
            if (BeyondBoundary(Vec2d(x, y)) || !CheckCollision(x, y, heading)) {
                return false;
            }
        }
        return true;
    };

    auto build_shortcut_path = [&](const VectorVec4d &input_path) {
        VectorVec4d anchors;
        anchors.reserve(input_path.size());
        anchors.push_back(input_path.front());
        std::size_t index = 0;
        while (index + 1 < input_path.size()) {
            std::size_t best = index + 1;
            for (std::size_t candidate = input_path.size() - 1; candidate > index + 1; --candidate) {
                if (segment_collision_free(input_path[index], input_path[candidate])) {
                    best = candidate;
                    break;
                }
            }
            anchors.push_back(input_path[best]);
            index = best;
        }

        VectorVec4d densified;
        densified.reserve(input_path.size());
        const double step = std::max(0.15, MAP_GRID_RESOLUTION_);
        for (std::size_t i = 0; i + 1 < anchors.size(); ++i) {
            const auto &a = anchors[i];
            const auto &b = anchors[i + 1];
            const double dx = b.x() - a.x();
            const double dy = b.y() - a.y();
            const double distance = std::hypot(dx, dy);
            const int segments = std::max(1, static_cast<int>(std::ceil(distance / step)));
            for (int seg = 0; seg < segments; ++seg) {
                const double ratio = static_cast<double>(seg) / static_cast<double>(segments);
                Vec4d pose = Vec4d::Zero();
                pose.x() = a.x() + dx * ratio;
                pose.y() = a.y() + dy * ratio;
                pose.w() = input_path.front().w();
                densified.emplace_back(pose);
            }
        }
        Vec4d tail = anchors.back();
        tail.w() = input_path.front().w();
        densified.emplace_back(tail);
        GetPathYaw(densified);
        return densified;
    };

    VectorVec4d path_smoothed = Smooth(path_raw);
    if (path_is_valid(path_smoothed)) {
        return path_smoothed;
    }

    LOG(WARNING) << "HybridAStar backend smoother produced invalid output, falling back to shortcut smoothing.";
    VectorVec4d shortcut_path = build_shortcut_path(path_raw);
    if (path_is_valid(shortcut_path)) {
        return shortcut_path;
    }
    return path_raw;
}

VectorVec4d HybridAStar::GetSmoothSegmentSplitPoints(VectorVec4d path_raw) {
    if (path_raw.empty()) {
        return {};
    }
    for (auto& point : path_raw) {
        if (std::abs(point[3]) < 1e-6) {
            point[3] = 1.0;
        }
    }
    GetPathYaw(path_raw);

    VectorVec4d split_points;
    std::vector<VectorVec4d> direction_segments = PathSegmentsByDirection(path_raw);
    for (const auto& direction_segment : direction_segments) {
        if (direction_segment.empty()) {
            continue;
        }
        split_points.push_back(direction_segment.front());
        const std::vector<int> split_indices = FindGeometrySplitIndices(direction_segment);
        for (const int split_idx : split_indices) {
            if (split_idx > 0 && split_idx < static_cast<int>(direction_segment.size()) - 1) {
                split_points.push_back(direction_segment[split_idx]);
            }
        }
    }
    return split_points;
}

void HybridAStar::Reset() {
    if (state_node_map_) {
        for (int i = 0; i < STATE_GRID_SIZE_X_; ++i) {
            if (state_node_map_[i] == nullptr)
                continue;

            for (int j = 0; j < STATE_GRID_SIZE_Y_; ++j) {
                if (state_node_map_[i][j] == nullptr)
                    continue;

                for (int k = 0; k < STATE_GRID_SIZE_PHI_; ++k) {
                    if (state_node_map_[i][j][k] != nullptr) {
                        delete state_node_map_[i][j][k];
                        state_node_map_[i][j][k] = nullptr;
                    }
                }
            }
        }
    }

    path_length_ = 0.0;
    terminal_node_ptr_ = nullptr;
}

//判断路径是否超越范围及碰撞，使用RS曲线链接，保存路径
bool HybridAStar::AnalyticExpansions(const StateNode::Ptr &current_node_ptr,
                                     const StateNode::Ptr &goal_node_ptr, double &length) {
    VectorVec4d rs_path_poses = rs_path_ptr_->GetRSPath(current_node_ptr->state_,
                                                        goal_node_ptr->state_,
                                                        move_step_size_, length);

    for (const auto &pose: rs_path_poses)
        if (BeyondBoundary(pose.head(2)) || !CheckCollision(pose.x(), pose.y(), pose.z())) {
            return false;
        };

    goal_node_ptr->intermediate_states_ = rs_path_poses;
    
    goal_node_ptr->parent_node_ = current_node_ptr;

    auto begin = goal_node_ptr->intermediate_states_.begin();
    goal_node_ptr->intermediate_states_.erase(begin);

    return true;
}


VectorVec4d HybridAStar::Smooth(VectorVec4d &path)
{
    clock_t time_start = clock();
    struct SegmentInfo {
        VectorVec4d segment;
        int expand_start = 0;
        int expand_end = 0;
    };

    std::vector<VectorVec4d> RawPath;
    std::vector<SegmentInfo> segment_infos;
    std::vector<VectorVec4d> direction_segments = PathSegmentsByDirection(path);
    int global_offset = 0;
    const int overlap_points = 3;
    const int shared_tail_points = overlap_points;
    for (const auto& direction_segment : direction_segments) {
        const int seg_size = static_cast<int>(direction_segment.size());
        if (seg_size == 0) {
            continue;
        }

        std::vector<int> split_indices = FindGeometrySplitIndices(direction_segment);
        std::vector<int> boundaries;
        boundaries.reserve(split_indices.size() + 2);
        boundaries.push_back(0);
        for (const int idx : split_indices) {
            if (idx > 0 && idx < seg_size - 1) {
                boundaries.push_back(idx);
            }
        }
        boundaries.push_back(seg_size - 1);

        for (size_t seg_idx = 0; seg_idx + 1 < boundaries.size(); ++seg_idx) {
            const int base_start = boundaries[seg_idx];
            const int base_end = boundaries[seg_idx + 1];
            if (base_end - base_start < 1) {
                continue;
            }
            const int expand_start =
                std::max(0, base_start - (seg_idx > 0 ? (shared_tail_points - 1) : 0));
            const int expand_end = base_end;
            SegmentInfo info;
            info.segment = VectorVec4d(direction_segment.begin() + expand_start,
                                       direction_segment.begin() + expand_end + 1);
            info.expand_start = global_offset + expand_start;
            info.expand_end = global_offset + expand_end;
            if (info.segment.size() >= 2) {
                segment_infos.push_back(info);
                RawPath.push_back(info.segment);
            }
        }
        global_offset += seg_size;
    }
    std::vector<VectorVec4d> smoothed_paths;  //总路径
    int solver_num=0;  //求解总次数
    std::cout<<"开始优化!!"<<std::endl;
    std::cout<<"分段数量: "<<RawPath.size()<<std::endl;
    for (size_t pathNumber = 0;pathNumber < RawPath.size(); pathNumber++)
    {
        if (RawPath[pathNumber].size()<6)   //若路径点不足六个，则不优化路线
        {
            for (size_t i = 0; i < RawPath[pathNumber].size(); i++)
            {
                smoothed_paths.push_back(RawPath[pathNumber]);
            }
            std::cout<<"小于6个，直接返回"<<std::endl;
            continue;
        }
        VectorVec4d opt_path=RawPath[pathNumber];  //待优化路线

        bool no_collsion=false;
        VectorVec4d smoothed_path;  //分段路径

        const int PathPointsNum=RawPath[pathNumber].size();
        std::vector<double> path_segment_lengths;
        path_segment_lengths.reserve(std::max(0, PathPointsNum - 1));
        for (int i = 0; i + 1 < PathPointsNum; ++i) {
            path_segment_lengths.push_back((RawPath[pathNumber][i + 1].head(2) - RawPath[pathNumber][i].head(2)).norm());
        }
        double nominal_interval_length = 0.0;
        if (path_segment_lengths.size() > 1) {
            for (std::size_t seg_idx = 0; seg_idx + 1 < path_segment_lengths.size(); ++seg_idx) {
                nominal_interval_length += path_segment_lengths[seg_idx];
            }
            nominal_interval_length /= static_cast<double>(path_segment_lengths.size() - 1);
        } else if (!path_segment_lengths.empty()) {
            nominal_interval_length = path_segment_lengths.front();
        }
        const bool has_short_tail =
            path_segment_lengths.size() >= 2 &&
            nominal_interval_length > 1e-6 &&
            path_segment_lengths.back() < 0.8 * nominal_interval_length;
        const int tail_relaxed_curvature_constraints =
            has_short_tail ? std::min(2, PathPointsNum - 2) : 0;
        const VectorVec4d* prev_smoothed_segment = nullptr;
        int overlap_count = 0;
        int hard_bind_points = 0;
        int prev_overlap_start = 0;
        if (pathNumber > 0 && !smoothed_paths.empty()) {
            prev_smoothed_segment = &smoothed_paths.back();
            overlap_count =
                std::max(0, segment_infos[pathNumber - 1].expand_end - segment_infos[pathNumber].expand_start + 1);
            overlap_count = std::min(overlap_count,
                                     std::min(PathPointsNum, static_cast<int>(prev_smoothed_segment->size())));
            hard_bind_points = std::min(2, overlap_count);
            prev_overlap_start = static_cast<int>(prev_smoothed_segment->size()) - overlap_count;
        }
        const int num_of_pos_variables_ = PathPointsNum * 2;                            //2*n
        const int num_of_slack_variables_ = PathPointsNum - 2;                          //n-2
        const int num_of_variables_ = num_of_pos_variables_ + num_of_slack_variables_;  //3n-2

        const int num_of_variable_constraints_ = num_of_variables_;   //3n-2
        const int num_of_curvature_constraints_ = PathPointsNum - 2;  //n-2
        const int num_of_constraints_expectStartEnd =num_of_variable_constraints_ + num_of_curvature_constraints_ ; //4n-4
        // Keep endpoint position locks as hard constraints. For later segments,
        // the first few overlap points are hard-bound to the previous optimized
        // segment, and tangent / second-difference continuity terms regularize
        // the local connection without any post-hoc seam interpolation.
        const int endpoint_lock_points = 1;
        const int endpoint_constraint_half = endpoint_lock_points * 2;
        const int endpoint_constraint_count = endpoint_constraint_half * 2;
        const int num_of_constraints_ = num_of_constraints_expectStartEnd + endpoint_constraint_count;
        const double w_smooth = 3e6;
        const double w_length = 10;
        const double w_ref = 1;
        const double w_endpoint_heading = fix_endpoint_heading_ ? 5e4 : 0.0;
        const double w_endpoint_curvature = fix_endpoint_heading_ ? 1.5e5 : 0.0;
        const double w_connection_tangent = 2e4;
        const double w_connection_curvature = 1e5;

                //ref初始化
        Eigen::VectorXd referenceline   = Eigen::VectorXd::Zero(2*PathPointsNum);  //2n
        Eigen::VectorXd referenceline_x = Eigen::VectorXd::Zero(PathPointsNum);
        Eigen::VectorXd referenceline_y = Eigen::VectorXd::Zero(PathPointsNum);

        Eigen::VectorXd QPSolution(num_of_variables_);  //求解结果
        Eigen::VectorXd last_pos_QPSolution(num_of_pos_variables_);  //xy求解结果

        QPSolution.setZero();
        for (int i = 0; i < PathPointsNum; i++)
        {
            referenceline_x(i) = RawPath[pathNumber][i][0];
            referenceline_y(i) = RawPath[pathNumber][i][1];
            referenceline(2*i) =RawPath[pathNumber][i][0];
            referenceline(2*i+1) =RawPath[pathNumber][i][1];
            QPSolution(2*i)   = RawPath[pathNumber][i][0];
            QPSolution(2*i+1) = RawPath[pathNumber][i][1];
        }

        const float x_lb = -1.5; const float x_ub = 1.5;  //xy矩形限制上下界
        const float y_lb = -1.5; const float y_ub = 1.5;

        Eigen::VectorXd dynamic_lb = Eigen::VectorXd::Constant(2 * PathPointsNum, x_lb);  //这里是区别与上述lb、ub，这里是正方形xy限制范围，是动态调整的
        Eigen::VectorXd dynamic_ub = Eigen::VectorXd::Constant(2 * PathPointsNum, x_ub);  //因为初始xy限制都是一样的，故都设置为x的范围

        double total_length=0.0;
        for (size_t i = 0; i < PathPointsNum - 1; i++) // 注意循环范围
        {
            size_t idx = 2 * i; // 每个点的起始索引
            total_length += std::sqrt((QPSolution[idx + 2] - QPSolution[idx]) *
                                    (QPSolution[idx + 2] - QPSolution[idx]) +
                                    (QPSolution[idx + 3] - QPSolution[idx + 1]) *
                                    (QPSolution[idx + 3] - QPSolution[idx + 1]));
        }
        // std::cout<<"total_length:"<<total_length<<std::endl;
        double average_interval_length =total_length / (PathPointsNum - 1); //每个点间隔0.1米
        double interval_sqr = average_interval_length * average_interval_length;
        double curvature_constraint_sqr = (interval_sqr * curvature_constraint_) *
                                            (interval_sqr * curvature_constraint_);
        int collsion_num=0;
        while (!no_collsion)
        {
            int sub_itr = 1; bool fconverged = false;
            double last_fvalue=0.0;  //上一次代价值
            
            // std::cout<<" -- 参数加载中 --"<<std::endl;
            Eigen::SparseMatrix<double> A1(num_of_variables_, num_of_variables_);
            Eigen::SparseMatrix<double> A2(num_of_variables_, num_of_variables_);
            // 创建稀疏矩阵对象
            Eigen::SparseMatrix<double> A3(num_of_variables_, num_of_variables_);
            // 遍历对角线上的元素，设置为1

            Eigen::MatrixXd A_cons_I = Eigen::MatrixXd::Identity(num_of_variable_constraints_, num_of_variable_constraints_);  //增加首尾位置限制
            Eigen::MatrixXd A_cons   = Eigen::MatrixXd::Zero(num_of_constraints_, num_of_variable_constraints_);  //增加首尾位置限制
            Eigen::VectorXd lb = Eigen::VectorXd::Zero(num_of_constraints_);
            Eigen::VectorXd ub = Eigen::VectorXd::Zero(num_of_constraints_);

            int pen_itr = 0;
            double w_slack = 5e5;
            if (const char* w_slack_env = std::getenv("HYBRID_ASTAR_W_SLACK_INIT")) {
                try {
                    w_slack = std::max(1e-9, std::stod(w_slack_env));
                } catch (...) {
                    w_slack = 5e5;
                }
            }
            if (const char* w_slack_env = std::getenv("HYBRID_ASTAR_INITIAL_W_SLACK")) {
                const double parsed_w_slack = std::atof(w_slack_env);
                if (std::isfinite(parsed_w_slack) && parsed_w_slack > 0.0) {
                    w_slack = parsed_w_slack;
                }
            }
            int sqp_num=0;
            while (pen_itr<sqp_pen_max_iter_)
            {
            //     /* code */            
                while (sub_itr < sqp_sub_max_iter_)
                {
                    // std::cout<<"sub_itr循环开始！！！"<<std::endl;
                    sub_itr++;
                    solver_num++;  //求解次数

                    A_cons.block(0, 0, num_of_variable_constraints_, num_of_variable_constraints_) = A_cons_I;

                    std::vector<std::vector<double>> lin_cache;  //曲率约束线性化系数
                    lin_cache=CalculateLinearizedFemPosParams(QPSolution,PathPointsNum);  

                    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(num_of_variables_, num_of_variables_);
                    Eigen::VectorXd f = Eigen::VectorXd::Zero(num_of_variables_);
                    
                    // std::cout<<" -- 条件赋值中 -- "<<std::endl;

                    for (int i = 0; i < PathPointsNum; i++)
                    {
                        int now_num=2*i;
                        f(now_num)      = -2 * w_ref * referenceline_x(i);
                        f(now_num + 1)  = -2 * w_ref * referenceline_y(i);
                        if (prev_smoothed_segment != nullptr && i < hard_bind_points) {
                            const auto& bind_pose = (*prev_smoothed_segment)[prev_overlap_start + i];
                            lb(now_num) = bind_pose.x();
                            ub(now_num) = bind_pose.x();
                            lb(now_num + 1) = bind_pose.y();
                            ub(now_num + 1) = bind_pose.y();
                        } else {
                            lb(now_num)     = referenceline_x(i) + dynamic_lb(now_num);
                            ub(now_num)     = referenceline_x(i) + dynamic_ub(now_num);
                            lb(now_num + 1) = referenceline_y(i) + dynamic_lb(now_num+1);
                            ub(now_num + 1) = referenceline_y(i) + dynamic_ub(now_num+1);
                        }
                    }

                    for (int i = 0; i < num_of_slack_variables_; i++)
                    {
                        lb(num_of_pos_variables_+i)=0.0;
                        ub(num_of_pos_variables_+i)=1e15;
                    }

                    for (int i = 0; i < num_of_curvature_constraints_; i++)
                    {
                        lb(num_of_variable_constraints_+i)=-1e15;
                        const bool relax_tail_constraint =
                            tail_relaxed_curvature_constraints > 0 &&
                            i >= num_of_curvature_constraints_ - tail_relaxed_curvature_constraints;
                        ub(num_of_variable_constraints_+i)=
                            relax_tail_constraint ? 1e15 : curvature_constraint_sqr-lin_cache[i][6];
                        // std::cout<<"total_length:"<<total_length<<std::endl;
                        // std::cout<<"curvature_constraint_sqr:"<<curvature_constraint_sqr<<" , lin_cache[i][6]:"<<lin_cache[i][6]<<std::endl;
                    }
                    
                    for (int i = num_of_pos_variables_; i < num_of_pos_variables_+num_of_slack_variables_; i++)
                    {
                        f(i)  = w_slack;
                    }
                    // std::cout<<"f:"<<f<<std::endl;

                    for (int i = 0; i < 2 * PathPointsNum - 4; i++)
                    {
                        A1.coeffRef(i, i) = 1;
                        A1.coeffRef(i, i + 2) = -2;
                        A1.coeffRef(i, i + 4) = 1;
                    }
                    // A2赋值
                    for (int i = 0; i < 2 * PathPointsNum - 2; i++)
                    {
                        A2.coeffRef(i, i) = 1;
                        A2.coeffRef(i, i + 2) = -1;
                    }

                    for (int i = 0; i < 2 * PathPointsNum; ++i)
                    {
                        A3.coeffRef(i, i) = 1.0; // 在(i, i)位置插入1.0
                    }
  

                    // 将所有元素插入矩阵中
                    // A3.makeCompressed();

                    for (int i=0; i < num_of_curvature_constraints_; i++)
                    {
                        A_cons(num_of_variables_+i,2*i  )=lin_cache[i][0];
                        A_cons(num_of_variables_+i,2*i+1)=lin_cache[i][1];
                        A_cons(num_of_variables_+i,2*i+2)=lin_cache[i][2];
                        A_cons(num_of_variables_+i,2*i+3)=lin_cache[i][3];
                        A_cons(num_of_variables_+i,2*i+4)=lin_cache[i][4];
                        A_cons(num_of_variables_+i,2*i+5)=lin_cache[i][5];
                        A_cons(num_of_variables_+i,num_of_pos_variables_+i)=-1.0;
                    }

                    // A_con限制条件赋值
                    for (int i = 0; i < endpoint_constraint_half; i=i+2)
                    {
                        A_cons(num_of_constraints_expectStartEnd + i , i ) = 1;   //start_x
                        A_cons(num_of_constraints_expectStartEnd+ i+1 , i+1 ) = 1;  //start_y

                        A_cons(
                            num_of_constraints_expectStartEnd + endpoint_constraint_half + i,
                            num_of_pos_variables_ - endpoint_constraint_half + i) = 1;
                        A_cons(
                            num_of_constraints_expectStartEnd + endpoint_constraint_half + i + 1,
                            num_of_pos_variables_ - endpoint_constraint_half + i + 1) = 1;
                        int point_num_now=i/2;
                        int end_point_index = PathPointsNum - endpoint_lock_points + point_num_now;

                        // if (RawPath.size()<2) //只有一段，起点终点约束为参考点
                        // {
                            // std::cout<<"pathNumber:"<<pathNumber+1<<"/"<<RawPath.size()<<" 只有一段!!"<<std::endl;
                            if (prev_smoothed_segment != nullptr && point_num_now < hard_bind_points) {
                                const auto& bind_pose = (*prev_smoothed_segment)[prev_overlap_start + point_num_now];
                                lb(num_of_constraints_expectStartEnd + i) = bind_pose.x();
                                ub(num_of_constraints_expectStartEnd + i) = bind_pose.x();
                                lb(num_of_constraints_expectStartEnd + i + 1) = bind_pose.y();
                                ub(num_of_constraints_expectStartEnd + i + 1) = bind_pose.y();
                            } else {
                                lb(num_of_constraints_expectStartEnd + i) = referenceline_x(point_num_now);
                                ub(num_of_constraints_expectStartEnd + i) = referenceline_x(point_num_now);
                                lb(num_of_constraints_expectStartEnd + i + 1) = referenceline_y(point_num_now);
                                ub(num_of_constraints_expectStartEnd + i + 1) = referenceline_y(point_num_now);
                            }

                            lb(num_of_constraints_expectStartEnd + endpoint_constraint_half + i) =
                                referenceline_x(end_point_index);
                            ub(num_of_constraints_expectStartEnd + endpoint_constraint_half + i) =
                                referenceline_x(end_point_index);
                            lb(num_of_constraints_expectStartEnd + endpoint_constraint_half + i + 1) =
                                referenceline_y(end_point_index);
                            ub(num_of_constraints_expectStartEnd + endpoint_constraint_half + i + 1) =
                                referenceline_y(end_point_index);
                        // }
                        // else if(pathNumber==0)  //多段且为起点
                        // {
                        //     std::cout<<"pathNumber:"<<pathNumber+1<<"/"<<RawPath.size()<<" 多段且为起点!!"<<std::endl;
                        //     lb(num_of_constraints_expectStartEnd + i)=referenceline_x(point_num_now);     ub(num_of_constraints_expectStartEnd+ i)=referenceline_x(point_num_now);  //最大限制xy
                        //     lb(num_of_constraints_expectStartEnd + i + 1)=referenceline_y(point_num_now); ub(num_of_constraints_expectStartEnd + i + 1)=referenceline_y(point_num_now);  //
                        
                        //     lb(num_of_constraints_expectStartEnd + i +6)=referenceline_x(PathPointsNum +point_num_now-3)+x_lb; ub(num_of_constraints_expectStartEnd + i +6)=referenceline_x(PathPointsNum +point_num_now-3)+x_ub;  //最大限制xy
                        //     lb(num_of_constraints_expectStartEnd + i +7)=referenceline_y(PathPointsNum +point_num_now-3)+y_lb; ub(num_of_constraints_expectStartEnd + i +7)=referenceline_y(PathPointsNum +point_num_now-3)+y_ub;  //
                        // }
                        // else if(pathNumber==RawPath.size()-1)  //多段且为最后一段，终点约束必须为参考点，起点则为上端优化起点
                        // {
                        //     std::cout<<"pathNumber:"<<pathNumber+1<<"/"<<RawPath.size()<<" 多段且为最后一段!!"<<std::endl;
                        //     std::cout<<"last_opt_xy_num:"<<last_opt_xy_num<<" ,last_opt_xy_num:"<<last_pos_QPSolution<<std::endl;
                        //     lb(num_of_constraints_expectStartEnd + i)=last_pos_QPSolution(last_opt_xy_num-6+i);     ub(num_of_constraints_expectStartEnd+ i)=last_pos_QPSolution(last_opt_xy_num-6+i);  //最大限制xy
                        //     lb(num_of_constraints_expectStartEnd + i + 1)=last_pos_QPSolution(last_opt_xy_num-5+i); ub(num_of_constraints_expectStartEnd + i + 1)=last_pos_QPSolution(last_opt_xy_num-5+i);  //
                        
                        //     lb(num_of_constraints_expectStartEnd + i +6)=referenceline_x(PathPointsNum +point_num_now-3);       ub(num_of_constraints_expectStartEnd + i +6)=referenceline_x(PathPointsNum +point_num_now-3);  //最大限制xy
                        //     lb(num_of_constraints_expectStartEnd + i +7)=referenceline_y(PathPointsNum +point_num_now-3);       ub(num_of_constraints_expectStartEnd + i +7)=referenceline_y(PathPointsNum +point_num_now-3);  //
                        // }
                        // // else //多段且不为最后一段，起点，终点都为上次优化的终点
                        // // {
                        // //     std::cout<<"其余段!!"<<std::endl;
                        // //     lb(num_of_constraints_-12 + i)=referenceline_x(i)+x_lb;     ub(num_of_constraints_-12+ i)=referenceline_x(i)+x_ub;  //最大限制xy
                        // //     lb(num_of_constraints_-12 + i + 1)=referenceline_y(i)+y_lb; ub(num_of_constraints_-12 + i + 1)=referenceline_y(i)+y_ub;  //
                        
                        // //     lb(num_of_constraints_-12 + i +6)=referenceline_x(PathPointsNum +i-6)+x_lb; ub(num_of_constraints_-12 + i +6)=referenceline_x(PathPointsNum +i-6)+x_ub;  //最大限制xy
                        // //     lb(num_of_constraints_-12 + i +7)=referenceline_y(PathPointsNum +i-5)+y_lb; ub(num_of_constraints_-12 + i +7)=referenceline_y(PathPointsNum +i-5)+y_ub;  //
                        // // }
                        // else
                        // {
                        //     std::cout<<"pathNumber:"<<pathNumber+1<<"/"<<RawPath.size()<<" 未使用!!"<<std::endl;
                        //     lb(num_of_constraints_expectStartEnd + i)=referenceline_x(point_num_now); ub(num_of_constraints_expectStartEnd+ i)=referenceline_x(point_num_now);  //最大限制xy
                        //     lb(num_of_constraints_expectStartEnd + i + 1)=referenceline_y(point_num_now); ub(num_of_constraints_expectStartEnd + i + 1)=referenceline_y(point_num_now);  //
                        
                        //     lb(num_of_constraints_expectStartEnd + i +6)=referenceline_x(PathPointsNum +point_num_now-3); ub(num_of_constraints_expectStartEnd + i +6)=referenceline_x(PathPointsNum +point_num_now-3);  //最大限制xy
                        //     lb(num_of_constraints_expectStartEnd + i +7)=referenceline_y(PathPointsNum +point_num_now-3); ub(num_of_constraints_expectStartEnd + i +7)=referenceline_y(PathPointsNum +point_num_now-3);  //
                        // }
                    }


                    // for (int i = 0; i < lb.size(); i++)
                    // {
                    //     std::cout<<"lb_ub:"<<lb(i)<<"<< x <<"<<ub(i)<<std::endl;
                    // }
                    // for (size_t i = 0; i < ; i++)
                    // {
                    //     /* code */
                    // }
                    
                    
                    // std::cout<<"lb:"<<std::endl;
                    // std::cout<<lb<<std::endl;
                    // std::cout<<"ub:"<<std::endl;
                    // std::cout<<ub<<std::endl;
                    // for (int i = 0; i < A_cons.rows(); ++i) {
                    //     std::cout << std::setw(2) << std::fixed << std::setprecision(1) << lb(i) << "<=";
                    //     for (int j = 0; j < A_cons.cols(); ++j) {
                    //             // 使用 setw 设置宽度为 10，fixed 设置为固定小数点，precision 控制小数点后位数
                    //             std::cout << std::setw(2) << std::fixed << std::setprecision(1) << A_cons(i, j) << " ";
                    //         }
                    //         std::cout << std::setw(2) << std::fixed << std::setprecision(1) << "<="<< ub(i) ;
                    //         std::cout << std::endl; // 每打印一行后换行
                    //     }

                    std::cout << std::endl; // 每打印一行后换行
                    // std::cout<<" -- 求解中 -- "<<std::endl;
                    H = 2 * (w_smooth * A1.transpose() * A1 + w_length * A2.transpose() * A2 + w_ref * A3);

                    auto add_soft_linear_tracking = [&](const std::vector<std::pair<int, double>>& terms,
                                                        double target,
                                                        double weight) {
                        if (weight <= 0.0 || terms.empty()) {
                            return;
                        }
                        for (const auto& term_i : terms) {
                            const int row = term_i.first;
                            const double coeff_i = term_i.second;
                            f(row) += -2.0 * weight * target * coeff_i;
                            for (const auto& term_j : terms) {
                                const int col = term_j.first;
                                const double coeff_j = term_j.second;
                                H(row, col) += 2.0 * weight * coeff_i * coeff_j;
                            }
                        }
                    };

                    if (w_endpoint_heading > 0.0 && PathPointsNum >= 3) {
                        const Vec2d end_ref_tangent =
                            RawPath[pathNumber][PathPointsNum - 1].head(2) -
                            RawPath[pathNumber][PathPointsNum - 2].head(2);
                        if (hard_bind_points == 0) {
                            const Vec2d start_ref_tangent =
                                RawPath[pathNumber][1].head(2) - RawPath[pathNumber][0].head(2);
                            add_soft_linear_tracking({{0, -1.0}, {2, 1.0}},
                                                     start_ref_tangent.x(),
                                                     w_endpoint_heading);
                            add_soft_linear_tracking({{1, -1.0}, {3, 1.0}},
                                                     start_ref_tangent.y(),
                                                     w_endpoint_heading);
                        }

                        const int end_prev_x = 2 * (PathPointsNum - 2);
                        const int end_prev_y = end_prev_x + 1;
                        const int end_last_x = 2 * (PathPointsNum - 1);
                        const int end_last_y = end_last_x + 1;
                        add_soft_linear_tracking({{end_prev_x, -1.0}, {end_last_x, 1.0}},
                                                 end_ref_tangent.x(),
                                                 w_endpoint_heading);
                        add_soft_linear_tracking({{end_prev_y, -1.0}, {end_last_y, 1.0}},
                                                 end_ref_tangent.y(),
                                                 w_endpoint_heading);
                    }

                    if (w_endpoint_curvature > 0.0 && PathPointsNum >= 4) {
                        const Vec2d end_ref_curvature =
                            RawPath[pathNumber][PathPointsNum - 1].head(2) -
                            2.0 * RawPath[pathNumber][PathPointsNum - 2].head(2) +
                            RawPath[pathNumber][PathPointsNum - 3].head(2);
                        if (hard_bind_points == 0) {
                            const Vec2d start_ref_curvature =
                                RawPath[pathNumber][2].head(2) -
                                2.0 * RawPath[pathNumber][1].head(2) +
                                RawPath[pathNumber][0].head(2);
                            add_soft_linear_tracking({{0, 1.0}, {2, -2.0}, {4, 1.0}},
                                                     start_ref_curvature.x(),
                                                     w_endpoint_curvature);
                            add_soft_linear_tracking({{1, 1.0}, {3, -2.0}, {5, 1.0}},
                                                     start_ref_curvature.y(),
                                                     w_endpoint_curvature);
                        }

                        const int end_prev2_x = 2 * (PathPointsNum - 3);
                        const int end_prev2_y = end_prev2_x + 1;
                        const int end_prev_x = 2 * (PathPointsNum - 2);
                        const int end_prev_y = end_prev_x + 1;
                        const int end_last_x = 2 * (PathPointsNum - 1);
                        const int end_last_y = end_last_x + 1;
                        add_soft_linear_tracking({{end_prev2_x, 1.0}, {end_prev_x, -2.0}, {end_last_x, 1.0}},
                                                 end_ref_curvature.x(),
                                                 w_endpoint_curvature);
                        add_soft_linear_tracking({{end_prev2_y, 1.0}, {end_prev_y, -2.0}, {end_last_y, 1.0}},
                                                 end_ref_curvature.y(),
                                                 w_endpoint_curvature);
                    }

                    if (pathNumber > 0 && PathPointsNum >= 3 && !smoothed_paths.empty()) {
                        const auto& prev_segment = smoothed_paths.back();
                        const auto& prev_info = segment_infos[pathNumber - 1];
                        const auto& curr_info = segment_infos[pathNumber];
                        int overlap_count = std::max(0, prev_info.expand_end - curr_info.expand_start + 1);
                        overlap_count = std::min(overlap_count,
                                                 std::min(static_cast<int>(prev_segment.size()), PathPointsNum));
                        if (overlap_count >= 3) {
                            const int prev_overlap_start = static_cast<int>(prev_segment.size()) - overlap_count;
                            const Vec2d prev_p0 = prev_segment[prev_overlap_start].head(2);
                            const Vec2d prev_p1 = prev_segment[prev_overlap_start + 1].head(2);
                            const Vec2d prev_p2 = prev_segment[prev_overlap_start + 2].head(2);
                            const Vec2d tangent_ref = prev_p1 - prev_p0;
                            const Vec2d curvature_ref = prev_p2 - 2.0 * prev_p1 + prev_p0;

                            add_soft_linear_tracking({{0, -1.0}, {2, 1.0}}, tangent_ref.x(), w_connection_tangent);
                            add_soft_linear_tracking({{1, -1.0}, {3, 1.0}}, tangent_ref.y(), w_connection_tangent);
                            add_soft_linear_tracking({{0, 1.0}, {2, -2.0}, {4, 1.0}},
                                                     curvature_ref.x(),
                                                     w_connection_curvature);
                            add_soft_linear_tracking({{1, 1.0}, {3, -2.0}, {5, 1.0}},
                                                     curvature_ref.y(),
                                                     w_connection_curvature);
                        }
                    }
                    
                    // for (int i = 0; i < H.rows(); ++i) {
                    //     for (int j = 0; j < H.cols(); ++j) {
                    //             // 使用 setw 设置宽度为 10，fixed 设置为固定小数点，precision 控制小数点后位数
                    //             std::cout << std::setw(2) << std::fixed << std::setprecision(1) << H(i, j) << " ";
                    //         }
                    //         std::cout << std::endl; // 每打印一行后换行
                    //     }
                    // osqp求解
                    int NumberOfVariables = num_of_variables_;   // A矩阵的列数
                    int NumberOfConstraints = num_of_constraints_; // A矩阵的行数

                    // 求解部分
                    OsqpEigen::Solver solver;
                    Eigen::SparseMatrix<double> H_osqp = H.sparseView(); // 密集矩阵转换为稀疏矩阵
                    H_osqp.makeCompressed();                             // 压缩稀疏行 (CSR) 格式
                    H_osqp.reserve(H.nonZeros());                        // 预分配非零元素数量
                    // std::cout<<" -- 求解"<<solver_num<<" 次中  -- "<<std::endl;
                    Eigen::SparseMatrix<double> linearMatrix = A_cons.sparseView();
                    linearMatrix.makeCompressed();                 // 压缩稀疏行 (CSR) 格式
                    linearMatrix.reserve(linearMatrix.nonZeros()); // 预分配非零元素数量

                    solver.settings()->setVerbosity(false); // 求解器信息输出控制
                    solver.settings()->setWarmStart(true);
                    solver.data()->setNumberOfVariables(NumberOfVariables);     // 设置A矩阵的列数，即n
                    solver.data()->setNumberOfConstraints(NumberOfConstraints); // 设置A矩阵的行数，即m
                    // std::cout<<" -- 求解kaishi  -- "<<std::endl;
                    if (!solver.data()->setHessianMatrix(H_osqp))
                        // return 1; //设置P矩阵
                        std::cout << "error1" << std::endl;
                    if (!solver.data()->setGradient(f))
                        // return 1; //设置q or f矩阵。当没有时设置为全0向量
                        std::cout << "error2" << std::endl;
                    if (!solver.data()->setLinearConstraintsMatrix(linearMatrix))
                        // return 1; //设置线性约束的A矩阵
                        std::cout << "error3" << std::endl;
                    if (!solver.data()->setLowerBound(lb))
                    { // return 1; //设置下边界
                        std::cout << "error4" << std::endl;
                    }
                    if (!solver.data()->setUpperBound(ub))
                    { // return 1; //设置上边界
                        std::cout << "error5" << std::endl;
                    }

                    // instantiate the solvers
                    if (!solver.initSolver())
                        // return 1;
                        std::cout << "error6" << std::endl;

                    solver.setPrimalVariable(QPSolution);
                    // Eigen::VectorXd QPSolution;

                    // solve the QP problem

                    OsqpEigen::ErrorExitFlag result = solver.solveProblem();

                    // std::cout<<" -- 求解jieshu 1  -- "<<std::endl;
                
                    QPSolution = solver.getSolution();

                    
                    OSQPWorkspace* work=solver.workspace().get();

                    double cur_fvalue = work->info->obj_val;
                    double ftol = std::abs((last_fvalue - cur_fvalue) / last_fvalue);
                    // std::cout<<"ftol:"<<ftol<<" cur_fvalue:"<<cur_fvalue<<" last_fvalue:"<<last_fvalue <<std::endl;
                    if (ftol < sqp_ftol_) {  //sqp_ftol_  1e-2
                        fconverged = true;
                        break;
                    }
                    last_fvalue=cur_fvalue;  //将这次结果赋值last_fvalue

                    total_length=0.0;
                    for (size_t i = 0; i < PathPointsNum - 1; i++) // 注意循环范围
                    {
                        size_t idx = 2 * i; // 每个点的起始索引
                        total_length += std::sqrt((QPSolution[idx + 2] - QPSolution[idx]) *
                                                (QPSolution[idx + 2] - QPSolution[idx]) +
                                                (QPSolution[idx + 3] - QPSolution[idx + 1]) *
                                                (QPSolution[idx + 3] - QPSolution[idx + 1]));
                    }
                    
                    average_interval_length =total_length / (PathPointsNum - 1); //每个点间隔0.1米
                    interval_sqr = average_interval_length * average_interval_length;
                    curvature_constraint_sqr = (interval_sqr * curvature_constraint_) *
                                                        (interval_sqr * curvature_constraint_);
                    // std::cout<<"sqp循环结束！！！"<<std::endl;
                    // std::cout<<"sqp循环结束！！！"<<std::endl;
                    // sqp_num
                }  //求解SQP过程
                // std::cout<<"SQP结束"<<std::endl;
                // std::cout<<"qpsolution:"<<QPSolution<<std::endl;
                float ctol = CalculateConstraintViolation(QPSolution,PathPointsNum,curvature_constraint_sqr);
                if (ctol <= sqp_ctol_ && QPSolution(0)<10000)  //这里是防止出现无解的情况
                {
                    std::cout<<"符合曲率约束！max_violation:"<<ctol<<std::endl;
                    // std::cout<<"num_of_variables_:"<<num_of_variables_<<" num_of_constraints_ :"<<num_of_constraints_<<" points_num:"<< PathPointsNum<<" QPSolution_size:"<<QPSolution.size()<<"lb ub_size:"<<lb.size()<<","<<ub.size()<<std::endl;
                    // std::cout<<"QPSolution:"<<QPSolution<<std::endl;
                    
                    break;
                }
                else
                {
                    std::cout<<"不满足曲率约束！max_violation:"<<ctol<<std::endl;
                }
                pen_itr++;w_slack=w_slack*10;
            }

            smoothed_path.clear();
            for (int i = 0; i < num_of_pos_variables_; i=i+2)
            {
                Vec4d temp_point;
                temp_point(0)=QPSolution(i);
                temp_point(1)=QPSolution(i+1);
                temp_point(2)=0;  //航向角暂时设置为0
                temp_point(3)=RawPath[pathNumber][0][3];
                smoothed_path.emplace_back(temp_point);
            }
            
            GetPathYaw(smoothed_path);
            
            
            float boundary_penalty_raw=1.5;
            bool Flag_Collision=true;
            for (int i = 1; i < smoothed_path.size()-1; i++)
            {   
                bool collision_detected = !CheckCollision(smoothed_path[i][0], smoothed_path[i][1], smoothed_path[i][2]);
                if (!collision_detected && use_simplified_collision_check_) {
                    const Vec2i prev_index = Coordinate2MapGridIndex(Vec2d(smoothed_path[i - 1][0], smoothed_path[i - 1][1]));
                    const Vec2i curr_index = Coordinate2MapGridIndex(Vec2d(smoothed_path[i][0], smoothed_path[i][1]));
                    const Vec2i next_index = Coordinate2MapGridIndex(Vec2d(smoothed_path[i + 1][0], smoothed_path[i + 1][1]));
                    if (!LineCheck(prev_index.x(), prev_index.y(), curr_index.x(), curr_index.y()) ||
                        !LineCheck(curr_index.x(), curr_index.y(), next_index.x(), next_index.y())) {
                        collision_detected = true;
                    }
                }
                //TODO: 变ub lb时候已经弃用referenceline_x，改用qpsolution了
                if(collision_detected) //碰撞
                {   
                    Flag_Collision=false;
                    // std::cout<<"第"<<i<<"个点碰撞！！！"<<"共"<<smoothed_path.size()<<"个点"<<std::endl;
                    int now_number=2*i;
                    if(smoothed_path[i][0]<=referenceline_x(i))  //x
                    {
                        dynamic_lb(now_number)   /= boundary_penalty_raw;
                        dynamic_lb(now_number-2) /= boundary_penalty_raw;
                        dynamic_lb(now_number+2) /= boundary_penalty_raw;

                        lb(now_number)   = referenceline_x(i)   + dynamic_lb(now_number); 
                        lb(now_number-2) = referenceline_x(i-1) + dynamic_lb(now_number-2);
                        lb(now_number+2) = referenceline_x(i+1) + dynamic_lb(now_number+2);
                        // std::cout<<"x_lb改变为:"<<dynamic_lb(now_number)<<std::endl;
                    }
                    else
                    {
                        dynamic_ub(now_number)   /= boundary_penalty_raw;
                        dynamic_ub(now_number-2) /= boundary_penalty_raw;
                        dynamic_ub(now_number+2) /= boundary_penalty_raw;

                        ub(now_number)   = referenceline_x(i)   + dynamic_ub(now_number);
                        ub(now_number-2) = referenceline_x(i-1) + dynamic_ub(now_number-2);
                        ub(now_number+2) = referenceline_x(i+1) + dynamic_ub(now_number+2);
                        // std::cout<<"x_ub改变为:"<<dynamic_ub(now_number)<<std::endl;
                    }

                    if(smoothed_path[i][1]<=referenceline_y(i))  //y
                    {
                        dynamic_lb(now_number+1) /= boundary_penalty_raw;
                        dynamic_lb(now_number-1) /= boundary_penalty_raw;
                        dynamic_lb(now_number+3) /= boundary_penalty_raw;

                        lb(now_number+1) = referenceline_y(i) + dynamic_lb(now_number+1);
                        lb(now_number-1) = referenceline_y(i-1) + dynamic_lb(now_number-1);
                        lb(now_number+3) = referenceline_y(i+1) + dynamic_lb(now_number+3);
                        // std::cout<<"y_lb改变为:"<<dynamic_lb(now_number+1)<<std::endl;
                    }
                    else
                    {
                        dynamic_ub(now_number+1) /= boundary_penalty_raw;
                        dynamic_ub(now_number-1) /= boundary_penalty_raw;
                        dynamic_ub(now_number+3) /= boundary_penalty_raw;

                        ub(now_number+1) = referenceline_y(i) + dynamic_ub(now_number+1);
                        ub(now_number-1) = referenceline_y(i-1) + dynamic_ub(now_number-1);
                        ub(now_number+3) = referenceline_y(i+1) + dynamic_ub(now_number+3);
                        // std::cout<<"y_ub改变为:"<<dynamic_ub(now_number+1)<<std::endl;
                    }
                }
            }
            std::cout<<"无碰撞！！"<<std::endl;
            
            if(Flag_Collision)  //无碰撞
            {
                for (int i = 0; i < last_pos_QPSolution.size(); i++)  //存储xy优化
                {
                    last_pos_QPSolution(i)=QPSolution(i);
                }
                break;
            }
            else if (collsion_num>10000)
            {
                break;
            }
            else
            {
                smoothed_path.clear();
            }
            std::cout <<"碰撞迭代次数："<<collsion_num<<std::endl;
            collsion_num++;
        } //collsion


        smoothed_paths.push_back(smoothed_path);
        std::cout << "Path optimization progress --100% " <<std::endl;
       
    }  //while path_num end
    std::cout<<"END!!!"<<std::endl;
    VectorVec4d return_smoothed_path;
    if (!smoothed_paths.empty()) {
        return_smoothed_path = smoothed_paths.front();
        for (size_t seg_idx = 1; seg_idx < smoothed_paths.size(); ++seg_idx) {
            const int overlap_count =
                std::max(0, segment_infos[seg_idx - 1].expand_end - segment_infos[seg_idx].expand_start + 1);
            const int clamped_overlap =
                std::min(overlap_count,
                         std::min(static_cast<int>(return_smoothed_path.size()),
                                  static_cast<int>(smoothed_paths[seg_idx].size())));
            const int bound_points = std::min(2, clamped_overlap);
            const int replaced_tail_points = std::max(0, clamped_overlap - bound_points);
            for (int pop = 0; pop < replaced_tail_points && !return_smoothed_path.empty(); ++pop) {
                return_smoothed_path.pop_back();
            }
            for (size_t j = static_cast<size_t>(std::max(0, bound_points)); j < smoothed_paths[seg_idx].size(); ++j) {
                if (!return_smoothed_path.empty()) {
                    const auto& prev = return_smoothed_path.back();
                    const auto& curr = smoothed_paths[seg_idx][j];
                    if (std::hypot(prev.x() - curr.x(), prev.y() - curr.y()) < 1e-6) {
                        continue;
                    }
                }
                return_smoothed_path.emplace_back(smoothed_paths[seg_idx][j]);
            }
        }
    }
    auto segment_collision_free = [&](const Vec4d &a, const Vec4d &b) {
        const double dx = b.x() - a.x();
        const double dy = b.y() - a.y();
        const double heading = std::atan2(dy, dx);
        const double distance = std::hypot(dx, dy);
        const int steps = std::max(2, static_cast<int>(std::ceil(distance / std::max(0.1, MAP_GRID_RESOLUTION_ * 0.5))));
        for (int i = 0; i <= steps; ++i) {
            const double ratio = static_cast<double>(i) / static_cast<double>(steps);
            const double x = a.x() + dx * ratio;
            const double y = a.y() + dy * ratio;
            if (BeyondBoundary(Vec2d(x, y)) || !CheckCollision(x, y, heading)) {
                return false;
            }
        }
        return true;
    };

    auto redistribute_local_spacing = [&](VectorVec4d &candidate, int left_anchor, int right_anchor) {
        const int span = right_anchor - left_anchor;
        if (span < 2) {
            return;
        }
        std::vector<Vec2d> local_points(span + 1);
        std::vector<double> local_arc(span + 1, 0.0);
        for (int local_idx = 0; local_idx <= span; ++local_idx) {
            local_points[local_idx] = candidate[left_anchor + local_idx].head(2);
            if (local_idx == 0) {
                continue;
            }
            local_arc[local_idx] =
                local_arc[local_idx - 1] + (local_points[local_idx] - local_points[local_idx - 1]).norm();
        }
        const double total_arc = local_arc.back();
        if (total_arc <= 1e-6) {
            return;
        }
        int upper_idx = 1;
        for (int local_idx = 1; local_idx < span; ++local_idx) {
            const double target_arc = total_arc * static_cast<double>(local_idx) / static_cast<double>(span);
            while (upper_idx < span && local_arc[upper_idx] < target_arc) {
                ++upper_idx;
            }
            const int lower_idx = std::max(0, upper_idx - 1);
            const double denom = std::max(1e-9, local_arc[upper_idx] - local_arc[lower_idx]);
            const double ratio =
                std::max(0.0, std::min(1.0, (target_arc - local_arc[lower_idx]) / denom));
            const Vec2d blended_point =
                (1.0 - ratio) * local_points[lower_idx] + ratio * local_points[upper_idx];
            candidate[left_anchor + local_idx].x() = blended_point.x();
            candidate[left_anchor + local_idx].y() = blended_point.y();
        }
    };
    auto evaluate_window_max_curvature = [&](const VectorVec4d &path, int left_anchor, int right_anchor) {
        if (path.size() < 3) {
            return 0.0;
        }
        double total_length = 0.0;
        for (size_t idx = 0; idx + 1 < path.size(); ++idx) {
            total_length += (path[idx + 1].head(2) - path[idx].head(2)).norm();
        }
        const double average_interval_length =
            total_length / std::max(1.0, static_cast<double>(path.size() - 1));
        if (average_interval_length <= 1e-6) {
            return 0.0;
        }
        const double inv_interval_sqr = 1.0 / (average_interval_length * average_interval_length);
        const int eval_start = std::max(1, left_anchor - 1);
        const int eval_end = std::min(static_cast<int>(path.size()) - 2, right_anchor + 1);
        double max_curvature = 0.0;
        for (int idx = eval_start; idx <= eval_end; ++idx) {
            const Vec2d second_diff =
                path[idx - 1].head(2) - 2.0 * path[idx].head(2) + path[idx + 1].head(2);
            max_curvature = std::max(max_curvature, second_diff.norm() * inv_interval_sqr);
        }
        return max_curvature;
    };
    auto validate_local_window = [&](const VectorVec4d &candidate, int left_anchor, int right_anchor) {
        for (int idx = left_anchor; idx <= right_anchor; ++idx) {
            const int prev_idx = std::max(0, idx - 1);
            const int next_idx = std::min(static_cast<int>(candidate.size()) - 1, idx + 1);
            double heading = candidate[idx].z();
            if (next_idx != prev_idx) {
                heading = std::atan2(candidate[next_idx].y() - candidate[prev_idx].y(),
                                     candidate[next_idx].x() - candidate[prev_idx].x());
            }
            if (BeyondBoundary(candidate[idx].head(2)) || !CheckCollision(candidate[idx].x(), candidate[idx].y(), heading)) {
                return false;
            }
            if (idx < right_anchor && !segment_collision_free(candidate[idx], candidate[idx + 1])) {
                return false;
            }
        }
        return true;
    };
    auto apply_seam_blend = [&](VectorVec4d &path_to_blend) {
        if (path_to_blend.size() < 8 || seam_indices.empty()) {
            return;
        }
        const int blend_half_window = 2;
        for (const int seam_idx : seam_indices) {
            if (seam_idx <= blend_half_window || seam_idx >= static_cast<int>(path_to_blend.size()) - blend_half_window - 1) {
                continue;
            }
            const int left_anchor = seam_idx - blend_half_window;
            const int right_anchor = seam_idx + blend_half_window;
            if (right_anchor - left_anchor < 4) {
                continue;
            }

            VectorVec4d candidate = path_to_blend;
            const Vec2d p0 = path_to_blend[left_anchor].head(2);
            const Vec2d p1 = path_to_blend[right_anchor].head(2);
            Vec2d left_dir = path_to_blend[left_anchor + 1].head(2) - p0;
            Vec2d right_dir = p1 - path_to_blend[right_anchor - 1].head(2);
            if (left_dir.norm() < 1e-6 || right_dir.norm() < 1e-6) {
                continue;
            }
            left_dir.normalize();
            right_dir.normalize();
            const double chord = std::max(1e-6, (p1 - p0).norm());
            const double tangent_scale = 0.2 * chord;
            const Vec2d m0 = left_dir * tangent_scale;
            const Vec2d m1 = right_dir * tangent_scale;

            const int span = right_anchor - left_anchor;
            for (int idx = left_anchor + 1; idx < right_anchor; ++idx) {
                const double t = static_cast<double>(idx - left_anchor) / static_cast<double>(span);
                const double t2 = t * t;
                const double t3 = t2 * t;
                const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
                const double h10 = t3 - 2.0 * t2 + t;
                const double h01 = -2.0 * t3 + 3.0 * t2;
                const double h11 = t3 - t2;
                const Vec2d blended = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
                candidate[idx].x() = blended.x();
                candidate[idx].y() = blended.y();
            }
            redistribute_local_spacing(candidate, left_anchor, right_anchor);

            bool valid = validate_local_window(candidate, left_anchor, right_anchor);
            if (valid) {
                const double max_before = evaluate_window_max_curvature(path_to_blend, left_anchor, right_anchor);
                const double max_after = evaluate_window_max_curvature(candidate, left_anchor, right_anchor);
                if (max_after > max_before || max_after > curvature_constraint_ * 1.05) {
                    valid = false;
                }
            }
            if (valid) {
                for (int idx = left_anchor + 1; idx < right_anchor; ++idx) {
                    path_to_blend[idx].x() = candidate[idx].x();
                    path_to_blend[idx].y() = candidate[idx].y();
                }
            }
        }
    };
    auto apply_seed_guided_patch = [&](VectorVec4d &path_to_patch) {
        if (path_to_patch.size() != path.size() || path_to_patch.size() < 10 || seam_indices.empty()) {
            return;
        }
        const int patch_half_window = 4;
        for (const int seam_idx : seam_indices) {
            if (seam_idx <= patch_half_window || seam_idx >= static_cast<int>(path_to_patch.size()) - patch_half_window - 1) {
                continue;
            }
            const int left_anchor = seam_idx - patch_half_window;
            const int right_anchor = seam_idx + patch_half_window;
            VectorVec4d candidate = path_to_patch;
            for (int idx = left_anchor + 1; idx < right_anchor; ++idx) {
                const double normalized_distance =
                    static_cast<double>(std::abs(idx - seam_idx)) / static_cast<double>(patch_half_window);
                const double alpha = 0.7 * std::max(0.0, 1.0 - normalized_distance);
                candidate[idx].x() =
                    (1.0 - alpha) * path_to_patch[idx].x() + alpha * path[idx].x();
                candidate[idx].y() =
                    (1.0 - alpha) * path_to_patch[idx].y() + alpha * path[idx].y();
            }
            redistribute_local_spacing(candidate, left_anchor, right_anchor);
            if (!validate_local_window(candidate, left_anchor, right_anchor)) {
                continue;
            }
            for (int idx = left_anchor + 1; idx < right_anchor; ++idx) {
                path_to_patch[idx].x() = candidate[idx].x();
                path_to_patch[idx].y() = candidate[idx].y();
            }
        }
    };

    apply_seam_blend(return_smoothed_path);
    apply_seed_guided_patch(return_smoothed_path);
    GetPathYaw(return_smoothed_path);
    
    std::cout<<"求解 "<<solver_num<<" 次完成优化！！！"<<std::endl;
    clock_t time_end2 = clock();
    double time_diff2 = static_cast<double>(time_end2 - time_start) / CLOCKS_PER_SEC;
    std::cout << "QP程序运行时间: " << time_diff2 << " 秒" << std::endl;
    return return_smoothed_path;
}

void HybridAStar::GetPathYaw(std::vector<VectorVec4d> &RawPath) {
    for (size_t i = 0; i < RawPath.size(); i++) { // 遍历不同路径段
        const int car_dir=RawPath[i][0][3]; 
        const int n=RawPath[i].size();
        // 判断行进方向：前进(1)或后退(-1)
        if (n<=6)
        {
            for (size_t j = 0; j < n - 1; j++) {
                // 判断当前点的前进方向
                if (RawPath[i][j][3] == 1) { // 前进方向
                    double yaw = atan2(RawPath[i][j + 1][1] - RawPath[i][j][1],
                                       RawPath[i][j + 1][0] - RawPath[i][j][0]);
                    RawPath[i][j][2] = yaw;
                } else if (RawPath[i][j][3] == -1) { // 后退方向
                    double yaw = atan2(RawPath[i][j][1] - RawPath[i][j + 1][1],
                                       RawPath[i][j][0] - RawPath[i][j + 1][0]);
                    RawPath[i][j][2] = yaw;
                }
            }
            // 计算最后一个点的航向角
            if (n > 1) {
                if (RawPath[i][n - 1][3] == 1) { // 前进方向
                    double yaw = atan2(RawPath[i][n - 1][1] - RawPath[i][n - 2][1],
                                       RawPath[i][n - 1][0] - RawPath[i][n - 2][0]);
                    RawPath[i][n - 1][2] = yaw;
                } else if (RawPath[i][n - 1][3] == -1) { // 后退方向
                    double yaw = atan2(RawPath[i][n - 2][1] - RawPath[i][n - 1][1],
                                       RawPath[i][n - 2][0] - RawPath[i][n - 1][0]);
                    RawPath[i][n - 1][2] = yaw;
                }
            }
        }
        else
        {
            for (size_t j = 2; j < RawPath[i].size() - 2; j++) { // 忽略开头和结尾的两个点
                if (car_dir == 1) { // 前进方向
                    // 计算平滑航向角
                    double yaw1 = atan2(RawPath[i][j][1] - RawPath[i][j - 2][1],
                                        RawPath[i][j][0] - RawPath[i][j - 2][0]);
                    double yaw2 = atan2(RawPath[i][j + 2][1] - RawPath[i][j][1],
                                        RawPath[i][j + 2][0] - RawPath[i][j][0]);

                    double smoothed_yaw = (yaw1 + yaw2 ) / 2.0;

                    RawPath[i][j][2] = smoothed_yaw;

                } else if (car_dir== -1) { // 后退方向
                    // 计算平滑航向角（方向相反，加 180° 或 π 修正）
                    double yaw1 = atan2(RawPath[i][j][1] - RawPath[i][j - 2][1],
                                        RawPath[i][j][0] - RawPath[i][j - 2][0]) + M_PI;
                    double yaw2 = atan2(RawPath[i][j+2][1] - RawPath[i][j][1],
                                        RawPath[i][j+2][0] - RawPath[i][j][0]) + M_PI;

                    // 保证角度在 [-π, π] 范围内
                    yaw1 = std::fmod(yaw1 + M_PI, 2 * M_PI) - M_PI;
                    yaw2 = std::fmod(yaw2 + M_PI, 2 * M_PI) - M_PI;

                    // 平均航向角，平滑结果
                    double smoothed_yaw = (yaw1 + yaw2) / 2.0;

                    // 更新 RawPath 的 yaw 值
                    RawPath[i][j][2] = smoothed_yaw;
                }
            }
            if (car_dir == 1)  //前进
            {
                RawPath[i][0][2] =atan2(RawPath[i][1][1] - RawPath[i][0][1],RawPath[i][1][0] - RawPath[i][0][0]);
                RawPath[i][1][2] =atan2(RawPath[i][2][1] - RawPath[i][1][1],RawPath[i][2][0] - RawPath[i][1][0]);
                
                RawPath[i][n-2][2] =atan2(RawPath[i][n-2][1] - RawPath[i][n-3][1],RawPath[i][n-2][0] - RawPath[i][n-3][0]);
                RawPath[i][n-1][2] =atan2(RawPath[i][n-1][1] - RawPath[i][n-2][1],RawPath[i][n-1][0] - RawPath[i][n-2][0]);
            }
            else    //倒车
            {
                RawPath[i][0][2] = atan2(RawPath[i][0][1] - RawPath[i][1][1],RawPath[i][0][0] - RawPath[i][1][0]);
                RawPath[i][1][2] = atan2(RawPath[i][1][1] - RawPath[i][2][1],RawPath[i][1][0] - RawPath[i][2][0]);
                RawPath[i][n - 2][2] = atan2(RawPath[i][n - 3][1] - RawPath[i][n - 2][1],RawPath[i][n - 3][0] - RawPath[i][n - 2][0]);
                RawPath[i][n - 1][2] = atan2(RawPath[i][n - 2][1] - RawPath[i][n - 1][1],RawPath[i][n - 2][0] - RawPath[i][n - 1][0]);
            }
        }
    }

    // for (int i = 0; i < RawPath.size(); i++) { // 遍历不同路径段
    //     for(int j=0;j<RawPath[i].size();j++)
    //     {
    //         std::cout<<"第 "<<i<<"段，yaw:"<<RawPath[i][j][2]<<std::endl;
    //     }
    // }
}

void HybridAStar::GetPathYaw(VectorVec4d &RawPath) {

        const int car_dir=RawPath[0][3]; 
        const int n=RawPath.size();
        // 判断行进方向：前进(1)或后退(-1)

        for (int i = 0; i < RawPath.size()-1; i++)
        {
                // 判断当前点的前进方向
                if (RawPath[i][3] == 1) { // 前进方向
                    double yaw = atan2(RawPath[i + 1][1] - RawPath[i][1],
                                       RawPath[i + 1][0] - RawPath[i][0]);
                    RawPath[i][2] = yaw;
                } else if (RawPath[i][3] == -1) { // 后退方向
                    double yaw = atan2(RawPath[i][1] - RawPath[i + 1][1],
                                       RawPath[i][0] - RawPath[i + 1][0]);
                    RawPath[i][2] = yaw;
                }
        }
        RawPath[RawPath.size()-1][2]=RawPath[RawPath.size()-2][2];
        
}

// void HybridAStar::GetPathYaw(VectorVec4d &RawPath) {
//     clock_t time_start = clock();

//         const int car_dir=RawPath[0][3]; 
//         const int n=RawPath.size();
//         // 判断行进方向：前进(1)或后退(-1)
//         if (n<=6)
//         {
//             for (size_t j = 0; j < n - 1; j++) {
//                 // 判断当前点的前进方向
//                 if (RawPath[j][3] == 1) { // 前进方向
//                     double yaw = atan2(RawPath[j + 1][1] - RawPath[j][1],
//                                        RawPath[j + 1][0] - RawPath[j][0]);
//                     RawPath[j][2] = yaw;
//                 } else if (RawPath[j][3] == -1) { // 后退方向
//                     double yaw = atan2(RawPath[j][1] - RawPath[j + 1][1],
//                                        RawPath[j][0] - RawPath[j + 1][0]);
//                     RawPath[j][2] = yaw;
//                 }
//             }
//             // 计算最后一个点的航向角
//             if (n > 1) {
//                 if (RawPath[n - 1][3] == 1) { // 前进方向
//                     double yaw = atan2(RawPath[n - 1][1] - RawPath[n - 2][1],
//                                        RawPath[n - 1][0] - RawPath[n - 2][0]);
//                     RawPath[n - 1][2] = yaw;
//                 } else if (RawPath[n - 1][3] == -1) { // 后退方向
//                     double yaw = atan2(RawPath[n - 2][1] - RawPath[n - 1][1],
//                                        RawPath[n - 2][0] - RawPath[n - 1][0]);
//                     RawPath[n - 1][2] = yaw;
//                 }
//             }
//         }
//         else
//         {
//             for (size_t j = 2; j < RawPath.size() - 2; j++) { // 忽略开头和结尾的两个点
//                 if (car_dir == 1) { // 前进方向
//                     // 计算平滑航向角
//                     double yaw1 = atan2(RawPath[j][1] - RawPath[j - 2][1],
//                                         RawPath[j][0] - RawPath[j - 2][0]);
//                     double yaw2 = atan2(RawPath[j + 2][1] - RawPath[j][1],
//                                         RawPath[j + 2][0] - RawPath[j][0]);

//                     double smoothed_yaw = (yaw1 + yaw2 ) / 2.0;

//                     RawPath[j][2] = smoothed_yaw;

//                 } else if (car_dir== -1) { // 后退方向
//                     // 计算平滑航向角（方向相反，加 180° 或 π 修正）
//                     double yaw1 = atan2(RawPath[j][1] - RawPath[j - 2][1],
//                                         RawPath[j][0] - RawPath[j - 2][0]) + M_PI;
//                     double yaw2 = atan2(RawPath[j+2][1] - RawPath[j][1],
//                                         RawPath[j+2][0] - RawPath[j][0]) + M_PI;

//                     // 保证角度在 [-π, π] 范围内
//                     yaw1 = std::fmod(yaw1 + M_PI, 2 * M_PI) - M_PI;
//                     yaw2 = std::fmod(yaw2 + M_PI, 2 * M_PI) - M_PI;

//                     // 平均航向角，平滑结果
//                     double smoothed_yaw = (yaw1 + yaw2) / 2.0;

//                     // 更新 RawPath 的 yaw 值
//                     RawPath[j][2] = smoothed_yaw;
//                 }
//             }
//             if (car_dir == 1)  //前进
//             {
//                 RawPath[0][2] =atan2(RawPath[1][1] - RawPath[0][1],RawPath[1][0] - RawPath[0][0]);
//                 RawPath[1][2] =atan2(RawPath[2][1] - RawPath[1][1],RawPath[2][0] - RawPath[1][0]);
                
//                 RawPath[n-2][2] =atan2(RawPath[n-2][1] - RawPath[n-3][1],RawPath[n-2][0] - RawPath[n-3][0]);
//                 RawPath[n-1][2] =atan2(RawPath[n-1][1] - RawPath[n-2][1],RawPath[n-1][0] - RawPath[n-2][0]);
//             }
//             else    //倒车
//             {
//                 RawPath[0][2] = atan2(RawPath[0][1] - RawPath[1][1],RawPath[0][0] - RawPath[1][0]);
//                 RawPath[1][2] = atan2(RawPath[1][1] - RawPath[2][1],RawPath[1][0] - RawPath[2][0]);
//                 RawPath[n - 2][2] = atan2(RawPath[n - 3][1] - RawPath[n - 2][1],RawPath[n - 3][0] - RawPath[n - 2][0]);
//                 RawPath[n - 1][2] = atan2(RawPath[n - 2][1] - RawPath[n - 1][1],RawPath[n - 2][0] - RawPath[n - 1][0]);
//             }
//         }
    
//     clock_t time_end2 = clock();
//     double time_diff2 = static_cast<double>(time_end2 - time_start) / CLOCKS_PER_SEC;
//     std::cout << "yaw程序运行时间: " << time_diff2 << " 秒" << std::endl;
// }

std::vector<VectorVec4d> HybridAStar::PathSegmentsByDirection(VectorVec4d& path) {
    std::vector<VectorVec4d> path_Segments;

    int direction=path[0][3];

    int i=0;
    while (i < path.size())
    {
        VectorVec4d path_Segment;
        for ( ; i < path.size(); i++)
        {
            if (direction!=path[i][3])
            {
                direction=path[i][3];
                break;
            }
            else
            {
                path_Segment.push_back(path[i]);
            }
        }
        path_Segments.push_back(path_Segment);
    }

    return path_Segments;
}

double HybridAStar::EstimatePointClearance(const Vec2d& pt, double max_radius) const {
    if (BeyondBoundary(pt)) {
        return 0.0;
    }
    const Vec2i center = Coordinate2MapGridIndex(pt);
    if (HasObstacle(center)) {
        return 0.0;
    }
    const int max_ring = std::max(1, static_cast<int>(std::ceil(max_radius / MAP_GRID_RESOLUTION_)));
    for (int ring = 1; ring <= max_ring; ++ring) {
        const int min_x = center.x() - ring;
        const int max_x = center.x() + ring;
        const int min_y = center.y() - ring;
        const int max_y = center.y() + ring;
        for (int x = min_x; x <= max_x; ++x) {
            if (x < 0 || x >= MAP_GRID_SIZE_X_ || min_y < 0 || min_y >= MAP_GRID_SIZE_Y_ ||
                max_y < 0 || max_y >= MAP_GRID_SIZE_Y_) {
                return std::max(0, ring - 1) * MAP_GRID_RESOLUTION_;
            }
            if (HasObstacle(x, min_y) || HasObstacle(x, max_y)) {
                return std::max(0, ring - 1) * MAP_GRID_RESOLUTION_;
            }
        }
        for (int y = min_y + 1; y < max_y; ++y) {
            if (y < 0 || y >= MAP_GRID_SIZE_Y_ || min_x < 0 || min_x >= MAP_GRID_SIZE_X_ ||
                max_x < 0 || max_x >= MAP_GRID_SIZE_X_) {
                return std::max(0, ring - 1) * MAP_GRID_RESOLUTION_;
            }
            if (HasObstacle(min_x, y) || HasObstacle(max_x, y)) {
                return std::max(0, ring - 1) * MAP_GRID_RESOLUTION_;
            }
        }
    }
    return max_radius;
}

std::vector<int> HybridAStar::FindGeometrySplitIndices(const VectorVec4d& path) {
    const int n = static_cast<int>(path.size());
    if (n < 12) {
        return {};
    }

    const char* override_csv = std::getenv("HYBRID_ASTAR_OVERRIDE_SPLIT_POINTS_CSV");
    if (override_csv != nullptr && std::string(override_csv).size() > 0) {
        const std::vector<Vec2d> override_points = ReadOverrideSplitPointsCsv(override_csv);
        const std::vector<int> matched =
            MatchOverrideSplitIndices(path, override_points, /*max_match_dist=*/2.0);
        if (!matched.empty()) {
            std::cout << "[split-override] using external split points from " << override_csv
                      << " matched_indices=";
            for (std::size_t i = 0; i < matched.size(); ++i) {
                if (i > 0) {
                    std::cout << ",";
                }
                std::cout << matched[i];
            }
            std::cout << std::endl;
            return matched;
        }
        std::cout << "[split-override] no valid matched split index from " << override_csv << std::endl;
    }

    VectorVec4d yaw_path = path;
    GetPathYaw(yaw_path);

    const char* legacy_split_env = std::getenv("HYBRID_ASTAR_USE_LEGACY_SPLITS");
    const bool use_legacy_splits =
        legacy_split_env != nullptr && std::string(legacy_split_env) == "1";

    const double heading_split_thresh =
        (use_legacy_splits ? 30.0 : 35.0) * M_PI / 180.0;
    const double cumulative_heading_split_thresh =
        (use_legacy_splits ? 60.0 : 70.0) * M_PI / 180.0;
    const double straight_heading_thresh = 8.0 * M_PI / 180.0;
    const double narrow_enter_clearance_thresh = 1.0;
    const double min_split_distance_m = 3.0;
    const double straight_min_length_m = 2.5;
    const double max_clearance_probe = std::max(1.5, 8.0 * MAP_GRID_RESOLUTION_);
    const int min_segment_points = std::max(8, static_cast<int>(std::ceil(1.0 / MAP_GRID_RESOLUTION_)));
    const int narrow_min_run_points =
        std::max(4, static_cast<int>(std::ceil(0.5 / std::max(0.1, MAP_GRID_RESOLUTION_))));
    const int straight_run_points =
        std::max(4, static_cast<int>(std::ceil(straight_min_length_m / std::max(0.1, MAP_GRID_RESOLUTION_))));
    const bool debug_splits = std::getenv("HYBRID_ASTAR_DEBUG_SPLITS") != nullptr;

    std::vector<double> heading_delta(n, 0.0);
    std::vector<double> clearances(n, max_clearance_probe);
    for (int i = 1; i < n; ++i) {
        heading_delta[i] = NormalizeAngleDiff(yaw_path[i].z() - yaw_path[i - 1].z());
    }
    for (int i = 0; i < n; ++i) {
        clearances[i] = EstimatePointClearance(yaw_path[i].head(2), max_clearance_probe);
    }
    std::vector<double> cumulative_s(n, 0.0);
    for (int i = 1; i < n; ++i) {
        cumulative_s[i] = cumulative_s[i - 1] + (yaw_path[i].head(2) - yaw_path[i - 1].head(2)).norm();
    }

    std::set<int> candidates;
    std::map<int, std::vector<std::string>> candidate_reasons;
    auto add_candidate = [&](int idx, const std::string& reason) {
        candidates.insert(idx);
        if (debug_splits) {
            candidate_reasons[idx].push_back(reason);
        }
    };
    if (use_legacy_splits) {
        for (int i = 2; i < n - 2; ++i) {
            if (std::abs(heading_delta[i]) >= heading_split_thresh) {
                add_candidate(i, "heading_jump_legacy");
            }
        }
        int last_heading_anchor = 0;
        for (int i = 1; i < n - 1; ++i) {
            if (i - last_heading_anchor < min_segment_points) {
                continue;
            }
            const double yaw_from_anchor =
                NormalizeAngleDiff(yaw_path[i].z() - yaw_path[last_heading_anchor].z());
            if (std::abs(yaw_from_anchor) >= cumulative_heading_split_thresh) {
                add_candidate(i, "cumulative_heading_legacy");
                last_heading_anchor = i;
            }
        }
    } else {
        std::vector<int> angle_triggers;
        for (int i = 2; i < n - 2; ++i) {
            if (std::abs(heading_delta[i]) >= heading_split_thresh) {
                angle_triggers.push_back(i);
            }
        }

        int last_heading_anchor = 0;
        for (int i = 1; i < n - 1; ++i) {
            if (i - last_heading_anchor < min_segment_points) {
                continue;
            }
            const double yaw_from_anchor =
                NormalizeAngleDiff(yaw_path[i].z() - yaw_path[last_heading_anchor].z());
            if (std::abs(yaw_from_anchor) >= cumulative_heading_split_thresh) {
                angle_triggers.push_back(i);
                last_heading_anchor = i;
            }
        }
        std::sort(angle_triggers.begin(), angle_triggers.end());
        angle_triggers.erase(std::unique(angle_triggers.begin(), angle_triggers.end()), angle_triggers.end());

        for (const int trigger_idx : angle_triggers) {
            int best_split_idx = -1;
            int straight_run = 0;
            int straight_start_idx = -1;
            for (int j = trigger_idx + 1; j < n - 1; ++j) {
                if (std::abs(heading_delta[j]) < straight_heading_thresh) {
                    if (straight_run == 0) {
                        straight_start_idx = j;
                    }
                    ++straight_run;
                } else {
                    straight_run = 0;
                    straight_start_idx = -1;
                }
                if (straight_start_idx > 0 &&
                    straight_run >= straight_run_points &&
                    cumulative_s[j] - cumulative_s[straight_start_idx] >= straight_min_length_m) {
                    best_split_idx = j;
                    break;
                }
            }
            if (best_split_idx > 0 && best_split_idx < n - 1) {
                add_candidate(best_split_idx, "post_turn_straight");
            }
        }
    }

    bool in_narrow = false;
    int narrow_start = -1;
    int narrow_enter_run = 0;
    int narrow_clear_run = 0;
    for (int i = 1; i < n - 1; ++i) {
        if (!in_narrow) {
            if (clearances[i] <= narrow_enter_clearance_thresh) {
                if (narrow_enter_run == 0) {
                    narrow_start = i;
                }
                ++narrow_enter_run;
                if (narrow_enter_run >= narrow_min_run_points) {
                    in_narrow = true;
                    narrow_clear_run = 0;
                    add_candidate(std::max(1, narrow_start - 1), "narrow_enter");
                }
            } else {
                narrow_enter_run = 0;
                narrow_start = -1;
            }
        } else if (clearances[i] > narrow_enter_clearance_thresh) {
            ++narrow_clear_run;
            if (narrow_clear_run >= narrow_min_run_points) {
                in_narrow = false;
                narrow_enter_run = 0;
                narrow_clear_run = 0;
                narrow_start = -1;
            }
        } else {
            narrow_clear_run = 0;
        }
    }

    std::vector<int> sorted_candidates(candidates.begin(), candidates.end());
    std::vector<int> filtered;
    filtered.reserve(sorted_candidates.size());
    for (const int idx : sorted_candidates) {
        if (idx < min_segment_points || idx > n - min_segment_points) {
            continue;
        }
        if (!filtered.empty() && idx - filtered.back() < (min_segment_points - 1)) {
            continue;
        }
        if (!filtered.empty() &&
            cumulative_s[idx] - cumulative_s[filtered.back()] < min_split_distance_m) {
            continue;
        }
        filtered.push_back(idx);
    }
    if (!filtered.empty() && n - filtered.back() < min_segment_points) {
        filtered.pop_back();
    }
    if (!filtered.empty() &&
        cumulative_s.back() - cumulative_s[filtered.back()] < min_split_distance_m) {
        filtered.pop_back();
    }
    const char* drop_last_split_env = std::getenv("HYBRID_ASTAR_DROP_LAST_SPLIT_POINT");
    if (drop_last_split_env != nullptr && std::string(drop_last_split_env) == "1" && !filtered.empty()) {
        filtered.pop_back();
    }
    if (debug_splits) {
        std::cout << "[split-debug] n=" << n
                  << " legacy=" << (use_legacy_splits ? 1 : 0)
                  << " heading_deg=" << heading_split_thresh * 180.0 / M_PI
                  << " cumulative_deg=" << cumulative_heading_split_thresh * 180.0 / M_PI
                  << " straight_deg=8"
                  << " straight_min_length_m=" << straight_min_length_m
                  << " straight_run_points=" << straight_run_points
                  << " narrow_enter=" << narrow_enter_clearance_thresh
                  << " min_split_distance_m=" << min_split_distance_m
                  << " narrow_run=" << narrow_min_run_points
                  << " min_segment_points=" << min_segment_points << std::endl;
        for (const int idx : filtered) {
            std::cout << "[split-debug] keep idx=" << idx
                      << " xy=(" << yaw_path[idx].x() << "," << yaw_path[idx].y() << ")"
                      << " yaw_deg=" << yaw_path[idx].z() * 180.0 / M_PI
                      << " clearance=" << clearances[idx]
                      << " heading_delta_deg=" << heading_delta[idx] * 180.0 / M_PI
                      << " reasons=";
            const auto it = candidate_reasons.find(idx);
            if (it == candidate_reasons.end() || it->second.empty()) {
                std::cout << "unknown";
            } else {
                for (std::size_t reason_idx = 0; reason_idx < it->second.size(); ++reason_idx) {
                    if (reason_idx > 0) {
                        std::cout << ",";
                    }
                    std::cout << it->second[reason_idx];
                }
            }
            std::cout << std::endl;
        }
    }
    return filtered;
}

std::vector<VectorVec4d> HybridAStar::SplitSegmentByGeometry(const VectorVec4d& path) {
    if (path.size() < 12) {
        return {path};
    }

    const std::vector<int> split_indices = FindGeometrySplitIndices(path);
    if (split_indices.empty()) {
        return {path};
    }

    std::vector<VectorVec4d> segments;
    segments.reserve(split_indices.size() + 1);
    int start = 0;
    for (const int split_idx : split_indices) {
        if (split_idx <= start || split_idx >= static_cast<int>(path.size()) - 1) {
            continue;
        }
        VectorVec4d segment(path.begin() + start, path.begin() + split_idx + 1);
        if (segment.size() >= 2) {
            segments.push_back(segment);
        }
        start = split_idx;
    }
    VectorVec4d tail(path.begin() + start, path.end());
    if (tail.size() >= 2) {
        segments.push_back(tail);
    }
    if (segments.empty()) {
        segments.push_back(path);
    }
    return segments;
}


//求解限制最大曲率约束限制系数
double HybridAStar::CalculateConstraintViolation(const Eigen::VectorXd &points,int PathPointsNum,double curvature_constraint_sqr) {

  double max_cviolation = 0.0;
  for (int index = 1; index < PathPointsNum-1; index++) {
        double x_f = points[2*(index-1)];
        double x_m = points[2*index];
        double x_l = points[2*(index+1)];
        double y_f = points[2*(index-1)+1];
        double y_m = points[2*index+1];
        double y_l = points[2*(index+1)+1];
    const double curvature_actual =
                        (-2.0 * x_m + x_f + x_l) * (-2.0 * x_m + x_f + x_l) +
                        (-2.0 * y_m + y_f + y_l) * (-2.0 * y_m + y_f + y_l);
    const double cviolation = curvature_actual - curvature_constraint_sqr;
    max_cviolation = std::max(max_cviolation, cviolation);
    // std::cout<<"cviolation:"<<cviolation<<",max_cviolation:"<<max_cviolation<<" ,.....:"<<curvature_actual<<std::endl;
  }
  std::cout<<"curvature_constraint_sqr:"<<curvature_constraint_sqr<<std::endl;
  return max_cviolation;
}
//个人认为是将曲率约束线性化，下面的有多个系数的式子是泰勒展开后的线性化中导数的计算
std::vector<std::vector<double>> HybridAStar::CalculateLinearizedFemPosParams(const Eigen::VectorXd &points,int PathPointsNum) {

    std::vector<std::vector<double>> points_params;
    for (int index = 1; index < PathPointsNum-1; index++)
    {
        double x_f = points[2*(index-1)];
        double x_m = points[2*index];
        double x_l = points[2*(index+1)];

        double y_f = points[2*(index-1)+1];
        double y_m = points[2*index+1];
        double y_l = points[2*(index+1)+1];

        double linear_term_x_f = 2.0 * x_f - 4.0 * x_m + 2.0 * x_l; 
        double linear_term_x_m = 8.0 * x_m - 4.0 * x_f - 4.0 * x_l;
        double linear_term_x_l = 2.0 * x_l - 4.0 * x_m + 2.0 * x_f;
        double linear_term_y_f = 2.0 * y_f - 4.0 * y_m + 2.0 * y_l;
        double linear_term_y_m = 8.0 * y_m - 4.0 * y_f - 4.0 * y_l;
        double linear_term_y_l = 2.0 * y_l - 4.0 * y_m + 2.0 * y_f;

        double linear_approx = (-2.0 * x_m + x_f + x_l) * (-2.0 * x_m + x_f + x_l) +
                                (-2.0 * y_m + y_f + y_l) * (-2.0 * y_m + y_f + y_l) +
                                -x_f * linear_term_x_f + -x_m * linear_term_x_m +
                                -x_l * linear_term_x_l + -y_f * linear_term_y_f +
                                -y_m * linear_term_y_m + -y_l * linear_term_y_l;

        std::vector<double> point_params{linear_term_x_f, linear_term_y_f, linear_term_x_m, linear_term_y_m,
                linear_term_x_l, linear_term_y_l, linear_approx};
        points_params.emplace_back(point_params);
    }
    return points_params;
}
