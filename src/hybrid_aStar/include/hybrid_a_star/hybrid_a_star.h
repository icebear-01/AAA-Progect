//注：01版本为采用sqp优化，将起点终点不变优化
#ifndef HYBRID_A_STAR_HYBRID_A_STAR_H
#define HYBRID_A_STAR_HYBRID_A_STAR_H

#include "rs_path.h"
#include "state_node.h"

#include <glog/logging.h>

#include <map>
#include <memory>

#include <sensor_msgs/PointCloud2.h>
#include <boost/make_shared.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <geometry_msgs/Pose2D.h>
#include "OsqpEigen/OsqpEigen.h"
#include <iomanip> 
#include <random>

class HybridAStar {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    HybridAStar() = delete;

    HybridAStar(double steering_angle, int steering_angle_discrete_num, double segment_length,
                int segment_length_discrete_num, double wheel_base, double steering_penalty,
                double reversing_penalty, double steering_change_penalty, double shot_distance,
                int grid_size_phi = 72);

    ~HybridAStar();

    void Init(double x_lower, double x_upper, double y_lower, double y_upper,
              double state_grid_resolution, double map_grid_resolution = 0.1);

    bool Search(const Vec4d &start_state, const Vec4d &goal_state);

    VectorVec4d GetSearchedTree();

    VectorVec4d GetPath(VectorVec4d &path_original);

    VectorVec4d SmoothPath(VectorVec4d path_raw);

    VectorVec4d GetSmoothSegmentSplitPoints(VectorVec4d path_raw);

    __attribute__((unused)) int GetVisitedNodesNumber() const { return visited_node_number_; }

    __attribute__((unused)) double GetPathLength() const;

    __attribute__((unused)) Vec2d CoordinateRounding(const Vec2d &pt) const;

    Vec2i Coordinate2MapGridIndex(const Vec2d &pt) const;

    void SetObstacle(double pt_x, double pt_y);

    void SetObstacle(unsigned int x, unsigned int y);

    void SetSimplifiedCollisionCheck(bool enable) { use_simplified_collision_check_ = enable; }
    void SetFixEndpointHeading(bool enable) { fix_endpoint_heading_ = enable; }

    pcl::PointCloud<pcl::PointXYZI>::Ptr node_points_watch = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

    /*!
     * Set vehicle shape
     * Consider the shape of the vehicle as a rectangle.
     * @param length vehicle length (a to c)
     * @param width vehicle width (a to d)
     * @param rear_axle_dist Length from rear axle to rear (a to b)
     *
     *         b
     *  a  ---------------- c
     *    |    |          |    Front
     *    |    |          |
     *  d  ----------------
     */
    void SetVehicleShape(double length, double width, double rear_axle_dist);

    void Reset();

private:
    inline bool HasObstacle(int grid_index_x, int grid_index_y) const;

    inline bool HasObstacle(const Vec2i &grid_index) const;

    bool CheckCollision(const double &x, const double &y, const double &theta);

    inline bool LineCheck(double x0, double y0, double x1, double y1);

    ////判断路径是否超越范围及碰撞，使用RS曲线链接，保存路径
    bool AnalyticExpansions(const StateNode::Ptr &current_node_ptr,
                            const StateNode::Ptr &goal_node_ptr, double &length);

    inline double ComputeG(const StateNode::Ptr &current_node_ptr, const StateNode::Ptr &neighbor_node_ptr) const;

    inline double ComputeH(const StateNode::Ptr &current_node_ptr, const StateNode::Ptr &terminal_node_ptr);

    inline Vec3i State2Index(const Vec3d &state) const;
    inline Vec3i State2Index(const Vec4d &state) const;

    inline Vec2d MapGridIndex2Coordinate(const Vec2i &grid_index) const;

    void GetNeighborNodes(const StateNode::Ptr &curr_node_ptr, std::vector<StateNode::Ptr> &neighbor_nodes);

    /*!
     * Simplified car model. Center of the rear axle
     * refer to: http://planning.cs.uiuc.edu/node658.html
     * @param step_size Length of discrete steps
     * @param phi Car steering angle
     * @param x Car position (world frame)
     * @param y Car position (world frame)
     * @param theta Car yaw (world frame)
     */
    inline void DynamicModel(const double &step_size, const double &phi, double &x, double &y, double &theta) const;

    static inline double Mod2Pi(const double &x);

    //超越地图限制判断
    bool BeyondBoundary(const Vec2d &pt) const;

    void ReleaseMemory();

    std::vector<VectorVec4d> PathSegmentsByDirection(VectorVec4d& path);   
    std::vector<VectorVec4d> SplitSegmentByGeometry(const VectorVec4d& path);
    std::vector<int> FindGeometrySplitIndices(const VectorVec4d& path);
    double EstimatePointClearance(const Vec2d& pt, double max_radius) const;

    VectorVec4d Smooth(VectorVec4d &path);  //hybrid a平滑函数

    void GetPathYaw(std::vector<VectorVec4d> &RawPath);

    void GetPathYaw(VectorVec4d &RawPath);

    double CalculateConstraintViolation(const Eigen::VectorXd &points,int PathPointsNum,double curvature_constraint_sqr);

    std::vector<std::vector<double>> CalculateLinearizedFemPosParams(const Eigen::VectorXd &points,int PathPointsNum) ;

private:
    uint8_t *map_data_ = nullptr;
    double STATE_GRID_RESOLUTION_{}, MAP_GRID_RESOLUTION_{};
    double ANGULAR_RESOLUTION_{};
    int STATE_GRID_SIZE_X_{}, STATE_GRID_SIZE_Y_{}, STATE_GRID_SIZE_PHI_{};
    int MAP_GRID_SIZE_X_{}, MAP_GRID_SIZE_Y_{};

    double map_x_lower_{}, map_x_upper_{}, map_y_lower_{}, map_y_upper_{};

    StateNode::Ptr terminal_node_ptr_ = nullptr;   //
    StateNode::Ptr ***state_node_map_ = nullptr;   //三维节点，相当于三维数组

    std::multimap<double, StateNode::Ptr> openset_;  //哈希表类型(键值对)

    double wheel_base_; //The distance between the front and rear axles
    double segment_length_;  // 单次搜索路径长度 1.6
    double move_step_size_;  // 单步搜索路径长度 
    double steering_radian_step_size_;   //每一小步的搜索度数 20 /6 
    double steering_radian_; //radian //单个栅格转弯的最大角度，角度转弧度 20°（弧度）
    double tie_breaker_;

    double shot_distance_;              //提前链接距离
    int segment_length_discrete_num_;   //每次步进多少份  8
    int steering_discrete_num_;         //最大角度分为几个角度搜索  默认为3+3=6  ,即左分叉3，右分叉3
    double steering_penalty_;            //2
    double reversing_penalty_;           //3
    double steering_change_penalty_;     //2

    double path_length_ = 0.0;
    const float curvature_constraint_=0.8;  //最大曲率限制

    // Settings of sqp
    int sqp_pen_max_iter_ = 100;    //100
    double sqp_ftol_ = 1e-2;  //0.01
    int sqp_sub_max_iter_ = 100;  //100
    double sqp_ctol_ = 1e-6;    // curvature constraint violation tolerance

    std::shared_ptr<RSPath> rs_path_ptr_;

    VecXd vehicle_shape_;   ////车的四个角点XY
    MatXd vehicle_shape_discrete_;   //车的轮廓分割点

    // debug
    double check_collision_use_time = 0.0;
    int num_check_collision = 0;
    int visited_node_number_ = 0;
    std::mt19937 gen;  // Mersenne Twister 随机数生成器
    std::uniform_real_distribution<> dis;  // [0.0, 1.0) 的均匀分布
    float max_distance_rs_shot=40;  //40m 
    float min_distance_rs_shot=5;  //5m
    float max_probability_rs_shot=1.0;  
    float min_probability_rs_shot=0.02;  
    const double lambda = 5;  //指数概率分布系数，越小则平均概率增大
    bool use_simplified_collision_check_ = false;
    bool fix_endpoint_heading_ = true;
 
};

#endif //HYBRID_A_STAR_HYBRID_A_STAR_H
