#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <tf/transform_datatypes.h>
#include <visualization_msgs/MarkerArray.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "hope/configs.h"
#include "hope/geometry.h"
#include "hope/parking_map_normal.h"
#include "hybrid_a_star/hybrid_a_star.h"

namespace {

constexpr double kEpsilon = 1e-9;

struct Bounds {
    double xmin = 0.0;
    double xmax = 0.0;
    double ymin = 0.0;
    double ymax = 0.0;
};

struct VehicleLikeObstacle {
    double center_x = 0.0;
    double center_y = 0.0;
    double yaw = 0.0;
    double length = 0.0;
    double width = 0.0;
};

double ParamDouble(const ros::NodeHandle& nh, const std::string& name, double fallback) {
    double value = fallback;
    nh.param(name, value, fallback);
    return value;
}

int ParamInt(const ros::NodeHandle& nh, const std::string& name, int fallback) {
    int value = fallback;
    nh.param(name, value, fallback);
    return value;
}

bool ParamBool(const ros::NodeHandle& nh, const std::string& name, bool fallback) {
    bool value = fallback;
    nh.param(name, value, fallback);
    return value;
}

std::string ParamString(const ros::NodeHandle& nh, const std::string& name, const std::string& fallback) {
    std::string value = fallback;
    nh.param(name, value, fallback);
    return value;
}

double NormalizeYaw(double yaw) {
    while (yaw > M_PI) {
        yaw -= 2.0 * M_PI;
    }
    while (yaw < -M_PI) {
        yaw += 2.0 * M_PI;
    }
    return yaw;
}

Vec4d ToPlannerState(const hope::State& state) {
    Vec4d out;
    out << state.loc.x, state.loc.y, NormalizeYaw(state.heading), 1.0;
    return out;
}

bool PointOnSegment(const hope::Point2& p, const hope::Point2& a, const hope::Point2& b, double tol) {
    const double vx = b.x - a.x;
    const double vy = b.y - a.y;
    const double wx = p.x - a.x;
    const double wy = p.y - a.y;
    const double len2 = vx * vx + vy * vy;
    if (len2 < kEpsilon) {
        return std::hypot(p.x - a.x, p.y - a.y) <= tol;
    }
    double t = (wx * vx + wy * vy) / len2;
    t = std::max(0.0, std::min(1.0, t));
    const double px = a.x + t * vx;
    const double py = a.y + t * vy;
    return std::hypot(p.x - px, p.y - py) <= tol;
}

bool PointInPolygon(const hope::Point2& p, const hope::Polygon2& poly, double edge_tol) {
    if (poly.size() < 3) {
        return false;
    }
    bool inside = false;
    for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
        const hope::Point2& a = poly[j];
        const hope::Point2& b = poly[i];
        if (PointOnSegment(p, a, b, edge_tol)) {
            return true;
        }
        const bool crosses = ((a.y > p.y) != (b.y > p.y)) &&
                             (p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y + kEpsilon) + a.x);
        if (crosses) {
            inside = !inside;
        }
    }
    return inside;
}

Bounds PolygonBounds(const hope::Polygon2& poly) {
    Bounds b;
    b.xmin = b.ymin = std::numeric_limits<double>::infinity();
    b.xmax = b.ymax = -std::numeric_limits<double>::infinity();
    for (const auto& pt : poly) {
        b.xmin = std::min(b.xmin, pt.x);
        b.xmax = std::max(b.xmax, pt.x);
        b.ymin = std::min(b.ymin, pt.y);
        b.ymax = std::max(b.ymax, pt.y);
    }
    return b;
}

double Distance(const hope::Point2& a, const hope::Point2& b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

bool SimilarLength(double a, double b, double max_relative_error) {
    const double denom = std::max(std::max(std::abs(a), std::abs(b)), kEpsilon);
    return std::abs(a - b) / denom <= max_relative_error;
}

bool TryFitVehicleLikeObstacle(const hope::Polygon2& poly, VehicleLikeObstacle& vehicle) {
    if (poly.size() != 4) {
        return false;
    }

    const double d01 = Distance(poly[0], poly[1]);
    const double d12 = Distance(poly[1], poly[2]);
    const double d23 = Distance(poly[2], poly[3]);
    const double d30 = Distance(poly[3], poly[0]);
    if (d01 < kEpsilon || d12 < kEpsilon || d23 < kEpsilon || d30 < kEpsilon) {
        return false;
    }
    if (!SimilarLength(d01, d23, 0.28) || !SimilarLength(d12, d30, 0.28)) {
        return false;
    }

    const auto edge_dot_ratio = [](const hope::Point2& a,
                                   const hope::Point2& b,
                                   const hope::Point2& c) {
        const double ux = b.x - a.x;
        const double uy = b.y - a.y;
        const double vx = c.x - b.x;
        const double vy = c.y - b.y;
        const double denom = std::max(std::hypot(ux, uy) * std::hypot(vx, vy), kEpsilon);
        return std::abs((ux * vx + uy * vy) / denom);
    };
    if (edge_dot_ratio(poly[0], poly[1], poly[2]) > 0.30 ||
        edge_dot_ratio(poly[1], poly[2], poly[3]) > 0.30 ||
        edge_dot_ratio(poly[2], poly[3], poly[0]) > 0.30 ||
        edge_dot_ratio(poly[3], poly[0], poly[1]) > 0.30) {
        return false;
    }

    const double side_a = 0.5 * (d01 + d23);
    const double side_b = 0.5 * (d12 + d30);
    const bool edge_a_is_long = side_a >= side_b;
    const double long_side = edge_a_is_long ? side_a : side_b;
    const double short_side = edge_a_is_long ? side_b : side_a;
    if (long_side < hope::kVehicleLength * 0.70 || long_side > hope::kVehicleLength * 1.45 ||
        short_side < hope::kVehicleWidth * 0.65 || short_side > hope::kVehicleWidth * 1.55) {
        return false;
    }

    vehicle.center_x = 0.0;
    vehicle.center_y = 0.0;
    for (const auto& pt : poly) {
        vehicle.center_x += pt.x;
        vehicle.center_y += pt.y;
    }
    vehicle.center_x *= 0.25;
    vehicle.center_y *= 0.25;
    vehicle.yaw = edge_a_is_long
                      ? std::atan2(poly[1].y - poly[0].y, poly[1].x - poly[0].x)
                      : std::atan2(poly[2].y - poly[1].y, poly[2].x - poly[1].x);
    vehicle.length = long_side;
    vehicle.width = short_side;
    return true;
}

void RasterizePolygonToPlanner(
    const hope::Polygon2& poly,
    double map_resolution,
    double obstacle_inflation,
    HybridAStar& planner) {
    if (poly.size() < 3) {
        return;
    }
    const Bounds b = PolygonBounds(poly);
    const double step = std::max(0.02, map_resolution * 0.5);
    const double pad = obstacle_inflation + map_resolution;
    for (double x = b.xmin - pad; x <= b.xmax + pad; x += step) {
        for (double y = b.ymin - pad; y <= b.ymax + pad; y += step) {
            hope::Point2 p{x, y};
            if (PointInPolygon(p, poly, obstacle_inflation)) {
                planner.SetObstacle(x, y);
            }
        }
    }

    for (std::size_t i = 0; i < poly.size(); ++i) {
        const hope::Point2& a = poly[i];
        const hope::Point2& c = poly[(i + 1) % poly.size()];
        const double length = std::hypot(c.x - a.x, c.y - a.y);
        const int n = std::max(1, static_cast<int>(std::ceil(length / step)));
        for (int k = 0; k <= n; ++k) {
            const double t = static_cast<double>(k) / static_cast<double>(n);
            const double x = a.x + t * (c.x - a.x);
            const double y = a.y + t * (c.y - a.y);
            planner.SetObstacle(x, y);
        }
    }
}

VectorVec3d ToPath3d(const VectorVec4d& path) {
    VectorVec3d out;
    out.reserve(path.size());
    for (const auto& pose : path) {
        Vec3d p;
        p << pose.x(), pose.y(), pose.z();
        out.push_back(p);
    }
    return out;
}

void PublishPath(const ros::Publisher& pub, const VectorVec4d& path, const std::string& frame_id) {
    nav_msgs::Path msg;
    msg.header.frame_id = frame_id;
    msg.header.stamp = ros::Time::now();
    msg.poses.reserve(path.size());
    for (const auto& pose : path) {
        geometry_msgs::PoseStamped ps;
        ps.header = msg.header;
        ps.pose.position.x = pose.x();
        ps.pose.position.y = pose.y();
        ps.pose.position.z = 0.05;
        ps.pose.orientation = tf::createQuaternionMsgFromYaw(pose.z());
        msg.poses.push_back(ps);
    }
    pub.publish(msg);
}

void AddPolygonMarker(visualization_msgs::MarkerArray& array,
                      int& id,
                      const std::string& frame_id,
                      const std::string& ns,
                      const hope::Polygon2& poly,
                      double z,
                      double alpha,
                      double r,
                      double g,
                      double b,
                      double outline_width = 0.04) {
    if (poly.size() < 3) {
        return;
    }
    visualization_msgs::Marker fill;
    fill.header.frame_id = frame_id;
    fill.header.stamp = ros::Time::now();
    fill.ns = ns + "_fill";
    fill.id = id++;
    fill.type = visualization_msgs::Marker::TRIANGLE_LIST;
    fill.action = visualization_msgs::Marker::ADD;
    fill.pose.orientation.w = 1.0;
    fill.color.a = alpha;
    fill.color.r = r;
    fill.color.g = g;
    fill.color.b = b;

    const hope::Point2 origin = poly.front();
    for (std::size_t i = 1; i + 1 < poly.size(); ++i) {
        geometry_msgs::Point p0;
        p0.x = origin.x;
        p0.y = origin.y;
        p0.z = z;
        geometry_msgs::Point p1;
        p1.x = poly[i].x;
        p1.y = poly[i].y;
        p1.z = z;
        geometry_msgs::Point p2;
        p2.x = poly[i + 1].x;
        p2.y = poly[i + 1].y;
        p2.z = z;
        fill.points.push_back(p0);
        fill.points.push_back(p1);
        fill.points.push_back(p2);
    }
    array.markers.push_back(fill);

    visualization_msgs::Marker outline;
    outline.header = fill.header;
    outline.ns = ns + "_outline";
    outline.id = id++;
    outline.type = visualization_msgs::Marker::LINE_STRIP;
    outline.action = visualization_msgs::Marker::ADD;
    outline.pose.orientation.w = 1.0;
    outline.scale.x = outline_width;
    outline.color.a = std::min(1.0, alpha + 0.25);
    outline.color.r = std::max(0.0, r - 0.15);
    outline.color.g = std::max(0.0, g - 0.15);
    outline.color.b = std::max(0.0, b - 0.15);
    for (const auto& pt : poly) {
        geometry_msgs::Point p;
        p.x = pt.x;
        p.y = pt.y;
        p.z = z + 0.03;
        outline.points.push_back(p);
    }
    outline.points.push_back(outline.points.front());
    array.markers.push_back(outline);
}

void AddDashedPolygonMarker(visualization_msgs::MarkerArray& array,
                            int& id,
                            const std::string& frame_id,
                            const std::string& ns,
                            const hope::Polygon2& poly,
                            double z,
                            double r,
                            double g,
                            double b,
                            double outline_width = 0.12,
                            double dash_length = 0.35,
                            double gap_length = 0.18) {
    if (poly.size() < 3) {
        return;
    }

    visualization_msgs::Marker dash;
    dash.header.frame_id = frame_id;
    dash.header.stamp = ros::Time::now();
    dash.ns = ns + "_dashed_outline";
    dash.id = id++;
    dash.type = visualization_msgs::Marker::LINE_LIST;
    dash.action = visualization_msgs::Marker::ADD;
    dash.pose.orientation.w = 1.0;
    dash.scale.x = outline_width;
    dash.color.a = 0.95;
    dash.color.r = r;
    dash.color.g = g;
    dash.color.b = b;

    const double period = std::max(dash_length + gap_length, kEpsilon);
    for (std::size_t i = 0; i < poly.size(); ++i) {
        const hope::Point2& a = poly[i];
        const hope::Point2& c = poly[(i + 1) % poly.size()];
        const double edge_length = Distance(a, c);
        if (edge_length < kEpsilon) {
            continue;
        }
        const double ux = (c.x - a.x) / edge_length;
        const double uy = (c.y - a.y) / edge_length;
        for (double s = 0.0; s < edge_length; s += period) {
            const double e = std::min(s + dash_length, edge_length);
            geometry_msgs::Point p0;
            p0.x = a.x + ux * s;
            p0.y = a.y + uy * s;
            p0.z = z;
            geometry_msgs::Point p1;
            p1.x = a.x + ux * e;
            p1.y = a.y + uy * e;
            p1.z = z;
            dash.points.push_back(p0);
            dash.points.push_back(p1);
        }
    }

    if (!dash.points.empty()) {
        array.markers.push_back(dash);
    }
}

void AddBoundaryMarker(visualization_msgs::MarkerArray& array,
                       int& id,
                       const std::string& frame_id,
                       const hope::MapBoundary& boundary) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = ros::Time::now();
    marker.ns = "hope_boundary";
    marker.id = id++;
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.06;
    marker.color.a = 0.9;
    marker.color.r = 0.1;
    marker.color.g = 0.1;
    marker.color.b = 0.1;

    const std::vector<hope::Point2> pts = {
        {boundary.xmin, boundary.ymin},
        {boundary.xmax, boundary.ymin},
        {boundary.xmax, boundary.ymax},
        {boundary.xmin, boundary.ymax},
        {boundary.xmin, boundary.ymin},
    };
    for (const auto& pt : pts) {
        geometry_msgs::Point p;
        p.x = pt.x;
        p.y = pt.y;
        p.z = 0.08;
        marker.points.push_back(p);
    }
    array.markers.push_back(marker);
}

void AddVehiclePathMarkers(visualization_msgs::MarkerArray& array,
                           int& id,
                           const std::string& frame_id,
                           const VectorVec4d& path,
                           unsigned int interval) {
    interval = std::max(1u, interval);
    for (unsigned int i = 0; i < path.size(); i += interval) {
        visualization_msgs::Marker vehicle;
        vehicle.header.frame_id = frame_id;
        vehicle.header.stamp = ros::Time::now();
        vehicle.ns = "planned_vehicle_footprint";
        vehicle.id = id++;
        vehicle.type = visualization_msgs::Marker::CUBE;
        vehicle.action = visualization_msgs::Marker::ADD;
        vehicle.scale.x = hope::kVehicleLength;
        vehicle.scale.y = hope::kVehicleWidth;
        vehicle.scale.z = 0.05;
        vehicle.pose.position.x = path[i].x() + std::cos(path[i].z()) * (hope::kVehicleLength * 0.5 - hope::kRearHang);
        vehicle.pose.position.y = path[i].y() + std::sin(path[i].z()) * (hope::kVehicleLength * 0.5 - hope::kRearHang);
        vehicle.pose.position.z = 0.08;
        vehicle.pose.orientation = tf::createQuaternionMsgFromYaw(path[i].z());
        vehicle.color.a = 0.12;
        vehicle.color.r = 0.05;
        vehicle.color.g = 0.45;
        vehicle.color.b = 0.95;
        array.markers.push_back(vehicle);
    }
}

void AddSearchedTreeMarkers(visualization_msgs::MarkerArray& array,
                            int& id,
                            const std::string& frame_id,
                            const VectorVec4d& searched_tree) {
    if (searched_tree.empty()) {
        return;
    }

    visualization_msgs::Marker edges;
    edges.header.frame_id = frame_id;
    edges.header.stamp = ros::Time::now();
    edges.ns = "searched_tree_edges";
    edges.id = id++;
    edges.type = visualization_msgs::Marker::LINE_LIST;
    edges.action = visualization_msgs::Marker::ADD;
    edges.pose.orientation.w = 1.0;
    edges.scale.x = 0.018;
    edges.color.a = 0.22;
    edges.color.r = 0.95;
    edges.color.g = 0.66;
    edges.color.b = 0.05;

    for (const auto& segment : searched_tree) {
        if (!std::isfinite(segment.x()) || !std::isfinite(segment.y()) ||
            !std::isfinite(segment.z()) || !std::isfinite(segment.w())) {
            continue;
        }
        geometry_msgs::Point p0;
        p0.x = segment.x();
        p0.y = segment.y();
        p0.z = 0.055;
        geometry_msgs::Point p1;
        p1.x = segment.z();
        p1.y = segment.w();
        p1.z = 0.055;
        edges.points.push_back(p0);
        edges.points.push_back(p1);
    }

    if (!edges.points.empty()) {
        array.markers.push_back(edges);
    }
}

void AddParkedVehicleMarker(visualization_msgs::MarkerArray& array,
                            int& id,
                            const std::string& frame_id,
                            const VehicleLikeObstacle& obstacle) {
    visualization_msgs::Marker body;
    body.header.frame_id = frame_id;
    body.header.stamp = ros::Time::now();
    body.ns = "hope_other_vehicle";
    body.id = id++;
    body.type = visualization_msgs::Marker::CUBE;
    body.action = visualization_msgs::Marker::ADD;
    body.scale.x = obstacle.length;
    body.scale.y = obstacle.width;
    body.scale.z = 0.18;
    body.pose.position.x = obstacle.center_x;
    body.pose.position.y = obstacle.center_y;
    body.pose.position.z = 0.12;
    body.pose.orientation = tf::createQuaternionMsgFromYaw(obstacle.yaw);
    body.color.a = 0.72;
    body.color.r = 0.42;
    body.color.g = 0.42;
    body.color.b = 0.42;
    array.markers.push_back(body);

    visualization_msgs::Marker heading;
    heading.header = body.header;
    heading.ns = "hope_other_vehicle_heading";
    heading.id = id++;
    heading.type = visualization_msgs::Marker::ARROW;
    heading.action = visualization_msgs::Marker::ADD;
    heading.scale.x = 0.55;
    heading.scale.y = 0.16;
    heading.scale.z = 0.16;
    heading.pose.position.x = obstacle.center_x + std::cos(obstacle.yaw) * obstacle.length * 0.18;
    heading.pose.position.y = obstacle.center_y + std::sin(obstacle.yaw) * obstacle.length * 0.18;
    heading.pose.position.z = 0.30;
    heading.pose.orientation = tf::createQuaternionMsgFromYaw(obstacle.yaw);
    heading.color.a = 0.90;
    heading.color.r = 0.20;
    heading.color.g = 0.20;
    heading.color.b = 0.20;
    array.markers.push_back(heading);
}

void PublishScenarioMarkers(const ros::Publisher& pub,
                            const hope::ParkingMapState& state,
                            const VectorVec4d& smoothed_path,
                            const VectorVec4d& searched_tree,
                            const std::string& frame_id,
                            unsigned int vehicle_interval) {
    visualization_msgs::MarkerArray array;
    visualization_msgs::Marker reset;
    reset.action = visualization_msgs::Marker::DELETEALL;
    reset.header.frame_id = frame_id;
    array.markers.push_back(reset);

    int id = 1;
    AddSearchedTreeMarkers(array, id, frame_id, searched_tree);
    for (const auto& obs : state.obstacles) {
        VehicleLikeObstacle vehicle_obstacle;
        if (TryFitVehicleLikeObstacle(obs.shape, vehicle_obstacle)) {
            AddPolygonMarker(array, id, frame_id, "hope_other_vehicle_footprint", obs.shape, 0.03, 0.28, 0.42, 0.42, 0.42);
            AddParkedVehicleMarker(array, id, frame_id, vehicle_obstacle);
        } else {
            AddPolygonMarker(array, id, frame_id, "hope_obstacle", obs.shape, 0.02, 0.72, 0.33, 0.33, 0.33, 0.12);
        }
    }
    AddPolygonMarker(array, id, frame_id, "hope_start_box", state.start_box, 0.04, 0.45, 1.0, 0.45, 0.05, 0.12);
    if (smoothed_path.empty()) {
        AddDashedPolygonMarker(array, id, frame_id, "hope_goal_box_unreached", state.dest_box, 0.10, 0.0, 0.70, 0.75, 0.12);
    } else {
        AddPolygonMarker(array, id, frame_id, "hope_goal_box", state.dest_box, 0.05, 0.50, 0.0, 0.70, 0.75, 0.12);
    }
    AddVehiclePathMarkers(array, id, frame_id, smoothed_path, vehicle_interval);

    pub.publish(array);
}

void PublishPose(const ros::Publisher& pub,
                 const hope::State& state,
                 const std::string& frame_id) {
    geometry_msgs::PoseStamped msg;
    msg.header.frame_id = frame_id;
    msg.header.stamp = ros::Time::now();
    msg.pose.position.x = state.loc.x;
    msg.pose.position.y = state.loc.y;
    msg.pose.position.z = 0.1;
    msg.pose.orientation = tf::createQuaternionMsgFromYaw(state.heading);
    pub.publish(msg);
}

class HopeExtremHybridAstarDemo {
public:
    explicit HopeExtremHybridAstarDemo(ros::NodeHandle& nh) : nh_(nh) {
        LoadParams();
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("hope_extrem_markers", 1, true);
        raw_path_pub_ = nh_.advertise<nav_msgs::Path>("path_raw", 1, true);
        smooth_path_pub_ = nh_.advertise<nav_msgs::Path>("path_smoothed", 1, true);
        start_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("start_pose", 1, true);
        goal_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("goal_pose", 1, true);
    }

    bool Run() {
        VectorVec4d raw_path;
        VectorVec4d smoothed_path;
        VectorVec4d searched_tree;
        hope::ParkingMapState solved_state;
        bool solved = false;

        hope::ParkingMapNormal generator(hope::ParseMapLevel(level_), static_cast<unsigned int>(seed_));
        const int forced_case_id = horizontal_parallel_case_ ? 1 : case_id_;
        for (int i = 0; i < warmup_cases_; ++i) {
            generator.Reset(forced_case_id);
        }
        for (int attempt = 1; attempt <= sample_attempts_ && ros::ok(); ++attempt) {
            generator.Reset(forced_case_id);
            const hope::ParkingMapState& state = generator.GetState();
            ROS_INFO_STREAM("HOPE attempt " << attempt
                            << " level=" << hope::MapLevelToString(state.level)
                            << " case_id=" << state.case_id
                            << " start=(" << state.start.loc.x << ", " << state.start.loc.y << ", " << state.start.heading << ")"
                            << " goal=(" << state.dest.loc.x << ", " << state.dest.loc.y << ", " << state.dest.heading << ")"
                            << " obstacles=" << state.obstacles.size());
            if (Plan(state, raw_path, smoothed_path, searched_tree)) {
                solved_state = state;
                solved = true;
                ROS_INFO_STREAM("Traditional Hybrid A* succeeded on attempt " << attempt
                                << ", raw points=" << raw_path.size()
                                << ", smoothed points=" << smoothed_path.size());
                break;
            }
            solved_state = state;
            ROS_WARN_STREAM("Traditional Hybrid A* failed on attempt " << attempt);
        }

        PublishPose(start_pub_, solved_state.start, world_frame_);
        PublishPose(goal_pub_, solved_state.dest, world_frame_);
        PublishPath(raw_path_pub_, raw_path, world_frame_);
        PublishPath(smooth_path_pub_, smoothed_path, world_frame_);
        PublishScenarioMarkers(marker_pub_, solved_state, smoothed_path, searched_tree, world_frame_, vehicle_marker_interval_);
        return solved;
    }

private:
    void LoadParams() {
        world_frame_ = ParamString(nh_, "world_frame", "world");
        level_ = ParamString(nh_, "level", "Extrem");
        case_id_ = ParamInt(nh_, "case_id", 1);
        horizontal_parallel_case_ = ParamBool(nh_, "horizontal_parallel_case", true);
        seed_ = ParamInt(nh_, "seed", 0);
        warmup_cases_ = ParamInt(nh_, "warmup_cases", 0);
        sample_attempts_ = ParamInt(nh_, "sample_attempts", 30);
        map_margin_ = ParamDouble(nh_, "map_margin", 1.0);
        map_grid_resolution_ = ParamDouble(nh_, "map_grid_resolution", 0.12);
        state_grid_resolution_ = ParamDouble(nh_, "state_grid_resolution", 0.6);
        obstacle_inflation_ = ParamDouble(nh_, "obstacle_inflation", 0.03);
        simplified_collision_check_ = ParamBool(nh_, "simplified_collision_check", false);
        fix_endpoint_heading_ = ParamBool(nh_, "fix_endpoint_heading", true);
        vehicle_marker_interval_ = static_cast<unsigned int>(std::max(1, ParamInt(nh_, "vehicle_marker_interval", 6)));

        steering_angle_ = ParamDouble(nh_, "planner/steering_angle", 42.0);
        steering_angle_discrete_num_ = ParamInt(nh_, "planner/steering_angle_discrete_num", 12);
        segment_length_ = ParamDouble(nh_, "planner/segment_length", 0.75);
        segment_length_discrete_num_ = ParamInt(nh_, "planner/segment_length_discrete_num", 10);
        wheel_base_ = ParamDouble(nh_, "planner/wheel_base", hope::kWheelBase);
        steering_penalty_ = ParamDouble(nh_, "planner/steering_penalty", 1.5);
        reversing_penalty_ = ParamDouble(nh_, "planner/reversing_penalty", 2.0);
        steering_change_penalty_ = ParamDouble(nh_, "planner/steering_change_penalty", 4.0);
        shot_distance_ = ParamDouble(nh_, "planner/shot_distance", 8.0);
        phi_grid_size_ = ParamInt(nh_, "planner/phi_grid_size", 72);
    }

    bool Plan(const hope::ParkingMapState& state,
              VectorVec4d& raw_path,
              VectorVec4d& smoothed_path,
              VectorVec4d& searched_tree) {
        raw_path.clear();
        smoothed_path.clear();
        searched_tree.clear();

        HybridAStar planner(
            steering_angle_,
            steering_angle_discrete_num_,
            segment_length_,
            segment_length_discrete_num_,
            wheel_base_,
            steering_penalty_,
            reversing_penalty_,
            steering_change_penalty_,
            shot_distance_,
            phi_grid_size_);

        planner.Init(state.boundary.xmin - map_margin_,
                     state.boundary.xmax + map_margin_,
                     state.boundary.ymin - map_margin_,
                     state.boundary.ymax + map_margin_,
                     state_grid_resolution_,
                     map_grid_resolution_);
        planner.SetVehicleShape(hope::kVehicleLength, hope::kVehicleWidth, hope::kRearHang);
        planner.SetSimplifiedCollisionCheck(simplified_collision_check_);
        planner.SetFixEndpointHeading(fix_endpoint_heading_);

        for (const auto& obs : state.obstacles) {
            RasterizePolygonToPlanner(obs.shape, map_grid_resolution_, obstacle_inflation_, planner);
        }

        const Vec4d start = ToPlannerState(state.start);
        const Vec4d goal = ToPlannerState(state.dest);
        if (!planner.Search(start, goal)) {
            searched_tree = planner.GetSearchedTree();
            return false;
        }
        searched_tree = planner.GetSearchedTree();
        smoothed_path = planner.GetPath(raw_path);
        return !raw_path.empty() && !smoothed_path.empty();
    }

    ros::NodeHandle nh_;
    ros::Publisher marker_pub_;
    ros::Publisher raw_path_pub_;
    ros::Publisher smooth_path_pub_;
    ros::Publisher start_pub_;
    ros::Publisher goal_pub_;

    std::string world_frame_ = "world";
    std::string level_ = "Extrem";
    int case_id_ = 1;
    bool horizontal_parallel_case_ = true;
    int seed_ = 0;
    int warmup_cases_ = 0;
    int sample_attempts_ = 30;
    double map_margin_ = 1.0;
    double map_grid_resolution_ = 0.12;
    double state_grid_resolution_ = 0.6;
    double obstacle_inflation_ = 0.03;
    bool simplified_collision_check_ = false;
    bool fix_endpoint_heading_ = true;
    unsigned int vehicle_marker_interval_ = 6;

    double steering_angle_ = 42.0;
    int steering_angle_discrete_num_ = 12;
    double segment_length_ = 0.75;
    int segment_length_discrete_num_ = 10;
    double wheel_base_ = hope::kWheelBase;
    double steering_penalty_ = 1.5;
    double reversing_penalty_ = 2.0;
    double steering_change_penalty_ = 4.0;
    double shot_distance_ = 8.0;
    int phi_grid_size_ = 72;
};

}  // namespace

int main(int argc, char** argv) {
    ros::init(argc, argv, "hope_extrem_hybrid_astar");
    ros::NodeHandle nh("~");
    HopeExtremHybridAstarDemo demo(nh);
    const bool solved = demo.Run();
    if (!solved) {
        ROS_ERROR("No solvable HOPE extrem horizontal case was found within sample_attempts.");
    }
    bool exit_after_run = false;
    nh.param("exit_after_run", exit_after_run, false);
    if (exit_after_run) {
        return solved ? 0 : 1;
    }
    ros::spin();
    return solved ? 0 : 1;
}
