#include "include/emplanner.hpp"

#include <ros/package.h>
#include <xmlrpcpp/XmlRpcValue.h>

#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct ObstacleSpec {
    int id = 0;
    double center_x = 0.0;
    double center_y = 0.0;
    double length = 1.0;
    double width = 0.8;
    double yaw = 0.0;
    double x_vel = 0.0;
    double y_vel = 0.0;
};

struct PathSample {
    int index = 0;
    double s = 0.0;
    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
    double kappa = 0.0;
};

struct GridPointSample {
    int index = 0;
    int col = 0;
    int row = 0;
    double s = 0.0;
    double l = 0.0;
    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
};

struct StCurveSample {
    int index = 0;
    double t = 0.0;
    double s = 0.0;
    double v = 0.0;
    double a = 0.0;
    bool is_feasible = true;
};

struct StGridSample {
    int index = 0;
    int col = 0;
    int row = 0;
    double t = 0.0;
    double s = 0.0;
    double v = 0.0;
    double a = 0.0;
    double cost = 0.0;
    bool is_possible = false;
};

double XmlRpcNumber(const XmlRpc::XmlRpcValue& value, double default_value)
{
    if (value.getType() == XmlRpc::XmlRpcValue::TypeDouble) {
        return static_cast<double>(value);
    }
    if (value.getType() == XmlRpc::XmlRpcValue::TypeInt) {
        return static_cast<int>(value);
    }
    return default_value;
}

std::string XmlRpcString(const XmlRpc::XmlRpcValue& value, const std::string& default_value)
{
    if (value.getType() == XmlRpc::XmlRpcValue::TypeString) {
        return static_cast<std::string>(value);
    }
    if (value.getType() == XmlRpc::XmlRpcValue::TypeInt) {
        return std::to_string(static_cast<int>(value));
    }
    if (value.getType() == XmlRpc::XmlRpcValue::TypeDouble) {
        std::ostringstream ss;
        ss << static_cast<double>(value);
        return ss.str();
    }
    if (value.getType() == XmlRpc::XmlRpcValue::TypeBoolean) {
        return static_cast<bool>(value) ? "true" : "false";
    }
    return default_value;
}

double GetStructNumber(const XmlRpc::XmlRpcValue& value,
                       const std::string& key,
                       double default_value)
{
    if (value.getType() == XmlRpc::XmlRpcValue::TypeStruct && value.hasMember(key)) {
        return XmlRpcNumber(value[key], default_value);
    }
    return default_value;
}

std::string Trim(const std::string& value)
{
    const size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return std::string();
    }
    const size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<double> ParseCsvDoubles(const std::string& csv)
{
    std::vector<double> values;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = Trim(item);
        if (item.empty()) {
            continue;
        }
        char* end_ptr = nullptr;
        const double value = std::strtod(item.c_str(), &end_ptr);
        if (end_ptr == item.c_str() || (end_ptr != nullptr && *end_ptr != '\0')) {
            continue;
        }
        values.push_back(value);
    }
    return values;
}

std::string ToLowerAscii(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool ResolveTurnAngleCase(const std::string& raw_case, double& turn_angle_deg)
{
    const std::string normalized = ToLowerAscii(Trim(raw_case));
    if (normalized.empty() || normalized == "custom" || normalized == "keep") {
        return false;
    }
    if (normalized == "0" || normalized == "0deg" || normalized == "straight" ||
        normalized == "line") {
        turn_angle_deg = 0.0;
        return true;
    }
    if (normalized == "30" || normalized == "30deg") {
        turn_angle_deg = 30.0;
        return true;
    }
    if (normalized == "60" || normalized == "60deg") {
        turn_angle_deg = 60.0;
        return true;
    }
    if (normalized == "90" || normalized == "90deg") {
        turn_angle_deg = 90.0;
        return true;
    }
    return false;
}

double GetIndexedValueOrDefault(const std::vector<double>& values,
                                size_t index,
                                double default_value)
{
    if (index < values.size()) {
        return values[index];
    }
    return default_value;
}

bool EnsureDirectory(const std::string& path)
{
    if (path.empty()) {
        return false;
    }

    std::string current;
    if (path.front() == '/') {
        current = "/";
    }

    std::stringstream ss(path);
    std::string item;
    while (std::getline(ss, item, '/')) {
        if (item.empty()) {
            continue;
        }
        if (!current.empty() && current.back() != '/') {
            current += "/";
        }
        current += item;
        if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
            std::cerr << "mkdir failed for " << current << ": " << std::strerror(errno) << std::endl;
            return false;
        }
    }
    return true;
}

bool WriteStraightTrajectoryFile(const std::string& file_path,
                                 double start_x,
                                 double start_y,
                                 double length,
                                 double step,
                                 double turn_x,
                                 double turn_angle_deg,
                                 double turn_arc_length,
                                 const std::string& turn_shape_case,
                                 double second_turn_gap,
                                 double second_turn_angle_deg,
                                 double second_turn_arc_length)
{
    if (!(length > 0.0) || !(step > 0.0)) {
        return false;
    }

    const size_t slash = file_path.find_last_of('/');
    if (slash != std::string::npos) {
        if (!EnsureDirectory(file_path.substr(0, slash))) {
            return false;
        }
    }

    std::ofstream output(file_path);
    if (!output.is_open()) {
        return false;
    }

    struct Pose2d {
        double x = 0.0;
        double y = 0.0;
        double yaw = 0.0;
    };
    struct TrajectorySegment {
        bool is_arc = false;
        double start_s = 0.0;
        double length = 0.0;
        double turn_angle_rad = 0.0;
        Pose2d start_pose;
    };
    const auto advance_straight = [](const Pose2d& start_pose, double ds) {
        Pose2d pose = start_pose;
        pose.x += ds * std::cos(start_pose.yaw);
        pose.y += ds * std::sin(start_pose.yaw);
        return pose;
    };
    const auto advance_arc = [&](const Pose2d& start_pose, double ds, double turn_angle_rad) {
        if (ds <= 1e-8 || std::fabs(turn_angle_rad) <= 1e-8) {
            return advance_straight(start_pose, ds);
        }
        const double curvature = turn_angle_rad / ds;
        const double end_yaw = start_pose.yaw + turn_angle_rad;
        Pose2d pose;
        pose.x = start_pose.x + (std::sin(end_yaw) - std::sin(start_pose.yaw)) / curvature;
        pose.y = start_pose.y - (std::cos(end_yaw) - std::cos(start_pose.yaw)) / curvature;
        pose.yaw = end_yaw;
        return pose;
    };
    const auto sample_segment_pose = [&](const TrajectorySegment& segment, double local_s) {
        const double clamped_local_s = std::max(0.0, std::min(local_s, segment.length));
        if (!segment.is_arc || std::fabs(segment.turn_angle_rad) <= 1e-8) {
            return advance_straight(segment.start_pose, clamped_local_s);
        }
        const double partial_turn_angle =
            segment.turn_angle_rad * clamped_local_s / std::max(segment.length, 1e-8);
        return advance_arc(segment.start_pose, clamped_local_s, partial_turn_angle);
    };

    output << std::fixed << std::setprecision(6);
    const int point_count = std::max(21, static_cast<int>(std::floor(length / step)) + 1);
    const std::string normalized_turn_shape_case = ToLowerAscii(Trim(turn_shape_case));
    const double first_turn_distance = std::min(std::max(0.0, turn_x - start_x), length);
    const double first_turn_angle_rad = turn_angle_deg * M_PI / 180.0;
    const double first_arc_length = std::max(0.0, std::min(turn_arc_length, length - first_turn_distance));
    bool use_second_turn = (normalized_turn_shape_case == "s_curve" || normalized_turn_shape_case == "s");
    double second_turn_angle_rad = second_turn_angle_deg * M_PI / 180.0;
    if (use_second_turn && std::fabs(second_turn_angle_rad) <= 1e-8) {
        second_turn_angle_rad = -first_turn_angle_rad;
    }
    double second_arc_length = second_turn_arc_length;
    if (use_second_turn && second_arc_length <= 1e-8) {
        second_arc_length = first_arc_length;
    }
    second_arc_length = std::max(0.0, second_arc_length);
    second_turn_gap = std::max(0.0, second_turn_gap);

    std::vector<TrajectorySegment> segments;
    segments.reserve(5);
    Pose2d pose;
    pose.x = start_x;
    pose.y = start_y;
    pose.yaw = 0.0;
    double consumed_s = 0.0;
    const auto append_straight_segment = [&](double segment_length) {
        const double clamped_length = std::max(0.0, std::min(segment_length, length - consumed_s));
        if (clamped_length <= 1e-8) {
            return;
        }
        TrajectorySegment segment;
        segment.is_arc = false;
        segment.start_s = consumed_s;
        segment.length = clamped_length;
        segment.start_pose = pose;
        segments.push_back(segment);
        pose = advance_straight(pose, clamped_length);
        consumed_s += clamped_length;
    };
    const auto append_arc_segment = [&](double segment_length, double turn_angle_rad_value) {
        const double clamped_length = std::max(0.0, std::min(segment_length, length - consumed_s));
        if (clamped_length <= 1e-8) {
            return;
        }
        if (std::fabs(turn_angle_rad_value) <= 1e-8) {
            append_straight_segment(clamped_length);
            return;
        }
        TrajectorySegment segment;
        segment.is_arc = true;
        segment.start_s = consumed_s;
        segment.length = clamped_length;
        segment.turn_angle_rad = turn_angle_rad_value;
        segment.start_pose = pose;
        segments.push_back(segment);
        pose = advance_arc(pose, clamped_length, turn_angle_rad_value);
        consumed_s += clamped_length;
    };

    append_straight_segment(first_turn_distance);
    append_arc_segment(first_arc_length, first_turn_angle_rad);
    if (use_second_turn) {
        append_straight_segment(second_turn_gap);
        append_arc_segment(second_arc_length, second_turn_angle_rad);
    }
    append_straight_segment(length - consumed_s);

    for (int i = 0; i < point_count; ++i) {
        const double path_s = std::min(length, static_cast<double>(i) * step);
        Pose2d sample_pose = pose;
        bool matched_segment = false;
        for (const auto& segment : segments) {
            if (path_s > segment.start_s + segment.length + 1e-8) {
                continue;
            }
            sample_pose = sample_segment_pose(segment, path_s - segment.start_s);
            matched_segment = true;
            break;
        }
        if (!matched_segment) {
            if (!segments.empty()) {
                const auto& last_segment = segments.back();
                sample_pose = sample_segment_pose(last_segment, last_segment.length);
            } else {
                sample_pose.x = start_x + path_s;
                sample_pose.y = start_y;
                sample_pose.yaw = 0.0;
            }
        }
        output << sample_pose.x << " " << sample_pose.y << "\n";
    }
    return true;
}

bool LoadFirstPoseFromPath(const std::string& file_path, double& x, double& y, double& yaw)
{
    std::ifstream input(file_path);
    if (!input.is_open()) {
        return false;
    }

    double x1 = 0.0;
    double y1 = 0.0;
    double x2 = 0.0;
    double y2 = 0.0;
    if (!(input >> x1 >> y1)) {
        return false;
    }

    x = x1;
    y = y1;
    if (input >> x2 >> y2) {
        yaw = std::atan2(y2 - y1, x2 - x1);
    } else {
        yaw = 0.0;
    }
    return true;
}

std::vector<Eigen::Vector2d> LoadTrajectoryXY(const std::string& file_path)
{
    std::vector<Eigen::Vector2d> points;
    std::ifstream input(file_path);
    if (!input.is_open()) {
        return points;
    }

    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream iss(line);
        double x = 0.0;
        double y = 0.0;
        if (!(iss >> x >> y)) {
            continue;
        }
        points.emplace_back(x, y);
    }
    return points;
}

std::vector<ObstacleSpec> LoadObstacleSpecsFromCsv(const std::string& file_path)
{
    std::vector<ObstacleSpec> specs;
    std::ifstream input(file_path);
    if (!input.is_open()) {
        return specs;
    }

    std::string line;
    bool first_row = true;
    while (std::getline(input, line)) {
        line = Trim(line);
        if (line.empty()) {
            continue;
        }
        if (first_row) {
            first_row = false;
            if (line.find("id,") == 0) {
                continue;
            }
        }

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> cells;
        while (std::getline(ss, token, ',')) {
            cells.push_back(Trim(token));
        }
        if (cells.size() < 6) {
            continue;
        }

        ObstacleSpec spec;
        spec.id = std::atoi(cells[0].c_str());
        spec.center_x = std::atof(cells[1].c_str());
        spec.center_y = std::atof(cells[2].c_str());
        spec.length = std::atof(cells[3].c_str());
        spec.width = std::atof(cells[4].c_str());
        spec.yaw = std::atof(cells[5].c_str());
        if (cells.size() > 6) {
            spec.x_vel = std::atof(cells[6].c_str());
        }
        if (cells.size() > 7) {
            spec.y_vel = std::atof(cells[7].c_str());
        }
        specs.push_back(spec);
    }
    return specs;
}

planning_msgs::Obstacle CreateBoxObstacle(const ObstacleSpec& spec)
{
    planning_msgs::Obstacle obstacle;
    obstacle.number = spec.id;
    obstacle.x = spec.center_x;
    obstacle.y = spec.center_y;
    obstacle.x_vel = spec.x_vel;
    obstacle.y_vel = spec.y_vel;
    obstacle.s_vel = 0.0;
    obstacle.l_vel = 0.0;
    obstacle.is_dynamic_obs = (std::hypot(spec.x_vel, spec.y_vel) > 1e-4);
    obstacle.max_length = spec.length;

    const double half_length = 0.5 * spec.length;
    const double half_width = 0.5 * spec.width;
    const double cos_yaw = std::cos(spec.yaw);
    const double sin_yaw = std::sin(spec.yaw);
    const double dx = half_length * cos_yaw;
    const double dy = half_length * sin_yaw;
    const double wx = -half_width * sin_yaw;
    const double wy = half_width * cos_yaw;

    const std::vector<std::pair<double, double>> corners = {
        {spec.center_x + dx + wx, spec.center_y + dy + wy},
        {spec.center_x + dx - wx, spec.center_y + dy - wy},
        {spec.center_x - dx + wx, spec.center_y - dy + wy},
        {spec.center_x - dx - wx, spec.center_y - dy - wy},
    };

    obstacle.max_x = corners.front().first;
    obstacle.min_x = corners.front().first;
    obstacle.max_y = corners.front().second;
    obstacle.min_y = corners.front().second;

    for (size_t i = 0; i < corners.size(); ++i) {
        obstacle.bounding_boxs[i].x = corners[i].first;
        obstacle.bounding_boxs[i].y = corners[i].second;
        obstacle.max_x = std::max(obstacle.max_x, corners[i].first);
        obstacle.min_x = std::min(obstacle.min_x, corners[i].first);
        obstacle.max_y = std::max(obstacle.max_y, corners[i].second);
        obstacle.min_y = std::min(obstacle.min_y, corners[i].second);
    }

    return obstacle;
}

std::vector<ObstacleSpec> LoadObstacleSpecs(ros::NodeHandle& pnh,
                                           const std::string& emplanner_pkg_path,
                                           double ref_x,
                                           double ref_y,
                                           double ref_yaw)
{
    std::vector<ObstacleSpec> specs;

    std::string obstacle_csv;
    std::string obstacle_source_scenario;
    pnh.param("obstacle_csv", obstacle_csv, std::string());
    pnh.param("obstacle_source_scenario", obstacle_source_scenario, std::string());
    if (obstacle_csv.empty() && !obstacle_source_scenario.empty() && !emplanner_pkg_path.empty()) {
        obstacle_csv =
            emplanner_pkg_path + "/benchmark_results/" + obstacle_source_scenario + "/obstacles.csv";
    }
    if (!obstacle_csv.empty()) {
        specs = LoadObstacleSpecsFromCsv(obstacle_csv);
        if (!specs.empty()) {
            return specs;
        }
        ROS_WARN_STREAM("Failed to load obstacle specs from csv: " << obstacle_csv
                        << ". Falling back to inline obstacle parameters.");
    }

    XmlRpc::XmlRpcValue raw_obstacles;
    if (pnh.getParam("obstacles", raw_obstacles) &&
        raw_obstacles.getType() == XmlRpc::XmlRpcValue::TypeArray) {
        for (int i = 0; i < raw_obstacles.size(); ++i) {
            const XmlRpc::XmlRpcValue& item = raw_obstacles[i];
            if (item.getType() != XmlRpc::XmlRpcValue::TypeStruct) {
                continue;
            }

            ObstacleSpec spec;
            spec.id = static_cast<int>(GetStructNumber(item, "id", i));
            spec.length = GetStructNumber(item, "length", 1.0);
            spec.width = GetStructNumber(item, "width", 0.8);
            spec.x_vel = GetStructNumber(item, "x_vel", 0.0);
            spec.y_vel = GetStructNumber(item, "y_vel", 0.0);
            spec.yaw = GetStructNumber(item, "yaw", ref_yaw);

            if (item.hasMember("center_x") && item.hasMember("center_y")) {
                spec.center_x = GetStructNumber(item, "center_x", ref_x);
                spec.center_y = GetStructNumber(item, "center_y", ref_y);
            } else {
                const double s_offset = GetStructNumber(item, "s_offset", 2.5 + i * 1.0);
                const double l_offset = GetStructNumber(item, "l_offset", 0.0);
                spec.center_x = ref_x + std::cos(ref_yaw) * s_offset - std::sin(ref_yaw) * l_offset;
                spec.center_y = ref_y + std::sin(ref_yaw) * s_offset + std::cos(ref_yaw) * l_offset;
            }
            specs.push_back(spec);
        }
    }

    if (!specs.empty()) {
        return specs;
    }

    std::string centers_x_csv;
    std::string centers_y_csv;
    pnh.param("obstacle_centers_x", centers_x_csv, std::string());
    pnh.param("obstacle_centers_y", centers_y_csv, std::string());
    const std::vector<double> centers_x = ParseCsvDoubles(centers_x_csv);
    const std::vector<double> centers_y = ParseCsvDoubles(centers_y_csv);
    if (!centers_x.empty() && centers_x.size() == centers_y.size()) {
        std::string lengths_csv;
        std::string widths_csv;
        std::string yaws_csv;
        std::string x_vels_csv;
        std::string y_vels_csv;
        double default_length = 1.2;
        double default_width = 0.8;
        double default_yaw = ref_yaw;
        double default_x_vel = 0.0;
        double default_y_vel = 0.0;
        pnh.param("default_obstacle_length", default_length, default_length);
        pnh.param("default_obstacle_width", default_width, default_width);
        pnh.param("obstacle_yaw", default_yaw, default_yaw);
        pnh.param("obstacle_x_vel", default_x_vel, default_x_vel);
        pnh.param("obstacle_y_vel", default_y_vel, default_y_vel);
        pnh.param("obstacle_lengths", lengths_csv, std::string());
        pnh.param("obstacle_widths", widths_csv, std::string());
        pnh.param("obstacle_yaws", yaws_csv, std::string());
        pnh.param("obstacle_x_vels", x_vels_csv, std::string());
        pnh.param("obstacle_y_vels", y_vels_csv, std::string());
        const std::vector<double> lengths = ParseCsvDoubles(lengths_csv);
        const std::vector<double> widths = ParseCsvDoubles(widths_csv);
        const std::vector<double> yaws = ParseCsvDoubles(yaws_csv);
        const std::vector<double> x_vels = ParseCsvDoubles(x_vels_csv);
        const std::vector<double> y_vels = ParseCsvDoubles(y_vels_csv);

        for (size_t i = 0; i < centers_x.size(); ++i) {
            ObstacleSpec spec;
            spec.id = static_cast<int>(i);
            spec.center_x = centers_x[i];
            spec.center_y = centers_y[i];
            spec.length = GetIndexedValueOrDefault(lengths, i, default_length);
            spec.width = GetIndexedValueOrDefault(widths, i, default_width);
            spec.yaw = GetIndexedValueOrDefault(yaws, i, default_yaw);
            spec.x_vel = GetIndexedValueOrDefault(x_vels, i, default_x_vel);
            spec.y_vel = GetIndexedValueOrDefault(y_vels, i, default_y_vel);
            specs.push_back(spec);
        }
        return specs;
    }

    double direct_center_x = 0.0;
    double direct_center_y = 0.0;
    if (pnh.getParam("obstacle_center_x", direct_center_x) &&
        pnh.getParam("obstacle_center_y", direct_center_y)) {
        ObstacleSpec direct_spec;
        direct_spec.id = 0;
        direct_spec.center_x = direct_center_x;
        direct_spec.center_y = direct_center_y;
        pnh.param("default_obstacle_length", direct_spec.length, 1.2);
        pnh.param("default_obstacle_width", direct_spec.width, 0.8);
        pnh.param("obstacle_yaw", direct_spec.yaw, ref_yaw);
        pnh.param("obstacle_x_vel", direct_spec.x_vel, 0.0);
        pnh.param("obstacle_y_vel", direct_spec.y_vel, 0.0);
        specs.push_back(direct_spec);
        return specs;
    }

    double default_s_offset = 2.5;
    double default_l_offset = 0.2;
    double default_length = 1.2;
    double default_width = 0.8;
    pnh.param("default_obstacle_s_offset", default_s_offset, default_s_offset);
    pnh.param("default_obstacle_l_offset", default_l_offset, default_l_offset);
    pnh.param("default_obstacle_length", default_length, default_length);
    pnh.param("default_obstacle_width", default_width, default_width);

    ObstacleSpec default_spec;
    default_spec.id = 0;
    default_spec.length = default_length;
    default_spec.width = default_width;
    default_spec.yaw = ref_yaw;
    default_spec.center_x =
        ref_x + std::cos(ref_yaw) * default_s_offset - std::sin(ref_yaw) * default_l_offset;
    default_spec.center_y =
        ref_y + std::sin(ref_yaw) * default_s_offset + std::cos(ref_yaw) * default_l_offset;
    specs.push_back(default_spec);
    return specs;
}

std::string ResolveEmplannerPackagePath()
{
    const char* env_pkg_path = std::getenv("EMPLANNER_PKG_PATH");
    if (env_pkg_path != nullptr && env_pkg_path[0] != '\0') {
        return std::string(env_pkg_path);
    }

    const std::string source_path = __FILE__;
    const std::string suffix = "/src/compare_benchmark.cpp";
    const size_t suffix_pos = source_path.rfind(suffix);
    if (suffix_pos != std::string::npos) {
        return source_path.substr(0, suffix_pos);
    }

    const std::string pkg_path = ros::package::getPath("emplanner");
    if (!pkg_path.empty()) {
        return pkg_path;
    }

    return std::string();
}

std::vector<PathSample> BuildSamplesFromXY(const std::vector<Eigen::Vector2d>& points,
                                           const Eigen::VectorXd* global_s = nullptr)
{
    std::vector<PathSample> samples;
    samples.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        const double x = points[i](0);
        const double y = points[i](1);
        if (!std::isfinite(x) || !std::isfinite(y)) {
            continue;
        }
        PathSample sample;
        sample.index = static_cast<int>(samples.size());
        sample.x = x;
        sample.y = y;
        if (global_s != nullptr && static_cast<Eigen::Index>(i) < global_s->size()) {
            const double s_value = (*global_s)(static_cast<Eigen::Index>(i));
            if (std::isfinite(s_value)) {
                sample.s = s_value;
            }
        }
        if (sample.s == 0.0 && !samples.empty()) {
            const double dx = sample.x - samples.back().x;
            const double dy = sample.y - samples.back().y;
            sample.s = samples.back().s + std::hypot(dx, dy);
        }
        if (!samples.empty() && sample.s < samples.back().s) {
            sample.s = samples.back().s;
        }
        samples.push_back(sample);
    }
    if (samples.size() < 3) {
        return samples;
    }

    for (size_t i = 0; i < samples.size(); ++i) {
        const size_t prev = (i == 0) ? 0 : i - 1;
        const size_t next = (i + 1 >= samples.size()) ? samples.size() - 1 : i + 1;
        samples[i].yaw = std::atan2(samples[next].y - samples[prev].y,
                                    samples[next].x - samples[prev].x);
    }

    if (samples.size() >= 5) {
        for (size_t i = 2; i + 2 < samples.size(); ++i) {
            const double ax = samples[i + 2].x - samples[i].x;
            const double ay = samples[i + 2].y - samples[i].y;
            const double bx = samples[i + 2].x - samples[i - 2].x;
            const double by = samples[i + 2].y - samples[i - 2].y;
            const double cx = samples[i].x - samples[i - 2].x;
            const double cy = samples[i].y - samples[i - 2].y;
            const double a = std::hypot(ax, ay);
            const double b = std::hypot(bx, by);
            const double c = std::hypot(cx, cy);
            if (a < 1e-6 || b < 1e-6 || c < 1e-6) {
                samples[i].kappa = 0.0;
                continue;
            }
            double cos_theta = (a * a + c * c - b * b) / (2.0 * a * c);
            cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
            const double theta = std::acos(cos_theta);
            samples[i].kappa = 2.0 * std::sin(theta) / b;
            if (!std::isfinite(samples[i].kappa)) {
                samples[i].kappa = 0.0;
            }
        }
        samples[0].kappa = samples[2].kappa;
        samples[1].kappa = samples[2].kappa;
        samples[samples.size() - 2].kappa = samples[samples.size() - 3].kappa;
        samples[samples.size() - 1].kappa = samples[samples.size() - 3].kappa;
    }

    return samples;
}

bool InterpolateReferencePoseAtS(const std::vector<PathSample>& reference_samples,
                                 double target_s,
                                 PathSample& pose)
{
    if (reference_samples.empty()) {
        return false;
    }
    if (target_s <= reference_samples.front().s) {
        pose = reference_samples.front();
        return true;
    }
    if (target_s >= reference_samples.back().s) {
        pose = reference_samples.back();
        return true;
    }

    for (size_t i = 1; i < reference_samples.size(); ++i) {
        if (reference_samples[i].s < target_s) {
            continue;
        }
        const PathSample& prev = reference_samples[i - 1];
        const PathSample& next = reference_samples[i];
        const double segment_ds = next.s - prev.s;
        const double ratio =
            (segment_ds > 1e-6) ? ((target_s - prev.s) / segment_ds) : 0.0;
        pose = prev;
        pose.s = target_s;
        pose.x = prev.x + ratio * (next.x - prev.x);
        pose.y = prev.y + ratio * (next.y - prev.y);
        pose.yaw = std::atan2(next.y - prev.y, next.x - prev.x);
        return true;
    }

    pose = reference_samples.back();
    return true;
}

std::vector<GridPointSample> BuildDpGridSamples(const std::vector<PathSample>& reference_samples,
                                                double start_s,
                                                double sample_s,
                                                double sample_l,
                                                int col_node_num,
                                                int row_node_num)
{
    std::vector<GridPointSample> grid_samples;
    if (reference_samples.empty() || !(sample_s > 0.0) || !(sample_l > 0.0) ||
        col_node_num <= 0 || row_node_num <= 0) {
        return grid_samples;
    }

    grid_samples.reserve(static_cast<size_t>(col_node_num) * static_cast<size_t>(row_node_num));
    for (int col = 0; col < col_node_num; ++col) {
        const double global_s = start_s + static_cast<double>(col + 1) * sample_s;
        PathSample ref_pose;
        if (!InterpolateReferencePoseAtS(reference_samples, global_s, ref_pose)) {
            continue;
        }
        for (int row = 0; row < row_node_num; ++row) {
            const double l =
                (static_cast<double>(row_node_num - 1) / 2.0 - static_cast<double>(row)) * sample_l;
            GridPointSample sample;
            sample.index = static_cast<int>(grid_samples.size());
            sample.col = col + 1;
            sample.row = row;
            sample.s = global_s;
            sample.l = l;
            sample.yaw = ref_pose.yaw;
            sample.x = ref_pose.x - std::sin(ref_pose.yaw) * l;
            sample.y = ref_pose.y + std::cos(ref_pose.yaw) * l;
            grid_samples.push_back(sample);
        }
    }
    return grid_samples;
}

std::vector<StCurveSample> BuildSpeedProfileSamples(const Speed_plan_points& speed_points)
{
    std::vector<StCurveSample> samples;
    const size_t count = std::min(
        std::min(speed_points.s.size(), speed_points.s_dot.size()),
        std::min(speed_points.s_dot2.size(), speed_points.s_time.size()));
    samples.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        StCurveSample sample;
        sample.index = static_cast<int>(i);
        sample.t = speed_points.s_time[i];
        sample.s = speed_points.s[i];
        sample.v = speed_points.s_dot[i];
        sample.a = speed_points.s_dot2[i];
        samples.push_back(sample);
    }
    return samples;
}

std::vector<StCurveSample> BuildSpeedDpPathSamples(const std::vector<double>& dp_path_s,
                                                   const std::vector<StCurveSample>& qp_samples,
                                                   int last_feasible_index)
{
    std::vector<StCurveSample> samples;
    samples.reserve(dp_path_s.size());
    for (size_t i = 0; i < dp_path_s.size(); ++i) {
        StCurveSample sample;
        sample.index = static_cast<int>(i);
        sample.s = dp_path_s[i];
        if (i < qp_samples.size()) {
            sample.t = qp_samples[i].t;
            sample.v = qp_samples[i].v;
            sample.a = qp_samples[i].a;
        } else if (!samples.empty() && i > 0) {
            sample.t = samples.back().t;
        }
        sample.is_feasible =
            last_feasible_index < 0 ? false : static_cast<int>(i) <= last_feasible_index;
        samples.push_back(sample);
    }
    return samples;
}

std::vector<StGridSample> BuildStGridSamples(const Speed_Plan_DP_ST_nodes& st_nodes)
{
    std::vector<StGridSample> samples;
    for (size_t col = 0; col < st_nodes.col_nodes.size(); ++col) {
        const auto& col_nodes = st_nodes.col_nodes[col].rol_nodes;
        for (size_t row = 0; row < col_nodes.size(); ++row) {
            const auto& node = col_nodes[row];
            StGridSample sample;
            sample.index = static_cast<int>(samples.size());
            sample.col = static_cast<int>(col);
            sample.row = static_cast<int>(row);
            sample.t = node.node_t;
            sample.s = node.node_s;
            sample.v = node.node_s_dot;
            sample.a = node.node_s_dot2;
            sample.cost = node.cost;
            sample.is_possible = node.is_possible;
            samples.push_back(sample);
        }
    }
    return samples;
}

double PathLength(const std::vector<PathSample>& samples)
{
    if (samples.empty()) {
        return 0.0;
    }
    return samples.back().s;
}

double MeanAbsKappa(const std::vector<PathSample>& samples)
{
    if (samples.empty()) {
        return 0.0;
    }
    double total = 0.0;
    size_t valid_count = 0;
    for (const auto& sample : samples) {
        if (!std::isfinite(sample.kappa)) {
            continue;
        }
        total += std::abs(sample.kappa);
        ++valid_count;
    }
    if (valid_count == 0) {
        return 0.0;
    }
    return total / static_cast<double>(valid_count);
}

double MaxAbsKappa(const std::vector<PathSample>& samples)
{
    double max_abs_kappa = 0.0;
    for (const auto& sample : samples) {
        if (!std::isfinite(sample.kappa)) {
            continue;
        }
        max_abs_kappa = std::max(max_abs_kappa, std::abs(sample.kappa));
    }
    return max_abs_kappa;
}

void WritePathCsv(const std::string& file_path, const std::vector<PathSample>& samples)
{
    std::ofstream output(file_path);
    output << "index,s,x,y,yaw,kappa\n";
    output << std::fixed << std::setprecision(6);
    for (const auto& sample : samples) {
        output << sample.index << ","
               << sample.s << ","
               << sample.x << ","
               << sample.y << ","
               << sample.yaw << ","
               << sample.kappa << "\n";
    }
}

void WriteObstacleCsv(const std::string& file_path, const std::vector<ObstacleSpec>& obstacles)
{
    std::ofstream output(file_path);
    output << "id,center_x,center_y,length,width,yaw,x_vel,y_vel\n";
    output << std::fixed << std::setprecision(6);
    for (const auto& obstacle : obstacles) {
        output << obstacle.id << ","
               << obstacle.center_x << ","
               << obstacle.center_y << ","
               << obstacle.length << ","
               << obstacle.width << ","
               << obstacle.yaw << ","
               << obstacle.x_vel << ","
               << obstacle.y_vel << "\n";
    }
}

void WriteGridCsv(const std::string& file_path, const std::vector<GridPointSample>& samples)
{
    std::ofstream output(file_path);
    output << "index,col,row,s,l,x,y,yaw\n";
    output << std::fixed << std::setprecision(6);
    for (const auto& sample : samples) {
        output << sample.index << ","
               << sample.col << ","
               << sample.row << ","
               << sample.s << ","
               << sample.l << ","
               << sample.x << ","
               << sample.y << ","
               << sample.yaw << "\n";
    }
}

void WriteSpeedProfileCsv(const std::string& file_path, const std::vector<StCurveSample>& samples)
{
    std::ofstream output(file_path);
    output << "index,t,s,v,a\n";
    output << std::fixed << std::setprecision(6);
    for (const auto& sample : samples) {
        output << sample.index << ","
               << sample.t << ","
               << sample.s << ","
               << sample.v << ","
               << sample.a << "\n";
    }
}

void WriteStDpPathCsv(const std::string& file_path, const std::vector<StCurveSample>& samples)
{
    std::ofstream output(file_path);
    output << "index,t,s,is_feasible\n";
    output << std::fixed << std::setprecision(6);
    for (const auto& sample : samples) {
        output << sample.index << ","
               << sample.t << ","
               << sample.s << ","
               << (sample.is_feasible ? 1 : 0) << "\n";
    }
}

void WriteStGridCsv(const std::string& file_path, const std::vector<StGridSample>& samples)
{
    std::ofstream output(file_path);
    output << "index,col,row,t,s,v,a,cost,is_possible\n";
    output << std::fixed << std::setprecision(6);
    for (const auto& sample : samples) {
        output << sample.index << ","
               << sample.col << ","
               << sample.row << ","
               << sample.t << ","
               << sample.s << ","
               << sample.v << ","
               << sample.a << ","
               << sample.cost << ","
               << (sample.is_possible ? 1 : 0) << "\n";
    }
}

void WriteStObstacleCsv(const std::string& file_path, const planning_msgs::ObstacleList& obstacle_list)
{
    std::ofstream output(file_path);
    output << "id,is_consider,is_dynamic,x,y,s,l,min_s,max_s,min_l,max_l,s_vel,l_vel,t_in,t_out,s_in,s_out,"
           << "corner0_s,corner0_l,corner1_s,corner1_l,corner2_s,corner2_l,corner3_s,corner3_l\n";
    output << std::fixed << std::setprecision(6);
    for (const auto& obstacle : obstacle_list.obstacles) {
        double raw_min_corner_s = std::numeric_limits<double>::infinity();
        for (int i = 0; i < 4; ++i) {
            raw_min_corner_s = std::min(raw_min_corner_s, static_cast<double>(obstacle.bounding_boxs_SL[i].x));
        }
        const double corner_s_offset = static_cast<double>(obstacle.min_s) - raw_min_corner_s;
        output << obstacle.number << ","
               << (obstacle.is_consider ? 1 : 0) << ","
               << (obstacle.is_dynamic_obs ? 1 : 0) << ","
               << obstacle.x << ","
               << obstacle.y << ","
               << obstacle.s << ","
               << obstacle.l << ","
               << obstacle.min_s << ","
               << obstacle.max_s << ","
               << obstacle.min_l << ","
               << obstacle.max_l << ","
               << obstacle.s_vel << ","
               << obstacle.l_vel << ","
               << obstacle.speed_plan_t_in << ","
               << obstacle.speed_plan_t_out << ","
               << obstacle.speed_plan_s_in << ","
               << obstacle.speed_plan_s_out << ","
               << obstacle.bounding_boxs_SL[0].x + corner_s_offset << ","
               << obstacle.bounding_boxs_SL[0].y << ","
               << obstacle.bounding_boxs_SL[1].x + corner_s_offset << ","
               << obstacle.bounding_boxs_SL[1].y << ","
               << obstacle.bounding_boxs_SL[2].x + corner_s_offset << ","
               << obstacle.bounding_boxs_SL[2].y << ","
               << obstacle.bounding_boxs_SL[3].x + corner_s_offset << ","
               << obstacle.bounding_boxs_SL[3].y << "\n";
    }
}

void WriteSummary(const std::string& file_path,
                  const std::string& scenario_name,
                  const std::string& turn_angle_case,
                  const std::string& turn_shape_case,
                  const std::string& dp_source,
                  bool qp_running_normally,
                  bool speed_plan_available,
                  double sample_s,
                  double sample_l,
                  int col_node_num,
                  int row_node_num,
                  double straight_turn_x,
                  double straight_turn_angle_deg,
                  double straight_turn_arc_length,
                  double second_turn_gap,
                  double second_turn_angle_deg,
                  double second_turn_arc_length,
                  int rl_dp_s_samples,
                  double rl_dp_s_max,
                  double w_qp_l,
                  double w_qp_dl,
                  double w_qp_ddl,
                  double w_qp_ref_dp,
                  double speed_plan_t_dt,
                  double st_s_min_step,
                  double st_lateral_limit,
                  double speed_reference,
                  double speed_dp_ref_vel_weight,
                  double speed_dp_hard_distance,
                  double dp_vel_max,
                  double dp_a_max,
                  double speed_qp_ref_s_weight,
                  double speed_qp_safe_distance,
                  double speed_qp_v_max,
                  double speed_qp_a_max,
                  double plan_time,
                  double speed_plan_distance,
                  double planner_total_ms,
                  double dp_sampling_ms,
                  double qp_optimization_ms,
                  double speed_planning_ms,
                  size_t obstacle_count,
                  const std::vector<PathSample>& dp_samples,
                  const std::vector<PathSample>& qp_samples)
{
    std::ofstream output(file_path);
    output << std::fixed << std::setprecision(6);
    output << "scenario_name: " << scenario_name << "\n";
    output << "turn_angle_case: " << turn_angle_case << "\n";
    output << "turn_shape_case: " << turn_shape_case << "\n";
    output << "dp_source: " << dp_source << "\n";
    output << "qp_running_normally: " << (qp_running_normally ? "true" : "false") << "\n";
    output << "speed_plan_available: " << (speed_plan_available ? "true" : "false") << "\n";
    output << "sample_s: " << sample_s << "\n";
    output << "sample_l: " << sample_l << "\n";
    output << "col_node_num: " << col_node_num << "\n";
    output << "row_node_num: " << row_node_num << "\n";
    output << "planning_s_horizon: " << sample_s * static_cast<double>(col_node_num) << "\n";
    output << "straight_turn_x: " << straight_turn_x << "\n";
    output << "straight_turn_angle_deg: " << straight_turn_angle_deg << "\n";
    output << "straight_turn_arc_length: " << straight_turn_arc_length << "\n";
    output << "second_turn_gap: " << second_turn_gap << "\n";
    output << "second_turn_angle_deg: " << second_turn_angle_deg << "\n";
    output << "second_turn_arc_length: " << second_turn_arc_length << "\n";
    output << "rl_dp_s_samples: " << rl_dp_s_samples << "\n";
    output << "rl_dp_s_max: " << rl_dp_s_max << "\n";
    output << "w_qp_l: " << w_qp_l << "\n";
    output << "w_qp_dl: " << w_qp_dl << "\n";
    output << "w_qp_ddl: " << w_qp_ddl << "\n";
    output << "w_qp_ref_dp: " << w_qp_ref_dp << "\n";
    output << "speed_plan_t_dt: " << speed_plan_t_dt << "\n";
    output << "st_s_min_step: " << st_s_min_step << "\n";
    output << "st_lateral_limit: " << st_lateral_limit << "\n";
    output << "speed_reference: " << speed_reference << "\n";
    output << "speed_dp_ref_vel_weight: " << speed_dp_ref_vel_weight << "\n";
    output << "speed_dp_hard_distance: " << speed_dp_hard_distance << "\n";
    output << "dp_vel_max: " << dp_vel_max << "\n";
    output << "dp_a_max: " << dp_a_max << "\n";
    output << "speed_qp_ref_s_weight: " << speed_qp_ref_s_weight << "\n";
    output << "speed_qp_safe_distance: " << speed_qp_safe_distance << "\n";
    output << "speed_qp_v_max: " << speed_qp_v_max << "\n";
    output << "speed_qp_a_max: " << speed_qp_a_max << "\n";
    output << "plan_time: " << plan_time << "\n";
    output << "speed_plan_distance: " << speed_plan_distance << "\n";
    output << "planner_total_ms: " << planner_total_ms << "\n";
    output << "dp_sampling_ms: " << dp_sampling_ms << "\n";
    output << "qp_optimization_ms: " << qp_optimization_ms << "\n";
    output << "speed_planning_ms: " << speed_planning_ms << "\n";
    output << "obstacle_count: " << obstacle_count << "\n";
    output << "dp_points: " << dp_samples.size() << "\n";
    output << "qp_points: " << qp_samples.size() << "\n";
    output << "dp_path_length: " << PathLength(dp_samples) << "\n";
    output << "qp_path_length: " << PathLength(qp_samples) << "\n";
    output << "dp_mean_abs_kappa: " << MeanAbsKappa(dp_samples) << "\n";
    output << "qp_mean_abs_kappa: " << MeanAbsKappa(qp_samples) << "\n";
    output << "dp_max_abs_kappa: " << MaxAbsKappa(dp_samples) << "\n";
    output << "qp_max_abs_kappa: " << MaxAbsKappa(qp_samples) << "\n";
}

}  // namespace

int main(int argc, char** argv)
{
    ros::init(argc, argv, "emplanner_compare_benchmark");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    int floor = 1;
    int task_type = 0;
    bool is_indoor = true;
    int plan_iterations = 0;
    bool advance_vehicle = false;
    double plan_step = 0.2;
    double car_speed = 0.0;
    double injected_car_stop = 0.0;
    std::string scenario_name = "default_case";
    pnh.param("floor", floor, floor);
    pnh.param("task_type", task_type, task_type);
    pnh.param("is_indoor", is_indoor, is_indoor);
    pnh.param("plan_iterations", plan_iterations, plan_iterations);
    pnh.param("advance_vehicle", advance_vehicle, advance_vehicle);
    pnh.param("plan_step", plan_step, plan_step);
    pnh.param("car_speed", car_speed, car_speed);
    pnh.param("car_stop_flag", injected_car_stop, injected_car_stop);
    pnh.param("scenario_name", scenario_name, scenario_name);

    const std::string emplanner_pkg_path = ResolveEmplannerPackagePath();
    if (emplanner_pkg_path.empty()) {
        ROS_ERROR("Failed to resolve emplanner package path.");
        return 1;
    }

    std::string output_dir = emplanner_pkg_path + "/benchmark_results/" + scenario_name;
    pnh.param("output_dir", output_dir, output_dir);

    if (!EnsureDirectory(output_dir)) {
        ROS_ERROR_STREAM("Failed to create output directory: " << output_dir);
        return 1;
    }

    const std::string default_trajectory_file = emplanner_pkg_path + "/text/trajectory.txt";
    std::string trajectory_file = default_trajectory_file;
    pnh.param("trajectory_file", trajectory_file, trajectory_file);

    bool use_straight_trajectory = true;
    double straight_start_x = 0.0;
    double straight_start_y = 0.0;
    double straight_length = 10.0;
    double straight_step = 0.1;
    double straight_turn_x = 1e9;
    double straight_turn_angle_deg = 0.0;
    double straight_turn_arc_length = 0.0;
    std::string turn_shape_case = "single_arc";
    double second_turn_gap = 0.8;
    double second_turn_angle_deg = 0.0;
    double second_turn_arc_length = 0.0;
    std::string turn_angle_case;
    pnh.param("use_straight_trajectory", use_straight_trajectory, use_straight_trajectory);
    pnh.param("straight_start_x", straight_start_x, straight_start_x);
    pnh.param("straight_start_y", straight_start_y, straight_start_y);
    pnh.param("straight_length", straight_length, straight_length);
    pnh.param("straight_step", straight_step, straight_step);
    pnh.param("straight_turn_x", straight_turn_x, straight_turn_x);
    pnh.param("straight_turn_angle_deg", straight_turn_angle_deg, straight_turn_angle_deg);
    pnh.param("straight_turn_arc_length", straight_turn_arc_length, straight_turn_arc_length);
    pnh.param("turn_shape_case", turn_shape_case, turn_shape_case);
    pnh.param("second_turn_gap", second_turn_gap, second_turn_gap);
    pnh.param("second_turn_angle_deg", second_turn_angle_deg, second_turn_angle_deg);
    pnh.param("second_turn_arc_length", second_turn_arc_length, second_turn_arc_length);
    XmlRpc::XmlRpcValue raw_turn_angle_case;
    if (pnh.getParam("turn_angle_case", raw_turn_angle_case)) {
        turn_angle_case = XmlRpcString(raw_turn_angle_case, std::string());
    }
    if (!turn_angle_case.empty()) {
        double resolved_turn_angle_deg = straight_turn_angle_deg;
        if (ResolveTurnAngleCase(turn_angle_case, resolved_turn_angle_deg)) {
            straight_turn_angle_deg = resolved_turn_angle_deg;
        } else {
            ROS_WARN_STREAM("Unsupported turn_angle_case: " << turn_angle_case
                            << ". Falling back to straight_turn_angle_deg="
                            << straight_turn_angle_deg);
        }
    }
    const std::string normalized_turn_shape_case = ToLowerAscii(Trim(turn_shape_case));
    if (normalized_turn_shape_case == "straight") {
        straight_turn_angle_deg = 0.0;
        second_turn_angle_deg = 0.0;
    } else if (normalized_turn_shape_case == "s_curve" || normalized_turn_shape_case == "s") {
        if (std::fabs(second_turn_angle_deg) <= 1e-8) {
            second_turn_angle_deg = -straight_turn_angle_deg;
        }
        if (second_turn_arc_length <= 1e-8) {
            second_turn_arc_length = straight_turn_arc_length;
        }
    } else if (normalized_turn_shape_case != "single_arc" &&
               normalized_turn_shape_case != "single" &&
               !normalized_turn_shape_case.empty()) {
        ROS_WARN_STREAM("Unsupported turn_shape_case: " << turn_shape_case
                        << ". Falling back to single_arc.");
        turn_shape_case = "single_arc";
    }
    if (use_straight_trajectory) {
        trajectory_file = output_dir + "/straight_trajectory.txt";
        if (!WriteStraightTrajectoryFile(trajectory_file,
                                         straight_start_x,
                                         straight_start_y,
                                         straight_length,
                                         straight_step,
                                         straight_turn_x,
                                         straight_turn_angle_deg,
                                         straight_turn_arc_length,
                                         turn_shape_case,
                                         second_turn_gap,
                                         second_turn_angle_deg,
                                         second_turn_arc_length)) {
            ROS_ERROR_STREAM("Failed to write straight trajectory file: " << trajectory_file);
            return 1;
        }
        pnh.setParam("trajectory_file", trajectory_file);
    }

    double default_x = 0.0;
    double default_y = 0.0;
    double default_yaw = 0.0;
    LoadFirstPoseFromPath(trajectory_file, default_x, default_y, default_yaw);

    double start_x = default_x;
    double start_y = default_y;
    double start_yaw = default_yaw;
    pnh.param("start_x", start_x, start_x);
    pnh.param("start_y", start_y, start_y);
    pnh.param("start_yaw", start_yaw, start_yaw);

    double summary_sample_s = 0.8;
    double summary_sample_l = 0.35;
    int summary_col_node_num = 6;
    int summary_row_node_num = 11;
    int summary_rl_dp_s_samples = 9;
    double summary_rl_dp_s_max = 8.0;
    double summary_w_qp_l = 800.0;
    double summary_w_qp_dl = 200.0;
    double summary_w_qp_ddl = 600.0;
    double summary_w_qp_ref_dp = 50.0;
    double summary_speed_plan_t_dt = 0.5;
    double summary_st_s_min_step = 0.2;
    double summary_st_lateral_limit = 1.0;
    double summary_speed_reference = 0.5;
    double summary_speed_dp_ref_vel_weight = 500.0;
    double summary_speed_dp_hard_distance = 0.4;
    double summary_dp_vel_max = 1.2;
    double summary_dp_a_max = 0.6;
    double summary_speed_qp_ref_s_weight = 0.0;
    double summary_speed_qp_safe_distance = 0.01;
    double summary_speed_qp_v_max = 1.2;
    double summary_speed_qp_a_max = 0.5;
    double summary_plan_time = 4.0;
    double summary_speed_plan_distance = 5.0;
    pnh.param("sample_s", summary_sample_s, summary_sample_s);
    pnh.param("sample_l", summary_sample_l, summary_sample_l);
    pnh.param("col_node_num", summary_col_node_num, summary_col_node_num);
    pnh.param("row_node_num", summary_row_node_num, summary_row_node_num);
    pnh.param("rl_dp_s_samples", summary_rl_dp_s_samples, summary_rl_dp_s_samples);
    pnh.param("rl_dp_s_max", summary_rl_dp_s_max, summary_rl_dp_s_max);
    pnh.param("w_qp_l", summary_w_qp_l, summary_w_qp_l);
    pnh.param("w_qp_dl", summary_w_qp_dl, summary_w_qp_dl);
    pnh.param("w_qp_ddl", summary_w_qp_ddl, summary_w_qp_ddl);
    pnh.param("w_qp_ref_dp", summary_w_qp_ref_dp, summary_w_qp_ref_dp);
    pnh.param("speed_plan_t_dt", summary_speed_plan_t_dt, summary_speed_plan_t_dt);
    pnh.param("st_s_min_step", summary_st_s_min_step, summary_st_s_min_step);
    pnh.param("st_lateral_limit", summary_st_lateral_limit, summary_st_lateral_limit);
    pnh.param("speed_reference", summary_speed_reference, summary_speed_reference);
    pnh.param("speed_dp_ref_vel_weight", summary_speed_dp_ref_vel_weight, summary_speed_dp_ref_vel_weight);
    pnh.param("speed_dp_hard_distance", summary_speed_dp_hard_distance, summary_speed_dp_hard_distance);
    pnh.param("dp_vel_max", summary_dp_vel_max, summary_dp_vel_max);
    pnh.param("dp_a_max", summary_dp_a_max, summary_dp_a_max);
    pnh.param("speed_qp_ref_s_weight", summary_speed_qp_ref_s_weight, summary_speed_qp_ref_s_weight);
    pnh.param("speed_qp_safe_distance", summary_speed_qp_safe_distance, summary_speed_qp_safe_distance);
    pnh.param("speed_qp_v_max", summary_speed_qp_v_max, summary_speed_qp_v_max);
    pnh.param("speed_qp_a_max", summary_speed_qp_a_max, summary_speed_qp_a_max);
    pnh.param("plan_time", summary_plan_time, summary_plan_time);
    pnh.param("speed_plan_distance", summary_speed_plan_distance, summary_speed_plan_distance);

    planning_msgs::car_scene scene;
    scene.floor = static_cast<int8_t>(floor);
    scene.task_type = static_cast<uint8_t>(task_type);
    scene.is_indoor = is_indoor;

    EMPlanner planner(nh, scene);

    planning_msgs::car_info car_state;
    car_state.x = start_x;
    car_state.y = start_y;
    car_state.yaw = start_yaw;
    car_state.speed = std::fabs(car_speed);
    car_state.speedDrietion = (car_speed >= 0.0) ? 1 : -1;
    car_state.turnAngle = 0.0;
    car_state.yawrate = 0.0;

    const std::vector<ObstacleSpec> obstacle_specs =
        LoadObstacleSpecs(pnh, emplanner_pkg_path, start_x, start_y, start_yaw);
    planning_msgs::ObstacleList obstacle_list;
    obstacle_list.header.frame_id = "map";
    obstacle_list.goal_point.header = obstacle_list.header;
    for (const auto& spec : obstacle_specs) {
        obstacle_list.obstacles.push_back(CreateBoxObstacle(spec));
    }

    planner.InjectSimCarStop(static_cast<float>(injected_car_stop));
    planner.InjectSimCarState(car_state);
    planner.InjectSimObstacleList(obstacle_list);
    planner.ReinitializeForCurrentInputs();

    for (int iteration = 0; iteration < std::max(0, plan_iterations); ++iteration) {
        planner.InjectSimCarState(car_state);
        planner.InjectSimObstacleList(obstacle_list);
        planner.Plan(scene);

        if (advance_vehicle) {
            car_state.x += std::cos(car_state.yaw) * car_speed * plan_step;
            car_state.y += std::sin(car_state.yaw) * car_speed * plan_step;
        }
    }

    if (!planner.HasLatestPlanResult()) {
        ROS_ERROR("Planner did not produce a valid result.");
        return 2;
    }

    const std::vector<PathSample> dp_samples =
        BuildSamplesFromXY(planner.GetLatestDpPathXY(), &planner.GetLatestDpPathSL().s);
    const std::vector<PathSample> qp_samples =
        BuildSamplesFromXY(planner.GetLatestQpPathXY(), &planner.GetLatestQpPathSL().s);
    const std::vector<PathSample> reference_samples =
        BuildSamplesFromXY(LoadTrajectoryXY(trajectory_file));
    const double grid_start_s =
        (!dp_samples.empty() ? dp_samples.front().s : (reference_samples.empty() ? 0.0 : reference_samples.front().s));
    const std::vector<GridPointSample> dp_grid_samples =
        BuildDpGridSamples(reference_samples,
                           grid_start_s,
                           summary_sample_s,
                           summary_sample_l,
                           summary_col_node_num,
                           summary_row_node_num);
    if (dp_samples.empty() || qp_samples.empty()) {
        ROS_ERROR("Planner result is missing DP or QP samples.");
        return 3;
    }

    const std::string dp_csv = output_dir + "/dp_path.csv";
    const std::string qp_csv = output_dir + "/qp_path.csv";
    const std::string reference_csv = output_dir + "/reference_path.csv";
    const std::string grid_csv = output_dir + "/dp_grid_points.csv";
    const std::string obstacle_csv = output_dir + "/obstacles.csv";
    const std::string speed_profile_csv = output_dir + "/speed_profile.csv";
    const std::string st_dp_path_csv = output_dir + "/st_dp_path.csv";
    const std::string st_grid_csv = output_dir + "/st_lattice.csv";
    const std::string st_obstacle_csv = output_dir + "/st_obstacles.csv";
    const std::string summary_txt = output_dir + "/summary.txt";
    WritePathCsv(dp_csv, dp_samples);
    WritePathCsv(qp_csv, qp_samples);
    if (!reference_samples.empty()) {
        WritePathCsv(reference_csv, reference_samples);
    }
    if (!dp_grid_samples.empty()) {
        WriteGridCsv(grid_csv, dp_grid_samples);
    }
    WriteObstacleCsv(obstacle_csv, obstacle_specs);
    const bool speed_plan_available = planner.HasLatestSpeedPlanResult();
    if (speed_plan_available) {
        const std::vector<StCurveSample> speed_profile_samples =
            BuildSpeedProfileSamples(planner.GetLatestSpeedQpPoints());
        const std::vector<StCurveSample> st_dp_path_samples =
            BuildSpeedDpPathSamples(planner.GetLatestSpeedDpPathS(),
                                    speed_profile_samples,
                                    planner.GetLatestSpeedDpLastFeasibleIndex());
        const std::vector<StGridSample> st_grid_samples =
            BuildStGridSamples(planner.GetLatestSpeedDpStNodes());
        if (!speed_profile_samples.empty()) {
            WriteSpeedProfileCsv(speed_profile_csv, speed_profile_samples);
        }
        if (!st_dp_path_samples.empty()) {
            WriteStDpPathCsv(st_dp_path_csv, st_dp_path_samples);
        }
        if (!st_grid_samples.empty()) {
            WriteStGridCsv(st_grid_csv, st_grid_samples);
        }
        WriteStObstacleCsv(st_obstacle_csv, planner.GetLatestSpeedObstacleListSL());
    }
    WriteSummary(summary_txt,
                 scenario_name,
                 turn_angle_case,
                 turn_shape_case,
                 planner.GetLatestDpSource(),
                 planner.GetLatestQpRunningNormally(),
                 speed_plan_available,
                 summary_sample_s,
                 summary_sample_l,
                 summary_col_node_num,
                 summary_row_node_num,
                 straight_turn_x,
                 straight_turn_angle_deg,
                 straight_turn_arc_length,
                 second_turn_gap,
                 second_turn_angle_deg,
                 second_turn_arc_length,
                 summary_rl_dp_s_samples,
                 summary_rl_dp_s_max,
                 summary_w_qp_l,
                 summary_w_qp_dl,
                 summary_w_qp_ddl,
                 summary_w_qp_ref_dp,
                 summary_speed_plan_t_dt,
                 summary_st_s_min_step,
                 summary_st_lateral_limit,
                 summary_speed_reference,
                 summary_speed_dp_ref_vel_weight,
                 summary_speed_dp_hard_distance,
                 summary_dp_vel_max,
                 summary_dp_a_max,
                 summary_speed_qp_ref_s_weight,
                 summary_speed_qp_safe_distance,
                 summary_speed_qp_v_max,
                 summary_speed_qp_a_max,
                 summary_plan_time,
                 summary_speed_plan_distance,
                 planner.GetLatestPlannerCycleMs(),
                 planner.GetLatestDpSamplingMs(),
                 planner.GetLatestQpOptimizationMs(),
                 planner.GetLatestSpeedPlanningMs(),
                 obstacle_specs.size(),
                 dp_samples,
                 qp_samples);

    std::cout << "benchmark_output_dir=" << output_dir << std::endl;
    std::cout << "turn_angle_case=" << (turn_angle_case.empty() ? "custom" : turn_angle_case)
              << std::endl;
    std::cout << "turn_shape_case=" << turn_shape_case << std::endl;
    std::cout << "dp_source=" << planner.GetLatestDpSource() << std::endl;
    std::cout << "speed_plan_available=" << (speed_plan_available ? "true" : "false") << std::endl;
    std::cout << "planner_total_ms=" << planner.GetLatestPlannerCycleMs() << std::endl;
    std::cout << "dp_sampling_ms=" << planner.GetLatestDpSamplingMs() << std::endl;
    std::cout << "qp_optimization_ms=" << planner.GetLatestQpOptimizationMs() << std::endl;
    std::cout << "speed_planning_ms=" << planner.GetLatestSpeedPlanningMs() << std::endl;
    std::cout << "dp_max_abs_kappa=" << MaxAbsKappa(dp_samples) << std::endl;
    std::cout << "qp_max_abs_kappa=" << MaxAbsKappa(qp_samples) << std::endl;
    std::cout << "dp_mean_abs_kappa=" << MeanAbsKappa(dp_samples) << std::endl;
    std::cout << "qp_mean_abs_kappa=" << MeanAbsKappa(qp_samples) << std::endl;
    return 0;
}
