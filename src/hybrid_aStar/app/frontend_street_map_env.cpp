#include <geometry_msgs/Point.h>
#include <nav_msgs/OccupancyGrid.h>
#include <planning_msgs/car_info.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <zip.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct StreetMaps {
    std::vector<float> data;
    int map_count{0};
    int height{0};
    int width{0};
};

uint16_t ReadLe16(const unsigned char* ptr) {
    return static_cast<uint16_t>(ptr[0]) | (static_cast<uint16_t>(ptr[1]) << 8);
}

uint32_t ReadLe32(const unsigned char* ptr) {
    return static_cast<uint32_t>(ptr[0]) | (static_cast<uint32_t>(ptr[1]) << 8) |
           (static_cast<uint32_t>(ptr[2]) << 16) | (static_cast<uint32_t>(ptr[3]) << 24);
}

std::string SplitArrayName(const std::string& split) {
    if (split == "train") {
        return "arr_0.npy";
    }
    if (split == "valid") {
        return "arr_4.npy";
    }
    return "arr_8.npy";
}

std::vector<char> ReadZipEntry(zip_t* archive, const std::string& entry_name) {
    zip_stat_t stat;
    if (zip_stat(archive, entry_name.c_str(), ZIP_FL_ENC_GUESS, &stat) != 0) {
        throw std::runtime_error("zip entry not found: " + entry_name);
    }
    zip_file_t* file = zip_fopen(archive, entry_name.c_str(), ZIP_FL_ENC_GUESS);
    if (file == nullptr) {
        throw std::runtime_error("failed to open zip entry: " + entry_name);
    }
    std::vector<char> buffer(static_cast<std::size_t>(stat.size));
    zip_int64_t total = 0;
    while (total < stat.size) {
        const zip_int64_t read_size = zip_fread(file, buffer.data() + total, stat.size - total);
        if (read_size < 0) {
            zip_fclose(file);
            throw std::runtime_error("failed reading zip entry: " + entry_name);
        }
        if (read_size == 0) {
            break;
        }
        total += read_size;
    }
    zip_fclose(file);
    if (total != stat.size) {
        throw std::runtime_error("short read for zip entry: " + entry_name);
    }
    return buffer;
}

StreetMaps ParseNpyFloat32Maps(const std::vector<char>& npy_bytes) {
    const unsigned char* raw = reinterpret_cast<const unsigned char*>(npy_bytes.data());
    const std::size_t size = npy_bytes.size();
    if (size < 16 || std::memcmp(raw, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("invalid npy header");
    }
    const uint8_t major = raw[6];
    std::size_t header_len = 0;
    std::size_t data_offset = 0;
    if (major == 1) {
        header_len = ReadLe16(raw + 8);
        data_offset = 10;
    } else if (major == 2) {
        header_len = ReadLe32(raw + 8);
        data_offset = 12;
    } else {
        throw std::runtime_error("unsupported npy version");
    }
    if (data_offset + header_len > size) {
        throw std::runtime_error("corrupted npy header length");
    }

    const std::string header(reinterpret_cast<const char*>(raw + data_offset), header_len);
    if (header.find("'descr': '<f4'") == std::string::npos &&
        header.find("\"descr\": \"<f4\"") == std::string::npos) {
        throw std::runtime_error("expected float32 npy tensor");
    }
    if (header.find("False") == std::string::npos) {
        throw std::runtime_error("fortran-order npy is unsupported");
    }

    const std::size_t shape_pos = header.find("shape");
    const std::size_t lparen = header.find('(', shape_pos);
    const std::size_t rparen = header.find(')', lparen);
    if (shape_pos == std::string::npos || lparen == std::string::npos || rparen == std::string::npos) {
        throw std::runtime_error("failed to parse npy shape");
    }
    std::string shape_text = header.substr(lparen + 1, rparen - lparen - 1);
    std::replace(shape_text.begin(), shape_text.end(), ',', ' ');
    std::stringstream ss(shape_text);

    StreetMaps maps;
    if (!(ss >> maps.map_count >> maps.height >> maps.width)) {
        throw std::runtime_error("expected shape [N,H,W] in npy");
    }
    const std::size_t payload_offset = data_offset + header_len;
    const std::size_t expected_values =
        static_cast<std::size_t>(maps.map_count) * static_cast<std::size_t>(maps.height) *
        static_cast<std::size_t>(maps.width);
    const std::size_t expected_bytes = expected_values * sizeof(float);
    if (payload_offset + expected_bytes > size) {
        throw std::runtime_error("npy payload truncated");
    }
    maps.data.resize(expected_values);
    std::memcpy(maps.data.data(), raw + payload_offset, expected_bytes);
    return maps;
}

StreetMaps LoadStreetMaps(const std::string& dataset_path, const std::string& split) {
    int err = 0;
    zip_t* archive = zip_open(dataset_path.c_str(), ZIP_RDONLY, &err);
    if (archive == nullptr) {
        throw std::runtime_error("failed to open npz dataset: " + dataset_path);
    }
    try {
        const std::vector<char> bytes = ReadZipEntry(archive, SplitArrayName(split));
        zip_close(archive);
        return ParseNpyFloat32Maps(bytes);
    } catch (...) {
        zip_close(archive);
        throw;
    }
}

inline std::size_t FlatIndex(int x, int y, int width) {
    return static_cast<std::size_t>(y * width + x);
}

class FrontendStreetMapEnvironment {
public:
    explicit FrontendStreetMapEnvironment(ros::NodeHandle& nh)
        : nh_(nh), rng_(static_cast<std::mt19937::result_type>(seed_)) {
        LoadParams();
        maps_ = LoadStreetMaps(dataset_path_, split_);
        map_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>(map_topic_, 1, true);
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(marker_topic_, 1, true);
        car_pose_pub_ = nh_.advertise<planning_msgs::car_info>(car_pose_topic_, 1, true);

        PublishMap();
        if (auto_refresh_) {
            timer_ = nh_.createTimer(ros::Duration(refresh_period_),
                                     &FrontendStreetMapEnvironment::TimerCallback,
                                     this);
        }
    }

private:
    void LoadParams() {
        world_frame_ = nh_.param<std::string>("frontend_street_map/world_frame", "velodyne");
        map_topic_ = nh_.param<std::string>("frontend_street_map/map_topic", "/guided_frontend_street_map");
        marker_topic_ =
            nh_.param<std::string>("frontend_street_map/marker_topic", "/guided_frontend_street_map/markers");
        car_pose_topic_ = nh_.param<std::string>("frontend_street_map/car_pose_topic", "/car_pos");
        const std::string default_dataset =
            ros::package::getPath("hybrid_a_star") +
            "/model_base_astar/neural-astar/planning-datasets/data/street/mixed_064_moore_c16.npz";
        dataset_path_ = nh_.param<std::string>("frontend_street_map/dataset_path", default_dataset);
        split_ = nh_.param<std::string>("frontend_street_map/split", "train");
        map_index_ = nh_.param("frontend_street_map/map_index", 0);
        random_index_ = nh_.param("frontend_street_map/random_index", false);
        resolution_ = nh_.param("frontend_street_map/resolution", 0.25);
        auto_refresh_ = nh_.param("frontend_street_map/auto_refresh", false);
        refresh_period_ = nh_.param("frontend_street_map/refresh_period", 5.0);
        publish_default_start_ = nh_.param("frontend_street_map/publish_default_start", true);
        default_start_yaw_ = nh_.param("frontend_street_map/default_start_yaw", 0.0);
        seed_ = nh_.param("frontend_street_map/seed", 123);
        rng_.seed(static_cast<std::mt19937::result_type>(seed_));
    }

    int PickMapIndex() {
        if (random_index_) {
            std::uniform_int_distribution<int> dis(0, maps_.map_count - 1);
            return dis(rng_);
        }
        return std::max(0, std::min(map_index_, maps_.map_count - 1));
    }

    geometry_msgs::Point CellCenter(int x, int y, double z = 0.0) const {
        geometry_msgs::Point p;
        p.x = origin_x_ + (static_cast<double>(x) + 0.5) * resolution_;
        p.y = origin_y_ + (static_cast<double>(y) + 0.5) * resolution_;
        p.z = z;
        return p;
    }

    void PublishDefaultStart() {
        if (!publish_default_start_) {
            return;
        }

        std::vector<std::pair<int, int>> free_cells;
        free_cells.reserve(static_cast<std::size_t>(width_ * height_));
        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                if (current_occ_[FlatIndex(x, y, width_)] == 0) {
                    free_cells.emplace_back(x, y);
                }
            }
        }
        if (free_cells.empty()) {
            ROS_WARN("street map has no free cells for default start pose");
            return;
        }

        std::uniform_int_distribution<int> dis(0, static_cast<int>(free_cells.size()) - 1);
        const auto& xy = free_cells[dis(rng_)];
        const geometry_msgs::Point pt = CellCenter(xy.first, xy.second);

        planning_msgs::car_info msg;
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = world_frame_;
        msg.x = pt.x;
        msg.y = pt.y;
        msg.yaw = default_start_yaw_;
        msg.speedDrietion = 1;
        msg.speed = 0.0;
        msg.turnAngle = 0.0;
        msg.yawrate = 0.0;
        car_pose_pub_.publish(msg);
    }

    void PublishMap() {
        current_map_index_ = PickMapIndex();
        width_ = maps_.width;
        height_ = maps_.height;
        origin_x_ = -0.5 * static_cast<double>(width_) * resolution_;
        origin_y_ = -0.5 * static_cast<double>(height_) * resolution_;
        current_occ_.assign(static_cast<std::size_t>(width_ * height_), 0);

        const std::size_t offset =
            static_cast<std::size_t>(current_map_index_) * static_cast<std::size_t>(width_ * height_);
        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                const float free_value = maps_.data[offset + FlatIndex(x, y, width_)];
                current_occ_[FlatIndex(x, y, width_)] = free_value > 0.5f ? 0 : 1;
            }
        }

        nav_msgs::OccupancyGrid grid;
        grid.header.stamp = ros::Time::now();
        grid.header.frame_id = world_frame_;
        grid.info.map_load_time = grid.header.stamp;
        grid.info.resolution = resolution_;
        grid.info.width = static_cast<uint32_t>(width_);
        grid.info.height = static_cast<uint32_t>(height_);
        grid.info.origin.position.x = origin_x_;
        grid.info.origin.position.y = origin_y_;
        grid.info.origin.orientation.w = 1.0;
        grid.data.resize(current_occ_.size());
        for (std::size_t i = 0; i < current_occ_.size(); ++i) {
            grid.data[i] = current_occ_[i] == 0 ? 0 : 100;
        }
        map_pub_.publish(grid);

        visualization_msgs::MarkerArray markers;

        visualization_msgs::Marker obstacle_marker;
        obstacle_marker.header.frame_id = world_frame_;
        obstacle_marker.header.stamp = grid.header.stamp;
        obstacle_marker.ns = "frontend_street_map";
        obstacle_marker.id = 0;
        obstacle_marker.type = visualization_msgs::Marker::CUBE_LIST;
        obstacle_marker.action = visualization_msgs::Marker::ADD;
        obstacle_marker.pose.orientation.w = 1.0;
        obstacle_marker.scale.x = resolution_;
        obstacle_marker.scale.y = resolution_;
        obstacle_marker.scale.z = 0.05;
        obstacle_marker.color.r = 0.62;
        obstacle_marker.color.g = 0.62;
        obstacle_marker.color.b = 0.62;
        obstacle_marker.color.a = 0.95;
        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                if (current_occ_[FlatIndex(x, y, width_)] != 0) {
                    obstacle_marker.points.push_back(CellCenter(x, y));
                }
            }
        }
        markers.markers.push_back(obstacle_marker);

        visualization_msgs::Marker boundary;
        boundary.header.frame_id = world_frame_;
        boundary.header.stamp = grid.header.stamp;
        boundary.ns = "frontend_street_map";
        boundary.id = 1;
        boundary.type = visualization_msgs::Marker::LINE_STRIP;
        boundary.action = visualization_msgs::Marker::ADD;
        boundary.pose.orientation.w = 1.0;
        boundary.scale.x = 0.08;
        boundary.color.r = 0.35;
        boundary.color.g = 0.35;
        boundary.color.b = 0.35;
        boundary.color.a = 1.0;
        boundary.points = {
            CellCenter(0, 0),
            CellCenter(width_ - 1, 0),
            CellCenter(width_ - 1, height_ - 1),
            CellCenter(0, height_ - 1),
            CellCenter(0, 0),
        };
        markers.markers.push_back(boundary);
        marker_pub_.publish(markers);

        PublishDefaultStart();
        ROS_INFO("Published C++ street map split=%s index=%d size=%dx%d resolution=%.3f",
                 split_.c_str(),
                 current_map_index_,
                 width_,
                 height_,
                 resolution_);
    }

    void TimerCallback(const ros::TimerEvent&) {
        PublishMap();
    }

    ros::NodeHandle nh_;
    ros::Publisher map_pub_;
    ros::Publisher marker_pub_;
    ros::Publisher car_pose_pub_;
    ros::Timer timer_;

    std::string world_frame_{"velodyne"};
    std::string map_topic_{"/guided_frontend_street_map"};
    std::string marker_topic_{"/guided_frontend_street_map/markers"};
    std::string car_pose_topic_{"/car_pos"};
    std::string dataset_path_;
    std::string split_{"train"};
    int map_index_{0};
    bool random_index_{false};
    double resolution_{0.25};
    bool auto_refresh_{false};
    double refresh_period_{5.0};
    bool publish_default_start_{true};
    double default_start_yaw_{0.0};
    int seed_{123};

    StreetMaps maps_;
    std::mt19937 rng_;
    int current_map_index_{0};
    int width_{0};
    int height_{0};
    double origin_x_{0.0};
    double origin_y_{0.0};
    std::vector<int> current_occ_;
};

}  // namespace

int main(int argc, char** argv) {
    ros::init(argc, argv, "frontend_street_map_env_cpp");
    ros::NodeHandle nh("~");
    try {
        FrontendStreetMapEnvironment env(nh);
        ros::spin();
    } catch (const std::exception& e) {
        ROS_FATAL_STREAM("frontend_street_map_env_cpp failed: " << e.what());
        return 1;
    }
    return 0;
}
