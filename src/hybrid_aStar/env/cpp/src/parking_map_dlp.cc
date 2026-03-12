#include "hope/parking_map_dlp.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>

#include "hope/geometry.h"

namespace hope {

namespace {
constexpr double kPi = 3.14159265358979323846;

std::string DefaultDlpPath() {
    return "data/dlp.bin";
}

template <typename T>
bool ReadBinary(std::ifstream& stream, T& value) {
    return static_cast<bool>(stream.read(reinterpret_cast<char*>(&value), sizeof(T)));
}

Polygon2 MakePolygon(std::vector<Point2>&& pts) {
    return pts;
}

}  // namespace

ParkingMapDlp::ParkingMapDlp(const std::string& path) {
    const std::string resolved = path.empty() ? DefaultDlpPath() : path;
    Load(resolved);
}

void ParkingMapDlp::Load(const std::string& path) {
    cases_.clear();
    multi_start_ = false;

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return;
    }

    uint32_t num_cases = 0;
    if (!ReadBinary(file, num_cases)) {
        return;
    }

    cases_.reserve(num_cases);
    for (uint32_t i = 0; i < num_cases; ++i) {
        DlpCase dlp_case;
        uint32_t num_starts = 0;
        if (!ReadBinary(file, num_starts)) {
            break;
        }
        dlp_case.starts.reserve(num_starts);
        for (uint32_t s = 0; s < num_starts; ++s) {
            double x = 0.0, y = 0.0, yaw = 0.0;
            ReadBinary(file, x);
            ReadBinary(file, y);
            ReadBinary(file, yaw);
            dlp_case.starts.emplace_back(x, y, yaw, 0.0, 0.0);
        }
        if (num_starts > 1) {
            multi_start_ = true;
        }

        double dx = 0.0, dy = 0.0, dyaw = 0.0;
        ReadBinary(file, dx);
        ReadBinary(file, dy);
        ReadBinary(file, dyaw);
        dlp_case.dest = State(dx, dy, dyaw, 0.0, 0.0);

        uint32_t num_obstacles = 0;
        ReadBinary(file, num_obstacles);
        dlp_case.obstacles.reserve(num_obstacles);
        for (uint32_t o = 0; o < num_obstacles; ++o) {
            uint32_t num_points = 0;
            ReadBinary(file, num_points);
            std::vector<Point2> pts;
            pts.reserve(num_points);
            for (uint32_t p = 0; p < num_points; ++p) {
                double px = 0.0, py = 0.0;
                ReadBinary(file, px);
                ReadBinary(file, py);
                pts.push_back({px, py});
            }
            dlp_case.obstacles.push_back(MakePolygon(std::move(pts)));
        }

        cases_.push_back(std::move(dlp_case));
    }
}

State ParkingMapDlp::FlipBoxOrientation(const State& target_state) const {
    const Point2 center = PolygonCentroid(target_state.CreateBox());
    const double new_x = 2.0 * center.x - target_state.loc.x;
    const double new_y = 2.0 * center.y - target_state.loc.y;
    return State(new_x, new_y, target_state.heading + kPi, target_state.speed, target_state.steering);
}

void ParkingMapDlp::FlipDestOrientation() {
    state_.dest = FlipBoxOrientation(state_.dest);
    state_.dest_box = state_.dest.CreateBox();
}

void ParkingMapDlp::FlipStartOrientation() {
    state_.start = FlipBoxOrientation(state_.start);
    state_.start_box = state_.start.CreateBox();
}

void ParkingMapDlp::FilterObstacles() {
    std::vector<Area> filtered;
    for (const auto& obs : state_.obstacles) {
        double x_max = -std::numeric_limits<double>::infinity();
        double x_min = std::numeric_limits<double>::infinity();
        double y_max = -std::numeric_limits<double>::infinity();
        double y_min = std::numeric_limits<double>::infinity();
        for (const auto& pt : obs.shape) {
            x_max = std::max(x_max, pt.x);
            x_min = std::min(x_min, pt.x);
            y_max = std::max(y_max, pt.y);
            y_min = std::min(y_min, pt.y);
        }
        if (!(x_max <= state_.boundary.xmin || x_min >= state_.boundary.xmax ||
              y_max <= state_.boundary.ymin || y_min >= state_.boundary.ymax)) {
            filtered.push_back(obs);
        }
    }
    state_.obstacles = filtered;
}

State ParkingMapDlp::Reset(int case_id, const std::string& path) {
    if (!path.empty()) {
        Load(path);
    }
    if (cases_.empty()) {
        return {};
    }

    if (case_id < 0) {
        case_id = static_cast<int>(rng_.Uniform(0.0, 1.0) * cases_.size());
    }
    if (case_id >= static_cast<int>(cases_.size())) {
        case_id = case_id % static_cast<int>(cases_.size());
    }

    state_.case_id = case_id;
    const DlpCase& dlp_case = cases_[case_id];

    int start_idx = 0;
    if (!dlp_case.starts.empty()) {
        start_idx = static_cast<int>(rng_.Uniform(0.0, 1.0) * dlp_case.starts.size());
        if (start_idx >= static_cast<int>(dlp_case.starts.size())) {
            start_idx = static_cast<int>(dlp_case.starts.size()) - 1;
        }
    }

    state_.start = dlp_case.starts.empty() ? State() : dlp_case.starts[start_idx];
    if (multi_start_) {
        state_.start.loc.x += rng_.Normal(0.0, 0.05);
        state_.start.loc.y += rng_.Normal(0.0, 0.05);
        state_.start.heading += rng_.Normal(0.0, 0.02);
    }
    state_.dest = dlp_case.dest;

    state_.start_box = state_.start.CreateBox();
    state_.dest_box = state_.dest.CreateBox();

    state_.boundary.xmin = std::floor(std::min(state_.start.loc.x, state_.dest.loc.x) - 20.0);
    state_.boundary.xmax = std::ceil(std::max(state_.start.loc.x, state_.dest.loc.x) + 20.0);
    state_.boundary.ymin = std::floor(std::min(state_.start.loc.y, state_.dest.loc.y) - 20.0);
    state_.boundary.ymax = std::ceil(std::max(state_.start.loc.y, state_.dest.loc.y) + 20.0);

    state_.obstacles.clear();
    for (const auto& obs : dlp_case.obstacles) {
        Area area;
        area.shape = obs;
        area.subtype = "obstacle";
        area.color = {150, 150, 150, 255};
        state_.obstacles.push_back(area);
    }

    FilterObstacles();

    if (rng_.Uniform(0.0, 1.0) > 0.5) {
        FlipDestOrientation();
    }
    if (rng_.Uniform(0.0, 1.0) > 0.5) {
        FlipStartOrientation();
    }

    state_.level = GetMapLevel(state_.start, state_.dest, state_.obstacles);

    return state_.start;
}

}  // namespace hope
