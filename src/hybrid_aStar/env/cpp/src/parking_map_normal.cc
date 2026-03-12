#include "hope/parking_map_normal.h"

#include <cmath>
#include <limits>

#include "hope/geometry.h"
#include "hope/vehicle.h"

namespace hope {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kProbHugeObst = 0.5;
constexpr int kNonCriticalCar = 3;
constexpr double kProbNonCriticalCar = 0.7;

Point2 GetRandPos(RandomGenerator& rng,
                  double origin_x,
                  double origin_y,
                  double angle_min,
                  double angle_max,
                  double radius_min,
                  double radius_max) {
    const double angle_mean = (angle_max + angle_min) / 2.0;
    const double angle_std = (angle_max - angle_min) / 4.0;
    const double radius_mean = (radius_min + radius_max) / 2.0;
    const double radius_std = (radius_max - radius_min) / 4.0;
    const double angle_rand = RandomGaussianNum(rng, angle_mean, angle_std, angle_min, angle_max);
    const double radius_rand = RandomGaussianNum(rng, radius_mean, radius_std, radius_min, radius_max);
    return {origin_x + std::cos(angle_rand) * radius_rand,
            origin_y + std::sin(angle_rand) * radius_rand};
}

Polygon2 MakePolygon(std::initializer_list<Point2> pts) {
    Polygon2 poly(pts.begin(), pts.end());
    return poly;
}

double MaxYInObstacles(const std::vector<Polygon2>& obstacles) {
    double max_y = -std::numeric_limits<double>::infinity();
    for (const auto& obs : obstacles) {
        for (const auto& pt : obs) {
            if (pt.y > max_y) {
                max_y = pt.y;
            }
        }
    }
    return max_y;
}

bool IntersectsAny(const Polygon2& poly, const std::vector<Polygon2>& obstacles) {
    for (const auto& obs : obstacles) {
        if (PolygonIntersects(poly, obs)) {
            return true;
        }
    }
    return false;
}

}  // namespace

ParkingMapNormal::ParkingMapNormal(MapLevel level, unsigned int seed)
    : level_(level), rng_(seed == 0 ? RandomGenerator() : RandomGenerator(seed)) {
    state_.level = level_;
}

State ParkingMapNormal::Reset(int case_id) {
    State start;
    State dest;
    std::vector<Polygon2> obstacles;

    const double roll = rng_.Uniform(0.0, 1.0);
    const bool use_bay = (case_id == 0 || (roll > 0.5 && case_id != 1))
                         && (level_ == MapLevel::Normal || level_ == MapLevel::Complex);

    if (use_bay) {
        while (!GenerateBayParkingCase(start, dest, obstacles)) {
        }
        state_.case_id = 0;
    } else {
        while (!GenerateParallelParkingCase(start, dest, obstacles)) {
        }
        state_.case_id = 1;
    }

    state_.start = start;
    state_.dest = dest;
    state_.start_box = start.CreateBox();
    state_.dest_box = dest.CreateBox();

    state_.boundary.xmin = std::floor(std::min(start.loc.x, dest.loc.x) - 10.0);
    state_.boundary.xmax = std::ceil(std::max(start.loc.x, dest.loc.x) + 10.0);
    state_.boundary.ymin = std::floor(std::min(start.loc.y, dest.loc.y) - 10.0);
    state_.boundary.ymax = std::ceil(std::max(start.loc.y, dest.loc.y) + 10.0);

    state_.obstacles.clear();
    for (const auto& obs : obstacles) {
        Area area;
        area.shape = obs;
        area.subtype = "obstacle";
        area.color = {150, 150, 150, 255};
        state_.obstacles.push_back(area);
    }

    return state_.start;
}

State ParkingMapNormal::FlipBoxOrientation(const State& target_state) const {
    const auto center = PolygonCentroid(target_state.CreateBox());
    const double new_x = 2.0 * center.x - target_state.loc.x;
    const double new_y = 2.0 * center.y - target_state.loc.y;
    const double new_heading = target_state.heading + kPi;
    return State(new_x, new_y, new_heading, target_state.speed, target_state.steering);
}

void ParkingMapNormal::FlipDestOrientation() {
    state_.dest = FlipBoxOrientation(state_.dest);
    state_.dest_box = state_.dest.CreateBox();
}

void ParkingMapNormal::FlipStartOrientation() {
    state_.start = FlipBoxOrientation(state_.start);
    state_.start_box = state_.start.CreateBox();
}

bool ParkingMapNormal::GenerateBayParkingCase(State& start,
                                              State& dest,
                                              std::vector<Polygon2>& obstacles) {
    const Point2 origin{0.0, 0.0};
    const double bay_half_len = 15.0;

    const LevelParams params = GetLevelParams(level_);
    const double max_bay_width = params.max_park_lot_width;
    const double min_bay_width = params.min_park_lot_width;
    const double bay_wall_dist = params.bay_park_wall_dist;
    const int n_obst = params.n_obstacle;

    const double max_lateral_space = max_bay_width - kVehicleWidth;
    const double min_lateral_space = min_bay_width - kVehicleWidth;

    bool generate_success = true;

    Polygon2 obstacle_back = MakePolygon({
        {origin.x + bay_half_len, origin.y},
        {origin.x + bay_half_len, origin.y - 1.0},
        {origin.x - bay_half_len, origin.y - 1.0},
        {origin.x - bay_half_len, origin.y},
    });

    const double dest_yaw = RandomGaussianNum(rng_, kPi / 2.0, kPi / 36.0, kPi * 5.0 / 12.0, kPi * 7.0 / 12.0);
    State dest_state(origin.x, origin.y, dest_yaw, 0.0, 0.0);
    const auto dest_box_coords = dest_state.CreateBox();
    const Point2 rb = dest_box_coords[0];
    const Point2 lb = dest_box_coords[3];
    const double min_dest_y = -std::min(rb.y, lb.y) + kMinDistToObst;
    const double dest_x = origin.x;
    const double dest_y = RandomGaussianNum(rng_, min_dest_y + 0.4, 0.2, min_dest_y, min_dest_y + 0.8);
    dest = State(dest_x, dest_y, dest_yaw, 0.0, 0.0);
    Polygon2 dest_box = dest.CreateBox();

    std::vector<Polygon2> non_critical_vehicle;

    Polygon2 obstacle_left;
    if (rng_.Uniform(0.0, 1.0) < kProbHugeObst) {
        const double max_dist_to_obst = max_lateral_space / 5.0 * 4.0;
        const double min_dist_to_obst = max_lateral_space / 5.0 * 1.0;
        const auto dest_box_coords2 = dest_box;
        const Point2 car_lf = dest_box_coords2[2];
        const Point2 car_lb = dest_box_coords2[3];
        const Point2 left_obst_rf = GetRandPos(rng_, car_lf.x, car_lf.y, kPi * 11.0 / 12.0, kPi * 13.0 / 12.0,
                                               min_dist_to_obst, max_dist_to_obst);
        const Point2 left_obst_rb = GetRandPos(rng_, car_lb.x, car_lb.y, kPi * 11.0 / 12.0, kPi * 13.0 / 12.0,
                                               min_dist_to_obst, max_dist_to_obst);
        obstacle_left = MakePolygon({
            left_obst_rf,
            left_obst_rb,
            {origin.x - bay_half_len, origin.y},
            {origin.x - bay_half_len, left_obst_rf.y},
        });
    } else {
        const double max_dist_to_obst = max_lateral_space / 5.0 * 4.0;
        const double min_dist_to_obst = max_lateral_space / 5.0 * 1.0;
        double left_car_x = origin.x - (kVehicleWidth + RandomUniformNum(rng_, min_dist_to_obst, max_dist_to_obst));
        double left_car_yaw = RandomGaussianNum(rng_, kPi / 2.0, kPi / 36.0, kPi * 5.0 / 12.0, kPi * 7.0 / 12.0);
        State left_state(left_car_x, origin.y, left_car_yaw, 0.0, 0.0);
        const auto left_box_coords = left_state.CreateBox();
        const double min_left_car_y = -std::min(left_box_coords[0].y, left_box_coords[3].y) + kMinDistToObst;
        double left_car_y = RandomGaussianNum(rng_, min_left_car_y + 0.4, 0.2, min_left_car_y, min_left_car_y + 0.8);
        left_state.loc.y = left_car_y;
        obstacle_left = left_state.CreateBox();

        for (int i = 0; i < kNonCriticalCar; ++i) {
            left_car_x -= (kVehicleWidth + kMinDistToObst + RandomUniformNum(rng_, min_dist_to_obst, max_dist_to_obst));
            left_car_y += RandomGaussianNum(rng_, 0.0, 0.05, -0.1, 0.1);
            left_car_yaw = RandomGaussianNum(rng_, kPi / 2.0, kPi / 36.0, kPi * 5.0 / 12.0, kPi * 7.0 / 12.0);
            State obs_state(left_car_x, left_car_y, left_car_yaw, 0.0, 0.0);
            if (rng_.Uniform(0.0, 1.0) < kProbNonCriticalCar) {
                non_critical_vehicle.push_back(obs_state.CreateBox());
            }
        }
    }

    const double dist_dest_to_left_obst = PolygonDistance(dest_box, obstacle_left);
    double min_dist_to_obst = std::max(min_lateral_space - dist_dest_to_left_obst, 0.0) + kMinDistToObst;
    double max_dist_to_obst = std::max(max_lateral_space - dist_dest_to_left_obst, 0.0) + kMinDistToObst;

    Polygon2 obstacle_right;
    if (rng_.Uniform(0.0, 1.0) < kProbHugeObst) {
        const Point2 car_rf = dest_box[1];
        const Point2 car_rb = dest_box[0];
        const Point2 right_obst_lf = GetRandPos(rng_, car_rf.x, car_rf.y, -kPi / 12.0, kPi / 12.0,
                                                min_dist_to_obst, max_dist_to_obst);
        const Point2 right_obst_lb = GetRandPos(rng_, car_rb.x, car_rb.y, -kPi / 12.0, kPi / 12.0,
                                                min_dist_to_obst, max_dist_to_obst);
        obstacle_right = MakePolygon({
            {origin.x + bay_half_len, right_obst_lf.y},
            {origin.x + bay_half_len, origin.y},
            right_obst_lb,
            right_obst_lf,
        });
    } else {
        double right_car_x = origin.x + (kVehicleWidth + RandomUniformNum(rng_, min_dist_to_obst, max_dist_to_obst));
        double right_car_yaw = RandomGaussianNum(rng_, kPi / 2.0, kPi / 36.0, kPi * 5.0 / 12.0, kPi * 7.0 / 12.0);
        State right_state(right_car_x, origin.y, right_car_yaw, 0.0, 0.0);
        const auto right_box_coords = right_state.CreateBox();
        const double min_right_car_y = -std::min(right_box_coords[0].y, right_box_coords[3].y) + kMinDistToObst;
        double right_car_y = RandomGaussianNum(rng_, min_right_car_y + 0.4, 0.2, min_right_car_y, min_right_car_y + 0.8);
        right_state.loc.y = right_car_y;
        obstacle_right = right_state.CreateBox();

        for (int i = 0; i < kNonCriticalCar; ++i) {
            right_car_x += (kVehicleWidth + kMinDistToObst + RandomUniformNum(rng_, min_dist_to_obst, max_dist_to_obst));
            right_car_y += RandomGaussianNum(rng_, 0.0, 0.05, -0.1, 0.1);
            right_car_yaw = RandomGaussianNum(rng_, kPi / 2.0, kPi / 36.0, kPi * 5.0 / 12.0, kPi * 7.0 / 12.0);
            State obs_state(right_car_x, right_car_y, right_car_yaw, 0.0, 0.0);
            if (rng_.Uniform(0.0, 1.0) < kProbNonCriticalCar) {
                non_critical_vehicle.push_back(obs_state.CreateBox());
            }
        }
    }

    const double dist_dest_to_right_obst = PolygonDistance(dest_box, obstacle_right);
    if (dist_dest_to_right_obst + dist_dest_to_left_obst < min_lateral_space ||
        dist_dest_to_right_obst + dist_dest_to_left_obst > max_lateral_space ||
        dist_dest_to_left_obst < kMinDistToObst || dist_dest_to_right_obst < kMinDistToObst) {
        generate_success = false;
    }

    obstacles = {obstacle_back, obstacle_left, obstacle_right};
    obstacles.insert(obstacles.end(), non_critical_vehicle.begin(), non_critical_vehicle.end());

    for (const auto& obst : obstacles) {
        if (PolygonIntersects(obst, dest_box)) {
            generate_success = false;
        }
    }

    const double max_obstacle_y = MaxYInObstacles(obstacles) + kMinDistToObst;
    std::vector<Polygon2> other_obstacles;

    if (rng_.Uniform(0.0, 1.0) < 0.2) {
        other_obstacles.push_back(MakePolygon({
            {origin.x - bay_half_len, bay_wall_dist + max_obstacle_y + kMinDistToObst},
            {origin.x + bay_half_len, bay_wall_dist + max_obstacle_y + kMinDistToObst},
            {origin.x + bay_half_len, bay_wall_dist + max_obstacle_y + kMinDistToObst + 0.1},
            {origin.x - bay_half_len, bay_wall_dist + max_obstacle_y + kMinDistToObst + 0.1},
        }));
    } else {
        Polygon2 other_obstacle_range = MakePolygon({
            {origin.x - bay_half_len, bay_wall_dist + max_obstacle_y},
            {origin.x + bay_half_len, bay_wall_dist + max_obstacle_y},
            {origin.x + bay_half_len, bay_wall_dist + max_obstacle_y + 8.0},
            {origin.x - bay_half_len, bay_wall_dist + max_obstacle_y + 8.0},
        });

        const double valid_x_min = origin.x - bay_half_len + 2.0;
        const double valid_x_max = origin.x + bay_half_len - 2.0;
        const double valid_y_min = bay_wall_dist + max_obstacle_y + 2.0;
        const double valid_y_max = bay_wall_dist + max_obstacle_y + 6.0;

        for (int i = 0; i < n_obst; ++i) {
            const double obs_x = RandomUniformNum(rng_, valid_x_min, valid_x_max);
            const double obs_y = RandomUniformNum(rng_, valid_y_min, valid_y_max);
            const double obs_yaw = rng_.Uniform(0.0, 1.0) * kPi * 2.0;
            State obs_state(obs_x, obs_y, obs_yaw, 0.0, 0.0);
            Polygon2 obs = obs_state.CreateBox();
            for (auto& pt : obs) {
                pt.x += rng_.Uniform(0.0, 0.5);
                pt.y += rng_.Uniform(0.0, 0.5);
            }
            if (PolygonIntersects(obs, other_obstacle_range)) {
                continue;
            }
            if (IntersectsAny(obs, other_obstacles)) {
                continue;
            }
            other_obstacles.push_back(obs);
        }
    }

    obstacles.insert(obstacles.end(), other_obstacles.begin(), other_obstacles.end());

    bool start_box_valid = false;
    const double valid_start_x_min = origin.x - bay_half_len / 2.0;
    const double valid_start_x_max = origin.x + bay_half_len / 2.0;
    const double valid_start_y_min = max_obstacle_y + 1.0;
    const double valid_start_y_max = bay_wall_dist + max_obstacle_y - 1.0;

    double start_x = 0.0;
    double start_y = 0.0;
    double start_yaw = 0.0;
    while (!start_box_valid) {
        start_box_valid = true;
        start_x = RandomUniformNum(rng_, valid_start_x_min, valid_start_x_max);
        start_y = RandomUniformNum(rng_, valid_start_y_min, valid_start_y_max);
        start_yaw = RandomGaussianNum(rng_, 0.0, kPi / 6.0, -kPi / 2.0, kPi / 2.0);
        if (rng_.Uniform(0.0, 1.0) < 0.5) {
            start_yaw += kPi;
        }
        State start_state(start_x, start_y, start_yaw, 0.0, 0.0);
        Polygon2 start_box = start_state.CreateBox();
        if (IntersectsAny(start_box, obstacles) || PolygonIntersects(start_box, dest_box)) {
            start_box_valid = false;
        }
    }

    if (std::cos(start_yaw) < 0.0) {
        const Point2 dest_box_center = PolygonCentroid(dest_box);
        dest = State(2.0 * dest_box_center.x - dest_x,
                     2.0 * dest_box_center.y - dest_y,
                     dest_yaw + kPi,
                     0.0,
                     0.0);
    }

    if (kDropOutObst > 0.0) {
        std::vector<Polygon2> kept;
        kept.reserve(obstacles.size());
        for (const auto& obs : obstacles) {
            if (rng_.Uniform(0.0, 1.0) >= kDropOutObst) {
                kept.push_back(obs);
            }
        }
        obstacles = kept;
    }

    start = State(start_x, start_y, start_yaw, 0.0, 0.0);
    return generate_success;
}

bool ParkingMapNormal::GenerateParallelParkingCase(State& start,
                                                   State& dest,
                                                   std::vector<Polygon2>& obstacles) {
    const Point2 origin{0.0, 0.0};
    const double bay_half_len = 18.0;

    const LevelParams params = GetLevelParams(level_);
    const double max_para_len = params.max_park_lot_len;
    const double min_para_len = params.min_park_lot_len;
    const double para_wall_dist = params.para_park_wall_dist;
    const int n_obst = params.n_obstacle;

    const double max_long_space = max_para_len - kVehicleLength;
    const double min_long_space = min_para_len - kVehicleLength;

    bool generate_success = true;

    Polygon2 obstacle_back = MakePolygon({
        {origin.x + bay_half_len, origin.y},
        {origin.x + bay_half_len, origin.y - 1.0},
        {origin.x - bay_half_len, origin.y - 1.0},
        {origin.x - bay_half_len, origin.y},
    });

    const double dest_yaw = RandomGaussianNum(rng_, 0.0, kPi / 36.0, -kPi / 12.0, kPi / 12.0);
    State dest_state(origin.x, origin.y, dest_yaw, 0.0, 0.0);
    const auto dest_box_coords = dest_state.CreateBox();
    const Point2 rb = dest_box_coords[0];
    const Point2 rf = dest_box_coords[1];
    const double min_dest_y = -std::min(rb.y, rf.y) + kMinDistToObst;
    const double dest_x = origin.x;
    const double dest_y = RandomGaussianNum(rng_, min_dest_y + 0.4, 0.2, min_dest_y, min_dest_y + 0.8);
    dest = State(dest_x, dest_y, dest_yaw, 0.0, 0.0);
    Polygon2 dest_box = dest.CreateBox();

    std::vector<Polygon2> non_critical_vehicle;

    Polygon2 obstacle_left;
    if (rng_.Uniform(0.0, 1.0) < kProbHugeObst) {
        const double max_dist_to_obst = max_long_space / 5.0 * 4.0;
        const double min_dist_to_obst = min_long_space / 5.0 * 1.0;
        const Point2 car_lb = dest_box[3];
        const Point2 car_rb = dest_box[0];
        const Point2 left_obst_rf = GetRandPos(rng_, car_lb.x, car_lb.y, kPi * 11.0 / 12.0, kPi * 13.0 / 12.0,
                                               min_dist_to_obst, max_dist_to_obst);
        const Point2 left_obst_rb = GetRandPos(rng_, car_rb.x, car_rb.y, kPi * 11.0 / 12.0, kPi * 13.0 / 12.0,
                                               min_dist_to_obst, max_dist_to_obst);
        obstacle_left = MakePolygon({
            left_obst_rf,
            left_obst_rb,
            {origin.x - bay_half_len, origin.y},
            {origin.x - bay_half_len, left_obst_rf.y},
        });
    } else {
        const double max_dist_to_obst = max_long_space / 5.0 * 4.0;
        const double min_dist_to_obst = min_long_space / 5.0 * 1.0;
        double left_car_x = origin.x - (kVehicleLength + RandomUniformNum(rng_, min_dist_to_obst, max_dist_to_obst));
        double left_car_yaw = RandomGaussianNum(rng_, 0.0, kPi / 36.0, -kPi / 12.0, kPi / 12.0);
        State left_state(left_car_x, origin.y, left_car_yaw, 0.0, 0.0);
        const auto left_box_coords = left_state.CreateBox();
        const double min_left_car_y = -std::min(left_box_coords[0].y, left_box_coords[1].y) + kMinDistToObst;
        double left_car_y = RandomGaussianNum(rng_, min_left_car_y + 0.4, 0.2, min_left_car_y, min_left_car_y + 0.8);
        left_state.loc.y = left_car_y;
        obstacle_left = left_state.CreateBox();

        for (int i = 0; i < kNonCriticalCar - 1; ++i) {
            left_car_x -= (kVehicleLength + kMinDistToObst + RandomUniformNum(rng_, min_dist_to_obst, max_dist_to_obst));
            left_car_y += RandomGaussianNum(rng_, 0.0, 0.05, -0.1, 0.1);
            left_car_yaw = RandomGaussianNum(rng_, 0.0, kPi / 36.0, -kPi / 12.0, kPi / 12.0);
            State obs_state(left_car_x, left_car_y, left_car_yaw, 0.0, 0.0);
            if (rng_.Uniform(0.0, 1.0) < kProbNonCriticalCar) {
                non_critical_vehicle.push_back(obs_state.CreateBox());
            }
        }
    }

    const double dist_dest_to_left_obst = PolygonDistance(dest_box, obstacle_left);
    double min_dist_to_obst = std::max(min_long_space - dist_dest_to_left_obst, 0.0) + kMinDistToObst;
    double max_dist_to_obst = std::max(max_long_space - dist_dest_to_left_obst, 0.0) + kMinDistToObst;

    Polygon2 obstacle_right;
    if (rng_.Uniform(0.0, 1.0) < 0.5) {
        const Point2 car_lf = dest_box[2];
        const Point2 car_rf = dest_box[1];
        const Point2 right_obst_lf = GetRandPos(rng_, car_lf.x, car_lf.y, -kPi / 12.0, kPi / 12.0,
                                                min_dist_to_obst, max_dist_to_obst);
        const Point2 right_obst_lb = GetRandPos(rng_, car_rf.x, car_rf.y, -kPi / 12.0, kPi / 12.0,
                                                min_dist_to_obst, max_dist_to_obst);
        obstacle_right = MakePolygon({
            {origin.x + bay_half_len, right_obst_lf.y},
            {origin.x + bay_half_len, origin.y},
            right_obst_lb,
            right_obst_lf,
        });
    } else {
        double right_car_x = origin.x + (kVehicleLength + RandomUniformNum(rng_, min_dist_to_obst, max_dist_to_obst));
        double right_car_yaw = RandomGaussianNum(rng_, 0.0, kPi / 36.0, -kPi / 12.0, kPi / 12.0);
        State right_state(right_car_x, origin.y, right_car_yaw, 0.0, 0.0);
        const auto right_box_coords = right_state.CreateBox();
        const double min_right_car_y = -std::min(right_box_coords[0].y, right_box_coords[1].y) + kMinDistToObst;
        double right_car_y = RandomGaussianNum(rng_, min_right_car_y + 0.4, 0.2, min_right_car_y, min_right_car_y + 0.8);
        right_state.loc.y = right_car_y;
        obstacle_right = right_state.CreateBox();

        for (int i = 0; i < kNonCriticalCar - 1; ++i) {
            right_car_x += (kVehicleLength + kMinDistToObst + RandomUniformNum(rng_, min_dist_to_obst, max_dist_to_obst));
            right_car_y += RandomGaussianNum(rng_, 0.0, 0.05, -0.1, 0.1);
            right_car_yaw = RandomGaussianNum(rng_, 0.0, kPi / 36.0, -kPi / 12.0, kPi / 12.0);
            State obs_state(right_car_x, right_car_y, right_car_yaw, 0.0, 0.0);
            if (rng_.Uniform(0.0, 1.0) < kProbNonCriticalCar) {
                non_critical_vehicle.push_back(obs_state.CreateBox());
            }
        }
    }

    const double dist_dest_to_right_obst = PolygonDistance(dest_box, obstacle_right);
    if (dist_dest_to_right_obst + dist_dest_to_left_obst < min_long_space ||
        dist_dest_to_right_obst + dist_dest_to_left_obst > max_long_space ||
        dist_dest_to_left_obst < kMinDistToObst || dist_dest_to_right_obst < kMinDistToObst) {
        generate_success = false;
    }

    obstacles = {obstacle_back, obstacle_left, obstacle_right};
    obstacles.insert(obstacles.end(), non_critical_vehicle.begin(), non_critical_vehicle.end());

    for (const auto& obst : obstacles) {
        if (PolygonIntersects(obst, dest_box)) {
            generate_success = false;
        }
    }

    const double max_obstacle_y = MaxYInObstacles(obstacles) + kMinDistToObst;
    std::vector<Polygon2> other_obstacles;

    if (rng_.Uniform(0.0, 1.0) < 0.2) {
        other_obstacles.push_back(MakePolygon({
            {origin.x - bay_half_len, para_wall_dist + max_obstacle_y + kMinDistToObst},
            {origin.x + bay_half_len, para_wall_dist + max_obstacle_y + kMinDistToObst},
            {origin.x + bay_half_len, para_wall_dist + max_obstacle_y + kMinDistToObst + 0.1},
            {origin.x - bay_half_len, para_wall_dist + max_obstacle_y + kMinDistToObst + 0.1},
        }));
    } else {
        Polygon2 other_obstacle_range = MakePolygon({
            {origin.x - bay_half_len, para_wall_dist + max_obstacle_y},
            {origin.x + bay_half_len, para_wall_dist + max_obstacle_y},
            {origin.x + bay_half_len, para_wall_dist + max_obstacle_y + 8.0},
            {origin.x - bay_half_len, para_wall_dist + max_obstacle_y + 8.0},
        });

        const double valid_x_min = origin.x - bay_half_len + 2.0;
        const double valid_x_max = origin.x + bay_half_len - 2.0;
        const double valid_y_min = para_wall_dist + max_obstacle_y + 2.0;
        const double valid_y_max = para_wall_dist + max_obstacle_y + 6.0;

        for (int i = 0; i < n_obst; ++i) {
            const double obs_x = RandomUniformNum(rng_, valid_x_min, valid_x_max);
            const double obs_y = RandomUniformNum(rng_, valid_y_min, valid_y_max);
            const double obs_yaw = rng_.Uniform(0.0, 1.0) * kPi * 2.0;
            State obs_state(obs_x, obs_y, obs_yaw, 0.0, 0.0);
            Polygon2 obs = obs_state.CreateBox();
            for (auto& pt : obs) {
                pt.x += rng_.Uniform(0.0, 0.5);
                pt.y += rng_.Uniform(0.0, 0.5);
            }
            if (PolygonIntersects(obs, other_obstacle_range)) {
                continue;
            }
            if (IntersectsAny(obs, other_obstacles)) {
                continue;
            }
            other_obstacles.push_back(obs);
        }
    }

    obstacles.insert(obstacles.end(), other_obstacles.begin(), other_obstacles.end());

    bool start_box_valid = false;
    const double valid_start_x_min = origin.x - bay_half_len / 2.0;
    const double valid_start_x_max = origin.x + bay_half_len / 2.0;
    const double valid_start_y_min = max_obstacle_y + 1.0;
    const double valid_start_y_max = para_wall_dist + max_obstacle_y - 1.0;

    double start_x = 0.0;
    double start_y = 0.0;
    double start_yaw = 0.0;
    while (!start_box_valid) {
        start_box_valid = true;
        start_x = RandomUniformNum(rng_, valid_start_x_min, valid_start_x_max);
        start_y = RandomUniformNum(rng_, valid_start_y_min, valid_start_y_max);
        start_yaw = RandomGaussianNum(rng_, 0.0, kPi / 6.0, -kPi / 2.0, kPi / 2.0);
        if (rng_.Uniform(0.0, 1.0) < 0.5) {
            start_yaw += kPi;
        }
        State start_state(start_x, start_y, start_yaw, 0.0, 0.0);
        Polygon2 start_box = start_state.CreateBox();
        if (IntersectsAny(start_box, obstacles) || PolygonIntersects(start_box, dest_box)) {
            start_box_valid = false;
        }
    }

    if (std::cos(start_yaw) < 0.0) {
        const Point2 dest_box_center = PolygonCentroid(dest_box);
        dest = State(2.0 * dest_box_center.x - dest_x,
                     2.0 * dest_box_center.y - dest_y,
                     dest_yaw + kPi,
                     0.0,
                     0.0);
    }

    if (kDropOutObst > 0.0) {
        std::vector<Polygon2> kept;
        kept.reserve(obstacles.size());
        for (const auto& obs : obstacles) {
            if (rng_.Uniform(0.0, 1.0) >= kDropOutObst) {
                kept.push_back(obs);
            }
        }
        obstacles = kept;
    }

    start = State(start_x, start_y, start_yaw, 0.0, 0.0);
    return generate_success;
}

}  // namespace hope
