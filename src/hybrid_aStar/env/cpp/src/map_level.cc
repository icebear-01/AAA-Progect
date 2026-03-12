#include "hope/map_level.h"

#include <cmath>
#include <limits>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/multi_point.hpp>

#include "hope/geometry.h"

namespace hope {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kExtremParkLotLength = (kVehicleLength * 1.2 < kVehicleLength + 0.9)
                                           ? (kVehicleLength * 1.2)
                                           : (kVehicleLength + 0.9);

using BoostPoint = boost::geometry::model::d2::point_xy<double>;
using BoostMultiPoint = boost::geometry::model::multi_point<BoostPoint>;
using BoostPolygon = boost::geometry::model::polygon<BoostPoint>;

Point2 RotatePoint(const Point2& pt, double cos_t, double sin_t) {
    return {cos_t * pt.x - sin_t * pt.y, sin_t * pt.x + cos_t * pt.y};
}

Polygon2 MinimumRotatedRectangle(const std::vector<Point2>& points) {
    if (points.empty()) {
        return {};
    }

    BoostMultiPoint mp;
    for (const auto& pt : points) {
        boost::geometry::append(mp, BoostPoint(pt.x, pt.y));
    }

    BoostPolygon hull;
    boost::geometry::convex_hull(mp, hull);
    const auto& hull_pts = hull.outer();
    if (hull_pts.size() < 2) {
        return points;
    }

    double best_area = std::numeric_limits<double>::infinity();
    double best_angle = 0.0;
    double best_min_x = 0.0;
    double best_max_x = 0.0;
    double best_min_y = 0.0;
    double best_max_y = 0.0;

    const int n = static_cast<int>(hull_pts.size()) - 1;  // last equals first
    for (int i = 0; i < n; ++i) {
        const auto& p1 = hull_pts[i];
        const auto& p2 = hull_pts[(i + 1) % n];
        const double dx = p2.x() - p1.x();
        const double dy = p2.y() - p1.y();
        const double angle = std::atan2(dy, dx);
        const double cos_t = std::cos(-angle);
        const double sin_t = std::sin(-angle);

        double min_x = std::numeric_limits<double>::infinity();
        double max_x = -std::numeric_limits<double>::infinity();
        double min_y = std::numeric_limits<double>::infinity();
        double max_y = -std::numeric_limits<double>::infinity();

        for (int j = 0; j < n; ++j) {
            const Point2 pt{hull_pts[j].x(), hull_pts[j].y()};
            const Point2 rot = RotatePoint(pt, cos_t, sin_t);
            min_x = std::min(min_x, rot.x);
            max_x = std::max(max_x, rot.x);
            min_y = std::min(min_y, rot.y);
            max_y = std::max(max_y, rot.y);
        }

        const double area = (max_x - min_x) * (max_y - min_y);
        if (area < best_area) {
            best_area = area;
            best_angle = angle;
            best_min_x = min_x;
            best_max_x = max_x;
            best_min_y = min_y;
            best_max_y = max_y;
        }
    }

    const double cos_t = std::cos(best_angle);
    const double sin_t = std::sin(best_angle);
    Polygon2 rect;
    rect.push_back(RotatePoint({best_min_x, best_min_y}, cos_t, sin_t));
    rect.push_back(RotatePoint({best_max_x, best_min_y}, cos_t, sin_t));
    rect.push_back(RotatePoint({best_max_x, best_max_y}, cos_t, sin_t));
    rect.push_back(RotatePoint({best_min_x, best_max_y}, cos_t, sin_t));
    return rect;
}

Point2 Midpoint(const Point2& a, const Point2& b) {
    return {(a.x + b.x) / 2.0, (a.y + b.y) / 2.0};
}

Point2 TranslatePoint(const Point2& pt, double heading, double dist) {
    return {pt.x + std::cos(heading) * dist, pt.y + std::sin(heading) * dist};
}

double DistancePointPolygon(const Point2& pt, const Polygon2& poly) {
    BoostPoint bp(pt.x, pt.y);
    BoostPolygon bpoly;
    for (const auto& p : poly) {
        bpoly.outer().push_back(BoostPoint(p.x, p.y));
    }
    if (poly.front().x != poly.back().x || poly.front().y != poly.back().y) {
        bpoly.outer().push_back(BoostPoint(poly.front().x, poly.front().y));
    }
    boost::geometry::correct(bpoly);
    return boost::geometry::distance(bp, bpoly);
}

int GetNearestObstacleIdx(const Point2& pt,
                          const std::vector<Polygon2>& obstacles,
                          double max_min_dist,
                          const std::vector<int>& skip) {
    double min_dist = max_min_dist;
    int nearest = -1;
    for (size_t i = 0; i < obstacles.size(); ++i) {
        if (std::find(skip.begin(), skip.end(), static_cast<int>(i)) != skip.end()) {
            continue;
        }
        const double dist = DistancePointPolygon(pt, obstacles[i]);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = static_cast<int>(i);
        }
    }
    return nearest;
}

struct SurroundingObstacles {
    int left = -1;
    int right = -1;
    int front = -1;
    int back = -1;
};

SurroundingObstacles GetSurroundingObstacles(const State& dest, const std::vector<Polygon2>& obstacles) {
    SurroundingObstacles result;

    const auto box = dest.CreateBox();
    const Point2 rb = box[0];
    const Point2 rf = box[1];
    const Point2 lf = box[2];
    const Point2 lb = box[3];

    const Point2 lc = Midpoint(lf, lb);
    const Point2 rc = Midpoint(rf, rb);
    const Point2 fc = Midpoint(lf, rf);
    const Point2 bc = Midpoint(lb, rb);

    std::vector<int> skip;
    result.left = GetNearestObstacleIdx(lc, obstacles, kVehicleLength / 2.0, skip);
    if (result.left >= 0) {
        skip.push_back(result.left);
    }
    result.right = GetNearestObstacleIdx(rc, obstacles, kVehicleLength / 2.0, skip);
    if (result.right >= 0) {
        skip.push_back(result.right);
    }
    result.front = GetNearestObstacleIdx(fc, obstacles, kVehicleLength / 2.0, skip);
    if (result.front >= 0) {
        skip.push_back(result.front);
    }
    result.back = GetNearestObstacleIdx(bc, obstacles, kVehicleLength / 2.0, skip);

    return result;
}

bool HasEnoughSpace(const State& pos, const std::vector<Polygon2>& obstacles, double width, double length) {
    const auto dest_box = pos.CreateBox();
    const auto sur = GetSurroundingObstacles(pos, obstacles);

    if (width > 0.0) {
        if (sur.left >= 0 && sur.right >= 0) {
            const double sum = PolygonDistance(dest_box, obstacles[sur.left]) +
                               PolygonDistance(dest_box, obstacles[sur.right]) + kVehicleWidth;
            if (sum < width) {
                return false;
            }
        }
    }

    if (length > 0.0) {
        if (sur.front >= 0 && sur.back >= 0) {
            const double sum = PolygonDistance(dest_box, obstacles[sur.front]) +
                               PolygonDistance(dest_box, obstacles[sur.back]) + kVehicleLength;
            if (sum < length) {
                return false;
            }
        }
    }

    return true;
}

bool CheckExtremLevel(const State& start, const State& dest, const std::vector<Polygon2>& obstacles) {
    const auto sur = GetSurroundingObstacles(dest, obstacles);
    const double dist_sd = std::hypot(start.loc.x - dest.loc.x, start.loc.y - dest.loc.y);

    if (dist_sd > 30.0) {
        if (sur.front >= 0 && sur.back >= 0 &&
            !HasEnoughSpace(dest, obstacles, 0.0, GetLevelParams(MapLevel::Normal).min_park_lot_len)) {
            return true;
        }
        if (sur.left >= 0 && sur.right >= 0 &&
            !HasEnoughSpace(dest, obstacles, GetLevelParams(MapLevel::Normal).min_park_lot_width, 0.0)) {
            return true;
        }
    }

    if (sur.front >= 0 && sur.back >= 0 &&
        !HasEnoughSpace(dest, obstacles, 0.0, kExtremParkLotLength)) {
        return true;
    }

    return false;
}

}  // namespace

MapLevel GetMapLevel(const State& start, const State& dest, const std::vector<Area>& obstacles) {
    if (obstacles.size() <= 1) {
        return MapLevel::Normal;
    }

    std::vector<Polygon2> obs_polys;
    obs_polys.reserve(obstacles.size());
    for (const auto& obs : obstacles) {
        obs_polys.push_back(obs.shape);
    }

    if (CheckExtremLevel(start, dest, obs_polys)) {
        return MapLevel::Extrem;
    }

    const double dist_sd = std::hypot(start.loc.x - dest.loc.x, start.loc.y - dest.loc.y);
    const bool distance_exceed = dist_sd > kMaxDriveDistance;

    const auto sur = GetSurroundingObstacles(dest, obs_polys);
    const bool has_left = sur.left >= 0;
    const bool has_right = sur.right >= 0;
    const bool has_front = sur.front >= 0;
    const bool has_back = sur.back >= 0;

    const auto dest_box = dest.CreateBox();
    const Point2 rb = dest_box[0];
    const Point2 rf = dest_box[1];
    const Point2 lf = dest_box[2];
    const Point2 lb = dest_box[3];

    if (has_left && has_right && !has_front) {
        if (distance_exceed || !HasEnoughSpace(dest, obs_polys, GetLevelParams(MapLevel::Normal).min_park_lot_width, 0.0)) {
            return MapLevel::Complex;
        }
        std::vector<Point2> free_pts;
        const double dest_heading = dest.heading;
        free_pts.push_back(TranslatePoint(lf, dest_heading, 0.2));
        free_pts.push_back(TranslatePoint(rf, dest_heading, 0.2));
        free_pts.push_back(TranslatePoint(lf, dest_heading, GetLevelParams(MapLevel::Normal).bay_park_wall_dist - 0.5));
        free_pts.push_back(TranslatePoint(rf, dest_heading, GetLevelParams(MapLevel::Normal).bay_park_wall_dist - 0.5));
        free_pts.push_back(start.loc);
        const Polygon2 free_space = MinimumRotatedRectangle(free_pts);

        bool free_space_valid = true;
        for (size_t i = 0; i < obs_polys.size(); ++i) {
            if (static_cast<int>(i) == sur.left || static_cast<int>(i) == sur.right) {
                continue;
            }
            if (PolygonIntersects(free_space, obs_polys[i])) {
                free_space_valid = false;
                break;
            }
        }
        return free_space_valid ? MapLevel::Normal : MapLevel::Complex;
    }

    if (has_front && has_back) {
        if (distance_exceed || !HasEnoughSpace(dest, obs_polys, 0.0, GetLevelParams(MapLevel::Normal).min_park_lot_len)) {
            return MapLevel::Complex;
        }
        double out_direction = dest.heading + kPi / 2.0;
        Point2 key_pt_front = rf;
        Point2 key_pt_back = rb;
        const double projection = std::cos(out_direction) * (start.loc.x - dest.loc.x) +
                                  std::sin(out_direction) * (start.loc.y - dest.loc.y);
        if (projection < 0.0) {
            out_direction += kPi;
            key_pt_front = rf;
            key_pt_back = rb;
        } else {
            key_pt_front = lf;
            key_pt_back = lb;
        }

        std::vector<Point2> free_pts;
        free_pts.push_back(TranslatePoint(key_pt_front, out_direction, 0.2));
        free_pts.push_back(TranslatePoint(key_pt_back, out_direction, 0.2));
        free_pts.push_back(TranslatePoint(key_pt_front, out_direction, GetLevelParams(MapLevel::Normal).para_park_wall_dist - 0.5));
        free_pts.push_back(TranslatePoint(key_pt_back, out_direction, GetLevelParams(MapLevel::Normal).para_park_wall_dist - 0.5));
        const auto start_box = start.CreateBox();
        free_pts.insert(free_pts.end(), start_box.begin(), start_box.end());
        free_pts.push_back(start.loc);

        const Polygon2 free_space = MinimumRotatedRectangle(free_pts);
        bool free_space_valid = true;
        for (size_t i = 0; i < obs_polys.size(); ++i) {
            if (static_cast<int>(i) == sur.front || static_cast<int>(i) == sur.back) {
                continue;
            }
            if (PolygonIntersects(free_space, obs_polys[i])) {
                free_space_valid = false;
                break;
            }
        }
        return free_space_valid ? MapLevel::Normal : MapLevel::Complex;
    }

    if ((!has_left || !has_right) && (!has_front || !has_back)) {
        return MapLevel::Normal;
    }

    return MapLevel::Complex;
}

}  // namespace hope
