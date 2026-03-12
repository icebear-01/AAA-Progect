#include "hope/geometry.h"

#include <cmath>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/multi_polygon.hpp>

namespace hope {

namespace bg = boost::geometry;
using BoostPoint = bg::model::d2::point_xy<double>;
using BoostPolygon = bg::model::polygon<BoostPoint>;
using BoostMultiPolygon = bg::model::multi_polygon<BoostPolygon>;

static BoostPolygon ToBoost(const Polygon2& poly) {
    BoostPolygon boost_poly;
    auto& ring = boost_poly.outer();
    ring.clear();
    if (poly.empty()) {
        return boost_poly;
    }
    for (const auto& pt : poly) {
        ring.emplace_back(pt.x, pt.y);
    }
    if (poly.front().x != poly.back().x || poly.front().y != poly.back().y) {
        ring.emplace_back(poly.front().x, poly.front().y);
    }
    bg::correct(boost_poly);
    return boost_poly;
}

Polygon2 ClosePolygon(const Polygon2& poly) {
    if (poly.empty()) {
        return poly;
    }
    Polygon2 closed = poly;
    if (poly.front().x != poly.back().x || poly.front().y != poly.back().y) {
        closed.push_back(poly.front());
    }
    return closed;
}

Polygon2 TransformPolygon(const Polygon2& poly, double cos_t, double sin_t, double tx, double ty) {
    Polygon2 out;
    out.reserve(poly.size());
    for (const auto& pt : poly) {
        const double x = cos_t * pt.x - sin_t * pt.y + tx;
        const double y = sin_t * pt.x + cos_t * pt.y + ty;
        out.push_back({x, y});
    }
    return out;
}

Polygon2 AffineTransformPolygon(const Polygon2& poly,
                                double a,
                                double b,
                                double d,
                                double e,
                                double xoff,
                                double yoff) {
    Polygon2 out;
    out.reserve(poly.size());
    for (const auto& pt : poly) {
        const double x = a * pt.x + b * pt.y + xoff;
        const double y = d * pt.x + e * pt.y + yoff;
        out.push_back({x, y});
    }
    return out;
}

bool PolygonIntersects(const Polygon2& a, const Polygon2& b) {
    return bg::intersects(ToBoost(a), ToBoost(b));
}

double PolygonDistance(const Polygon2& a, const Polygon2& b) {
    return bg::distance(ToBoost(a), ToBoost(b));
}

double PolygonIntersectionArea(const Polygon2& a, const Polygon2& b) {
    const BoostPolygon pa = ToBoost(a);
    const BoostPolygon pb = ToBoost(b);
    BoostMultiPolygon output;
    bg::intersection(pa, pb, output);
    double area = 0.0;
    for (const auto& poly : output) {
        area += std::abs(bg::area(poly));
    }
    return area;
}

double PolygonArea(const Polygon2& poly) {
    const BoostPolygon bp = ToBoost(poly);
    return std::abs(bg::area(bp));
}

Point2 PolygonCentroid(const Polygon2& poly) {
    const BoostPolygon bp = ToBoost(poly);
    BoostPoint c;
    bg::centroid(bp, c);
    return {c.x(), c.y()};
}

}  // namespace hope
