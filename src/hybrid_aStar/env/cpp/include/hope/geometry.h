#ifndef HOPE_GEOMETRY_H
#define HOPE_GEOMETRY_H

#include <array>
#include <vector>

namespace hope {

struct Point2 {
    double x;
    double y;
};

using Polygon2 = std::vector<Point2>;

Polygon2 ClosePolygon(const Polygon2& poly);
Polygon2 TransformPolygon(const Polygon2& poly, double cos_t, double sin_t, double tx, double ty);
Polygon2 AffineTransformPolygon(const Polygon2& poly,
                                double a,
                                double b,
                                double d,
                                double e,
                                double xoff,
                                double yoff);

bool PolygonIntersects(const Polygon2& a, const Polygon2& b);
double PolygonDistance(const Polygon2& a, const Polygon2& b);
double PolygonIntersectionArea(const Polygon2& a, const Polygon2& b);
double PolygonArea(const Polygon2& poly);
Point2 PolygonCentroid(const Polygon2& poly);

}  // namespace hope

#endif  // HOPE_GEOMETRY_H
