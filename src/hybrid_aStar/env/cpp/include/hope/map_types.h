#ifndef HOPE_MAP_TYPES_H
#define HOPE_MAP_TYPES_H

#include <string>
#include <vector>

#include "hope/configs.h"
#include "hope/geometry.h"
#include "hope/vehicle.h"

namespace hope {

struct Area {
    Polygon2 shape;
    std::string subtype;
    Color color{0, 0, 0, 255};
};

struct MapBoundary {
    double xmin = 0.0;
    double xmax = 0.0;
    double ymin = 0.0;
    double ymax = 0.0;
};

struct ParkingMapState {
    int case_id = 0;
    State start;
    State dest;
    Polygon2 start_box;
    Polygon2 dest_box;
    MapBoundary boundary;
    std::vector<Area> obstacles;
    MapLevel level = MapLevel::Normal;
};

}  // namespace hope

#endif  // HOPE_MAP_TYPES_H
