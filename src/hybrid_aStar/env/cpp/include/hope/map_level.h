#ifndef HOPE_MAP_LEVEL_H
#define HOPE_MAP_LEVEL_H

#include <vector>

#include "hope/configs.h"
#include "hope/map_types.h"

namespace hope {

MapLevel GetMapLevel(const State& start, const State& dest, const std::vector<Area>& obstacles);

}  // namespace hope

#endif  // HOPE_MAP_LEVEL_H
