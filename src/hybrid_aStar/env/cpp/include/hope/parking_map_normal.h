#ifndef HOPE_PARKING_MAP_NORMAL_H
#define HOPE_PARKING_MAP_NORMAL_H

#include <vector>

#include "hope/configs.h"
#include "hope/map_types.h"
#include "hope/random_utils.h"

namespace hope {

class ParkingMapNormal {
public:
    explicit ParkingMapNormal(MapLevel level = MapLevel::Normal, unsigned int seed = 0);

    State Reset(int case_id = -1);
    const ParkingMapState& GetState() const { return state_; }

    void FlipDestOrientation();
    void FlipStartOrientation();

private:
    bool GenerateBayParkingCase(State& start, State& dest, std::vector<Polygon2>& obstacles);
    bool GenerateParallelParkingCase(State& start, State& dest, std::vector<Polygon2>& obstacles);

    State FlipBoxOrientation(const State& target_state) const;

    MapLevel level_;
    RandomGenerator rng_;
    ParkingMapState state_;
};

}  // namespace hope

#endif  // HOPE_PARKING_MAP_NORMAL_H
