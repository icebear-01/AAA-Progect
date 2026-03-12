#ifndef HOPE_PARKING_MAP_DLP_H
#define HOPE_PARKING_MAP_DLP_H

#include <string>
#include <vector>

#include "hope/map_level.h"
#include "hope/map_types.h"
#include "hope/random_utils.h"

namespace hope {

struct DlpCase {
    std::vector<State> starts;
    State dest;
    std::vector<Polygon2> obstacles;
};

class ParkingMapDlp {
public:
    explicit ParkingMapDlp(const std::string& path = std::string());

    State Reset(int case_id = -1, const std::string& path = std::string());
    const ParkingMapState& GetState() const { return state_; }
    int NumCases() const { return static_cast<int>(cases_.size()); }

private:
    void Load(const std::string& path);
    void FilterObstacles();
    void FlipDestOrientation();
    void FlipStartOrientation();
    State FlipBoxOrientation(const State& target_state) const;

    std::vector<DlpCase> cases_;
    bool multi_start_ = false;
    RandomGenerator rng_;
    ParkingMapState state_;
};

}  // namespace hope

#endif  // HOPE_PARKING_MAP_DLP_H
