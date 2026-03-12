#ifndef HOPE_REEDS_SHEPP_H
#define HOPE_REEDS_SHEPP_H

#include <vector>

namespace hope {

struct ReedsSheppPath {
    std::vector<double> lengths;
    std::vector<char> ctypes;
    double total_length = 0.0;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> yaw;
    std::vector<int> directions;
};

ReedsSheppPath CalcOptimalPath(double sx,
                               double sy,
                               double syaw,
                               double gx,
                               double gy,
                               double gyaw,
                               double maxc,
                               double step_size = 0.2);

std::vector<ReedsSheppPath> CalcAllPaths(double sx,
                                        double sy,
                                        double syaw,
                                        double gx,
                                        double gy,
                                        double gyaw,
                                        double maxc,
                                        double step_size = 0.2);

}  // namespace hope

#endif  // HOPE_REEDS_SHEPP_H
