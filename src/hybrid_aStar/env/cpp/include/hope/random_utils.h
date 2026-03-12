#ifndef HOPE_RANDOM_UTILS_H
#define HOPE_RANDOM_UTILS_H

#include <random>

namespace hope {

class RandomGenerator {
public:
    RandomGenerator();
    explicit RandomGenerator(unsigned int seed);

    double Uniform(double low, double high);
    double Normal(double mean, double stddev);

private:
    std::mt19937 engine_;
};

double RandomGaussianNum(RandomGenerator& rng, double mean, double stddev, double clip_low, double clip_high);
double RandomUniformNum(RandomGenerator& rng, double clip_low, double clip_high);

}  // namespace hope

#endif  // HOPE_RANDOM_UTILS_H
