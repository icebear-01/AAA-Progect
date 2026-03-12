#include "hope/random_utils.h"

#include <algorithm>
#include <chrono>

namespace hope {

RandomGenerator::RandomGenerator() {
    const auto seed = static_cast<unsigned int>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    engine_.seed(seed);
}

RandomGenerator::RandomGenerator(unsigned int seed) { engine_.seed(seed); }

double RandomGenerator::Uniform(double low, double high) {
    std::uniform_real_distribution<double> dist(low, high);
    return dist(engine_);
}

double RandomGenerator::Normal(double mean, double stddev) {
    std::normal_distribution<double> dist(mean, stddev);
    return dist(engine_);
}

double RandomGaussianNum(RandomGenerator& rng,
                         double mean,
                         double stddev,
                         double clip_low,
                         double clip_high) {
    double value = rng.Normal(mean, stddev);
    return std::clamp(value, clip_low, clip_high);
}

double RandomUniformNum(RandomGenerator& rng, double clip_low, double clip_high) {
    return rng.Uniform(clip_low, clip_high);
}

}  // namespace hope
