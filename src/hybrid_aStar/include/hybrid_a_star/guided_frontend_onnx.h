#ifndef HYBRID_A_STAR_GUIDED_FRONTEND_ONNX_H
#define HYBRID_A_STAR_GUIDED_FRONTEND_ONNX_H

#include "type.h"

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

namespace guided_frontend {

struct GridAstarOptions {
    double lambda_guidance{1.0};
    double diagonal_cost{1.4142135623730951};
    double heuristic_weight{1.0};
    double guidance_bonus_threshold{0.5};
    bool allow_corner_cut{false};
    std::string heuristic_mode{"octile"};
    std::string guidance_integration_mode{"g_cost"};
};

struct GridAstarResult {
    bool success{false};
    int expanded_nodes{0};
    std::vector<Vec2i> path_xy;
    std::vector<Vec2i> expanded_xy;
};

class GuidanceCostMapOnnx {
public:
    GuidanceCostMapOnnx(const std::string& model_path,
                       int intra_threads = 1,
                       int inter_threads = 1);

    std::vector<float> Infer(const std::vector<int>& occupancy,
                             int width,
                             int height,
                             const Vec2i& start_xy,
                             const Vec2i& goal_xy,
                             float start_yaw,
                             float goal_yaw);

private:
    std::vector<float> MakeOneHotMap(int width, int height, const Vec2i& xy) const;
    Vec2i ScaleGridCoord(const Vec2i& xy,
                        int src_width,
                        int src_height,
                        int dst_width,
                        int dst_height) const;
    std::vector<float> ResizeBinaryMap(const std::vector<int>& src,
                                       int src_width,
                                       int src_height,
                                       int dst_width,
                                       int dst_height) const;
    std::vector<float> ResizeFloatMap(const std::vector<float>& src,
                                      int src_width,
                                      int src_height,
                                      int dst_width,
                                      int dst_height) const;

private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<std::string> input_names_storage_;
    std::vector<const char*> input_names_;
    std::vector<std::string> output_names_storage_;
    std::vector<const char*> output_names_;
    int model_width_{-1};
    int model_height_{-1};
};

GridAstarResult RunGuidedGridAstar(const std::vector<int>& occupancy,
                                   int width,
                                   int height,
                                   const Vec2i& start_xy,
                                   const Vec2i& goal_xy,
                                   const std::vector<float>& guidance_cost,
                                   const GridAstarOptions& options);

}  // namespace guided_frontend

#endif  // HYBRID_A_STAR_GUIDED_FRONTEND_ONNX_H
