#include "dp_infer.h"

#include <array>
#include <stdexcept>

namespace {
Ort::SessionOptions CreateSessionOptions() {
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return opts;
}
}  // namespace

DPInference::DPInference(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "dp_infer"),
      session_(env_, model_path.c_str(), CreateSessionOptions()) {
    Ort::AllocatedStringPtr input_name = session_.GetInputNameAllocated(0, allocator_);
    input_name_ = input_name.get();
    Ort::AllocatedStringPtr logits_name = session_.GetOutputNameAllocated(0, allocator_);
    logits_name_ = logits_name.get();
    Ort::AllocatedStringPtr value_name = session_.GetOutputNameAllocated(1, allocator_);
    value_name_ = value_name.get();

    auto type_info = session_.GetInputTypeInfo(0);
    auto input_info = type_info.GetTensorTypeAndShapeInfo();
    auto input_shape = input_info.GetShape();
    if (input_shape.size() < 2 || input_shape[1] <= 0) {
        throw std::runtime_error("Could not determine feature_dim from model input.");
    }
    feature_dim_ = static_cast<int>(input_shape[1]);
}

InferenceOutput DPInference::Run(const std::vector<float>& input) {
    if (input.size() != static_cast<size_t>(feature_dim_)) {
        throw std::runtime_error("Input feature size mismatch.");
    }
    std::array<int64_t, 2> dims{1, feature_dim_};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        const_cast<float*>(input.data()),
        input.size(),
        dims.data(),
        dims.size());

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {logits_name_.c_str(), value_name_.c_str()};
    auto outputs = session_.Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 2);

    const Ort::Value& logits_val = outputs[0];
    auto logits_info = logits_val.GetTensorTypeAndShapeInfo();
    const size_t logits_count = logits_info.GetElementCount();
    if (logits_count == 0) {
        throw std::runtime_error("Logits output is empty.");
    }
    const float* logits_data = logits_val.GetTensorData<float>();
    std::vector<float> logits(logits_data, logits_data + logits_count);

    const Ort::Value& value_val = outputs[1];
    auto value_info = value_val.GetTensorTypeAndShapeInfo();
    const size_t value_count = value_info.GetElementCount();
    if (value_count == 0) {
        throw std::runtime_error("Value output is empty.");
    }
    const float* value_data = value_val.GetTensorData<float>();

    InferenceOutput result;
    result.logits = std::move(logits);
    result.value = value_data[0];
    return result;
}
