#pragma once

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

// logits 对应当前 s 列的各个 l 栅格点分数。
struct InferenceOutput {
    std::vector<float> logits;
    float value = 0.0f;
};

class DPInference {
public:
    explicit DPInference(const std::string& model_path);

    int feature_dim() const { return feature_dim_; }
    const std::string& input_name() const { return input_name_; }
    const std::string& logits_name() const { return logits_name_; }
    const std::string& value_name() const { return value_name_; }

    InferenceOutput Run(const std::vector<float>& input);

private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string input_name_;
    std::string logits_name_;
    std::string value_name_;
    int feature_dim_ = 0;
};
