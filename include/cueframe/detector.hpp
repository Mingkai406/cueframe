#pragma once
#include "cueframe/core.hpp"
#include <opencv2/dnn.hpp>
namespace cueframe {
struct Inference {
  std::vector<Detection> detections;
  double preprocess_ms = 0, inference_ms = 0, postprocess_ms = 0;
};
class Detector {
public:
  explicit Detector(const std::string &model, float threshold = .35f);
  Inference run(const cv::Mat &bgr);

private:
  cv::dnn::Net net_;
  float threshold_;
};
} // namespace cueframe
