#include "cueframe/detector.hpp"
#include <chrono>
#include <cmath>
#include <opencv2/imgproc.hpp>
namespace cueframe {
namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
const std::vector<std::string> labels = {"person",        "bicycle",      "car",
                                         "motorcycle",    "airplane",     "bus",
                                         "train",         "truck",        "boat",
                                         "traffic light", "fire hydrant", "stop sign",
                                         "parking meter", "bench",        "bird",
                                         "cat",           "dog",          "horse",
                                         "sheep",         "cow",          "elephant",
                                         "bear",          "zebra",        "giraffe",
                                         "backpack",      "umbrella",     "handbag",
                                         "tie",           "suitcase",     "frisbee",
                                         "skis",          "snowboard",    "sports ball",
                                         "kite",          "baseball bat", "baseball glove",
                                         "skateboard",    "surfboard",    "tennis racket",
                                         "bottle",        "wine glass",   "cup",
                                         "fork",          "knife",        "spoon",
                                         "bowl",          "banana",       "apple",
                                         "sandwich",      "orange",       "broccoli",
                                         "carrot",        "hot dog",      "pizza",
                                         "donut",         "cake",         "chair",
                                         "couch",         "potted plant", "bed",
                                         "dining table",  "toilet",       "tv",
                                         "laptop",        "mouse",        "remote",
                                         "keyboard",      "cell phone",   "microwave",
                                         "oven",          "toaster",      "sink",
                                         "refrigerator",  "book",         "clock",
                                         "vase",          "scissors",     "teddy bear",
                                         "hair drier",    "toothbrush"};
} // namespace
Detector::Detector(const std::string &model, float t)
    : net_(cv::dnn::readNetFromONNX(model)), threshold_(t) {
  net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}
Inference Detector::run(const cv::Mat &bgr) {
  if (bgr.empty() || bgr.type() != CV_8UC3)
    throw std::invalid_argument("expected nonempty BGR8 image");
  auto start = Clock::now();
  float ratio = std::min(640.f / bgr.cols, 640.f / bgr.rows);
  cv::Mat resized, padded(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));
  cv::resize(bgr, resized,
             cv::Size(std::max(1, int(bgr.cols * ratio)), std::max(1, int(bgr.rows * ratio))));
  resized.copyTo(padded(cv::Rect(0, 0, resized.cols, resized.rows)));
  // OpenCV Zoo YOLOX: RGB, raw 0..255, top-left letterbox, 114 padding.
  auto blob =
      cv::dnn::blobFromImage(padded, 1.0, cv::Size(640, 640), cv::Scalar(), true, false, CV_32F);
  auto pre = Clock::now();
  net_.setInput(blob);
  cv::Mat out = net_.forward();
  auto inf = Clock::now();
  if (out.total() != 8400 * 85 || out.type() != CV_32F)
    throw std::runtime_error("unexpected YOLOX output shape");
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> classes;
  int row = 0;
  for (int stride : {8, 16, 32})
    for (int y = 0; y < 640 / stride; ++y)
      for (int x = 0; x < 640 / stride; ++x, ++row) {
        const float *p = out.ptr<float>() + row * 85;
        int cls = int(std::max_element(p + 5, p + 85) - (p + 5));
        float score = p[4] * p[5 + cls];
        if (!std::isfinite(score) || score < threshold_)
          continue;
        float cx = (p[0] + x) * stride / ratio, cy = (p[1] + y) * stride / ratio;
        float w = std::exp(p[2]) * stride / ratio, h = std::exp(p[3]) * stride / ratio;
        if (!std::isfinite(cx + cy + w + h))
          continue;
        int x1 = int(std::clamp(cx - w / 2, 0.f, float(bgr.cols))),
            y1 = int(std::clamp(cy - h / 2, 0.f, float(bgr.rows)));
        int x2 = int(std::clamp(cx + w / 2, 0.f, float(bgr.cols))),
            y2 = int(std::clamp(cy + h / 2, 0.f, float(bgr.rows)));
        if (x2 <= x1 || y2 <= y1)
          continue;
        boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
        scores.push_back(score);
        classes.push_back(cls);
      }
  std::vector<int> keep;
  cv::dnn::NMSBoxesBatched(boxes, scores, classes, threshold_, .5f, keep);
  Inference result;
  for (auto i : keep) {
    auto b = boxes[i];
    result.detections.push_back({classes[i],
                                 labels[classes[i]],
                                 scores[i],
                                 {float(b.x) / bgr.cols, float(b.y) / bgr.rows,
                                  float(b.width) / bgr.cols, float(b.height) / bgr.rows}});
    if (result.detections.size() == 256)
      break;
  }
  auto end = Clock::now();
  result.preprocess_ms = ms(start, pre);
  result.inference_ms = ms(pre, inf);
  result.postprocess_ms = ms(inf, end);
  return result;
}
} // namespace cueframe
