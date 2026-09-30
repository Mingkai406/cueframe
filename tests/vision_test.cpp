#include "cueframe/detector.hpp"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
using namespace cueframe;
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      return 2;
    cv::setNumThreads(2);
    Detector model(argv[1]);
    auto im = cv::imread(argv[2]);
    auto first = model.run(im);
    bool cup = false;
    for (auto &d : first.detections)
      if (d.label == "cup" && d.confidence > .5 && d.box.x > .2 && d.box.x < .4 && d.box.w > .3 &&
          d.box.w < .6)
        cup = true;
    if (!cup)
      throw std::runtime_error("reference cup or inverse-letterbox geometry missing");
    cv::Mat portrait(800, 600, CV_8UC3, cv::Scalar(114, 114, 114));
    im.copyTo(portrait(cv::Rect(0, 200, 600, 400)));
    auto second = model.run(portrait);
    cup = false;
    for (auto &d : second.detections) {
      auto b = d.box;
      if (b.x < 0 || b.y < 0 || b.x + b.w > 1.001 || b.y + b.h > 1.001)
        throw std::runtime_error("box outside frame");
      if (d.label == "cup" && b.y > .2 && b.y < .4)
        cup = true;
    }
    if (!cup)
      throw std::runtime_error("portrait letterbox projection failed");
    bool rejected = false;
    try {
      model.run(cv::Mat());
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("empty frame accepted");
    std::cout << "PASS real ONNX inference, landscape/portrait geometry, empty input\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
