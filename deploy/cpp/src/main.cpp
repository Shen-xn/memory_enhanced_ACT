#include "act_pipeline.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

std::vector<float> ParseQpos(const std::string& text) {
  std::vector<float> values;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    values.push_back(std::stof(item));
  }
  return values;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "Usage: act_infer_demo <deploy_dir> <rgb_path> <depth_path> <qpos_csv> [device]\n";
    return 1;
  }

  try {
    const std::string deploy_dir = argv[1];
    const std::string rgb_path = argv[2];
    const std::string depth_path = argv[3];
    const std::string qpos_csv = argv[4];
    const std::string device = argc > 5 ? argv[5] : "cpu";

    cv::Mat bgr = cv::imread(rgb_path, cv::IMREAD_COLOR);
    cv::Mat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
    if (bgr.empty() || depth.empty()) {
      throw std::runtime_error("Failed to read rgb/depth inputs.");
    }

    ActPipeline pipeline(deploy_dir, device);
    const auto trajectory = pipeline.Predict(bgr, depth, ParseQpos(qpos_csv), true);

    for (size_t i = 0; i < trajectory.size(); ++i) {
      std::cout << "step[" << i << "]:";
      for (float value : trajectory[i]) {
        std::cout << ' ' << value;
      }
      std::cout << '\n';
    }
  } catch (const std::exception& exc) {
    std::cerr << "Inference failed: " << exc.what() << '\n';
    return 2;
  }

  return 0;
}
