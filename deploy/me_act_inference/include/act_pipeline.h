#pragma once

#include <opencv2/opencv.hpp>
#include <torch/script.h>

#include <string>
#include <vector>

struct DeployConfig {
  int target_width = 640;
  int target_height = 480;
  int pad_left = 0;
  int pad_top = 40;
  float depth_clip_min = 0.0f;
  float depth_clip_max = 800.0f;
  int state_dim = 6;
  int num_queries = 10;
  bool use_memory_image_input = false;
  bool has_me_block = false;
  int me_block_num_classes = 3;
};

class ActPipeline {
 public:
  explicit ActPipeline(const std::string& deploy_dir, const std::string& device = "cpu");

  std::vector<std::vector<float>> Predict(
      const cv::Mat& bgr,
      const cv::Mat& depth,
      const std::vector<float>& qpos,
      bool use_me_block = true);

  std::vector<std::vector<float>> PredictFromFourChannel(
      const cv::Mat& four_channel_bgra,
      const std::vector<float>& qpos,
      bool use_me_block = true);

  void ResetMemory();
  bool UsesMemoryImageInput() const { return config_.use_memory_image_input; }
  bool HasMeBlock() const { return me_block_loaded_; }

 private:
  DeployConfig config_;
  torch::Device device_;
  torch::jit::script::Module act_module_;
  torch::jit::script::Module me_block_module_;
  bool me_block_loaded_ = false;
  torch::Tensor prev_memory_;
  torch::Tensor prev_scores_;

  static DeployConfig LoadConfig(const std::string& path);
  static torch::Device ParseDevice(const std::string& device);
  static cv::Mat NormalizeDepth(const cv::Mat& depth, float clip_min, float clip_max);
  static cv::Mat BuildFourChannelImage(const cv::Mat& bgr, const cv::Mat& depth, const DeployConfig& config);
  torch::Tensor MatToTensor(const cv::Mat& image) const;
  torch::Tensor QposToTensor(const std::vector<float>& qpos) const;
  std::vector<std::vector<float>> TensorToTrajectory(const torch::Tensor& tensor) const;
  void EnsureMemoryState();
};
