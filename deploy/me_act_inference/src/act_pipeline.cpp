#include "act_pipeline.h"

#include <filesystem>
#include <stdexcept>
#include <torch/cuda.h>

namespace {

std::string JoinPath(const std::string& a, const std::string& b) {
  return (std::filesystem::path(a) / b).string();
}

}  // namespace

ActPipeline::ActPipeline(const std::string& deploy_dir, const std::string& device)
    : config_(LoadConfig(JoinPath(deploy_dir, "deploy_config.yml"))),
      device_(ParseDevice(device)),
      act_module_(torch::jit::load(JoinPath(deploy_dir, "act_inference.pt"), device_)) {
  act_module_.eval();
}

std::vector<std::vector<float>> ActPipeline::Predict(
    const cv::Mat& bgr,
    const cv::Mat& depth,
    const std::vector<float>& qpos) {
  const cv::Mat four_channel = BuildFourChannelImage(bgr, depth, config_);
  return PredictFromFourChannel(four_channel, qpos);
}

std::vector<std::vector<float>> ActPipeline::PredictFromFourChannel(
    const cv::Mat& four_channel_bgra,
    const std::vector<float>& qpos) {
  if (four_channel_bgra.empty() || four_channel_bgra.channels() != 4) {
    throw std::invalid_argument("PredictFromFourChannel expects a non-empty BGRA image.");
  }
  const torch::Tensor image_tensor = MatToTensor(four_channel_bgra);
  const torch::Tensor qpos_tensor = QposToTensor(qpos);
  torch::NoGradGuard no_grad;
  torch::Tensor actions = act_module_.forward({qpos_tensor, image_tensor}).toTensor();
  return TensorToTrajectory(actions.squeeze(0).to(torch::kCPU));
}

cv::Mat ActPipeline::BuildDebugFourChannelImage(const cv::Mat& bgr, const cv::Mat& depth) const {
  return BuildFourChannelImage(bgr, depth, config_);
}

DeployConfig ActPipeline::LoadConfig(const std::string& path) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    throw std::runtime_error("Failed to open deploy config: " + path);
  }

  DeployConfig cfg;
  fs["target_width"] >> cfg.target_width;
  fs["target_height"] >> cfg.target_height;
  fs["pad_left"] >> cfg.pad_left;
  fs["pad_top"] >> cfg.pad_top;
  fs["depth_clip_min"] >> cfg.depth_clip_min;
  fs["depth_clip_max"] >> cfg.depth_clip_max;
  fs["state_dim"] >> cfg.state_dim;
  fs["num_queries"] >> cfg.num_queries;
  if (!fs["image_channels"].empty()) {
    fs["image_channels"] >> cfg.image_channels;
  }
  return cfg;
}

torch::Device ActPipeline::ParseDevice(const std::string& device) {
  if (device == "cuda") {
    if (!torch::cuda::is_available()) {
      throw std::runtime_error("device='cuda' was requested, but LibTorch reports CUDA is not available.");
    }
    return torch::Device(torch::kCUDA);
  }
  return torch::Device(torch::kCPU);
}

cv::Mat ActPipeline::NormalizeDepth(const cv::Mat& depth, float clip_min, float clip_max) {
  if (depth.empty() || depth.channels() != 1) {
    throw std::invalid_argument("Depth image must be non-empty and single-channel.");
  }
  cv::Mat depth_f32;
  depth.convertTo(depth_f32, CV_32F);
  cv::min(depth_f32, clip_max, depth_f32);
  cv::max(depth_f32, clip_min, depth_f32);
  depth_f32 = (depth_f32 - clip_min) * (255.0f / (clip_max - clip_min));
  cv::Mat depth_u8;
  depth_f32.convertTo(depth_u8, CV_8U);
  return depth_u8;
}

cv::Mat ActPipeline::BuildFourChannelImage(const cv::Mat& bgr, const cv::Mat& depth, const DeployConfig& config) {
  if (bgr.empty() || bgr.channels() != 3) {
    throw std::invalid_argument("BGR image must be non-empty and 3-channel.");
  }

  cv::Mat bgr_resized;
  cv::resize(bgr, bgr_resized, cv::Size(config.target_width, config.target_height), 0.0, 0.0, cv::INTER_LINEAR);

  const cv::Mat depth_u8 = NormalizeDepth(depth, config.depth_clip_min, config.depth_clip_max);
  cv::Mat depth_aligned = cv::Mat::zeros(config.target_height, config.target_width, CV_8UC1);
  const int copy_w = std::min(depth_u8.cols, config.target_width - config.pad_left);
  const int copy_h = std::min(depth_u8.rows, config.target_height - config.pad_top);
  if (copy_w > 0 && copy_h > 0) {
    depth_u8(cv::Rect(0, 0, copy_w, copy_h))
        .copyTo(depth_aligned(cv::Rect(config.pad_left, config.pad_top, copy_w, copy_h)));
  }

  std::vector<cv::Mat> channels;
  cv::split(bgr_resized, channels);
  channels.push_back(depth_aligned);
  cv::Mat four_channel;
  cv::merge(channels, four_channel);
  return four_channel;
}

torch::Tensor ActPipeline::MatToTensor(const cv::Mat& image) const {
  cv::Mat continuous = image.isContinuous() ? image : image.clone();
  auto tensor = torch::from_blob(
                    continuous.data,
                    {1, continuous.rows, continuous.cols, continuous.channels()},
                    torch::TensorOptions().dtype(torch::kUInt8))
                    .to(torch::kFloat32)
                    .div_(255.0f)
                    .permute({0, 3, 1, 2})
                    .clone();
  return tensor.to(device_);
}

torch::Tensor ActPipeline::QposToTensor(const std::vector<float>& qpos) const {
  if (static_cast<int>(qpos.size()) != config_.state_dim) {
    throw std::invalid_argument("Unexpected qpos dimension.");
  }
  auto tensor = torch::from_blob(
                    const_cast<float*>(qpos.data()),
                    {1, config_.state_dim},
                    torch::TensorOptions().dtype(torch::kFloat32))
                    .clone();
  return tensor.to(device_);
}

std::vector<std::vector<float>> ActPipeline::TensorToTrajectory(const torch::Tensor& tensor) const {
  const torch::Tensor cpu = tensor.contiguous();
  const int64_t steps = cpu.size(0);
  const int64_t dims = cpu.size(1);
  std::vector<std::vector<float>> trajectory(steps, std::vector<float>(dims, 0.0f));
  const auto accessor = cpu.accessor<float, 2>();
  for (int64_t i = 0; i < steps; ++i) {
    for (int64_t j = 0; j < dims; ++j) {
      trajectory[i][j] = accessor[i][j];
    }
  }
  return trajectory;
}
