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

  const std::string me_block_path = JoinPath(deploy_dir, "me_block_inference.pt");
  if (config_.has_me_block && std::filesystem::exists(me_block_path)) {
    me_block_module_ = torch::jit::load(me_block_path, device_);
    me_block_module_.eval();
    me_block_loaded_ = true;
  }

  ResetMemory();
}

std::vector<std::vector<float>> ActPipeline::Predict(
    const cv::Mat& bgr,
    const cv::Mat& depth,
    const std::vector<float>& qpos,
    bool use_me_block) {
  // Deployment receives raw BGR and depth. Convert them to the same BGRA
  // tensor contract used by ACT training before calling the model.
  const cv::Mat four_channel = BuildFourChannelImage(bgr, depth, config_);
  return PredictFromFourChannel(four_channel, qpos, use_me_block);
}

std::vector<std::vector<float>> ActPipeline::PredictFromFourChannel(
    const cv::Mat& four_channel_bgra,
    const std::vector<float>& qpos,
    bool use_me_block) {
  if (four_channel_bgra.empty() || four_channel_bgra.channels() != 4) {
    throw std::invalid_argument("PredictFromFourChannel expects a non-empty BGRA image.");
  }

  const torch::Tensor image_tensor = MatToTensor(four_channel_bgra);
  const torch::Tensor qpos_tensor = QposToTensor(qpos);
  torch::Tensor memory_tensor = torch::zeros_like(image_tensor);

  if (config_.use_memory_image_input) {
    EnsureMemoryState();
    if (use_me_block && me_block_loaded_) {
      // me_block returns the rendered memory image plus updated recurrent
      // state. The state stays in this pipeline until ResetMemory() is called.
      auto outputs = me_block_module_.forward({image_tensor, prev_memory_, prev_scores_}).toTuple();
      memory_tensor = outputs->elements()[0].toTensor();
      prev_memory_ = outputs->elements()[1].toTensor();
      prev_scores_ = outputs->elements()[2].toTensor();
    } else if (use_me_block) {
      throw std::runtime_error("me_block was requested, but me_block_inference.pt is not loaded.");
    } else {
      ResetMemory();
    }
  }

  torch::NoGradGuard no_grad;
  torch::Tensor actions;
  if (config_.use_memory_image_input) {
    actions = act_module_.forward({qpos_tensor, image_tensor, memory_tensor}).toTensor();
  } else {
    actions = act_module_.forward({qpos_tensor, image_tensor}).toTensor();
  }

  return TensorToTrajectory(actions.squeeze(0).to(torch::kCPU));
}

void ActPipeline::ResetMemory() {
  // State shape is [batch, class, channel, height, width] for memory images and
  // [batch, class, height, width] for scores. Batch is always 1 in deployment.
  prev_memory_ = torch::zeros(
      {1, config_.me_block_num_classes, 4, config_.target_height, config_.target_width},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  prev_scores_ = torch::zeros(
      {1, config_.me_block_num_classes, config_.target_height, config_.target_width},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));
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
  int use_memory = 0;
  int has_me_block = 0;
  fs["use_memory_image_input"] >> use_memory;
  fs["has_me_block"] >> has_me_block;
  if (!fs["me_block_num_classes"].empty()) {
    fs["me_block_num_classes"] >> cfg.me_block_num_classes;
  }
  cfg.use_memory_image_input = use_memory != 0;
  cfg.has_me_block = has_me_block != 0;
  return cfg;
}

torch::Device ActPipeline::ParseDevice(const std::string& device) {
  if (device == "cuda") {
    if (!torch::cuda::is_available()) {
      throw std::runtime_error(
          "ROS parameter device='cuda' was requested, but LibTorch reports CUDA is not available. "
          "Install a CUDA-enabled LibTorch/PyTorch build on Jetson or set device='cpu'.");
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
  // Match data_process_2.py: place normalized depth with left/top padding into
  // the final RGB-sized frame, cropping any overflow at right/bottom.
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
  // Clone after from_blob so TorchScript execution never depends on cv::Mat
  // storage lifetime.
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

void ActPipeline::EnsureMemoryState() {
  if (!prev_memory_.defined() || !prev_scores_.defined()) {
    ResetMemory();
  }
}
