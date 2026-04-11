#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <chrono>
#include <cmath>
#include <future>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "act_pipeline.h"
#include "ros_robot_controller_msgs/msg/get_bus_servo_cmd.hpp"
#include "ros_robot_controller_msgs/msg/servo_position.hpp"
#include "ros_robot_controller_msgs/msg/servos_position.hpp"
#include "ros_robot_controller_msgs/srv/get_bus_servo_state.hpp"

using namespace std::chrono_literals;

namespace {

enum class RunState {
  IDLE,
  INITIALIZING,
  RUNNING,
  ESTOP,
  FAULT,
};

std::string ToString(RunState state) {
  switch (state) {
    case RunState::IDLE:
      return "IDLE";
    case RunState::INITIALIZING:
      return "INITIALIZING";
    case RunState::RUNNING:
      return "RUNNING";
    case RunState::ESTOP:
      return "ESTOP";
    case RunState::FAULT:
      return "FAULT";
  }
  return "UNKNOWN";
}

template <typename T>
std::string JoinVector(const std::vector<T>& values) {
  std::ostringstream oss;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << values[i];
  }
  return oss.str();
}

}  // namespace

class MeActInferenceNode : public rclcpp::Node {
 public:
  MeActInferenceNode()
      : Node("me_act_inference_node"),
        rng_(std::random_device{}()) {
    DeclareParameters();
    LoadParameters();

    pipeline_ = std::make_unique<ActPipeline>(deploy_dir_, device_);

    callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    servo_command_pub_ =
        create_publisher<ros_robot_controller_msgs::msg::ServosPosition>(servo_command_topic_, 10);
    rclcpp::ClientOptions client_options;
    client_options.callback_group = callback_group_;
    servo_state_client_ =
        create_client<ros_robot_controller_msgs::srv::GetBusServoState>(servo_state_service_, client_options);

    rgb_sub_.subscribe(this, rgb_topic_);
    depth_sub_.subscribe(this, depth_topic_);
    sync_ = std::make_shared<Synchronizer>(SyncPolicy(sync_queue_size_), rgb_sub_, depth_sub_);
    sync_->registerCallback(
        std::bind(&MeActInferenceNode::OnSyncedImages, this, std::placeholders::_1, std::placeholders::_2));

    start_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/start", std::bind(&MeActInferenceNode::HandleStart, this, std::placeholders::_1, std::placeholders::_2));
    stop_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/stop", std::bind(&MeActInferenceNode::HandleStop, this, std::placeholders::_1, std::placeholders::_2));
    estop_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/emergency_stop",
        std::bind(&MeActInferenceNode::HandleEmergencyStop, this, std::placeholders::_1, std::placeholders::_2));
    initialize_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/initialize",
        std::bind(&MeActInferenceNode::HandleInitialize, this, std::placeholders::_1, std::placeholders::_2));

    control_timer_ = create_wall_timer(
        std::chrono::milliseconds(control_period_ms_),
        std::bind(&MeActInferenceNode::OnControlTimer, this),
        callback_group_);

    state_ = enable_inference_on_start_ ? RunState::RUNNING : RunState::IDLE;
    if (enable_me_block_ && !pipeline_->UsesMemoryImageInput()) {
      RCLCPP_WARN(get_logger(), "enable_me_block=true, but exported ACT is a single-image model. me_block will be ignored.");
    }
    if (enable_me_block_ && pipeline_->UsesMemoryImageInput() && !pipeline_->HasMeBlock()) {
      throw std::runtime_error("enable_me_block=true, but deploy_dir does not contain me_block_inference.pt.");
    }
    RCLCPP_INFO(
        get_logger(),
        "me_act_inference_node ready. state=%s deploy_dir=%s enable_me_block=%s",
        ToString(state_).c_str(),
        deploy_dir_.c_str(),
        enable_me_block_ ? "true" : "false");
  }

 private:
  struct SyncedFrame {
    cv::Mat rgb_bgr;
    cv::Mat depth_raw;
    rclcpp::Time rgb_stamp;
    rclcpp::Time depth_stamp;
    rclcpp::Time synced_stamp;
    uint64_t frame_id = 0;
  };

  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image,
      sensor_msgs::msg::Image>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

  void DeclareParameters() {
    declare_parameter<std::string>("deploy_dir", "");
    declare_parameter<std::string>("device", "cpu");
    declare_parameter<std::string>("rgb_topic", "/depth_cam/rgb/image_raw");
    declare_parameter<std::string>("depth_topic", "/depth_cam/depth/image_raw");
    declare_parameter<std::string>("servo_command_topic", "/ros_robot_controller/bus_servo/set_position");
    declare_parameter<std::string>("servo_state_service", "/ros_robot_controller/bus_servo/get_state");
    declare_parameter<int>("control_period_ms", 200);
    declare_parameter<int>("command_duration_ms", 220);
    declare_parameter<int>("max_frame_age_ms", 250);
    declare_parameter<int>("max_state_image_skew_ms", 150);
    declare_parameter<int>("sync_queue_size", 10);
    declare_parameter<bool>("enable_inference_on_start", false);
    declare_parameter<bool>("enable_me_block", false);
    declare_parameter<std::vector<int64_t>>("servo_ids", {1, 2, 3, 4, 5, 10});
    declare_parameter<std::vector<int64_t>>("init_center", {500, 560, 120, 180, 500, 240});
    declare_parameter<int>("init_random_range", 100);
    declare_parameter<std::vector<int64_t>>("physical_min", {0, 0, 0, 0, 0, 100});
    declare_parameter<std::vector<int64_t>>("physical_max", {1000, 1000, 1000, 1000, 1000, 700});
  }

  void LoadParameters() {
    deploy_dir_ = get_parameter("deploy_dir").as_string();
    device_ = get_parameter("device").as_string();
    rgb_topic_ = get_parameter("rgb_topic").as_string();
    depth_topic_ = get_parameter("depth_topic").as_string();
    servo_command_topic_ = get_parameter("servo_command_topic").as_string();
    servo_state_service_ = get_parameter("servo_state_service").as_string();
    control_period_ms_ = get_parameter("control_period_ms").as_int();
    command_duration_ms_ = get_parameter("command_duration_ms").as_int();
    max_frame_age_ms_ = get_parameter("max_frame_age_ms").as_int();
    max_state_image_skew_ms_ = get_parameter("max_state_image_skew_ms").as_int();
    sync_queue_size_ = get_parameter("sync_queue_size").as_int();
    enable_inference_on_start_ = get_parameter("enable_inference_on_start").as_bool();
    enable_me_block_ = get_parameter("enable_me_block").as_bool();
    servo_ids_ = ToIntVector(get_parameter("servo_ids").as_integer_array());
    init_center_ = ToIntVector(get_parameter("init_center").as_integer_array());
    init_random_range_ = get_parameter("init_random_range").as_int();
    physical_min_ = ToIntVector(get_parameter("physical_min").as_integer_array());
    physical_max_ = ToIntVector(get_parameter("physical_max").as_integer_array());

    if (deploy_dir_.empty()) {
      throw std::runtime_error("Parameter deploy_dir must not be empty.");
    }
    if (servo_ids_.size() != 6 || init_center_.size() != 6 || physical_min_.size() != 6 || physical_max_.size() != 6) {
      throw std::runtime_error("servo_ids/init_center/physical_min/physical_max must all have length 6.");
    }
  }

  static std::vector<int> ToIntVector(const std::vector<int64_t>& values) {
    std::vector<int> out;
    out.reserve(values.size());
    for (const auto value : values) {
      out.push_back(static_cast<int>(value));
    }
    return out;
  }

  void OnSyncedImages(
      const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg) {
    try {
      const auto rgb_cv = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
      auto depth_cv = cv_bridge::toCvCopy(depth_msg, depth_msg->encoding);

      SyncedFrame frame;
      frame.rgb_bgr = rgb_cv->image.clone();
      frame.depth_raw = depth_cv->image.clone();
      frame.rgb_stamp = rgb_msg->header.stamp;
      frame.depth_stamp = depth_msg->header.stamp;
      frame.synced_stamp =
          frame.depth_stamp.nanoseconds() > frame.rgb_stamp.nanoseconds() ? frame.depth_stamp : frame.rgb_stamp;
      frame.frame_id = ++frame_counter_;

      std::lock_guard<std::mutex> lock(frame_mutex_);
      latest_frame_ = std::move(frame);
    } catch (const std::exception& exc) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "Image conversion failed: %s", exc.what());
    }
  }

  void OnControlTimer() {
    if (state_ == RunState::ESTOP || state_ == RunState::IDLE || state_ == RunState::FAULT) {
      return;
    }

    const auto now = get_clock()->now();
    if (state_ == RunState::INITIALIZING) {
      if (now < initialize_until_) {
        return;
      }
      {
        std::lock_guard<std::mutex> lock(pipeline_mutex_);
        pipeline_->ResetMemory();
      }
      tick_id_ = 0;
      state_ = RunState::RUNNING;
      RCLCPP_INFO(get_logger(), "Initialization finished. Switching to RUNNING.");
      return;
    }

    if (now < next_command_allowed_at_) {
      return;
    }

    auto frame = GetLatestFrame();
    if (!frame.has_value()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "No synced RGB/depth frame available yet.");
      return;
    }

    if ((now - frame->synced_stamp) > rclcpp::Duration::from_seconds(max_frame_age_ms_ / 1000.0)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Latest frame is too old. Skip this tick.");
      return;
    }

    const auto tick = ++tick_id_;
    const auto state_query_started = get_clock()->now();
    const auto qpos = QueryServoPositions();
    const auto state_received = get_clock()->now();
    if (!qpos.has_value()) {
      EnterFault("Failed to query servo states.");
      return;
    }

    const auto skew = state_received - frame->synced_stamp;
    if (std::abs(skew.nanoseconds()) > static_cast<int64_t>(max_state_image_skew_ms_) * 1000LL * 1000LL) {
      RCLCPP_WARN(
          get_logger(),
          "Tick %llu skipped due to image/state skew: %.1f ms",
          static_cast<unsigned long long>(tick),
          skew.nanoseconds() / 1e6);
      return;
    }

    try {
      const auto infer_started = get_clock()->now();
      std::vector<std::vector<float>> trajectory;
      {
        std::lock_guard<std::mutex> lock(pipeline_mutex_);
        trajectory = pipeline_->Predict(frame->rgb_bgr, frame->depth_raw, *qpos, enable_me_block_);
      }
      const auto infer_finished = get_clock()->now();
      if (trajectory.empty() || trajectory.front().size() != servo_ids_.size()) {
        EnterFault("ACT returned an empty trajectory or wrong action dimension.");
        return;
      }

      PublishServoCommand(trajectory.front());
      next_command_allowed_at_ = now + rclcpp::Duration::from_seconds(command_duration_ms_ / 1000.0);

      RCLCPP_INFO(
          get_logger(),
          "tick=%llu frame=%llu frame_age=%.1fms state_wait=%.1fms infer=%.1fms qpos=[%s] cmd0=[%s]",
          static_cast<unsigned long long>(tick),
          static_cast<unsigned long long>(frame->frame_id),
          (now - frame->synced_stamp).nanoseconds() / 1e6,
          (state_received - state_query_started).nanoseconds() / 1e6,
          (infer_finished - infer_started).nanoseconds() / 1e6,
          JoinVector(*qpos).c_str(),
          JoinVector(trajectory.front()).c_str());
    } catch (const std::exception& exc) {
      EnterFault(exc.what());
    }
  }

  std::optional<SyncedFrame> GetLatestFrame() {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return latest_frame_;
  }

  std::optional<std::vector<float>> QueryServoPositions() {
    if (!servo_state_client_->wait_for_service(500ms)) {
      RCLCPP_WARN(get_logger(), "Servo state service not available: %s", servo_state_service_.c_str());
      return std::nullopt;
    }

    auto request = std::make_shared<ros_robot_controller_msgs::srv::GetBusServoState::Request>();
    request->cmd.reserve(servo_ids_.size());
    for (const auto servo_id : servo_ids_) {
      ros_robot_controller_msgs::msg::GetBusServoCmd cmd;
      cmd.id = static_cast<uint8_t>(servo_id);
      cmd.get_position = 1;
      request->cmd.push_back(cmd);
    }

    auto future = servo_state_client_->async_send_request(request);
    if (future.wait_for(500ms) != std::future_status::ready) {
      RCLCPP_WARN(get_logger(), "Timeout while querying bus servo state.");
      return std::nullopt;
    }

    const auto response = future.get();
    if (!response->success || response->state.empty()) {
      RCLCPP_WARN(get_logger(), "GetBusServoState returned no valid state.");
      return std::nullopt;
    }

    std::vector<float> qpos;
    qpos.reserve(servo_ids_.size());
    for (const auto& bus_state : response->state) {
      for (const auto position : bus_state.position) {
        qpos.push_back(static_cast<float>(position));
      }
    }

    if (qpos.size() != servo_ids_.size()) {
      RCLCPP_WARN(
          get_logger(),
          "Expected %zu servo positions, but got %zu from GetBusServoState.",
          servo_ids_.size(),
          qpos.size());
      return std::nullopt;
    }

    return qpos;
  }

  void PublishServoCommand(const std::vector<float>& action) {
    ros_robot_controller_msgs::msg::ServosPosition msg;
    msg.duration = static_cast<float>(command_duration_ms_) / 1000.0f;
    msg.position.reserve(servo_ids_.size());

    for (size_t i = 0; i < servo_ids_.size(); ++i) {
      ros_robot_controller_msgs::msg::ServoPosition servo;
      servo.id = servo_ids_[i];
      servo.position = ClampToPhysicalRange(static_cast<int>(std::lround(action[i])), i);
      msg.position.push_back(servo);
    }

    servo_command_pub_->publish(msg);
  }

  int ClampToPhysicalRange(int value, size_t index) const {
    return std::max(physical_min_[index], std::min(value, physical_max_[index]));
  }

  std::vector<float> SampleInitializationPose() {
    std::uniform_int_distribution<int> dist(-init_random_range_, init_random_range_);
    std::vector<float> pose;
    pose.reserve(init_center_.size());
    for (size_t i = 0; i < init_center_.size(); ++i) {
      const int candidate = init_center_[i] + dist(rng_);
      pose.push_back(static_cast<float>(ClampToPhysicalRange(candidate, i)));
    }
    return pose;
  }

  void HandleStart(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (state_ == RunState::FAULT) {
      response->success = false;
      response->message = "Node is in FAULT state. Restart node or reinitialize after fixing the cause.";
      return;
    }
    if (state_ == RunState::INITIALIZING) {
      response->success = false;
      response->message = "Node is initializing. Wait until initialization finishes.";
      return;
    }
    state_ = RunState::RUNNING;
    response->success = true;
    response->message = "Inference started.";
  }

  void HandleStop(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    state_ = RunState::IDLE;
    {
      std::lock_guard<std::mutex> lock(pipeline_mutex_);
      pipeline_->ResetMemory();
    }
    next_command_allowed_at_ = get_clock()->now();
    response->success = true;
    response->message = "Inference stopped.";
  }

  void HandleEmergencyStop(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    state_ = RunState::ESTOP;
    {
      std::lock_guard<std::mutex> lock(pipeline_mutex_);
      pipeline_->ResetMemory();
    }
    next_command_allowed_at_ = get_clock()->now();
    response->success = true;
    response->message = "Emergency stop activated. No more motion commands will be sent.";
    RCLCPP_WARN(get_logger(), "Emergency stop activated.");
  }

  void HandleInitialize(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    try {
      const auto pose = SampleInitializationPose();
      PublishServoCommand(pose);
      {
        std::lock_guard<std::mutex> lock(pipeline_mutex_);
        pipeline_->ResetMemory();
      }
      next_command_allowed_at_ = get_clock()->now() + rclcpp::Duration::from_seconds(command_duration_ms_ / 1000.0);
      initialize_until_ = get_clock()->now() + rclcpp::Duration::from_seconds((command_duration_ms_ + 300) / 1000.0);
      state_ = RunState::INITIALIZING;
      response->success = true;
      response->message = "Initialization command sent.";
      RCLCPP_INFO(get_logger(), "Initialization pose sent: [%s]", JoinVector(pose).c_str());
    } catch (const std::exception& exc) {
      response->success = false;
      response->message = exc.what();
      EnterFault(exc.what());
    }
  }

  void EnterFault(const std::string& reason) {
    state_ = RunState::FAULT;
    {
      std::lock_guard<std::mutex> lock(pipeline_mutex_);
      pipeline_->ResetMemory();
    }
    RCLCPP_ERROR(get_logger(), "Entering FAULT state: %s", reason.c_str());
  }

  std::string deploy_dir_;
  std::string device_;
  std::string rgb_topic_;
  std::string depth_topic_;
  std::string servo_command_topic_;
  std::string servo_state_service_;
  int control_period_ms_ = 200;
  int command_duration_ms_ = 220;
  int max_frame_age_ms_ = 250;
  int max_state_image_skew_ms_ = 150;
  int sync_queue_size_ = 10;
  bool enable_inference_on_start_ = false;
  bool enable_me_block_ = false;
  std::vector<int> servo_ids_;
  std::vector<int> init_center_;
  int init_random_range_ = 100;
  std::vector<int> physical_min_;
  std::vector<int> physical_max_;

  RunState state_ = RunState::IDLE;
  uint64_t frame_counter_ = 0;
  uint64_t tick_id_ = 0;

  std::mutex frame_mutex_;
  std::mutex pipeline_mutex_;
  std::optional<SyncedFrame> latest_frame_;
  std::mt19937 rng_;

  std::unique_ptr<ActPipeline> pipeline_;

  message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  std::shared_ptr<Synchronizer> sync_;

  rclcpp::Publisher<ros_robot_controller_msgs::msg::ServosPosition>::SharedPtr servo_command_pub_;
  rclcpp::Client<ros_robot_controller_msgs::srv::GetBusServoState>::SharedPtr servo_state_client_;

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stop_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr estop_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr initialize_srv_;
  rclcpp::TimerBase::SharedPtr control_timer_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;

  rclcpp::Time next_command_allowed_at_{0, 0, RCL_ROS_TIME};
  rclcpp::Time initialize_until_{0, 0, RCL_ROS_TIME};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<MeActInferenceNode>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception& exc) {
    fprintf(stderr, "me_act_inference_node failed to start: %s\n", exc.what());
  }
  rclcpp::shutdown();
  return 0;
}
