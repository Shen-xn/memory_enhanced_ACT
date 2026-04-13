#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "ros_robot_controller_msgs/msg/servo_position.hpp"
#include "ros_robot_controller_msgs/msg/servos_position.hpp"

using namespace std::chrono_literals;

namespace {

std::vector<std::string> ListSortedImages(const std::string& dir) {
  std::vector<std::string> out;
  if (!std::filesystem::exists(dir)) {
    return out;
  }
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto ext = entry.path().extension().string();
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".JPG" || ext == ".JPEG" || ext == ".PNG") {
      out.push_back(entry.path().string());
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<std::string> SplitCsvLine(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string item;
  while (std::getline(ss, item, ',')) {
    out.push_back(item);
  }
  return out;
}

}  // namespace

class MeActReplayNode : public rclcpp::Node {
 public:
  MeActReplayNode() : Node("me_act_replay_node") {
    DeclareParameters();
    LoadParameters();

    servo_command_pub_ =
        create_publisher<ros_robot_controller_msgs::msg::ServosPosition>(servo_command_topic_, 10);
    rgb_pub_ = create_publisher<sensor_msgs::msg::Image>(rgb_topic_, 10);
    depth_pub_ = create_publisher<sensor_msgs::msg::Image>(depth_topic_, 10);

    start_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/start", std::bind(&MeActReplayNode::HandleStart, this, std::placeholders::_1, std::placeholders::_2));
    stop_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/stop", std::bind(&MeActReplayNode::HandleStop, this, std::placeholders::_1, std::placeholders::_2));

    LoadReplayData();

    timer_ = create_wall_timer(
        std::chrono::milliseconds(publish_period_ms_),
        std::bind(&MeActReplayNode::OnTimer, this));

    RCLCPP_INFO(
        get_logger(),
        "me_act_replay_node ready. frames=%zu running=%s task_dir=%s",
        std::min(rgb_paths_.size(), qpos_.size()),
        running_ ? "true" : "false",
        task_dir_.c_str());
  }

 private:
  void DeclareParameters() {
    declare_parameter<std::string>("task_dir", "");
    declare_parameter<std::string>("rgb_dirname", "rgb");
    declare_parameter<std::string>("depth_dirname", "depth_normalized");
    declare_parameter<std::string>("states_filename", "states_filtered.csv");
    declare_parameter<std::string>("rgb_topic", "/depth_cam/rgb/image_raw");
    declare_parameter<std::string>("depth_topic", "/depth_cam/depth/image_raw");
    declare_parameter<std::string>("servo_command_topic", "/ros_robot_controller/bus_servo/set_position");
    declare_parameter<int>("publish_period_ms", 200);
    declare_parameter<int>("command_duration_ms", 220);
    declare_parameter<bool>("loop", false);
    declare_parameter<bool>("start_on_launch", false);
    declare_parameter<std::vector<int64_t>>("servo_ids", {1, 2, 3, 4, 5, 10});
  }

  void LoadParameters() {
    task_dir_ = get_parameter("task_dir").as_string();
    rgb_dirname_ = get_parameter("rgb_dirname").as_string();
    depth_dirname_ = get_parameter("depth_dirname").as_string();
    states_filename_ = get_parameter("states_filename").as_string();
    rgb_topic_ = get_parameter("rgb_topic").as_string();
    depth_topic_ = get_parameter("depth_topic").as_string();
    servo_command_topic_ = get_parameter("servo_command_topic").as_string();
    publish_period_ms_ = get_parameter("publish_period_ms").as_int();
    command_duration_ms_ = get_parameter("command_duration_ms").as_int();
    loop_ = get_parameter("loop").as_bool();
    running_ = get_parameter("start_on_launch").as_bool();
    servo_ids_ = ToIntVector(get_parameter("servo_ids").as_integer_array());

    if (task_dir_.empty()) {
      throw std::runtime_error("task_dir must be set to a task_* folder.");
    }
    if (servo_ids_.size() != 6) {
      throw std::runtime_error("servo_ids must have length 6.");
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

  void LoadReplayData() {
    const auto rgb_dir = (std::filesystem::path(task_dir_) / rgb_dirname_).string();
    const auto depth_dir = (std::filesystem::path(task_dir_) / depth_dirname_).string();
    const auto csv_path = (std::filesystem::path(task_dir_) / states_filename_).string();

    rgb_paths_ = ListSortedImages(rgb_dir);
    depth_paths_ = ListSortedImages(depth_dir);
    qpos_ = LoadQpos(csv_path);

    if (rgb_paths_.empty()) {
      throw std::runtime_error("No RGB images found in " + rgb_dir);
    }
    if (depth_paths_.empty()) {
      throw std::runtime_error("No depth images found in " + depth_dir);
    }
    if (qpos_.empty()) {
      throw std::runtime_error("No qpos rows found in " + csv_path);
    }
  }

  std::vector<std::vector<float>> LoadQpos(const std::string& csv_path) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open " + csv_path);
    }

    std::string line;
    if (!std::getline(file, line)) {
      return {};
    }

    auto header = SplitCsvLine(line);
    auto find_col = [&](const std::string& name) {
      auto it = std::find(header.begin(), header.end(), name);
      if (it == header.end()) {
        return -1;
      }
      return static_cast<int>(std::distance(header.begin(), it));
    };

    const int idx_j1 = find_col("j1");
    const int idx_j2 = find_col("j2");
    const int idx_j3 = find_col("j3");
    const int idx_j4 = find_col("j4");
    const int idx_j5 = find_col("j5");
    const int idx_j10 = find_col("j10");
    if (idx_j1 < 0 || idx_j2 < 0 || idx_j3 < 0 || idx_j4 < 0 || idx_j5 < 0 || idx_j10 < 0) {
      throw std::runtime_error("states_filtered.csv missing joint columns j1..j10");
    }

    std::vector<std::vector<float>> rows;
    while (std::getline(file, line)) {
      if (line.empty()) {
        continue;
      }
      auto cols = SplitCsvLine(line);
      if (static_cast<int>(cols.size()) <= std::max({idx_j1, idx_j2, idx_j3, idx_j4, idx_j5, idx_j10})) {
        continue;
      }
      std::vector<float> row = {
          std::stof(cols[idx_j1]),
          std::stof(cols[idx_j2]),
          std::stof(cols[idx_j3]),
          std::stof(cols[idx_j4]),
          std::stof(cols[idx_j5]),
          std::stof(cols[idx_j10]),
      };
      rows.push_back(row);
    }
    return rows;
  }

  void OnTimer() {
    if (!running_) {
      return;
    }

    const size_t total = std::min({rgb_paths_.size(), depth_paths_.size(), qpos_.size()});
    if (total == 0) {
      return;
    }

    if (frame_index_ >= total) {
      if (loop_) {
        frame_index_ = 0;
      } else {
        running_ = false;
        RCLCPP_INFO(get_logger(), "Replay finished.");
        return;
      }
    }

    const auto& rgb_path = rgb_paths_[frame_index_];
    const auto& depth_path = depth_paths_[frame_index_];
    const auto& qpos = qpos_[frame_index_];

    cv::Mat rgb = cv::imread(rgb_path, cv::IMREAD_COLOR);
    cv::Mat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
    if (rgb.empty() || depth.empty()) {
      RCLCPP_WARN(get_logger(), "Failed to load frame %zu", frame_index_);
      frame_index_++;
      return;
    }

    PublishImages(rgb, depth);
    PublishServoCommand(qpos);
    frame_index_++;
  }

  void PublishImages(const cv::Mat& rgb, const cv::Mat& depth) {
    auto rgb_msg = cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::BGR8, rgb).toImageMsg();
    rgb_msg->header.stamp = now();
    rgb_pub_->publish(*rgb_msg);

    cv::Mat depth_mono;
    if (depth.channels() == 1) {
      depth_mono = depth;
    } else {
      cv::cvtColor(depth, depth_mono, cv::COLOR_BGR2GRAY);
    }
    auto depth_msg = cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::MONO8, depth_mono).toImageMsg();
    depth_msg->header.stamp = rgb_msg->header.stamp;
    depth_pub_->publish(*depth_msg);
  }

  void PublishServoCommand(const std::vector<float>& action) {
    ros_robot_controller_msgs::msg::ServosPosition msg;
    msg.duration = static_cast<float>(command_duration_ms_) / 1000.0f;
    msg.position.reserve(servo_ids_.size());

    for (size_t i = 0; i < servo_ids_.size(); ++i) {
      ros_robot_controller_msgs::msg::ServoPosition servo;
      servo.id = servo_ids_[i];
      servo.position = static_cast<int>(std::lround(action[i]));
      msg.position.push_back(servo);
    }

    servo_command_pub_->publish(msg);
  }

  void HandleStart(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    running_ = true;
    response->success = true;
    response->message = "Replay started.";
  }

  void HandleStop(
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    running_ = false;
    response->success = true;
    response->message = "Replay stopped.";
  }

  std::string task_dir_;
  std::string rgb_dirname_;
  std::string depth_dirname_;
  std::string states_filename_;
  std::string rgb_topic_;
  std::string depth_topic_;
  std::string servo_command_topic_;
  int publish_period_ms_ = 200;
  int command_duration_ms_ = 220;
  bool loop_ = false;
  bool running_ = false;
  std::vector<int> servo_ids_;

  std::vector<std::string> rgb_paths_;
  std::vector<std::string> depth_paths_;
  std::vector<std::vector<float>> qpos_;
  size_t frame_index_ = 0;

  rclcpp::Publisher<ros_robot_controller_msgs::msg::ServosPosition>::SharedPtr servo_command_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stop_srv_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<MeActReplayNode>();
    rclcpp::spin(node);
  } catch (const std::exception& exc) {
    fprintf(stderr, "me_act_replay_node failed to start: %s\n", exc.what());
  }
  rclcpp::shutdown();
  return 0;
}
