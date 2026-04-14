cmake_minimum_required(VERSION 3.8)
project(me_act_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(ros_robot_controller_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

add_executable(me_act_inference_node
  src/act_pipeline.cpp
  src/me_act_inference_node.cpp
)

add_executable(me_act_replay_node
  src/me_act_replay_node.cpp
)

target_include_directories(me_act_inference_node PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(me_act_replay_node PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(me_act_inference_node
  rclcpp
  sensor_msgs
  std_srvs
  cv_bridge
  message_filters
  ros_robot_controller_msgs
)

ament_target_dependencies(me_act_replay_node
  rclcpp
  sensor_msgs
  std_srvs
  cv_bridge
  ros_robot_controller_msgs
)

target_link_libraries(me_act_inference_node
  ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
)

target_link_libraries(me_act_replay_node
  ${OpenCV_LIBS}
)

install(TARGETS me_act_inference_node me_act_replay_node
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/me_act_inference_node_py.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  scripts/act_pipeline_py.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

ament_package()
