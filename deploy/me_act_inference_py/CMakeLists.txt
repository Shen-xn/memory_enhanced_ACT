cmake_minimum_required(VERSION 3.8)
project(me_act_inference_py)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

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

ament_package()
