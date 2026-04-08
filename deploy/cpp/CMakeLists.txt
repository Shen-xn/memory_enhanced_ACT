cmake_minimum_required(VERSION 3.18)
project(me_act_deploy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_library(act_pipeline
  src/act_pipeline.cpp
)
target_include_directories(act_pipeline PUBLIC include)
target_link_libraries(act_pipeline PUBLIC ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_compile_definitions(act_pipeline PUBLIC TORCH_API_INCLUDE_EXTENSION_H)

add_executable(act_infer_demo src/main.cpp)
target_link_libraries(act_infer_demo PRIVATE act_pipeline)

if(MSVC)
  target_compile_options(act_pipeline PRIVATE /W4)
  target_compile_options(act_infer_demo PRIVATE /W4)
else()
  target_compile_options(act_pipeline PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(act_infer_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()
