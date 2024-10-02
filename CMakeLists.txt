cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(NeuralNetworkModelInference)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Torch REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(${PROJECT_SOURCE_DIR}/src/torchscript_nn_in_c/)

add_library(
    torchscript_nn_in_c SHARED
    src/torchscript_nn_in_c/torchscript_nn.cpp
    src/torchscript_nn_in_c/c_wrapper.cpp
)
target_link_libraries(torchscript_nn_in_c "${TORCH_LIBRARIES}")
set_property(TARGET torchscript_nn_in_c PROPERTY CXX_STANDARD 17)
add_executable(main src/main.c)
target_link_libraries(main torchscript_nn_in_c)
