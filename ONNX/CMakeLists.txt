cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(NeuralNetworkModelInference)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${PROJECT_SOURCE_DIR}/src/)

include_directories("${ONNXRUNTIME_ROOTDIR}")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

add_executable(main src/main.c src/utils.c src/onnx_nn.c)
target_link_libraries(main onnxruntime)
