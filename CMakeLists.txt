cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(NeuralNetworkModelInference)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(${PROJECT_SOURCE_DIR}/src/onnx_nn_in_c/)

include_directories("${ONNXRUNTIME_ROOTDIR}")
# = CPU Build ==================================================================
# include_directories("${ONNXRUNTIME_ROOTDIR}/include")
# link_directories("${ONNXRUNTIME_ROOTDIR}/lib")
# = CPU + iGPU Build ===========================================================
# include_directories("${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
# link_directories("${ONNXRUNTIME_ROOTDIR}/build/Linux/RelWithDebInfo")
# ==============================================================================

add_library(
    onnx_nn_in_c SHARED
    src/onnx_nn_in_c/onnx_nn.cpp
    src/onnx_nn_in_c/c_wrapper.cpp
)
set_property(TARGET onnx_nn_in_c PROPERTY CXX_STANDARD 17)
add_executable(main src/main.c)
target_link_libraries(main onnx_nn_in_c onnxruntime)
