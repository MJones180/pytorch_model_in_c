#ifndef ONNX_NN_H
#define ONNX_NN_H

#include <constants.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

class NN_Model {
    std::string model_dir_path;
    // https://github.com/microsoft/onnxruntime/issues/4131#issuecomment-682796289
    std::unique_ptr<Ort::Session> onnx_session = nullptr;
    // Variables that are needed to run the model
    Ort::MemoryInfo onnx_mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    std::vector<std::int64_t> onnx_input_shape = {1, 1, IPS, IPS};
    std::vector<const char*> onnx_input_name = {NN_INPUT_NAME};
    std::vector<const char*> onnx_output_name = {NN_OUTPUT_NAME};
    // The base field that should be subtracted off
    double base_field[IPS][IPS];
    // There should be a single value for all inputs.
    double input_mmd; // Input max min diff
    double input_mx;  // Input min x
    // Each output should have its own value
    double output_mmd[OVS]; // Output max min diff
    double output_mx[OVS];  // Output min x

  private:
    // Functions that will be called by the constructor to get things set up
    void load_model();
    void load_base_field();
    void load_norm_data();

    // Pre-processing
    void subtract_base_field(double data[IPS][IPS]);
    void normalize(double data[IPS][IPS]);
    // Call the TorchScript model
    double* model_inference(double data[IPS][IPS]);
    // Post-processing
    void denormalize(double* data);

  public:
    NN_Model(std::string model_path);
    double* run_zernike_model(double input_pixels[IPS][IPS]);
    ~NN_Model(){};
};

#endif
