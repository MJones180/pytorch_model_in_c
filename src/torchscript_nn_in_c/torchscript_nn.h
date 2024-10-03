#ifndef TORCHSCRIPT_NN_H
#define TORCHSCRIPT_NN_H

#include <constants.h>
#include <torch/script.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

class NN_Model {
    std::string model_dir_path;
    torch::jit::script::Module nn_model_obj;
    // The base field that should be subtracted off
    float base_field[INPUT_PIXEL_SIZE][INPUT_PIXEL_SIZE];
    // There should be a single value for all inputs
    float input_max_min_diff;
    float input_min_x;
    // Each output should have its own value
    float output_max_min_diff[OUTPUT_PIXEL_SIZE];
    float output_min_x[OUTPUT_PIXEL_SIZE];

  private:
    void load_model();
    void load_base_field();
    void load_norm_data();

    float* call_model(float data[INPUT_PIXEL_SIZE][INPUT_PIXEL_SIZE]);
    void denorm(float* data);

  public:
    NN_Model(std::string model_path);
    float* run_model(float data[INPUT_PIXEL_SIZE][INPUT_PIXEL_SIZE]);
};

#endif
