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
    double base_field[INPUT_PIXEL_SIZE][INPUT_PIXEL_SIZE];
    // There should be a single value for all inputs
    double input_max_min_diff;
    double input_min_x;
    // Each output should have its own value
    double output_max_min_diff[OUTPUT_PIXEL_SIZE];
    double output_min_x[OUTPUT_PIXEL_SIZE];

  private:
    void load_model();
    void load_base_field();
    void load_norm_data();

    double* call_model(double data[INPUT_PIXEL_SIZE][INPUT_PIXEL_SIZE]);
    void subtract_base_field(double data[INPUT_PIXEL_SIZE][INPUT_PIXEL_SIZE]);
    void norm(double data[INPUT_PIXEL_SIZE][INPUT_PIXEL_SIZE]);
    void denorm(double* data);

  public:
    NN_Model(std::string model_path);
    double* run_model(double data[INPUT_PIXEL_SIZE][INPUT_PIXEL_SIZE]);
};

#endif
