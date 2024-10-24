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
};

#endif
