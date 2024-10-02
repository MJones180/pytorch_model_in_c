#ifndef TORCHSCRIPT_NN_H
#define TORCHSCRIPT_NN_H

#include <constants.h>
#include <torch/script.h>

#include <iostream>
#include <memory>
#include <string>

class NN_Model {
    std::string model_location;
    torch::jit::script::Module nn_model_obj;

  public:
    NN_Model(std::string model_path);
    float* run_model(float data[INPUT_PIXEL_SIZE][INPUT_PIXEL_SIZE]);
};

#endif
