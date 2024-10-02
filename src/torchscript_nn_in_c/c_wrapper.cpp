#include <cstdlib>

#include "c_wrapper.h"
#include "torchscript_nn.h"

#ifdef __cplusplus
extern "C" {
#endif

static NN_Model* NN_Model_instance = NULL;

void load_model(const char* model_path) {
    NN_Model_instance = new NN_Model(model_path);
}

float* run_model(float data[INPUT_PIXEL_SIZE][INPUT_PIXEL_SIZE]) {
    return NN_Model_instance->run_model(data);
}

#ifdef __cplusplus
}
#endif
