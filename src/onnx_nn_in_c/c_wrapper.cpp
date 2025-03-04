#include <cstdlib>

#include "c_wrapper.h"
#include "onnx_nn.h"

#ifdef __cplusplus
extern "C" {
#endif

static NN_Model* NN_Model_instance = NULL;

void load_model(const char* model_path, int core_count) {
    NN_Model_instance = new NN_Model(model_path, core_count);
}

float* run_zernike_model(float input_pixels[IPS][IPS]) {
    return NN_Model_instance->run_zernike_model(input_pixels);
}

void close_model() { delete NN_Model_instance; }

#ifdef __cplusplus
}
#endif
