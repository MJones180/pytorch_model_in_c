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

double* run_zernike_model(double input_pixels[IPS][IPS]) {
    return NN_Model_instance->run_zernike_model(input_pixels);
}

#ifdef __cplusplus
}
#endif
