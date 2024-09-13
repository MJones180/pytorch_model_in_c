#include <cstdlib>

#include "c_wrapper.h"
#include "torchscript_nn.h"

#ifdef __cplusplus
extern "C" {
#endif

static NN_Model *NN_Model_instance = NULL;

void load_model(const char *filename) {
    NN_Model_instance = new NN_Model(filename);
}

float* run_model(float data[1][1][32][32]) {
    return NN_Model_instance->run_model(data);
}

#ifdef __cplusplus
}
#endif
