#ifndef C_WRAPPER_H
#define C_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <constants.h>

void load_model(const char* model_path, int core_count);
float* run_zernike_model(float input_pixels[IPS][IPS]);
void close_model();

#ifdef __cplusplus
}
#endif

#endif
