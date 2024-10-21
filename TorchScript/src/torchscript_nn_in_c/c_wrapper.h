#ifndef C_WRAPPER_H
#define C_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <constants.h>

void load_model(const char* model_path);
double* run_zernike_model(double input_pixels[IPS][IPS]);

#ifdef __cplusplus
}
#endif

#endif
