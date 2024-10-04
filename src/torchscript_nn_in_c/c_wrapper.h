#ifndef C_WRAPPER_H
#define C_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <constants.h>

void load_model(const char* model_path);
double* run_model(double data[INPUT_PIXEL_SIZE][INPUT_PIXEL_SIZE]);

#ifdef __cplusplus
}
#endif

#endif
