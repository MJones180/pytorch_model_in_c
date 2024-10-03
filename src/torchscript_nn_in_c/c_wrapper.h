#ifndef C_WRAPPER_H
#define C_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <constants.h>

void load_model(const char* model_path);
float* run_model(float* data);

#ifdef __cplusplus
}
#endif

#endif
