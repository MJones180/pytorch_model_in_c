#ifndef C_WRAPPER_H
#define C_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

void load_model(const char *filename);
float* run_model(float data[1][1][32][32]);

#ifdef __cplusplus
}
#endif

#endif
