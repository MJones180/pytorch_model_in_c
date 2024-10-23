#ifndef CONSTANT_H
#define CONSTANT_H

// Number of pixels for both X and Y on the camera (must be square)
const int IPS = 32;
const int IPS2 = 32 * 32;
// Number of values being outputted from the model (Zernike terms)
const int OVS = 23;

// Name of the input and output fields for the ONNX model
const char* const NN_INPUT_NAME = "input";
const char* const NN_OUTPUT_NAME = "output";

#endif
