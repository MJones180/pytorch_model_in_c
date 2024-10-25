# ONNX

## Installation

This runs an ONNX model using the ONNX Runtime library.
The library is used from within C++ (with a C wrapper), but can also be used directly from C.

First, the C/C++ `onnxruntime` library must be downloaded ( https://onnxruntime.ai/getting-started ).
Once this is downloaded, it must be unzipped and placed at `./libraries/onnxruntime`.

Next, CMake must be installed to build everything.
If CMake is not installed, then help for installing it can be found at ( https://askubuntu.com/a/1157132 ).

Building the code expects that ZSH is being used (not Bash).
To build, run `zsh run_build.zsh` from within the root.
This script contains the absolute path to the `onnxruntime` library.

# Inference

The executable can be run from the `build` directory (`cd build`).

The main script (`main`) can be run using the following signature:

    ./main ../../models/<model> <action>;

On some machines, the number of cores can be capped by adding `taskset` at the start of the command:

    # Uses only 2 cores
    taskset --cpu-list 1,2 ./main ../../models/<model> <action>;

Example commands can be found in the `example_commands.txt`.

## Model

The follow models are shipped with this repo for testing:
- `exported_data_groups_approx_2_epoch285`: slowest and most accurate.
- `exported_inference_speedup_v2_1_epoch325`: faster and a tiny bit less acurrate.
- `exported_inference_speedup_v1_2_epoch267`: fastest and a tiny bit more less acurrate.
Any other PyTorch model can be used, it will just need to be converted using the `export_model.py` script from the `uml_picture_d` repository.
Under the hood, this is using the `torch.onnx.export` function to convert the PyTorch model to an ONNX model.

For this code, it is required that the model only takes one input array and one output array.
The names and sizes of these arrays can be found in the `constants.h` file.

### Normalization

For the normalization data, there should be a single value for all inputs and each output should have its own value.
Additionally, the model should have been trained on data that was normalized between -1 and 1.

## Timing

Results are for 5,000 iterations of the models shipped with this repo.
All timings were taken using an AMD Ryzen 9 7950X CPU (no GPU).
Additionally, all times reported are in milliseconds.

- exported_data_groups_approx_2_epoch285: 0.24
- exported_inference_speedup_v2_1_epoch325: 0.18

Note: total logical cores = physical cores * threads per core.
Note: for some reason, using the `taskset --cpu-list ...` does not seem to work on all machines for the ONNX Runtime library.
