# ONNX

## Installation

This runs an ONNX model using the ONNX Runtime C library.

First, the C `onnxruntime` library must be downloaded ( https://onnxruntime.ai/getting-started ).
Once this is downloaded, it must be unzipped and placed at `./libraries/onnxruntime`.

Next, CMake must be installed to build everything.
If CMake is not installed, then help for installing it can be found at ( https://askubuntu.com/a/1157132 ).

Building the code expects that ZSH is being used (not Bash).
To build, run `zsh run_build.zsh` from within the root.
This script contains the absolute path to the `onnxruntime` library.

# Inference

The executable can be run by using the following two commands:

    cd build;
    # Uses only 1 core
    taskset --cpu-list 1 ./main ../../models/exported_data_groups_approx_2_epoch285 5000;

Note: if multiple cores can be used, then it should be `taskset --cpu-list 1,2,...`.
The commands above are also run by the `run_build.zsh` script to ensure everything is working properly.

## Model

The `exported_data_groups_approx_2_epoch285` model is shipped with this repo for testing.
Any other PyTorch model can be used, it will just need to be converted using the `export_model.py` script from the `uml_picture_d` repository.
Under the hood, this is using the `torch.onnx.export` function to convert the PyTorch model to an ONNX model.

### Normalization

For the normalization data, there should be a single value for all inputs and each output should have its own value.
Additionally, the model should have been trained on data that was normalized between -1 and 1.

## Timing

Results are for 5,000 iterations of the model shipped with this repo (`exported_data_groups_approx_2_epoch285`).
All timings were taken using an AMD Ryzen 9 7950X CPU (no GPU).
Additionally, all times reported are in milliseconds.

...

Note: total logical cores = physical cores * threads per core.
