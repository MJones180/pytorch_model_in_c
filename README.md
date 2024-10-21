# PyTorch Model in C

## Installation

This repository runs a C++ TorchScript model in C.

First, the C++ `libtorch` library must be installed.
Instructions for doing this can be found at ( https://pytorch.org/cppdocs/installing.html ).
Once this is downloaded, it must be placed at `./libraries/libtorch`.

Next, CMake must be installed to build everything.
If CMake is not installed, then help for installing it can be found at ( https://askubuntu.com/a/1157132 ).

The overview of how to run C++ code from within a C wrapper was taken from ( https://stackoverflow.com/a/29192858 ).
Additionally, the basic code for running the model in C++ was taken from ( https://pytorch.org/tutorials/advanced/cpp_export.html#a-minimal-c-application ).

Building the code expects that ZSH is being used (not Bash).
To build, run `zsh run_build.zsh` from within the root.
This script contains the absolute path to the `libtorch` library.

# Inference

The executable can be run by using the following two commands:

    cd build;
    # Uses only 1 core
    taskset --cpu-list 1 ./main ../models/ts_data_groups_approx_2_epoch285 5000;

Note: if multiple cores can be used, then it should be `taskset --cpu-list 1,2,...`.
The commands above are also run by the `run_build.zsh` script to ensure everything is working properly.

## Model

The `ts_data_groups_approx_2_epoch285` model is shipped with this repository for testing.
Any other PyTorch model can be used, it will just need to be converted using the `export_model.py` script from the `uml_picture_d` repository.
Under the hood, this is using the `torch.jit.trace` function to convert the PyTorch model to a TorchScript model.

### Normalization

For the normalization data, there should be a single value for all inputs and each output should have its own value.
Additionally, the model should have been trained on data that was normalized between -1 and 1.

## Timing

Results are for 5,000 iterations of the model shipped with this repo (`ts_data_groups_approx_2_epoch285`).
All timings were taken using an AMD Ryzen 9 7950X CPU (no GPU).
Additionally, all times reported are in milliseconds.

- Python:
    - 1 logical core: 2.431
    - 32 logical cores: 1.111
- C:
    - 1 logical core: 2.1 to 2.3
    - 2 logical cores – separate physical cores: 1.3 to 1.5
    - 32 local cores: ~1

Note: total logical cores = physical cores * threads per core.
