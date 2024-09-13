# PyTorch Model in C

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

The executable can be run by using the following two commands:

    cd build;
    ./main ../data_groups_approx_v2_2_epoch238.pt;

The `../data_groups_approx_v2_2_epoch238.pt` is shipped with this repository for testing.
Any other PyTorch model can be used, it will just need to be converted using the `create_torchscript_model.py` script from the `uml_picture_d` repository.
