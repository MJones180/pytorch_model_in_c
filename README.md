# ONNX

## Installation

This runs an ONNX model using the ONNX Runtime library.
The library is used from within C++ (with a C wrapper), but can also be used directly from C.

CMake must be installed to build everything.
If CMake is not installed, then help for installing it can be found at ( https://askubuntu.com/a/1157132 ).

ONNX Runtime (and potentially OpenVINO) must be installed:
- **CPU Only**: The C/C++ `onnxruntime` library must be downloaded ( https://onnxruntime.ai/getting-started ).  Once this is downloaded, it must be unzipped and placed at `./libraries/onnxruntime`.
- **CPU and iGPU**: For the ability to use the integrated GPU on an Intel chip, use the following steps taken from ( https://onnxruntime.ai/docs/build/eps.html#openvino ):
    1. Install OpenVINO Release 2024.3 using APT. A link for 2024.6 instructions can be found at ( https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?PACKAGE=OPENVINO_BASE&VERSION=v_2024_6_0&OP_SYSTEM=LINUX&DISTRIBUTION=APT ).
    2. Intel GPU library downloads. ( https://github.com/intel/compute-runtime/releases/tag/24.35.30872.22 )
    3. The `setupvars.sh` step can be ignored since using apt to install OpenVINO.
    4. Download and unzip the ONNX Runtime (v5.4) source code (tar.gz) to `./libraries/onnxruntime`. ( https://github.com/intel/onnxruntime/releases/tag/v5.4 )
    5. Inside of the unzipped ONNX folder, run: `./build.sh --config RelWithDebInfo --use_openvino GPU --build_shared_lib --build_wheel`

After ONNX Runtime is installed, the correct ONNX Runtime statements need to be uncommented in `CMakeLists.txt`.
*Note, the iGPU instructions work for Ubuntu 22.04 with an Intel Xeon CPU E3-1505L v6 @ 2.20GHz (Intel HD Graphics P630 GPU).*

Building the code for this repo expects that ZSH is being used (not Bash).
To build, run `zsh run_build.zsh` from within the root.

# Inference

The executable can be run from the `build` directory (`cd build`).

The main script (`main`) can be run using the following signature:

    ./main ../models/<model> <core count> <action>;

If a value of `0` is passed for the `core count`, then `ONNX Runtime` decides how many cores to use.
If a value of `-1` is passed for the `core count`, then the integrated GPU is used.

On some machines, the number of cores can be capped by adding `taskset` at the start of the command:

    # Uses only 2 cores
    taskset --cpu-list 1,2 ./main ../models/<model> <core count> <action>;

Example commands can be found in the `example_commands.txt`.

## Model

Many models are shipped with this repo, including `exported_weighted_aberration_ranges_local_v4_epoch15`.

Any other PyTorch model can be used, it will just need to be converted using the `export_model.py` script from the `uml_picture_d` repository.
Under the hood, this is using the `torch.onnx.export` function to convert the PyTorch model to an ONNX model.

For this code, it is required that the model only takes one input array and one output array.
The names and sizes of these arrays can be found in the `constants.h` file.

### Normalization

For the normalization data, there should be a single value for all inputs and each output should have its own value.
Additionally, the model should have been trained on data that was normalized between -1 and 1.

## Timing

- Total logical cores = physical cores * threads per core.
- For some reason, using the `taskset --cpu-list ...` does not seem to work on all machines.

## Notes

- This repo originally used TorchScript, but ONNX Runtime was switched to since it is faster and is still being actively developed.
- Many models are shipped with this repo (primarily to make inference speed testing easier). That means, this repo is sadly bulkier than it needs to be.
