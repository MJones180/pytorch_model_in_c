#!/usr/bin/env zsh

# Absolute path to the onnxruntime folder
ONNXRUNTIME_ROOTDIR=${PWD}/libraries/onnxruntime;

rm -rf build;
mkdir build;
cd build;
echo "*\n!.gitignore" > .gitignore;

echo "";
echo "Script running in: ${PWD}";
echo "onnxruntime Path: ${ONNXRUNTIME_ROOTDIR}";
echo "";

# Configure the project
cmake -DONNXRUNTIME_ROOTDIR=${ONNXRUNTIME_ROOTDIR} ..;
# Build the project
cmake --build .;

# Run a test to ensure everything works
./main ../models/exported_weighted_aberration_ranges_local_v4_epoch15 1 VALIDATE_OUTPUTS;
