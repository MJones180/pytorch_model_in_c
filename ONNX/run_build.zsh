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

# Run a test to ensure everything works (use only 1 core)
# time taskset --cpu-list 1 ./main ../../models/ts_data_groups_approx_2_epoch285 1;