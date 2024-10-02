#!/usr/bin/env zsh
CMAKE_PREFIX_PATH=${PWD}/libraries/libtorch/share/cmake/Torch;
rm -rf build;
mkdir build;
cd build;
echo "Script running in: ${PWD}"
echo "Cmake Prefix: ${CMAKE_PREFIX_PATH}"
echo "*\n!.gitignore" > .gitignore;
# Absolute path to the ./libraries/libtorch folder
cmake -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH} ..;
cmake --build . --config Release;
# Run a test to ensure everything works (use only 1 core)
time taskset --cpu-list 1 ./main ../models/data_groups_approx_v2_2_epoch238;
