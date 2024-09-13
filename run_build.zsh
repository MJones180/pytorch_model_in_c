rm -rf build;
mkdir build;
cd build;
echo "*\n!.gitignore" > .gitignore;
# Absolute path to the ./libraries/libtorch folder
cmake -DCMAKE_PREFIX_PATH=/home/michael-jones/Documents/pytorch_model_in_c/libraries/libtorch ..;
cmake --build . --config Release;
