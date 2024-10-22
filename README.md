# PyTorch Model in C

I know having two completely separate folders is bad etiquette, but it is just a tad bit easier.

## TorchScript

A C wrapper around the C++ TorchScript library.
Runs python PyTorch models that were converted to TorchScript via the `torch.jit.trace` function.

## ONNX

A C wrapper around the C++ ONNX runtime library.
Runs python PyTorch models that were converted to ONNX via the `torch.onnx.export` function.

## Models

The models shared between the two folders.
Any other PyTorch model can be used, it will just need to be converted using the `export_model.py` script from the `uml_picture_d` repository.
