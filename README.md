# PyTorch Model in C

I know having two completely separate folders is bad etiquette, but it is just a tad bit easier.
The reason for this is TorchScript requires C++ while ONNX is just in C.

## TorchScript

A C wrapper around the C++ TorchScript library.
Used to run python PyTorch models that were converted to TorchScript via the `torch.jit.trace` function.

## ONNX

C code that uses the ONNX runtime library.
Used to run python PyTorch models that were converted to ONNX via the `torch.onnx.export` function.
