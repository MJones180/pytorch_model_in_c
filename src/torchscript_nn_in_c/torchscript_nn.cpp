#include <torchscript_nn.h>

NN_Model::NN_Model(std::string filename) {
    model_filename = filename;
    try {
        nn_model_obj = torch::jit::load(filename);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }

    std::cout << "Model loaded from: " << filename << "\n";

    // Should load in normalization data here
}

float* NN_Model::run_model(float data[1][1][32][32]) {
    // Create a vector of inputs.
    torch::Tensor inputs_tensor = torch::from_blob(data, {1,1,32,32});

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(inputs_tensor);

    // Execute the model and turn its output into a tensor.
    at::Tensor model_output = nn_model_obj.forward(inputs).toTensor();

    // https://stackoverflow.com/a/36784891
    float* output_cpp = new float[23];
    // https://www.simonwenkel.com/notes/software_libraries/pytorch/data_transfer_to_and_from_pytorch.html
    std::memcpy(output_cpp, model_output.data_ptr(), sizeof(float)*model_output.numel());
    return output_cpp;
}
