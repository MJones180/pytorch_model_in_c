#include <torchscript_nn.h>

void NN_Model::load_model() {
    std::string file_path = model_dir_path + "/model.pt";
    try {
        nn_model_obj = torch::jit::load(file_path);
    } catch (const c10::Error& e) {
        std::cerr << "error loading in the model\n";
    }
    std::cout << "Model loaded from: " << file_path << "\n";
}

void NN_Model::load_base_field() {
    std::string file_path = model_dir_path + "/base_field.txt";
    std::ifstream fs;
    fs.open(file_path);
    if (!fs)
        std::cerr << "error loading in the base field\n";
    std::cout << "Base field loaded from: " << file_path << "\n";
    for (int i = 0; i < IPS; i++)
        for (int j = 0; j < IPS; j++)
            fs >> base_field[i][j];
    fs.close();
}

void NN_Model::load_norm_data() {
    std::string file_path = model_dir_path + "/norm_data.txt";
    std::ifstream fs;
    fs.open(file_path);
    if (!fs)
        std::cerr << "error loading in the norm data\n";
    std::cout << "Norm data loaded from: " << file_path << "\n";
    // First two lines are input norm data.
    fs >> input_mmd;
    fs >> input_mx;
    // Last two lines are output norm data.
    for (int i = 0; i < OVS; i++)
        fs >> output_mmd[i];
    for (int i = 0; i < OVS; i++)
        fs >> output_mx[i];
    fs.close();
}

NN_Model::NN_Model(std::string model_path) {
    model_dir_path = model_path;
    load_model();
    load_base_field();
    load_norm_data();
}

double* NN_Model::model_inference(double data[IPS][IPS]) {
    // The TorchScript model expects all data to be floats instead of doubles.
    float data_as_float[IPS][IPS];
    for (int i = 0; i < IPS; i++)
        for (int j = 0; j < IPS; j++)
            data_as_float[i][j] = (float)data[i][j];
    // The data being passed to this function is a single row with one channel.
    // However, the model expects a 4D array of the shape:
    //  (batch size, channels, input pixels, input pixels)
    // That means, we need to add two empty dimensions at the start.
    torch::Tensor inputs_tensor =
        torch::from_blob(data_as_float, {1, 1, IPS, IPS});
    // The model expects a vector, not a Tensor, so we need to convert it.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(inputs_tensor);
    // Run the model and convert its output into a tensor.
    at::Tensor model_output = nn_model_obj.forward(inputs).toTensor();
    // We cannot allocate the array locally, instead we need to dynamically
    // allocate the array memory. (https://stackoverflow.com/a/36784891)
    double* output_cpp = new double[OVS];
    // The values should be returned as type double, not float.
    for (int i = 0; i < OVS; i++)
        output_cpp[i] = model_output[0][i].item<double>();
    return output_cpp;
}

void NN_Model::subtract_base_field(double data[IPS][IPS]) {
    // Subtract the base field off of each of the pixels.
    for (int i = 0; i < IPS; i++)
        for (int j = 0; j < IPS; j++)
            data[i][j] -= base_field[i][j];
}

void NN_Model::normalize(double data[IPS][IPS]) {
    // Normalize the data between -1 and 1.
    for (int i = 0; i < IPS; i++)
        for (int j = 0; j < IPS; j++)
            data[i][j] = 2 * (data[i][j] - input_mx) / input_mmd - 1;
}

void NN_Model::denormalize(double data[OVS]) {
    // Denormalize the data from being between -1 and 1.
    for (int i = 0; i < OVS; i++)
        data[i] = (((data[i] + 1) / 2) * output_mmd[i]) + output_mx[i];
}

double* NN_Model::run_zernike_model(double input_pixels[IPS][IPS]) {
    // Create a copy of the data so that the original is not mutated.
    // This function only accepts one row of data at a time (hence being 2D).
    double data_copy[IPS][IPS];
    std::memcpy(data_copy, input_pixels, sizeof(double) * IPS * IPS);
    // Pre-processing steps.
    subtract_base_field(data_copy); // Subtract off the base field
    normalize(data_copy);           // Normalize the data between -1 and 1
    // Call the model.
    double* model_output = model_inference(data_copy);
    // Post-processing steps.
    denormalize(model_output); // Denormalize the data
    return model_output;
}
