#include <torchscript_nn.h>

void NN_Model::load_model() {
    std::string model_location = model_dir_path + "/model.pt";
    try {
        nn_model_obj = torch::jit::load(model_location);
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }
    std::cout << "Model loaded from: " << model_location << "\n";
}

void NN_Model::load_base_field() {
    std::string base_field_path = model_dir_path + "/base_field.txt";
    std::cout << "Opening the base field: " << base_field_path << "\n";
    std::ifstream source;
    source.open(base_field_path);
    if (!source) {
        std::cerr << "error loading in base field\n";
    }
    for (int i = 0; i < INPUT_PIXEL_SIZE; i++) {
        for (int j = 0; j < INPUT_PIXEL_SIZE; j++) {
            source >> base_field[i][j];
        }
    }
    source.close();
}

void NN_Model::load_norm_data() {
    std::string norm_path = model_dir_path + "/norm_data.txt";
    std::cout << "Opening the norm data: " << norm_path << "\n";
    std::ifstream source;
    source.open(norm_path);
    if (!source) {
        std::cerr << "error loading in norm data\n";
    }
    source >> input_max_min_diff;
    source >> input_min_x;
    for (int i = 0; i < OUTPUT_PIXEL_SIZE; i++) {
        source >> output_max_min_diff[i];
    }
    for (int i = 0; i < OUTPUT_PIXEL_SIZE; i++) {
        source >> output_min_x[i];
    }
    source.close();
}

NN_Model::NN_Model(std::string model_path) {
    model_dir_path = model_path;
    load_model();
    load_base_field();
    load_norm_data();
}

double* NN_Model::call_model(double data[INPUT_PIXEL_SIZE][INPUT_PIXEL_SIZE]) {

    // The converted TorchScript model expects floats instead of doubles
    float data_float[INPUT_PIXEL_SIZE][INPUT_PIXEL_SIZE];
    for (int i = 0; i < INPUT_PIXEL_SIZE; i++) {
        for (int j = 0; j < INPUT_PIXEL_SIZE; j++) {
            data_float[i][j] = (float)data[i][j];
        }
    }

    // The data coming in is assumed to be a batch size of one with one channel.
    // That means, it should just be a 2D array. However, to run through
    // TorchScript we will need to make it 4D (batch size, channels, *pixels).
    torch::Tensor inputs_tensor = torch::from_blob(
        data_float, {1, 1, INPUT_PIXEL_SIZE, INPUT_PIXEL_SIZE});

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(inputs_tensor);

    // Execute the model and turn its output into a tensor.
    at::Tensor model_output = nn_model_obj.forward(inputs).toTensor();

    // https://stackoverflow.com/a/36784891
    double* output_cpp = new double[OUTPUT_PIXEL_SIZE];
    for (int i = 0; i < OUTPUT_PIXEL_SIZE; i++) {
        output_cpp[i] = model_output[0][i].item<double>();
    }
    return output_cpp;
}

void NN_Model::subtract_base_field(
    double data[INPUT_PIXEL_SIZE][INPUT_PIXEL_SIZE]) {
    for (int i = 0; i < INPUT_PIXEL_SIZE; i++) {
        for (int j = 0; j < INPUT_PIXEL_SIZE; j++) {
            data[i][j] -= base_field[i][j];
        }
    }
}

void NN_Model::norm(double data[INPUT_PIXEL_SIZE][INPUT_PIXEL_SIZE]) {
    for (int i = 0; i < INPUT_PIXEL_SIZE; i++) {
        for (int j = 0; j < INPUT_PIXEL_SIZE; j++) {
            data[i][j] =
                2 * (data[i][j] - input_min_x) / input_max_min_diff - 1;
        }
    }
}

void NN_Model::denorm(double data[OUTPUT_PIXEL_SIZE]) {
    for (int i = 0; i < OUTPUT_PIXEL_SIZE; i++) {
        data[i] =
            (((data[i] + 1) / 2) * output_max_min_diff[i]) + output_min_x[i];
    }
}

double* NN_Model::run_model(double data[INPUT_PIXEL_SIZE][INPUT_PIXEL_SIZE]) {
    double data_copied[INPUT_PIXEL_SIZE][INPUT_PIXEL_SIZE];
    std::memcpy(data_copied, data,
                sizeof(double) * INPUT_PIXEL_SIZE * INPUT_PIXEL_SIZE);
    subtract_base_field(data_copied);
    norm(data_copied);
    double* result = call_model(data_copied);
    denorm(result);
    return result;
}
