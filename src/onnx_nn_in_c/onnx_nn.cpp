#include <onnx_nn.h>

void NN_Model::load_model(int core_count) {
    std::string file_path = model_dir_path + "/model.onnx";
    try {
        Ort::SessionOptions session_options;
        if (core_count == -1) {
            std::cout << "Using the integrated GPU\n";
            std::unordered_map<std::string, std::string> options;
            options["device_type"] = "GPU";
            options["precision"] = "FP32";
            session_options.AppendExecutionProvider("OpenVINO", options);
            // May potentially provide better performance for OpenVINO
            // https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#onnxruntime-graph-level-optimization
            session_options.SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_DISABLE_ALL);
        } else {
            std::cout << "Using " << core_count << " CPU cores\n";
            session_options.SetIntraOpNumThreads(core_count);
        }
        // https://github.com/microsoft/onnxruntime/issues/4131#issuecomment-682796289
        onnx_session = std::move(std::make_unique<Ort::Session>(
            env, file_path.c_str(), session_options));
    } catch (const std::exception& e) {
        std::cerr << "error loading in the model\n";
        exit(-1);
    }
    std::cout << "Model loaded from: " << file_path << "\n";
}

void NN_Model::load_base_field() {
    std::string file_path = model_dir_path + "/base_field.txt";
    std::ifstream fs;
    fs.open(file_path);
    if (!fs) {
        std::cerr << "error loading in the base field\n";
        exit(-1);
    }
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
    if (!fs) {
        std::cerr << "error loading in the norm data\n";
        exit(-1);
    }
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

NN_Model::NN_Model(std::string model_path, int core_count) {
    model_dir_path = model_path;
    load_model(core_count);
    load_base_field();
    load_norm_data();
}

float* NN_Model::model_inference(float data[IPS][IPS]) {
    // Code outline for doing this taken from the example located at:
    //     github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/model-explorer/model-explorer.cpp
    // The data needs to be converted to a flattened 1D vector of floats.
    std::vector<float> flat_float_data;
    for (int i = 0; i < IPS; i++)
        for (int j = 0; j < IPS; j++)
            flat_float_data.push_back((float)data[i][j]);
    // The data needs to be converted to the type Value
    std::vector<Ort::Value> input_tensor;
    input_tensor.emplace_back(Ort::Value::CreateTensor<float>(
        onnx_mem_info, flat_float_data.data(), flat_float_data.size(),
        onnx_input_shape.data(), onnx_input_shape.size()));
    try {
        return onnx_session
            ->Run(Ort::RunOptions{nullptr}, onnx_input_name.data(),
                  input_tensor.data(), onnx_input_name.size(),
                  onnx_output_name.data(), onnx_output_name.size())[0]
            .GetTensorMutableData<float>();
    } catch (const Ort::Exception& exception) {
        std::cerr << "error running the model: " << exception.what() << "\n";
        exit(-1);
    }
}

void NN_Model::subtract_base_field(float data[IPS][IPS]) {
    // Subtract the base field off of each of the pixels.
    for (int i = 0; i < IPS; i++)
        for (int j = 0; j < IPS; j++)
            data[i][j] -= base_field[i][j];
}

void NN_Model::normalize(float data[IPS][IPS]) {
    // Normalize the data between -1 and 1.
    for (int i = 0; i < IPS; i++)
        for (int j = 0; j < IPS; j++)
            data[i][j] = 2 * (data[i][j] - input_mx) / input_mmd - 1;
}

void NN_Model::denormalize(float data[OVS]) {
    // Denormalize the data from being between -1 and 1.
    for (int i = 0; i < OVS; i++)
        data[i] = (((data[i] + 1) / 2) * output_mmd[i]) + output_mx[i];
}

float* NN_Model::run_zernike_model(float input_pixels[IPS][IPS]) {
    // Create a copy of the data so that the original is not mutated.
    // This function only accepts one row of data at a time (hence being 2D).
    float data_copy[IPS][IPS];
    std::memcpy(data_copy, input_pixels, sizeof(float) * IPS2);

    // Only perform the input normalization if the input_mmd is not equal to 0
    if (input_mmd != 0) {
        // Pre-processing steps.
        subtract_base_field(data_copy); // Subtract off the base field
        normalize(data_copy);           // Normalize the data between -1 and 1
    }

    // Call the model.
    float* model_output = model_inference(data_copy);
    // Post-processing steps.
    denormalize(model_output); // Denormalize the data
    return model_output;
}
