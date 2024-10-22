#include <onnx_nn.h>

void NN_Model::load_model() {
    std::string file_path = model_dir_path + "/model.onnx";
    try {

        Ort::SessionOptions session_options{nullptr};
        auto envLocal =
            std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");
        env = std::move(envLocal);
        auto sessionLocal = std::make_unique<Ort::Session>(
            *env, file_path.c_str(), session_options);
        session = std::move(sessionLocal);

    } catch (const std::exception& e) {
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
    // Code for doing this taken from
    // https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/model-explorer/model-explorer.cpp

    // print name/shape of inputs
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_names(
        {session->GetInputNameAllocated(0, allocator).get()});
    std::vector<std::int64_t> input_shapes(
        {session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()});

    // print name/shape of outputs
    std::vector<std::string> output_names(
        {session->GetOutputNameAllocated(0, allocator).get()});

    float data_as_float_flattened[IPS * IPS];
    for (int i = 0; i < IPS; i++)
        for (int j = 0; j < IPS; j++)
            data_as_float_flattened[i * IPS + j] = (float)data[i][j];

    std::vector<float> input_tensor_values(std::begin(data_as_float_flattened),
                                           std::end(data_as_float_flattened));
    std::vector<Ort::Value> input_tensors;

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        mem_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shapes.data(), input_shapes.size()));

    // pass data through model
    std::vector<const char*> input_names_char(input_names.size(), nullptr);
    std::transform(std::begin(input_names), std::end(input_names),
                   std::begin(input_names_char),
                   [&](const std::string& str) { return str.c_str(); });

    std::vector<const char*> output_names_char(output_names.size(), nullptr);
    std::transform(std::begin(output_names), std::end(output_names),
                   std::begin(output_names_char),
                   [&](const std::string& str) { return str.c_str(); });

    try {
        std::vector<Ort::Value> output_tensors =
            session->Run(Ort::RunOptions{nullptr}, input_names_char.data(),
                         input_tensors.data(), input_names_char.size(),
                         output_names_char.data(), output_names_char.size());

        // float* arr = output_tensors.front().GetTensorMutableData<float>();
        float* arr = output_tensors[0].GetTensorMutableData<float>();

        // We cannot allocate the array locally, instead we need to dynamically
        // allocate the array memory. (https://stackoverflow.com/a/36784891)
        double* output_cpp = new double[OVS];
        // The values should be returned as type double, not float.
        for (int i = 0; i < OVS; i++)
            output_cpp[i] = arr[i];
        return output_cpp;

    } catch (const Ort::Exception& exception) {
        std::cout << "ERROR running model inference: " << exception.what()
                  << std::endl;
        exit(-1);
    }
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
