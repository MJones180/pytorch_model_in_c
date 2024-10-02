#include <c_wrapper.h>
#include <constants.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// https://stackoverflow.com/a/8465083
char* concat(const char* s1, const char* s2) {
    char* result = malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        fprintf(stderr,
                "Usage: main <path to model folder> <benchmark row count>\n");
        return -1;
    }

    const char* model_path = argv[1];

    printf("Loading in the example input and output rows\n");
    char* data_path = concat(model_path, "/example_data/");
    char* input_row_path = concat(data_path, "input_line.txt");
    char* output_row_path = concat(data_path, "output_line.txt");

    // https://stackoverflow.com/a/7152018
    FILE* text_file = fopen(input_row_path, "r");
    float inputs[INPUT_PIXEL_SIZE][INPUT_PIXEL_SIZE];
    double double_value;
    for (int i = 0; i < INPUT_PIXEL_SIZE; i++) {
        for (int j = 0; j < INPUT_PIXEL_SIZE; j++) {
            fscanf(text_file, "%lf", &double_value);
            inputs[i][j] = double_value;
        }
    }
    fclose(text_file);

    text_file = fopen(output_row_path, "r");
    float truth_output[OUTPUT_PIXEL_SIZE];
    for (int i = 0; i < OUTPUT_PIXEL_SIZE; i++) {
        fscanf(text_file, "%lf", &double_value);
        truth_output[i] = double_value;
    }
    fclose(text_file);

    printf("Calling the function to load in the model\n");
    load_model(model_path);

    printf("Calling the model to verify its outputs\n");
    float* model_output = run_model(inputs);
    printf("Truth Outputs, Model Outputs\n");
    for (int i = 0; i < OUTPUT_PIXEL_SIZE; i++) {
        printf("%f, %f\n", *(truth_output + i), *(model_output + i));
    }

    int iterations = strtol(argv[2], NULL, 10);
    printf("Running %d iterations to test speed\n", iterations);
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
        run_model(inputs);
    }
    gettimeofday(&end, NULL);
    // https://stackoverflow.com/a/55346612
    double total_time =
        end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6;
    printf("Average time per iteration: %f seconds\n", total_time / iterations);

    // Proper output:
    //  1.0341
    //  0.66958
    //  1.031
    // -0.83508
    //  0.53434
    //  0.39728
    // -0.9179
    // -0.48178
    //  0.21093
    // -0.00046546
    //  0.22018
    //  0.94071
    // -0.68948
    //  0.17338
    //  0.17838
    //  0.26740
    // -0.59733
    // -0.12139
    //  0.18144
    //  0.47307
    // -0.016768
    //  0.35732
    //  0.23316
}
