#include <c_wrapper.h>
#include <constants.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// Concat two strings (https://stackoverflow.com/a/8465083).
char* concat(const char* s1, const char* s2) {
    char* result = malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

// A function to pretty print the steps.
void print_step(const char* text) { printf("\n%s...\n", text); }

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        fprintf(stderr,
                "Usage: main <path to model folder> <benchmark row count>\n");
        return -1;
    }

    // =========================================================================
    // Load in the example input and output row of data.
    // =========================================================================
    print_step("Loading in the example input and output rows");
    const char* model_path = argv[1];
    char* data_path = concat(model_path, "/example_data/");
    char* input_row_path = concat(data_path, "input_line.txt");
    // Using what the Python model outputs as opposed to the truth output to
    // ensure that the results align correctly.
    char* output_row_path = concat(data_path, "python_output_line.txt");
    // Load the input values from the file (https://stackoverflow.com/a/7152018)
    FILE* text_file = fopen(input_row_path, "r");
    double inputs[IPS][IPS];
    for (int i = 0; i < IPS; i++)
        for (int j = 0; j < IPS; j++)
            fscanf(text_file, "%lf", &inputs[i][j]);
    fclose(text_file);
    // Load the output values from the file.
    text_file = fopen(output_row_path, "r");
    double python_output[OVS];
    for (int i = 0; i < OVS; i++)
        fscanf(text_file, "%lf", &python_output[i]);
    fclose(text_file);

    // =========================================================================
    // Load in the model.
    // =========================================================================
    print_step("Loading in the model");
    load_model(model_path);

    // =========================================================================
    // Call the model on the example data to verify the model is working.
    // =========================================================================
    print_step("Calling the model to verify its outputs");
    double* model_output = run_zernike_model(inputs);
    printf("Python Model Output, C++ Model Output\n");
    for (int i = 0; i < OVS; i++)
        printf("%.16f, %.16f\n", *(python_output + i), *(model_output + i));

    // =========================================================================
    // Run the model benchmark (one row at a time).
    // =========================================================================
    print_step("Benchmarking the model's performance");
    // Grab the number of iterations from the CLI command.
    int iterations = strtol(argv[2], NULL, 10);
    printf("Using %d iterations\n", iterations);
    // Grab the real time elapsed by running all the iterations
    // (https://stackoverflow.com/a/55346612).
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++)
        run_zernike_model(inputs);
    gettimeofday(&end, NULL);
    double total_time =
        end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6;
    printf("Average time per iteration: %f seconds\n", total_time / iterations);
}
