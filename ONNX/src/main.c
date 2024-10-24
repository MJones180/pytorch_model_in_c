#include <c_wrapper.h>
#include <constants.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/time.h>

// Actions that can be done in this script
static const char VALIDATE_OUTPUTS[] = "validate_outputs";
static const char BENCHMARK_1[] = "benchmark_1";
static const char BENCHMARK_2[] = "benchmark_2";

// Concat two strings (https://stackoverflow.com/a/8465083).
char* concat(const char* s1, const char* s2) {
    int size = sizeof(char) * (strlen(s1) + strlen(s2) + 1);
    char* concat_str = malloc(size);
    snprintf(concat_str, size, "%s%s", s1, s2);
    return concat_str;
}

// A function to pretty print the steps.
void print_step(const char* text) { printf("\n%s...\n", text); }

// I will remember this function name and the weird true/false is flipped
int str_match(const char* str_1, const char* str_2) {
    return !strcasecmp(str_1, str_2);
}

int verify_args(int arg_count, const char* args[]) {
    // Every command will start with this
    char* base_str = "Usage: main <path to model folder>";
    // The following is the usage message that is shown if no action or an
    // incorrect action is given
    char usage_str[256];
    snprintf(usage_str, sizeof(usage_str),
             "%s <action> ...\nValid actions: %s, %s, %s\n", base_str,
             VALIDATE_OUTPUTS, BENCHMARK_1, BENCHMARK_2);
    // The minimum number of arguments that must be passed (model path and
    // action to perform)
    int min_arg_count = 3;
    // Verify the model path and action are passed
    if (arg_count < min_arg_count) {
        fprintf(stderr, "%s", usage_str);
        return -1;
    }
    const char* action = args[2];
    if (str_match(action, VALIDATE_OUTPUTS)) {
        if (arg_count != min_arg_count) {
            fprintf(stderr, "%s %s\n", base_str, VALIDATE_OUTPUTS);
            return -1;
        }
    } else if (str_match(action, BENCHMARK_1)) {
        if (arg_count != min_arg_count + 1) {
            fprintf(stderr, "%s %s <iterations>\n", base_str, BENCHMARK_1);
            return -1;
        }
    } else if (str_match(action, BENCHMARK_2)) {
        if (arg_count != min_arg_count + 2) {
            fprintf(stderr, "%s %s <calling frequency (Hz)> <total time>\n",
                    base_str, BENCHMARK_2);
            return -1;
        }
    } else {
        fprintf(stderr, "%s", usage_str);
        return -1;
    }
    return 1;
}

int main(int argc, const char* argv[]) {
    if (verify_args(argc, argv) == -1)
        return -1;
    const char* model_path = argv[1];
    const char* action = argv[2];
    printf("Model Path: %s\n", model_path);
    printf("Action: %s\n", action);

    // =========================================================================
    // Load in the example input and output row of data.
    // =========================================================================
    print_step("Loading in the example input and output rows");
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

    if (str_match(action, VALIDATE_OUTPUTS)) {
        // =========================================================================
        // Call the model on the example data to verify the model is working.
        // =========================================================================
        print_step("Calling the model to verify its outputs");
        double* model_output = run_zernike_model(inputs);
        printf("Python Model Output, C++ Model Output\n");
        for (int i = 0; i < OVS; i++)
            printf("%.16f, %.16f\n", *(python_output + i), *(model_output + i));
    }

    if (str_match(action, BENCHMARK_1)) {
        // =========================================================================
        // Run the model benchmark (one row at a time).
        // =========================================================================
        print_step("Benchmarking the model's performance");
        // Grab the number of iterations from the CLI command.
        int iterations = strtol(argv[3], NULL, 10);
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
        printf("Average time per iteration: %f seconds\n",
               total_time / iterations);
    }

    if (str_match(action, BENCHMARK_2)) {
        // =========================================================================
        // Run the model benchmark (one row at a time).
        // =========================================================================
        print_step("Benchmarking the model's performance");
        // Grab the number of iterations from the CLI command.
        int iterations = strtol(argv[3], NULL, 10);
        int calling_frequency = strtol(argv[4], NULL, 10);
        float calling_period = (float)1 / calling_frequency;
        printf("Using %d iterations\n", iterations);
        printf("Using %d frequency (Hz)\n", calling_frequency);
        printf("Using %f period (s)\n", calling_period);
        // Grab the real time elapsed by running all the iterations
        // (https://stackoverflow.com/a/55346612).
        struct timeval start, end;
        gettimeofday(&start, NULL);
        for (int i = 0; i < iterations; i++)
            run_zernike_model(inputs);
        gettimeofday(&end, NULL);
        double total_time =
            end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6;
        printf("Average time per iteration: %f seconds\n",
               total_time / iterations);
    }

    // =========================================================================
    // Close the model
    // =========================================================================
    close_model();
    print_step("Model closed");
}
