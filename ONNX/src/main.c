#include <c_wrapper.h>
#include <constants.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/time.h>
#include <time.h>

// Actions that can be done in this script
static const char VALIDATE_OUTPUTS[] = "validate_outputs";
static const char BENCHMARK_ITER_COUNT[] = "benchmark_iter_count";
static const char BENCHMARK_FREQ[] = "benchmark_freq";

// Concat two strings (https://stackoverflow.com/a/8465083).
char* concat(const char* s1, const char* s2) {
    int size = sizeof(char) * (strlen(s1) + strlen(s2) + 1);
    char* concat_str = malloc(size);
    snprintf(concat_str, size, "%s%s", s1, s2);
    return concat_str;
}

// Grab the current time (https://stackoverflow.com/a/55346612)
double get_current_time() {
    struct timeval current_time_val;
    gettimeofday(&current_time_val, NULL);
    return current_time_val.tv_sec + current_time_val.tv_usec / 1e6;
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
             VALIDATE_OUTPUTS, BENCHMARK_ITER_COUNT, BENCHMARK_FREQ);
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
    } else if (str_match(action, BENCHMARK_ITER_COUNT)) {
        if (arg_count != min_arg_count + 1) {
            fprintf(stderr, "%s %s <iterations>\n", base_str,
                    BENCHMARK_ITER_COUNT);
            return -1;
        }
    } else if (str_match(action, BENCHMARK_FREQ)) {
        if (arg_count != min_arg_count + 2) {
            fprintf(stderr, "%s %s <total time (s)> <frequency (Hz)>\n",
                    base_str, BENCHMARK_FREQ);
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
        printf("PyTorch Model Output, C ONNX Model Output\n");
        for (int i = 0; i < OVS; i++)
            printf("%.16f, %.16f\n", *(python_output + i), *(model_output + i));
    }

    if (str_match(action, BENCHMARK_ITER_COUNT)) {
        // =========================================================================
        // Run the model benchmark (one row at a time).
        // =========================================================================
        print_step("Benchmarking the model's performance");
        // Grab the number of iterations from the CLI command.
        int iterations = strtol(argv[3], NULL, 10);
        printf("Using %d iterations\n", iterations);
        double start = get_current_time();
        for (int i = 0; i < iterations; i++)
            run_zernike_model(inputs);
        double average_time = (get_current_time() - start) / iterations;
        printf("Average time per iteration: %f seconds\n", average_time);
    }

    if (str_match(action, BENCHMARK_FREQ)) {
        // =========================================================================
        // Run the model benchmark (at a given frequency).
        // =========================================================================
        print_step("Benchmarking the model's performance");
        // Grab the number of iterations from the CLI command.
        int total_time = strtol(argv[3], NULL, 10);
        int frequency = strtol(argv[4], NULL, 10);
        double period = (double)1 / frequency;
        printf("Total Time (s): %i\n", total_time);
        printf("Frequency (Hz): %d\n", frequency);
        printf("Period (s): %f\n", period);
        double start_time = get_current_time();
        double end_time = get_current_time() + total_time;
        double current_time = get_current_time();
        int total_model_calls = 0;
        double total_compute_time = 0;
        double total_sleep_time = 0;
        while (current_time < end_time) {
            double compute_start = get_current_time();
            run_zernike_model(inputs);
            total_model_calls += 1;
            double compute_time = get_current_time() - compute_start;
            total_compute_time += compute_time;
            double sleep_time = period - compute_time;
            total_sleep_time += sleep_time;
            // https://stackoverflow.com/a/7684399
            nanosleep((const struct timespec[]){{0, sleep_time * 1e9}}, NULL);
            current_time = get_current_time();
        }
        double actual_total_time = get_current_time() - start_time;
        printf("Total actual time: %f\n", actual_total_time);
        printf("Total time accounted for: %f\n",
               total_compute_time + total_sleep_time);
        printf("Total model calls: %d\n", total_model_calls);
        printf("Total model call time: %f\n", total_compute_time);
        printf("Total sleep time: %f\n", total_sleep_time);
        printf("Average time per iteration: %f seconds\n",
               total_compute_time / total_model_calls);
    }

    // =========================================================================
    // Close the model
    // =========================================================================
    print_step("Closing the model");
    close_model();
}
