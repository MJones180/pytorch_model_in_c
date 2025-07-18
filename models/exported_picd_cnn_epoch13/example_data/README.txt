Data taken from the 11th row of the processed dataset:

    picd_instrument_data_25k_10nm_raw_processed

Code snippet to export the 11th row of the input and output to text files:

    # From inside of the processed dataset folder
    from h5py import File
    import numpy as np
    datafile = File('data.h5', 'r')
    np.savetxt('input_line.txt', datafile['inputs'][10][0], fmt='%.16f')
    np.savetxt('output_line.txt', datafile['outputs'][10], fmt='%.16f')

For this data, since it was preprocessed using the `preprocess_data_bare` script,
no normalization is done. Additionally, the base field has not been subtracted off.

The `python_output_line.txt` file is manually exported from the `model_test` script.
This can be done by adding the following two lines after the model output denormalization is done:

    np.savetxt('python_output_line.txt', outputs_model[10], fmt='%.16f')
    quit()

Then, of course, the command to run the `model_test` script is:

    python3 main.py model_test picd_cnn last picd_instrument_data_25k_10nm_raw_processed --inputs-need-diff --inputs-need-norm
