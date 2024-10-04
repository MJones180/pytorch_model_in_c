Data taken from the first row of:

    random_10nm_med_processed

Code snippet to export the first row of the input and output to text files:

    # From inside of the processed dataset folder
    from h5py import File
    import numpy as np
    datafile = File('data.h5', 'r')
    np.savetxt('input_line.txt', datafile['inputs'][10][0], fmt='%.16f')
    np.savetxt('output_line.txt', datafile['outputs'][10], fmt='%.16f')

For this data, since it was preprocessed using the `preprocess_data_bare` script,
no normalization is done. Additionally, the base field has not been subtracted off.

The `python_output_line.txt` file is manually exported from the `model_test` script.
