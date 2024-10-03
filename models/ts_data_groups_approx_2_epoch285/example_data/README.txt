Data taken from the first row of:

    val_fixed_2000_and_random_group_ranges_approx

Code snippet to export the first row of the input and output to text files:

    # From inside of the processed dataset folder
    from h5py import File
    import numpy as np
    datafile = File('data.h5', 'r')
    input_line = datafile['inputs'][0][0]
    output_line = datafile['outputs'][0]
    np.savetxt('input_line.txt', input_line, fmt='%.8f')
    np.savetxt('output_line.txt', output_line, fmt='%.8f')
