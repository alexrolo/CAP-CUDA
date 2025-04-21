#include "gauss_jordan_cuda.cuh"

__global__ void gauss_jordan_no_swap(unsigned int size, double *matrix)
{
    unsigned int const current_column = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int actual_columns = size + 1;
    const double divisor = *(matrix + current_column * actual_columns + current_column);
    for (unsigned int i = 0 ; i < actual_columns ; i++)
        *(matrix + current_column * actual_columns + i) /= divisor;

    for (unsigned int i = 0; i < size; i++)
        if (i != current_column)
        {
            const double multiplier = *(matrix + i * actual_columns + current_column)*(-1);
            for (unsigned int j = 0 ; j < actual_columns ; j++)
            {
                const double newValue = (
                        *(matrix + i * actual_columns + j) +
                        *(matrix + current_column * actual_columns + j)
                        * multiplier);
                *(matrix + i * actual_columns + j) = newValue;
            }
        }
}

__global__ void gauss_jordan(unsigned int size, double *matrix)
{
    // unsigned int const current_column = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int actual_columns = size + 1;
    
    // if (current_column >= size)
        // return; // Out of bounds

    for (unsigned int current_column = 0 ; current_column < size ; current_column++){
    // Find maximum in current column for partial pivoting
    unsigned int max_row = 0, current_row = 0;
    double max_value = 0;
    for (current_row = current_column; current_row < size; current_row++)
    {
        double current_value = *(matrix + current_row * actual_columns + current_column);
        if (current_value < 0)
            current_value = -current_value; // Take absolute value for comparison
        if (current_value > max_value)
        {
            max_row = current_row;
            max_value = current_value;
        }
    }

    // Swap rows
    if (max_row != current_column)
    {
        for (unsigned int i = 0; i < actual_columns; i++)
        {
            unsigned int current_idx = current_column * actual_columns + i;
            unsigned int max_idx = max_row * actual_columns + i;
            double temp = *(matrix + current_idx);
            *(matrix + current_idx) = *(matrix + max_idx);
            *(matrix + max_idx) = temp;
        }
    }

    // Make the diagonal element equal to 1
    double divisor = *(matrix + current_column * actual_columns + current_column);
    for (unsigned int i = 0; i < actual_columns; i++)
        *(matrix + current_column * actual_columns + i) /= divisor; // Normalize the pivot row

    // Make zeros in the current column
    for (unsigned int i = 0; i < size; i++)
    {
        if (i != current_column)
        {
            double multiplier = *(matrix + i * actual_columns + current_column) / *(matrix + current_column * actual_columns + current_column);
            for (unsigned int j = 0; j <= size; j++)
                *(matrix + i * actual_columns + j) -= multiplier * *(matrix + current_column * actual_columns + j); // Eliminate the current column
        }
    }
}
}

