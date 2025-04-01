#include "gauss_jordan.h"

void gauss_jordan(unsigned int size, double **mat)
{
    char* buffer = (char*)malloc(128);

    unsigned int current_column = 0;
    for (current_column = 0; current_column < size; current_column++)
    {
        // Find maximum in current column for partial pivoting
        unsigned int max_row = 0, current_row = 0;
        double max_value = 0;
        for (current_row = current_column; current_row < size; current_row++)
        {
            double current_value = *(*(mat + current_row) + current_column);
            if (current_value < 0)
                current_value = -current_value; // Take absolute value for comparison
            sprintf(buffer, "Checking row %d, column %d: value = %f (max is %f)\n", current_row, current_column, current_value, max_value);
            log_message(buffer);
            if (current_value > max_value)
            {
                max_row = current_row;
                max_value = current_value;
                sprintf(buffer, "New max found at row %d: %f\n", max_row, max_value);
                log_message(buffer);
            }
        }

        sprintf(buffer, "Maximum element in column %d is %f at row %d\n", current_column, max_value, max_row);
        log_message(buffer);

        // Swap rows
        if (max_row != current_column)
        {
            double *temp = mat[current_column];
            mat[current_column] = mat[max_row];
            mat[max_row] = temp;
            sprintf(buffer, "Swapping rows %d and %d\n", current_column, max_row);
            log_message(buffer);
            // print_equation_system(size, mat);
        } else {
            sprintf(buffer, "No need to swap rows %d and %d\n", current_column, max_row);
            log_message(buffer);
        }

        // Make the diagonal element equal to 1
        double divisor = *(*(mat + current_column) + current_column);
        sprintf(buffer, "Divisor for row %d is %f\n", current_column, divisor);
        log_message(buffer);
        for (unsigned int i = 0 ; i <= size ; i++)
            *(*(mat + current_column) + i) /= divisor; // Normalize the pivot row
        // print_equation_system(size, mat);

        // Make zeros in the current column
        for (unsigned int i = 0; i < size; i++)
        {
            if (i != current_column)
            {
                double multiplier = *(*(mat + i) + current_column) / *(*(mat + current_column) + current_column);
                sprintf(buffer, "Multiplier for row %d is %f\n", i, multiplier);
                log_message(buffer);
                for (unsigned int j = 0; j <= size; j++)
                    *(*(mat + i) + j) -= multiplier * *(*(mat + current_column) + j); // Eliminate the current column
                // print_equation_system(size, mat);
            }
        }
    }

    free(buffer);
}