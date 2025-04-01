#include "gauss_jordan.h"

int maximum_element_in_column(unsigned int size, unsigned int column, double **mat)
{
    unsigned int max_row = 0, current_row = 0;
    double max_value = 0;
    for (current_row = column; current_row < size; current_row++)
    {
        double current_value = *(*(mat + current_row) + column);
        if (current_value < 0)
            current_value = -current_value; // Take absolute value for comparison
        printf("Checking row %d, column %d: value = %f (max is %f)\n", current_row, column, current_value, max_value);
        if (current_value > max_value)
        {
            max_row = current_row;
            max_value = current_value;
            printf("New max found at row %d: %f\n", max_row, max_value);
        }
    }

    printf("Maximum element in column %d is %f at row %d\n", column, max_value, max_row);
    return max_row;
}

void gauss_jordan(unsigned int size, double **mat)
{
    unsigned int current_column = 0;
    for (current_column = 0; current_column < size; current_column++)
    {
        // Find maximum in current column for partial pivoting
        unsigned int max_row = maximum_element_in_column(size, current_column, mat);

        // Swap rows
        if (max_row != current_column)
        {
            double *temp = mat[current_column];
            mat[current_column] = mat[max_row];
            mat[max_row] = temp;
            printf("Swapping rows %d and %d\n", current_column, max_row);
            print_equation_system(size, mat);
        } else {
            printf("No need to swap rows %d and %d\n", current_column, max_row);
        }

        // Make the diagonal element equal to 1
        double divisor = *(*(mat + current_column) + current_column);
        printf("Divisor for row %d is %f\n", current_column, divisor);
        for (unsigned int i = 0 ; i <= size ; i++)
            *(*(mat + current_column) + i) /= divisor; // Normalize the pivot row
        print_equation_system(size, mat);

        // Make zeros in the current column
        for (unsigned int i = 0; i < size; i++)
        {
            if (i != current_column)
            {
                double multiplier = *(*(mat + i) + current_column) / *(*(mat + current_column) + current_column);
                printf("Multiplier for row %d is %f\n", i, multiplier);
                for (unsigned int j = 0; j <= size; j++)
                    *(*(mat + i) + j) -= multiplier * *(*(mat + current_column) + j); // Eliminate the current column
                print_equation_system(size, mat);
            }
        }
    }
}