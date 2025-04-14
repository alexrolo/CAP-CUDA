#include <time.h>

#include "functions.h"

void gauss_jordan(unsigned int size, double *matrix)
{
    unsigned int current_column = 0;
    const unsigned int actual_columns = size + 1;
    for (current_column = 0; current_column < size; current_column++)
    {
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

int main(int argc, char *argv[])
{
    unsigned int size; // Square matrix
    double *matrix, *original_matrix = NULL, *sol;
    clock_t start, end;
    double seconds;

    // Check if the required arguments are provided
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <size>\n", argv[0]);
        return 1;
    }

    if ((size = atoi(argv[1])) < 1)
    {
        fprintf(stderr, "Invalid size: %s\n", argv[1]);
        return 1;
    }

    // Allocate memory
    matrix = allocate_matrix(size);
    sol = (double *)malloc(size * sizeof(double));

    // Generate random matrix
    srand(time(NULL));
    generate_matrix(size, matrix);
    original_matrix = copy_matrix(size, matrix, original_matrix);

    unsigned int i;

    start = clock();
    gauss_jordan(size, matrix);
    end = clock();

    // The solution is in the last column
    for (i = 0; i < size; i++)
        sol[i] = *(matrix + i * (size + 1) + size);

    seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Execution time (seconds): %.5f\n", seconds);

    // The solution is in the last column
    printf("System solution:\n");
    for (i = 0; i < size; i++)
        printf("x%d = %.3f\n", i, sol[i]);

    printf("Checking against original matrix:\n");
    print_equation_system(size, original_matrix);

    // Check if the solution is correct
    if (check_equation_system(size, original_matrix, sol))
        printf("The solution is correct.\n");
    else
        printf("The solution is incorrect.\n");

    // Free matrix
    free(matrix);
    free(original_matrix);
    free(sol);

    return 0;
}