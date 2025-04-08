#include <time.h>

#include "functions.h"

void gauss_jordan(unsigned int size, double **mat)
{
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
            if (current_value > max_value)
            {
                max_row = current_row;
                max_value = current_value;
            }
        }

        // Swap rows
        if (max_row != current_column)
        {
            double *temp = mat[current_column];
            mat[current_column] = mat[max_row];
            mat[max_row] = temp;
        }

        // Make the diagonal element equal to 1
        double divisor = *(*(mat + current_column) + current_column);
        for (unsigned int i = 0 ; i <= size ; i++)
            *(*(mat + current_column) + i) /= divisor; // Normalize the pivot row

        // Make zeros in the current column
        for (unsigned int i = 0; i < size; i++)
        {
            if (i != current_column)
            {
                double multiplier = *(*(mat + i) + current_column) / *(*(mat + current_column) + current_column);
                for (unsigned int j = 0; j <= size; j++)
                    *(*(mat + i) + j) -= multiplier * *(*(mat + current_column) + j); // Eliminate the current column
            }
        }
    }
}

int main(int argc, char *argv[])
{
    unsigned int size; // Square matrix
    double **mat, *sol;
    clock_t start, end;
    double seconds;
    char* buffer = (char*)malloc(128);

    // Check if the required arguments are provided
    if (argc != 2)
    {
        sprintf(buffer, "Usage: %s <size>\n", argv[0]);
        log_message(buffer);
        return 1;
    }

    if ((size = atoi(argv[1])) < 1)
    {
        log_message("Error: size must be greater than 0.\n");
        return 1;
    }

    // Allocate memory
    mat = allocate_matrix(size);
    sol = (double *)malloc(size * sizeof(double));

    // Generate random matrix
    srand(time(NULL));
    generate_matrix(size, mat);

    unsigned int i;

    start = clock();
    gauss_jordan(size, mat);
    end = clock();

    // The solution is in the last column
    for (i = 0; i < size; i++)
        sol[i] = mat[i][size];

    seconds = (double)(end - start) / CLOCKS_PER_SEC;
    sprintf(buffer, "Execution time (seconds): %.5f\n", seconds);
    log_message(buffer);

    // The solution is in the last column
    log_message("System solution:\n");
    for (unsigned int i = 0; i < size; i++)
    {
        sprintf(buffer, "x%d = %.3f\n", i, mat[i][size]);
        log_message(buffer);
        sol[i] = mat[i][size];
    }

    // Check if the solution is correct
    if (check_equation_system(size, mat, sol))
        log_message("The solution is correct.\n");
    else
        log_message("The solution is incorrect.\n");

    // Free matrix
    free_matrix(size, mat);
    free(sol);
    free(buffer);

    return 0;
}