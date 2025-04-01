#include <time.h>

#include "functions.h"

/**
 * Function to perform Gauss-Jordan elimination
 * 
 * @param size The size of the matrix
 * @param mat The matrix to be transformed
 */
void gaussJordan(int size, float **mat)
{
    unsigned int current_column = 0;
    for (current_column = 0; current_column < size; current_column++)
    {
        // Find maximum in current column for partial pivoting


        // Swap rows

        // Make the diagonal element equal to 1

        // Make zeros in the current column

    }
}

int main(int argc, char *argv[])
{
    unsigned int size; // Square matrix
    float **mat;
    clock_t start, end;
    double seconds;

    // Check if the required arguments are provided
    if (argc != 2)
    {
        printf("Usage: %s <size>\n", argv[0]);
        return 1;
    }

    if ((size = atoi(argv[1])) < 1)
    {
        printf("Error: size must be greater than 0.\n");
        return 1;
    }

    // Allocate memory
    mat = allocate_matrix(size);

    // Generate random matrix
    srand(time(NULL));
    generate_matrix(size, mat);

    printf("Equation system:\n");
    print_equation_system(size, mat);

    // Apply Gauss-Jordan CPU
    start = clock();
    gaussJordan(size, mat);
    end = clock();
    seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Execution time (seconds): %.5f\n", seconds);

    // The solution is in the last column
    printf("System solution:\n");
    for (unsigned int i = 0; i < size; i++)
    {
        printf("x%d = %.3f\n", i, mat[i][size]);
    }

    // Free matrix
    free_matrix(size, mat);

    return 0;
}