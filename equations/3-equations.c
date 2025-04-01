#include <time.h>

#include "functions.h"
#include "gauss_jordan.h"

int main(int argc, char *argv[])
{
    unsigned int size; // Square matrix
    double **mat, *sol;
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
    sol = (double *)malloc(size * sizeof(double));

    printf("Equation system:\n");
    print_equation_system(size, mat);

    // Apply Gauss-Jordan CPU
    start = clock();
    gauss_jordan(size, mat);
    end = clock();
    seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Execution time (seconds): %.5f\n", seconds);

    printf("Resulting system:\n");
    print_equation_system(size, mat);

    // The solution is in the last column
    printf("System solution:\n");
    for (unsigned int i = 0; i < size; i++)
    {
        printf("x%d = %.3f\n", i, mat[i][size]);
        sol[i] = mat[i][size];
    }

    // Check if the solution is correct
    if (check_equation_system(size, mat, sol))
        printf("The solution is correct.\n");
    else
        printf("The solution is incorrect.\n");

    // Free matrix
    free_matrix(size, mat);
    free(sol);

    return 0;
}