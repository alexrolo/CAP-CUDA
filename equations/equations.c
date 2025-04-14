#include <time.h>

#include "gauss_jordan.h"
#include "functions.h"

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