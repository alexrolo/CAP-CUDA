#include "functions.h"

float **allocate_matrix(int size)
{
    float **matrix = (float **)malloc(size * sizeof(float *));
    if (matrix == NULL)
    {
        fprintf(stderr, "Error: Unable to allocate memory for rows.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size; i++)
    {
        matrix[i] = (float *)malloc((size + 1) * sizeof(float)); // IMPORTANTE: N+1 columnas
        if (matrix[i] == NULL)
        {
            fprintf(stderr, "Error: Could not allocate memory for row %d.\n", i);
            exit(EXIT_FAILURE);
        }
    }

    return matrix;
}

void free_matrix(int m, float **matrix)
{
    for (int i = 0; i < m; i++)
        free(matrix[i]);
    free(matrix);
}

void generate_matrix(int size, float **matrix)
{
    for (int i = 0; i < size; i++)
        for (int j = 0; j <= size; j++)
            matrix[i][j] = (rand() % 10) + 1; // Generate numbers between 0 and 9
}

void print_equation_system(int size, float **mat)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        { // Not including the independent column
            printf("%5.1f x%d  ", mat[i][j], j);
            if (j < size - 1)
                printf("+ ");
        }
        printf(" = %5.1f\n", mat[i][size]); // Last column (independent term)
    }
    printf("\n");
}

int check_equation_system(int size, float **mat, float *sol)
{
    for (int i = 0; i < size; i++)
    {
        float sum = 0;
        for (int j = 0; j < size; j++)
            sum += mat[i][j] * sol[j];
        if (fabs(sum - mat[i][size]) > 1e-6) // Check if the absolute difference is small
            return 0; // Solution is incorrect
    }
    return 1; // Solution is correct
}
