#include "functions.h"

double *allocate_matrix(unsigned int size)
{
    double *matrix = (double *)malloc(size * size * sizeof(double *));
    if (matrix == NULL)
    {
        fprintf(stderr, "Error: Unable to allocate memory for rows.\n");
        exit(EXIT_FAILURE);
    }

    return matrix;
}

void generate_matrix(unsigned int size, double *matrix)
{
    const unsigned int total_size = size * size;
    for (unsigned int i = 0; i < total_size; i++)
        *(matrix + i) = (rand() % 10) + 1; // Generate numbers between 0 and 9
}

void print_equation_system(unsigned int size, double *matrix)
{
    const unsigned int total_size = size * size;
    for (unsigned int i = 0; i < total_size; i++)
    {
        printf("%5.4f x%d ", *(matrix + i), i % size);
        if (i % size < size - 1)
            printf("+ ");
        if (i % size == size - 1)
            printf("\n");
    }
    printf("\n");
}

int check_equation_system(unsigned int size, double *matrix, double *solution)
{
    const unsigned int total_size = size * size;
    double sum = 0;
    for (unsigned int i = 0; i < total_size; i++)
    {
        // Reset sum for each row
        if (i % size == 0)
            sum = 0;

        // Calculate the sum of the products
        sum += *(matrix + i) * *(solution + size + (i % size) * size);

        // Check if the absolute difference is small
        if (fabs(sum - *(matrix + i * size + size)) > 1e-6)
            return 0; // Solution is incorrect
    }
    return 1; // Solution is correct
}

void copy_matrix(unsigned int size, double *src, double *dest)
{
    dest = allocate_matrix(size);
    for (unsigned int i = 0; i < size; i++)
        *(dest+i) = *(src+i);
}
