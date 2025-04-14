#include "functions.h"

double *allocate_matrix(unsigned int size)
{
    double *matrix = (double *)malloc(size * (size + 1) * sizeof(double));
    if (matrix == NULL)
    {
        fprintf(stderr, "Error: Unable to allocate memory for rows.\n");
        exit(EXIT_FAILURE);
    }

    return matrix;
}

void generate_matrix(unsigned int size, double *matrix)
{
    const unsigned int total_size = size * (size + 1);
    for (unsigned int i = 0; i < total_size; i++)
        *(matrix + i) = (rand() % 10) + 1; // Generate numbers between 0 and 9
}

void print_equation_system(unsigned int size, double *matrix)
{
    const unsigned int actual_columns = size + 1;
    const unsigned int total_size = size * actual_columns;
    for (unsigned int i = 0; i < total_size; i++)
    {
        if (i % actual_columns + 1 == actual_columns)
            printf("= %5.4f\n", *(matrix + i));
        else
            printf("+ %5.4f x%d ", *(matrix + i), i % actual_columns);
    }
    printf("\n");
}

int check_equation_system(unsigned int size, double *matrix, double *solution)
{
    const unsigned int actual_columns = size + 1;
    const unsigned int total_size = size * actual_columns;
    double sum = 0;
    for (unsigned int i = 0; i < total_size; i++)
    {
        if (i % actual_columns != actual_columns - 1)
        {
            // Reset sum for each row
            if (i % actual_columns == 0)
                sum = 0;

            if (i % actual_columns == actual_columns - 1)
            {
                // Check if the absolute difference is small
                if (fabs(sum - *(matrix + i * actual_columns + size)) > 1e-6)
                    return 0; // Solution is incorrect
            }
            // Calculate the sum of the products
            else
            {
                const double value = *(matrix + i) * *(solution + i % actual_columns);
                sum += value;
            }
        }
    }
    return 1; // Solution is correct
}

double *copy_matrix(unsigned int size, double *src, double *dest)
{
    dest = allocate_matrix(size);
    const unsigned int total_size = size * (size + 1);
    for (unsigned int i = 0; i < total_size; i++)
        *(dest + i) = *(src + i);
    return dest;
}
