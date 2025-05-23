#include <iostream>
#include <vector>

#include "functions.h"
#include "functions_cuda.cuh"
#include "gauss_jordan_cuda.cuh"

int main(int argc, char **argv)
{
    std::vector<unsigned int> sizes = {
	250, 500, 750, 1000, 1250, 1500, 1750, 2000,
	2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000};
    const unsigned int iterations = 32;

    double *matrix, *original_matrix = NULL, *sol;
    clock_t start, end;
    double seconds;
    unsigned int total_iterations = 0;

    bool was_valid = false;

    // Generate random matrix
    srand(time(NULL));

    std::cout << "SIZE;ARCH;TIME;SUCC" << std::endl;
    for (auto size : sizes)
    {
	seconds = 0;
	total_iterations = 0;
	for (unsigned int iteration = 0; iteration < iterations; iteration++)
	{
	    was_valid = false;
    	    while (!was_valid)
	    {
		total_iterations++;
		// Allocate memory
		matrix = allocate_matrix(size);
		sol = (double *)malloc(size * sizeof(double));
		generate_matrix(size, matrix);
		original_matrix = copy_matrix(size, matrix, original_matrix);

		start = clock();
		gauss_jordan(size, matrix);
		end = clock();

		// The solution is in the last column
		for (unsigned int i = 0; i < size; i++)
		    sol[i] = *(matrix + i * (size + 1) + size);

		seconds += (double)(end - start) / CLOCKS_PER_SEC;

		// Check if the solution is correct
		was_valid = check_equation_system(size, matrix, sol);

		// Free matrix
		free(matrix);
		free(original_matrix);
		free(sol);
	    }
	}
	const double success_rate = (double)iterations / total_iterations;
	std::cout << size << ";" << "GPU" << ";" << seconds / iterations << ";" << success_rate << std::endl;
    }
}
