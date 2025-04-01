#include <time.h>

#include "cuda_functions.cuh"
#include "functions.h"
#include "gauss_jordan.h"

double* solve_equation_with_cpu(unsigned int size, double **mat, double* sol)
{
    unsigned int i;

    // Apply Gauss-Jordan elimination
    gauss_jordan(size, mat);

    // The solution is in the last column
    for (i = 0; i < size; i++)
        sol[i] = mat[i][size];

    return sol;
}

double* solve_equation_with_gpu(unsigned int size, double** mat, double* sol)
{
    unsigned int i;
    double** d_mat;
    double* d_sol;
    cudaEvent_t start, end;
    float ms;

    // Allocate memory on the GPU
    size_t total_size = size * size * sizeof(double*);
    CudaMalloc((void**) &d_mat, total_size);
    CudaMalloc((void**) &d_sol, size * sizeof(double));
    CudaMemcpy(d_mat, mat, total_size, cudaMemcpyHostToDevice);
    
    // TODO: Define dim3 grid dimensions
    // dim3 numThreads...
    // dim3 numBlocks...

    CudaEventCreate(&start);
    CudaEventCreate(&end);

    cudaEventRecord(start);
    // TODO: Call the kernel function

    CudaEventSynchronize(end);
    CudaEventRecord(end);
    CudaEventElapsedTime(&ms, start, end);

    CudaMemcpy(sol, d_sol, size * sizeof(double), cudaMemcpyDeviceToHost);

    CudaFree(d_mat);
    CudaFree(d_sol);
    CudaEventDestroy(start);
    CudaEventDestroy(end);

    for (i = 0; i < size ; i++)
        sol[i] = mat[i][size];

    return sol;
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

    log_message("Equation system:\n");
    // print_equation_system(size, mat);

    start = clock();
    sol = solve_equation_with_cpu(size, mat, sol);
    end = clock();
    seconds = (double)(end - start) / CLOCKS_PER_SEC;
    sprintf(buffer, "Execution time (seconds): %.5f\n", seconds);
    log_message(buffer);

    log_message("Resulting system:\n");
    // print_equation_system(size, mat);

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

    return 0;
}