#include <iostream>
#include <vector>

#include "functions.h"
#include "functions_cuda.cuh"
#include "gauss_jordan_cuda.cuh"

double *solve_equation_with_gpu(unsigned int size, unsigned int threads, double *matrix, double *sol)
{
    double *d_matrix;
    cudaEvent_t start, end;
    float ms;

    // Allocate memory on the GPU
    size_t total_size = size * (size + 1) * sizeof(double);
    CudaMalloc((void **)&d_matrix, total_size);
    CudaMemcpy(d_matrix, matrix, total_size, cudaMemcpyHostToDevice);

    // Define dim3 grid dimensions
    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    CudaEventCreate(&start);
    CudaEventCreate(&end);

    CudaEventRecord(start);
    // TODO: Call the kernel function
    gauss_jordan<<<numBlocks, threadsPerBlock>>>(size, d_matrix);

    CudaEventRecord(end);
    CudaEventSynchronize(end);
    CudaEventElapsedTime(&ms, start, end);

    CudaFree(d_matrix);
    CudaEventDestroy(start);
    CudaEventDestroy(end);

    for (unsigned int i = 0; i < size; i++)
        sol[i] = *(matrix + i * (size + 1) + size);

    return sol;
}

int main(int argc, char **argv)
{
    std::vector<unsigned int> sizes = {
        2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    const unsigned int iterations = 32;

    double *matrix, *original_matrix = NULL, *sol, *d_matrix;
    clock_t start, end;
    double seconds;
    unsigned int total_iterations = 0;

    bool was_valid = false;

    // Generate random matrix
    srand(time(NULL));

    std::cout << "SIZE;ARCH;THRS;TIME;SUCC" << std::endl;
    for (auto size : sizes)
    {
        const unsigned int half_size = size / 2;
        for (unsigned int threads = 1; threads <= half_size; threads++)
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
                    solve_equation_with_gpu(size, threads, matrix, sol);
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
            std::cout << size << ";" << "GPU" << ";" << threads << ";" << seconds / iterations << ";" << success_rate << std::endl;
        }
    }
}