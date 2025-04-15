#include "functions.cuh"

#define ONLY_KERNEL_TIME 1

/**
 * Main funtction
 * @brief Benchmark for CUDA matrix multiplication
 * @details This program benchmarks the performance of CUDA matrix multiplication
 * using different matrix sizes and thread configurations.
 * @return 0 on success, non-zero on failure
 */
int main()
{
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    cudaEvent_t start, end;
    clock_t main_start, main_end;
    float elapsed_time, ms;
    int matrix_sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 636, 774, 892, 1024, 1280, 2048, 4096, 8192};
    int threads_per_block[] = {2, 4, 8, 16, 32};
    int iterations = 32;

    main_start = clock();

    int sizes_count = sizeof(matrix_sizes) / sizeof(matrix_sizes[0]);
    int threads_count = sizeof(threads_per_block) / sizeof(threads_per_block[0]);

    printf("Matrix size;Threads per block;Time(ms);Time(s)\n");

    for (int i = 0; i < sizes_count; i++)
    {
        int matrix_size = matrix_sizes[i];
        init_matrices(matrix_size, &A, &B, &C);

        for (int j = 0; j < threads_count; j++)
        {    
            int block_size = threads_per_block[j];
            elapsed_time = 0.0f;

            // Define grid and block dimensions for kernel launch
            dim3 threadsPerBlock(block_size, block_size);
            dim3 numBlocks((matrix_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (matrix_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

            // Launch kernel k times
            for (int k = 0; k < iterations; k++)
            {
                // Create CUDA events for timing
                CudaEventCreate(&start);
                CudaEventCreate(&end);
                
                // Stuff to be timed
                if (ONLY_KERNEL_TIME)
                {
                    // Measure only kernel execution time
                    cuda_malloc_and_copy(&d_A, &d_B, &d_C, A, B, C, matrix_size);
                    CudaEventRecord(start);
                    mul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, matrix_size, matrix_size);
                    CudaEventRecord(end);
                    CudaEventSynchronize(end);
                    CudaEventElapsedTime(&ms, start, end);
                    CudaMemcpy(C, d_C, matrix_size * matrix_size * sizeof(int), cudaMemcpyDeviceToHost);
                }
                else
                {
                    // Measure total time including memory allocation and copying
                    CudaEventRecord(start);
                    cuda_malloc_and_copy(&d_A, &d_B, &d_C, A, B, C, matrix_size);
                    mul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, matrix_size, matrix_size);
                    CudaMemcpy(C, d_C, matrix_size * matrix_size * sizeof(int), cudaMemcpyDeviceToHost);
                    CudaEventRecord(end);
                    CudaEventSynchronize(end);
                    CudaEventElapsedTime(&ms, start, end);
                }

                // Get elapsed time
                elapsed_time += ms;
                cuda_free_matrices(d_A, d_B, d_C);

                // Destroy CUDA events
                CudaEventDestroy(start);
                CudaEventDestroy(end);
            }
            
            elapsed_time /= iterations;
            printf("%d;%d;%f;%f\n", matrix_size, block_size, elapsed_time, elapsed_time / 1000);
        }

        free_matrices(A, B, C);
    }

    main_end = clock();

    printf("\nTotal execution time: %f seconds", (double)(main_end - main_start) / CLOCKS_PER_SEC);
    printf("\nIterations: %d", iterations);
    if (ONLY_KERNEL_TIME)
        printf("\nExperiment WITHOUT taking into account CUDA memory allocation and matrices copying (host-device and device-host). Only kernel execution time.");
    else
        printf("\nExperiment taking into account CUDA memory allocation and matrices copying (host-device and device-host).");
    printf("\nBenchmark completed.");

    return 0;
}