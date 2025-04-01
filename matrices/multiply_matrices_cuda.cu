#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_functions.cuh"

/**
 * Allocate memory for a matrix of size rows x n
 * @param rows Number of rows
 * @param n Number of columns
 * @return Pointer to the allocated matrix
 */
int *allocate_matrix(int rows, int n)
{
    int *matrix = (int *)calloc(rows * n, sizeof(int));
    if (matrix == NULL)
    {
        fprintf(stderr, "Error: Could not allocate memory for the matrix.\n");
        return NULL;
    }
    return matrix;
}

/**
 * Generate a random matrix of size rows x n
 * @param rows Number of rows
 * @param n Number of columns
 * @param matrix Pointer to the matrix to be filled
 */
void generate_matrix(int rows, int n, int *matrix)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < n; j++)
        {
            matrix[i * n + j] = rand() % 10; // Generate numbers between 0 and 9
        }
    }
}

/**
 * Kernel function to multiply two matrices in GPU
 * @param A Pointer to the first matrix
 * @param B Pointer to the second matrix
 * @param C Pointer to the result matrix
 * @param rows Number of rows in matrices
 * @param cols Number of columns in matrices
 */
__global__ void mul(int *A, int *B, int *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        int sum = 0;
        for (int k = 0; k < cols; k++)
        {
            sum += A[row * cols + k] * B[k * cols + col];
        }
        C[row * cols + col] = sum;
    }
}

/**
 * Process command line arguments
 * @param argc Number of arguments
 * @param argv Array of arguments
 * @param rows Pointer to the number of rows
 * @param columns Pointer to the number of columns
 */
void process_arguments(int argc, char *argv[], int *rows, int *columns, int *threads)
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <rows> <columns> <threads_per_block\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    *rows = atoi(argv[1]);
    *columns = atoi(argv[2]);
    *threads = atoi(argv[3]);

    if (*rows <= 0 || *columns <= 0)
    {
        fprintf(stderr, "Error: Invalid matrix size.\n");
        exit(EXIT_FAILURE);
    }

    if (*threads <= 0 || *threads > 1024)
    {
        fprintf(stderr, "Error: Invalid number of threads per block.\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * Main function
 * @param argc Number of arguments
 * @param argv Array of arguments
 * @return 0 on success, non-zero on failure
 */
int main(int argc, char *argv[])
{
    int rows, columns, threads;
    size_t size;
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    cudaEvent_t start, end;
    float ms;

    // Process command line arguments
    process_arguments(argc, argv, &rows, &columns, &threads);

    // Allocate memory for matrices [CPU]
    A = allocate_matrix(rows, columns);
    B = allocate_matrix(rows, columns);
    C = allocate_matrix(rows, columns);
    if (A == NULL || B == NULL || C == NULL)
    {
        fprintf(stderr, "Error: Could not allocate memory for the matrices.\n");
        exit(EXIT_FAILURE);
    }

    // Generate random matrices
    srand(time(NULL));
    generate_matrix(rows, columns, A);
    generate_matrix(rows, columns, B);

    // Allocate memory for matrices [GPU]
    size = rows * columns * sizeof(int);
    CudaMalloc((void **)&d_A, size);
    CudaMalloc((void **)&d_B, size);
    CudaMalloc((void **)&d_C, size);

    // Copy matrices from CPU to GPU
    CudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    CudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((columns + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Create CUDA events for timing
    CudaEventCreate(&start);
    CudaEventCreate(&end);

    // Multiply matrices on GPU measuring time
    CudaEventRecord(start, 0);
    mul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, rows, columns);
    CudaEventRecord(end, 0);

    // Synchronize and calculate elapsed time
    CudaEventSynchronize(end);
    CudaEventElapsedTime(&ms, start, end);

    // Copy result matrix from GPU to CPU
    CudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print elapsed time
    printf("Time (milliseconds): %.5f\n", ms);

    // Free device memory
    CudaFree(d_A);
    CudaFree(d_B);
    CudaFree(d_C);
    free(A);
    free(B);
    free(C);

    // Destroy CUDA events
    CudaEventDestroy(start);
    CudaEventDestroy(end);

    return 0;
}