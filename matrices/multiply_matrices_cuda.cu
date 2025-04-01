#include <stdio.h>
#include <cuda.h>
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
 * @param rows Number of rows in the first matrix
 * @param cols Number of columns in the second matrix
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
void process_arguments(int argc, char *argv[], int *rows, int *columns)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <rows> <columns>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    *rows = atoi(argv[1]);
    *columns = atoi(argv[2]);

    if (*rows <= 0 || *columns <= 0)
    {
        fprintf(stderr, "Error: Invalid matrix size.\n");
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[])
{
    int rows, columns;
    size_t size;
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    cudaEvent_t start, end;
    float ms;

    // Process command line arguments
    process_arguments(argc, argv, &rows, &columns);

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
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy matrices from CPU to GPU
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((columns + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Create CUDA events for timing
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Multiply matrices on GPU measuring time
    cudaEventRecord(start);
    mul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, rows, columns);
    cudaEventRecord(end);

    // Synchronize and calculate elapsed time
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);

    // Copy result matrix from GPU to CPU
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print elapsed time
    printf("Time (milliseconds): %.5f\n", ms);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}