#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>

/**
 * Function to allocate a matrix of size rows x n
 * Will return NULL if the allocation fails
 * 
 * @param rows Number of rows
 * @param n Number of columns
 * 
 * @return Pointer to the allocated matrix
 */
int *allocate_matrix(int rows, int n)
{
    // We use calloc to allocate and initialize the memory to zero.
    int *matrix = (int *)calloc(rows * n, sizeof(int));
    if (matrix == NULL)
    {
        fprintf(stderr, "Error: Could not allocate memory for the matrix.\n");
        return NULL;
    }

    return matrix;
}

/**
 * Function to generate a random matrix of size rows x n
 * 
 * @param rows Number of rows
 * @param n Number of columns
 * @param matrix Pointer to the matrix to be filled
 */
void generate_matrix(int rows, int n, int *matrix)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < n; j++)
            matrix[i * n + j] = rand() % 10; // Generate numbers between 0 and 9
}

/**
 * Function to safely allocate memory for a matrix on the GPU
 * Will exit if the allocation fails
 * 
 * @param rows Number of rows
 * @param columns Number of columns
 * @param matrix Pointer to the matrix to be allocated
 */
void safe_cuda_alloc(unsigned int rows, unsigned int columns, unsigned int*matrix)
{
    // Allocate memory for the matrix in GPU
    size_t size = rows * columns * sizeof(int);
    cudaError_t err = cudaMalloc((void **)&matrix, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Could not allocate memory for the matrix on GPU.\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * Check for CUDA errors
 * Will exit if an error is found
 */
void check_cuda_error()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * Kernel function to multiply two matrices
 * 
 * @param A Pointer to the first matrix
 * @param B Pointer to the second matrix
 * @param C Pointer to the result matrix
 * @param rows Number of rows in the matrices
 * @param columns Number of columns in the matrices
 */
__global__ void mul(unsigned int *A, unsigned int *B, unsigned int *C, unsigned int rows, unsigned int cols)
{
    double tmp = 0.0;
    unsigned int current_row = 0, current_column = 0, x = 0;
    unsigned int a = 0, b = 0;
    for (current_row = 0 ; current_row < rows ; current_row++) 
    {
		for (current_column = 0 ; current_column < columns ; current_column++)
        {
			tmp = 0.0;
			for (x = 0 ; x < rows ; x++)
            {
                a = *(*(A+current_row)+x); 
                b = *(*(B+x)+current_column);
                tmp += a * b;
			}
            *(*(C+current_row)+current_column) = tmp;
		}	 
    }

    return;
}

int main(int argc, char *argv[])
{
    unsigned int rows, columns;              // Matrix size (rows x columns), and block size for z-order
    size_t size;                   // Matrix size in bytes
    unsigned int *A, *B, *C;                // CPU matrices
    unsigned int *d_A, *d_B, *d_C;          // GPU matrices
    cudaEvent_t start, end;        // To measure time
    double ms;                     // Time in ms

    CudaEventCreate(&start);
    CudaEventCreate(&end);
    check_cuda_error();

    // Check if the required arguments are provided
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s <rows> <columns>\n", argv[0]);
        return 1;
    }
    // Read the matrix size from command line arguments
    rows = atoi(argv[1]);
    columns = atoi(argv[2]);

    // Allocate memory for matrices in CPU
    A = allocate_matrix(rows, columns);
    B = allocate_matrix(rows, columns);
    C = allocate_matrix(rows, columns);

    // Generate random matrices
    srand(time(NULL)); // Initialize the random number generator once
    generate_matrix(rows, columns, A);
    generate_matrix(rows, columns, B);

    // Allocate memory for matrices in GPU
    size = rows * columns * sizeof(int);
    safe_cuda_alloc(rows, columns, d_A);
    safe_cuda_alloc(rows, columns, d_B);
    safe_cuda_alloc(rows, columns, d_C);
    check_cuda_error();

    // Copy matrices from the CPU to the GPU
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    check_cuda_error();

    // Launch kernels multiplication and measure times
    mul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, rows, columns, block_size);
    check_cuda_error();

    // Copy C from GPU to CPU
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    check_cuda_error();
    printf("Time (seconds): %.5f\n", ms / 1000);

    // Free the memory of the matrices in CPU and GPU
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    check_cuda_error();

    // Destroy time events
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    check_cuda_error();

    return 0;
}