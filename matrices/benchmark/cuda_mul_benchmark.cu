#include "functions.cuh"

/**
 * Allocates memory for all matrices and initializes them
 * @param matrix_size Size of the matrices
 * @param A Pointer to the first matrix
 * @param B Pointer to the second matrix
 * @param C Pointer to the result matrix
 */
void init_matrices(int matrix_size, int **A, int **B, int **C)
{
    *A = allocate_matrix(matrix_size, matrix_size);
    *B = allocate_matrix(matrix_size, matrix_size);
    *C = allocate_matrix(matrix_size, matrix_size);

    if (*A == NULL || *B == NULL || *C == NULL)
    {
        fprintf(stderr, "Error: Could not allocate memory for the matrices.\n");
        exit(EXIT_FAILURE);
    }

    generate_matrix(matrix_size, matrix_size, *A);
    generate_matrix(matrix_size, matrix_size, *B);
    fill_matrix(matrix_size, matrix_size, *C, 0);
}

/**
 * Allocates memory on the device and copies matrices from host to device
 * @param d_A Pointer to the first matrix on device
 * @param d_B Pointer to the second matrix on device
 * @param d_C Pointer to the result matrix on device
 * @param A Pointer to the first matrix on host
 * @param B Pointer to the second matrix on host
 * @param C Pointer to the result matrix on host
 * @param matrix_size Size of the matrices
 */
void cuda_malloc_and_copy(int **d_A, int **d_B, int **d_C, int *A, int *B, int *C, int matrix_size)
{
    CudaMalloc((void **)d_A, matrix_size * matrix_size * sizeof(int));
    CudaMalloc((void **)d_B, matrix_size * matrix_size * sizeof(int));
    CudaMalloc((void **)d_C, matrix_size * matrix_size * sizeof(int));

    CudaMemcpy(*d_A, A, matrix_size * matrix_size * sizeof(int), cudaMemcpyHostToDevice);
    CudaMemcpy(*d_B, B, matrix_size * matrix_size * sizeof(int), cudaMemcpyHostToDevice);

    if (*d_A == NULL || *d_B == NULL || *d_C == NULL)
    {
        fprintf(stderr, "Error: Could not allocate memory on the device.\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * Free device memory for matrices
 * @param d_A Pointer to the first matrix on device
 * @param d_B Pointer to the second matrix on device
 * @param d_C Pointer to the result matrix on device
 */
void cuda_free_matrices(int *d_A, int *d_B, int *d_C)
{
    CudaFree(d_A);
    CudaFree(d_B);
    CudaFree(d_C);
}

/**
 * Free host memory for matrices
 * @param A Pointer to the first matrix on host
 * @param B Pointer to the second matrix on host
 * @param C Pointer to the result matrix on host
 */
void free_matrices(int *A, int *B, int *C)
{
    free(A);
    free(B);
    free(C);
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
 * Main funtction
 * @return 0 on success, non-zero on failure
 */
int main()
{
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    cudaEvent_t start, end;
    int matrix_sizes[] = {1024, 2048, 4096, 8192, 16384};
    int threads_per_block[] = {2, 4, 8, 16, 32};
    int iterations = 32;

    int sizes_count = sizeof(matrix_sizes) / sizeof(matrix_sizes[0]);
    int threads_count = sizeof(threads_per_block) / sizeof(threads_per_block[0]);

    printf("Matrix size;Threads per block;Time (ms)\n");

    for (int i = 0; i < sizes_count; i++)
    {
        for (int j = 0; j < threads_count; j++)
        {
            float elapsed_time, ms;
            int matrix_size = matrix_sizes[i];
            int block_size = threads_per_block[j];

            // Allocate and initialize matrices
            init_matrices(matrix_size, &A, &B, &C);

            // Create CUDA events for timing
            CudaEventCreate(&start);
            CudaEventCreate(&end);

            // Define grid and block dimensions
            dim3 threadsPerBlock(block_size, block_size);
            dim3 numBlocks((matrix_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (matrix_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

            // Launch kernel k times
            for (int k = 0; k < iterations; k++)
            {
                CudaEventRecord(start);
                cuda_malloc_and_copy(&d_A, &d_B, &d_C, A, B, C, matrix_size);
                mul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, matrix_size, matrix_size);
                CudaMemcpy(C, d_C, matrix_size * matrix_size * sizeof(int), cudaMemcpyDeviceToHost);
                CudaEventRecord(end);
                CudaEventSynchronize(end);
                CudaEventElapsedTime(&ms, start, end);
                elapsed_time += ms;
                cuda_free_matrices(d_A, d_B, d_C);
            }

            free_matrices(A, B, C);

            CudaEventDestroy(start);
            CudaEventDestroy(end);

            elapsed_time /= iterations;
            printf("%d;%d;%f\n", matrix_size, block_size, elapsed_time);
        }
    }
}