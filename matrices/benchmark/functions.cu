#include "functions.cuh"

void CudaEventCreate(cudaEvent_t *event)
{
    cudaError_t err = cudaEventCreate(event);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to create CUDA event %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void CudaEventDestroy(cudaEvent_t event)
{
    cudaError_t err = cudaEventDestroy(event);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to destroy CUDA event %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void CudaMalloc(void **devPtr, size_t size)
{
    cudaError_t err = cudaMalloc(devPtr, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to allocate device memory %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void CudaFree(void *devPtr)
{
    cudaError_t err = cudaFree(devPtr);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to free device memory %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void CudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    cudaError_t err = cudaMemcpy(dst, src, count, kind);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to copy memory %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void CudaEventRecord(cudaEvent_t event)
{
    cudaError_t err = cudaEventRecord(event);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to record CUDA event %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
void CudaEventSynchronize(cudaEvent_t event)
{
    cudaError_t err = cudaEventSynchronize(event);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to synchronize CUDA event %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void CudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    cudaError_t err = cudaEventElapsedTime(ms, start, end);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to calculate elapsed time %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void CudaDeviceSynchronize()
{
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to synchronize device %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void checkCudaError()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: CUDA error %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

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

void generate_matrix(int rows, int cols, int *matrix)
{
    srand(time(NULL));
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix[i * cols + j] = rand() % 10; // Generate numbers between 0 and 9
        }
    }
}

void print_matrix(int rows, int columns, int *matrix)
{
    for (int i = 0; i < rows; i++)
    {
        printf("| ");
        for (int j = 0; j < columns; j++)
        {
            printf("%d ", matrix[i * columns + j]);
        }
        printf("|\n");
    }
    printf("\n");
}

void fill_matrix(int rows, int columns, int *matrix, int value)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            matrix[i * columns + j] = value;
        }
    }
}
