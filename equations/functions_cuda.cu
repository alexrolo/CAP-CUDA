#include "functions_cuda.cuh"

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

double *allocate_matrix(unsigned int size)
{
    double *matrix = (double *)malloc(size * size * sizeof(double *));
    if (matrix == NULL)
    {
        fprintf(stderr, "Error: Unable to allocate memory for rows.\n");
        exit(EXIT_FAILURE);
    }

    return matrix;
}

void generate_matrix(unsigned int size, double *matrix)
{
    const unsigned int total_size = size * size;
    for (unsigned int i = 0; i < total_size; i++)
        *(matrix + i) = (rand() % 10) + 1; // Generate numbers between 0 and 9
}

void print_equation_system(unsigned int size, double *matrix)
{
    const unsigned int total_size = size * size;
    for (unsigned int i = 0; i < total_size; i++)
    {
        printf("%5.4f x%d ", *(matrix + i), i % size);
        if (i % size < size - 1)
            printf("+ ");
        if (i % size == size - 1)
            printf("\n");
    }
    printf("\n");
}

int check_equation_system(unsigned int size, double *matrix, double *solution)
{
    const unsigned int total_size = size * size;
    double sum = 0;
    for (unsigned int i = 0; i < total_size; i++)
    {
        // Reset sum for each row
        if (i % size == 0)
            sum = 0;

        // Calculate the sum of the products
        sum += *(matrix + i) * *(solution + size + (i % size) * size);

        // Check if the absolute difference is small
        if (fabs(sum - *(matrix + i * size + size)) > 1e-6)
            return 0; // Solution is incorrect
    }
    return 1; // Solution is correct
}

void copy_matrix(unsigned int size, double *src, double *dest)
{
    dest = allocate_matrix(size);
    for (unsigned int i = 0; i < size; i++)
        *(dest+i) = *(src+i);
}

