#include "cuda_functions.cuh"

void CudaEventCreate(cudaEvent_t *event)
{
    cudaError_t err = cudaEventCreate(event);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to create CUDA event\n");
        exit(EXIT_FAILURE);
    }
}

void CudaEventDestroy(cudaEvent_t event)
{
    cudaError_t err = cudaEventDestroy(event);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to destroy CUDA event\n");
        exit(EXIT_FAILURE);
    }
}

void CudaMalloc(void **devPtr, size_t size)
{
    cudaError_t err = cudaMalloc(devPtr, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to allocate device memory\n");
        exit(EXIT_FAILURE);
    }
}

void CudaFree(void *devPtr)
{
    cudaError_t err = cudaFree(devPtr);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to free device memory\n");
        exit(EXIT_FAILURE);
    }
}

void CudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    cudaError_t err = cudaMemcpy(dst, src, count, kind);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to copy memory\n");
        exit(EXIT_FAILURE);
    }
}

void CudaEventRecord(cudaEvent_t event)
{
    cudaError_t err = cudaEventRecord(event);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to record CUDA event\n");
        exit(EXIT_FAILURE);
    }
}
void CudaEventSynchronize(cudaEvent_t event)
{
    cudaError_t err = cudaEventSynchronize(event);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to synchronize CUDA event\n");
        exit(EXIT_FAILURE);
    }
}

void CudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    cudaError_t err = cudaEventElapsedTime(ms, start, end);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to calculate elapsed time\n");
        exit(EXIT_FAILURE);
    }
}

void CudaDeviceSynchronize()
{
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to synchronize device\n");
        exit(EXIT_FAILURE);
    }
}
