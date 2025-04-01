#include "cuda_functions.cuh"

/**
 * cudaEventCreate function with error handling
 * @param event Pointer to the event to be created
 */
void CudaEventCreate(cudaEvent_t *event)
{
    cudaError_t err = cudaEventCreate(event);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to create CUDA event\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * cudaEventDestroy function with error handling
 * @param event Event to be destroyed
 */
void CudaEventDestroy(cudaEvent_t event)
{
    cudaError_t err = cudaEventDestroy(event);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to destroy CUDA event\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * cudaMalloc function with error handling
 * @param devPtr Pointer to the device memory
 * @param size Size of the memory to be allocated
 */
void CudaMalloc(void **devPtr, size_t size)
{
    cudaError_t err = cudaMalloc(devPtr, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to allocate device memory\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * cudaFree function with error handling
 * @param devPtr Pointer to the device memory to be freed
 */
void CudaFree(void *devPtr)
{
    cudaError_t err = cudaFree(devPtr);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to free device memory\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * cudaMemcpy function with error handling
 * @param dst Destination pointer
 * @param src Source pointer
 * @param count Size of the memory to be copied
 * @param kind Type of memory copy (Host to Device, Device to Host, etc.)
 */
void CudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    cudaError_t err = cudaMemcpy(dst, src, count, kind);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to copy memory\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * cudaEventRecord function with error handling
 * @param event Event to be recorded
 * @param stream Stream in which the event is recorded
 */
void CudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    cudaError_t err = cudaEventRecord(event, stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to record CUDA event\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * cudaEventSynchronize function with error handling
 * @param event Event to be synchronized
 */
void CudaEventSynchronize(cudaEvent_t event)
{
    cudaError_t err = cudaEventSynchronize(event);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to synchronize CUDA event\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * cudaEventElapsedTime function with error handling
 * @param ms Pointer to store the elapsed time in milliseconds
 * @param start Start event
 * @param end End event
 */
void CudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    cudaError_t err = cudaEventElapsedTime(ms, start, end);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to calculate elapsed time\n");
        exit(EXIT_FAILURE);
    }
}

