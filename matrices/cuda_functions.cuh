#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

/**
 * cudaEventCreate function with error handling
 * @param event Pointer to the event to be created
 */
void CudaEventCreate(cudaEvent_t *event);

/**
 * cudaEventDestroy function with error handling
 * @param event Event to be destroyed
 */
void CudaEventDestroy(cudaEvent_t event);

/**
 * cudaMalloc function with error handling
 * @param devPtr Pointer to the device memory
 * @param size Size of the memory to be allocated
 */
void CudaMalloc(void **devPtr, size_t size);

/**
 * cudaFree function with error handling
 * @param devPtr Pointer to the device memory to be freed
 */
void CudaFree(void *devPtr);

/**
 * cudaMemcpy function with error handling
 * @param dst Destination pointer
 * @param src Source pointer
 * @param count Size of the memory to be copied
 * @param kind Type of memory copy (Host to Device, Device to Host, etc.)
 */
void CudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);

/**
 * cudaEventRecord function with error handling
 * @param event Event to be recorded
 * @param stream Stream in which the event is recorded
 */
void CudaEventRecord(cudaEvent_t event);

/**
 * cudaEventSynchronize function with error handling
 * @param event Event to be synchronized
 */
void CudaEventSynchronize(cudaEvent_t event);

/**
 * cudaEventElapsedTime function with error handling
 * @param ms Pointer to store the elapsed time in milliseconds
 * @param start Start event
 * @param end End event
 */
void CudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);

/**
 * cudaDeviceSynchronize function with error handling
 */
void CudaDeviceSynchronize();

#endif