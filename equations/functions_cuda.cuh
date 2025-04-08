#ifndef FUNCTIONS_CUDA_H
#define FUNCTIONS_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

void CudaEventCreate(cudaEvent_t *event);
void CudaEventDestroy(cudaEvent_t event);
void CudaMalloc(void **devPtr, size_t size);
void CudaFree(void *devPtr);
void CudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
void CudaEventRecord(cudaEvent_t event, cudaStream_t stream);
void CudaEventSynchronize(cudaEvent_t event);
void CudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);

#endif // FUNCTIONS_CUDA_H