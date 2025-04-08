#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

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

/**
 * Checks for CUDA errors and prints the error message if any
 */
void checkCudaError();

/**
 * Allocate memory for a matrix of size rows x cols
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Pointer to the allocated matrix
 */
int *allocate_matrix(int rows, int cols);

/**
 * Generate a random matrix of size rows x cols
 * @param rows Number of rows
 * @param cols Number of columns
 * @param matrix Pointer to the matrix to be filled
 */
void generate_matrix(int rows, int cols, int *matrix);

/**
 * Prints a matrix to the console
 * @param rows Number of rows
 * @param cols Number of columns
 * @param matrix Pointer to the matrix to be printed
 */
void print_matrix(int rows, int cols, int *matrix);

/**
 * Fills a matrix with a specific value
 * @param rows Number of rows
 * @param cols Number of columns
 * @param matrix Pointer to the matrix to be filled
 * @param value Value to fill the matrix with
 */
void fill_matrix(int rows, int cols, int *matrix, int value);



#endif