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

/**
 * Allocates host (CPU) memory for all matrices and initializes them
 * @param matrix_size Size of the matrices
 * @param A Pointer to the first matrix
 * @param B Pointer to the second matrix
 * @param C Pointer to the result matrix
 */
void init_matrices(int matrix_size, int **A, int **B, int **C);

/**
 * Allocates memory on the device (GPU) and copies matrices from host to device
 * @param d_A Pointer to the first matrix on device
 * @param d_B Pointer to the second matrix on device
 * @param d_C Pointer to the result matrix on device
 * @param A Pointer to the first matrix on host
 * @param B Pointer to the second matrix on host
 * @param C Pointer to the result matrix on host
 * @param matrix_size Size of the matrices
 */
void cuda_malloc_and_copy(int **d_A, int **d_B, int **d_C, int *A, int *B, int *C, int matrix_size);

/**
 * Free device memory for device (GPU) matrices
 * @param d_A Pointer to the first matrix on device
 * @param d_B Pointer to the second matrix on device
 * @param d_C Pointer to the result matrix on device
 */
void cuda_free_matrices(int *d_A, int *d_B, int *d_C);

/**
 * Free host (CPU) memory for matrices
 * @param A Pointer to the first matrix on host
 * @param B Pointer to the second matrix on host
 * @param C Pointer to the result matrix on host
 */
void free_matrices(int *A, int *B, int *C);

/**
 * Kernel function to multiply two matrices in GPU
 * @param A Pointer to the first matrix
 * @param B Pointer to the second matrix
 * @param C Pointer to the result matrix
 * @param rows Number of rows in matrices
 * @param cols Number of columns in matrices
 */
__global__ void mul(int *A, int *B, int *C, int rows, int cols);

#endif