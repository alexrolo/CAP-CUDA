#ifndef FUNCTIONS_CUDA_H
#define FUNCTIONS_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
 * Function to allocate memory for a square matrix of size n x n.
 * Will exit if memory allocation fails.
 * 
 * @param size The size of the matrix
 * 
 * @return A pointer to the allocated matrix
 */
double *allocate_matrix(unsigned int size);

/**
 * Function to generate a random matrix of size n x n
 * 
 * @param size The size of the matrix
 * @param matrix The matrix to be filled with random values
 */
void generate_matrix(unsigned int size, double *matrix);

/**
 * Function to print the system of equations
 * 
 * @param size The size of the matrix
 * @param mat The matrix to be printed
 */
void print_equation_system(unsigned int size, double *matrix);

/**
 * Function to check a solution of a system of equations
 * 
 * @param size The size of the matrix
 * @param mat The matrix to be checked
 * @param sol The solution to be checked
 * 
 * @return 1 if the solution is correct, 0 otherwise
 */
int check_equation_system(unsigned int size, double *matrix, double *solution);

/**
 * Function to copy a matrix
 * 
 * @param size The size of the matrix
 * @param src The source matrix
 * @param dest The destination matrix
 */
void copy_matrix(unsigned int size, double *src, double *dest);

#endif