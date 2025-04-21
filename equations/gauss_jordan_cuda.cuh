#ifndef GAUSS_JORDAN_CUDA_H
#define GAUSS_JORDAN_CUDA_H

#include <cuda.h>
#include "functions_cuda.cuh"

void gauss_jordan(unsigned int size, double *matrix);

#endif // GAUSS_JORDAN_CUDA_H
