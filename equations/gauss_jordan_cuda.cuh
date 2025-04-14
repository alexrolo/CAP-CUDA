#ifndef GAUSS_JORDAN_CUDA_H
#define GAUSS_JORDAN_CUDA_H

#include <cuda.h>

__global__ void gauss_jordan(unsigned int size, double *matrix);

#endif // GAUSS_JORDAN_CUDA_H