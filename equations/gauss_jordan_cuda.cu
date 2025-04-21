#include "gauss_jordan_cuda.cuh"

__global__ void normalize_row(
    unsigned int columns,
    double* matrix,
    unsigned int pivot_row
) {
    unsigned int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < columns)
    {
	const double divisor = *(matrix + pivot_row * columns + pivot_row);
	*(matrix + pivot_row * columns + thread_id) /= divisor;
    }
}

__global__ void eliminate_column(
    unsigned int size,
    double* matrix,
    unsigned int pivot_row 
) {
    const unsigned int column = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
    const unsigned int columns = size + 1;

    if (row >= size || column >= columns || row == pivot_row)
	return;

    const double pivot = *(matrix + pivot_row * columns + pivot_row);
    if (pivot == 0)
	return;

    const double factor = *(matrix + row * columns + pivot_row) / pivot;
    const double pivot_row_value = *(matrix + pivot_row * columns + column);
    *(matrix + row * columns + column) -= factor * pivot_row_value;
}

void gauss_jordan(
    unsigned int size,
    double *matrix
) {
    const unsigned int columns = size + 1;
    const unsigned int threads_per_block = 256;
    const unsigned int normalize_blocks = (columns + threads_per_block - 1) / threads_per_block;
    size_t bytes = sizeof(double) * size * columns;
    
    double* d_matrix;
    CudaMalloc((void**) &d_matrix, bytes);
    CudaMemcpy(d_matrix, matrix, bytes, cudaMemcpyHostToDevice);

    for (
	unsigned int current_column = 0 ;
	current_column < size ;
	current_column++
    ) {
	normalize_row<<<normalize_blocks, threads_per_block>>>(
	    columns,
	    d_matrix,
	    current_column);
	checkCudaError();	
	CudaDeviceSynchronize();
	
	dim3 threads_per_block(16, 16);
	dim3 num_blocks(
	    (columns + threads_per_block.x - 1) / threads_per_block.x,
	    (size + threads_per_block.y - 1) / threads_per_block.y
	);
	eliminate_column<<<num_blocks, threads_per_block>>>(
	    size,
	    d_matrix,
	    current_column);
	checkCudaError();	
	CudaDeviceSynchronize();
    }

    CudaMemcpy(matrix, d_matrix, bytes, cudaMemcpyDeviceToHost);
    CudaFree(d_matrix);
}

