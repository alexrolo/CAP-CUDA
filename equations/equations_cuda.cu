#include "functions_cuda.cuh"

__global__ void gauss_jordan(unsigned int size, double *matrix)
{
    unsigned int const current_column = blockIdx.x * blockDim.x + threadIdx.x;

    if (current_column >= size)
        return; // Out of bounds

    // Find maximum in current column for partial pivoting
    unsigned int max_row = 0, current_row = 0;
    double max_value = 0;
    for (current_row = current_column; current_row < size; current_row++)
    {
        double current_value = *(matrix + current_row + current_column * size);
        if (current_value < 0)
            current_value = -current_value; // Take absolute value for comparison
        if (current_value > max_value)
        {
            max_row = current_row;
            max_value = current_value;
        }
    }

    // Swap rows
    if (max_row != current_column)
        for (current_row = 0; current_row < size; current_row++)
        {
            double temp = *(matrix + current_column + current_row * size);
            *(matrix + current_column + current_row * size) = *(matrix + max_row + current_row * size);
            *(matrix + max_row + current_row * size) = temp;
        }

    // Make the diagonal element equal to 1
    double divisor = *(matrix + current_column + current_column * size);
    for (unsigned int i = 0; i <= size; i++)
        *(matrix + current_column * size + i) /= divisor; // Normalize the pivot row

    // Make zeros in the current column
    for (unsigned int i = 0; i < size; i++)
    {
        if (i != current_column)
        {
            double multiplier = *(matrix + i * size + current_column) / *(matrix + current_column * size + current_column);
            for (unsigned int j = 0; j <= size; j++)
                *(matrix + i * size + j) -= multiplier * *(matrix + current_column * size + j); // Eliminate the current column
        }
    }
}

double* solve_equation_with_gpu(unsigned int size, double* matrix, double* sol)
{

    double* d_matrix;
    cudaEvent_t start, end;
    float ms;

    // Allocate memory on the GPU
    size_t total_size = size * size * sizeof(double*);
    CudaMalloc((void**) &d_matrix, total_size);
    CudaMemcpy(d_matrix, matrix, total_size, cudaMemcpyHostToDevice);
    
    // TODO: Define dim3 grid dimensions
    const unsigned int threads = 32;
    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    CudaEventCreate(&start);
    CudaEventCreate(&end);

    CudaEventRecord(start);
    // TODO: Call the kernel function
    gauss_jordan<<<numBlocks, threadsPerBlock>>>(size, d_matrix);

    CudaEventRecord(end);
    CudaEventSynchronize(end);
    CudaEventElapsedTime(&ms, start, end);

    CudaFree(d_matrix);
    CudaEventDestroy(start);
    CudaEventDestroy(end);

    for (unsigned int i = 0; i < size ; i++)
        sol[i] = *(matrix + i * size + size);

    return sol;
}

int main(int argc, char *argv[])
{
    unsigned int size; // Square matrix
    double *matrix, *sol;
    clock_t start, end;
    double seconds;
    char* buffer = (char*)malloc(128);

    // Check if the required arguments are provided
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <size>\n", argv[0]);
        return 1;
    }

    if ((size = atoi(argv[1])) < 1)
    {
        fprintf(stderr, "Invalid size: %s\n", argv[1]);
        return 1;
    }

    // Allocate memory
    matrix = allocate_matrix(size);
    sol = (double *)malloc(size * sizeof(double));

    // Generate random matrix
    srand(time(NULL));
    generate_matrix(size, matrix);

    unsigned int i;

    start = clock();
    solve_equation_with_gpu(size, matrix, sol);
    end = clock();

    // The solution is in the last column
    for (i = 0; i < size; i++)
	*(sol + i) = *(matrix + i * size + size);

    seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Execution time (seconds): %.5f\n", seconds);

    // The solution is in the last column
    printf("System solution:\n");
    for (unsigned int i = 0; i < size; i++)
    {
        printf("x%d = %.3f\n", i, *(sol + i));
    }

    // Check if the solution is correct
    if (check_equation_system(size, matrix, sol))
        printf("The solution is correct.\n");
    else
        printf("The solution is incorrect.\n");

    // Free matrix
    free(matrix);
    free(sol);
    free(buffer);

    return 0;
}
