#include "functions_cuda.cuh"

__global__ void gauss_jordan(unsigned int size, double *matrix)
{
    unsigned int const current_column = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int actual_columns = size + 1;

    if (current_column >= size)
        return; // Out of bounds

    // Find maximum in current column for partial pivoting
    unsigned int max_row = 0, current_row = 0;
    double max_value = 0;
    for (current_row = current_column; current_row < size; current_row++)
    {
        double current_value = *(matrix + current_row * actual_columns + current_column);
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
    {
        for (unsigned int i = 0; i < actual_columns; i++)
        {
            unsigned int current_idx = current_column * actual_columns + i;
            unsigned int max_idx = max_row * actual_columns + i;
            double temp = *(matrix + current_idx);
            *(matrix + current_idx) = *(matrix + max_idx);
            *(matrix + max_idx) = temp;
        }
    }

    // Make the diagonal element equal to 1
    double divisor = *(matrix + current_column * actual_columns + current_column);
    for (unsigned int i = 0; i < actual_columns; i++)
        *(matrix + current_column * actual_columns + i) /= divisor; // Normalize the pivot row

    // Make zeros in the current column
    for (unsigned int i = 0; i < size; i++)
    {
        if (i != current_column)
        {
            double multiplier = *(matrix + i * actual_columns + current_column) / *(matrix + current_column * actual_columns + current_column);
            for (unsigned int j = 0; j <= size; j++)
                *(matrix + i * actual_columns + j) -= multiplier * *(matrix + current_column * actual_columns + j); // Eliminate the current column
        }
    }
}

double *solve_equation_with_gpu(unsigned int size, unsigned int threads, double *matrix, double *sol)
{

    double *d_matrix;
    cudaEvent_t start, end;
    float ms;

    // Allocate memory on the GPU
    size_t total_size = size * size * sizeof(double *);
    CudaMalloc((void **)&d_matrix, total_size);
    CudaMemcpy(d_matrix, matrix, total_size, cudaMemcpyHostToDevice);

    // Define dim3 grid dimensions
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

    for (unsigned int i = 0; i < size; i++)
        sol[i] = *(matrix + i * size + size);

    return sol;
}

int main(int argc, char *argv[])
{
    unsigned int size, threads; // Square matrix
    double *matrix, *sol;
    double *original_matrix = NULL;
    clock_t start, end;
    double seconds;

    // Check if the required arguments are provided
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <size> <threads_per_block>\n", argv[0]);
        return 1;
    }

    if ((size = atoi(argv[1])) < 1)
    {
        fprintf(stderr, "Invalid size: %s\n", argv[1]);
        return 1;
    }

    if ((threads = atoi(argv[2])) < 1)
    {
        fprintf(stderr, "Invalid threads per block: %s\n", argv[2]);
        return 1;
    }

    // Allocate memory
    matrix = allocate_matrix(size);
    sol = (double *)malloc(size * sizeof(double));

    // Generate random matrix
    srand(time(NULL));
    generate_matrix(size, matrix);
    original_matrix = copy_matrix(size, matrix, original_matrix);

    unsigned int i;

    start = clock();
    solve_equation_with_gpu(size, threads, matrix, sol);
    end = clock();

    // The solution is in the last column
    for (i = 0; i < size; i++)
        *(sol + i) = *(matrix + i * (size + 1) + size);

    seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Execution time (seconds): %.5f\n", seconds);

    // The solution is in the last column
    if (size < 16)
    {
        printf("System solution:\n");
        for (unsigned int i = 0; i < size; i++)
            printf("x%d = %.3f\n", i, *(sol + i));
    } else 
    {
        printf("System solution is too large to print.\n");
    }

    printf("Checking against original matrix:\n");
    if (size < 16)
        print_equation_system(size, matrix);
    else
        printf("Matrix is too large to print.\n");

    // Check if the solution is correct
    if (check_equation_system(size, original_matrix, sol))
        printf("The solution is correct.\n");
    else
        printf("The solution is incorrect.\n");

    // Free matrix
    free(matrix);
    free(original_matrix);
    free(sol);

    return 0;
}
