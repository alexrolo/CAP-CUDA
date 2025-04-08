CC = gcc
NVCC = nvcc

CFLAGS = -Wall -Wextra -pedantic -std=c99 -Werror -g -pg -O0 -fsanitize=address -fsanitize=undefined -Wshadow
CUDAFLAGS = -Xcompiler -fopenmp

C_TARGET = equations
C_OBJ = functions.o equations.o
CUDA_TARGET = equations_cuda
CUDA_OBJ = functions.o functions_cuda.o equations_cuda.o

all: $(CUDA_TARGET) $(C_TARGET)

$(C_TARGET): $(C_OBJ)
	$(CC) $(CFLAGS) -o $@ $^

%.o : %.c
	$(NVCC) $(CUDAFLAGS) -c $<

equations_cuda : functions.o functions_cuda.o
	$(NVCC) $(CUDAFLAGS) equations_cuda.cu -o $@ functions.o functions_cuda.o

equations_cuda.o : equations_cuda.cu functions_cuda.o
	$(NVCC) $(CUDAFLAGS) -c $<

functions_cuda.o : functions_cuda.cu
	$(NVCC) $(CUDAFLAGS) -c $<

clean:
	rm -f $(CUDA_TARGET) $(C_TARGET) $(C_OBJ) $(CUDA_OBJ)
	rm -f *.s
	rm -f *.gcno
	rm -f *.gcda
	rm -f gmon.out
	rm -f *.out
	rm -f *.txt
	rm -rf .asan*
	rm -rf .ubsan*
	rm -rf .valgrind*

rebuild: clean all
