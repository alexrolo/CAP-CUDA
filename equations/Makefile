CC = g++
NVCC = nvcc

CFLAGS = -Wall -Wextra -pedantic -Werror -g -pg -O0 -Wshadow
CUDAFLAGS = -Xcompiler -fopenmp

CPP_TARGET = comparer
CPP_OBJ = functions.o gauss_jordan.o comparer.o
C_TARGET = equations 
C_OBJ = functions.o gauss_jordan.o equations.o
CUDA_TARGET = equations_cuda
CUDA_OBJ = functions.o functions_cuda.o gauss_jordan_cuda.o equations_cuda.o

all: $(CUDA_TARGET) $(C_TARGET) $(CPP_TARGET)

$(CPP_TARGET): $(CPP_OBJ)
	$(CC) $(CFLAGS) -o $@ $^

$(C_TARGET): $(C_OBJ)
	$(CC) $(CFLAGS) -o $@ $^

%.o : %.c
	$(CC) $(CFLAGS) -c $<

$(CUDA_TARGET) : $(CUDA_OBJ)
	$(NVCC) $(CUDAFLAGS) -o $@ $^

equations_cuda.o : equations_cuda.cu functions_cuda.o gauss_jordan_cuda.o
	$(NVCC) $(CUDAFLAGS) -c $<

gauss_jordan_cuda.o : gauss_jordan_cuda.cu
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
