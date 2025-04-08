CC = gcc
NVCC = nvcc

CFLAGS = -Wall -Wextra -pedantic -std=c99 -Werror -g -pg -O0 -fsanitize=address -fsanitize=undefined -Wshadow
CUDAFLAGS = -Werror all-warnings -g -pg -O0 -Xcompiler -fopenmp

C_TARGET = equations
C_OBJ = functions.o equations.o
CUDA_TARGET = equations_cuda
CUDA_OBJ = functions.o cuda_functions.o equations_cuda.o



all: $(CUDA_TARGET) $(C_TARGET)

$(C_TARGET): $(C_OBJ)
	$(CC) $(CFLAGS) -o $@ $^

$(CUDA_TARGET): $(CUDA_OBJ)
	$(NVCC) $(CUDAFLAGS) -o $@ $^

functions.o: functions.c
	$(CC) $(CFLAGS) -c $<

cuda_functions.o: cuda_functions.cu
	$(NVCC) $(CUDAFLAGS) -c $<

%_cuda.o : %_cuda.cu
	$(NVCC) $(CUDAFLAGS) -o $@ $^

%.o : %.c
	$(CC) $(CFLAGS) -o $@ $^

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
