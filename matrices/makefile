NVCC = nvcc
CUDA_FLAGS = -Xcompiler -fopenmp

TARGET = multiply_matrices_cuda
OBJ = cuda_functions.o multiply_matrices_cuda.o

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(CUDA_FLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -c $<

clean:
	rm -f $(TARGET) $(OBJ)

rebuild: clean all
