NVCC = nvcc
CUDA_FLAGS = -Xcompiler -fopenmp

TARGET = cuda_mul_benchmark
OBJ = functions.o cuda_mul_benchmark.o

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(CUDA_FLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -c $<

clean:
	rm -f $(TARGET) $(OBJ)

rebuild: clean all
