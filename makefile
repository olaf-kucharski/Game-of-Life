CC = gcc
NVCC = nvcc
CFLAGS = -O2
OMPFLAG = -fopenmp

CPU_TARGET = gol-cpu
GPU_TARGET = gol-gpu

all: $(CPU_TARGET) $(GPU_TARGET)

$(CPU_TARGET): gol-cpu.c
	$(CC) $(CFLAGS) $(OMPFLAG) -o $(CPU_TARGET) gol-cpu.c

$(GPU_TARGET): gol-gpu.cu
	$(NVCC) $(CFLAGS) -o $(GPU_TARGET) gol-gpu.cu

# Default parameters: 20 rows, 40 columns, 10 steps
run-cpu: $(CPU_TARGET)
	./$(CPU_TARGET) 20 40 10

run-gpu: $(GPU_TARGET)
	./$(GPU_TARGET) 20 40 10

clean:
	rm -f $(CPU_TARGET) $(GPU_TARGET)
