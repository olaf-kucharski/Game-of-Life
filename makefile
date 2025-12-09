CC = gcc
MPICC = mpicc
NVCC = nvcc
CFLAGS = -O2
OMPFLAG = -fopenmp

CPU_TARGET = gol-cpu
GPU_TARGET = gol-gpu
MPI_TARGET = gol-mpi

all: $(CPU_TARGET) $(GPU_TARGET) $(MPI_TARGET)

$(CPU_TARGET): gol-cpu.c
	$(CC) $(CFLAGS) $(OMPFLAG) -o $(CPU_TARGET) gol-cpu.c

$(GPU_TARGET): gol-gpu.cu
	$(NVCC) $(CFLAGS) -o $(GPU_TARGET) gol-gpu.cu

$(MPI_TARGET): gol-mpi.c
	$(MPICC) $(CFLAGS) -o $(MPI_TARGET) gol-mpi.c

# Default parameters: 20 rows, 40 columns, 10 steps
run-cpu: $(CPU_TARGET)
	./$(CPU_TARGET) 20 40 10

run-gpu: $(GPU_TARGET)
	./$(GPU_TARGET) 20 40 10

run-mpi: $(MPI_TARGET)
	mpirun -np 2 ./$(MPI_TARGET) 20 40 10 2

clean:
	rm -f $(CPU_TARGET) $(GPU_TARGET) $(MPI_TARGET)
