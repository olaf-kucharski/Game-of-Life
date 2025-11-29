#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <cuda_runtime.h>

// A macro for CUDA error checking
#define CHECK_CUDA(err) do { cudaError_t e = (err); if (e != cudaSuccess) { \
    fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

// --- Bit manipulation helpers ---
static inline size_t words_for_bits(size_t bits) {
    return (bits + 31) / 32;
}

static inline void set_bit_host(uint32_t *arr, size_t i, int v) {
    size_t word = i / 32;
    int bit = i % 32;
    if (v) arr[word] |= (1u << bit);
    else arr[word] &= ~(1u << bit);
}

static inline int get_bit_host(const uint32_t *arr, size_t i) {
    size_t word = i / 32;
    int bit = i % 32;
    return (arr[word] >> bit) & 1u;
}

// --- CUDA kernel ---
__global__ void step_kernel(const uint32_t *cur_bits, uint32_t *next_bits, int H, int W, int total_cells) {

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_cells) return;

    int r = gid / W;
    int c = gid % W;

    int live = 0;
    for (int dr = -1; dr <= 1; ++dr) {
        int rr = (r + dr + H) % H;
        for (int dc = -1; dc <= 1; ++dc) {
            int cc = (c + dc + W) % W;
            if (dr == 0 && dc == 0) continue;
            int idx = rr * W + cc;
            size_t w = idx / 32;
            int b = idx % 32;
            uint32_t word = cur_bits[w];
            live += (word >> b) & 1u;
        }
    }

    int idx_cell = r * W + c;
    size_t wcur = idx_cell / 32;
    int bcur = idx_cell % 32;
    int state = (cur_bits[wcur] >> bcur) & 1u;

    // Game of Life rules
    int new_state = (state && (live == 2 || live == 3)) || (!state && live == 3);

    if (new_state) {
        // Atomic OR is necessary here because multiple threads might write to the same uint32 word
        uint32_t mask = 1u << bcur;
        atomicOr(&next_bits[wcur], mask);
    }
}

void print_board_bits(const uint32_t *bits, int H, int W) {
    // Limiting printing for large boards during performance tests
    if (H > 50 || W > 50) return;
    
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            int idx = r*W + c;
            int v = get_bit_host(bits, idx);
            putchar(v ? 'O' : '.');
        }
        putchar('\n');
    }
}

// --- Main program (CPU host) ---
int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: %s H W steps\n", argv[0]);
        return 1;
    }

    cudaDeviceProp prop;
    int dev = 0;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    printf("Running on GPU: %s with %d multiprocessors, %zu bytes of global memory\n",
           prop.name, prop.multiProcessorCount, prop.totalGlobalMem);

    int H = atoi(argv[1]);
    int W = atoi(argv[2]);
    int steps = atoi(argv[3]);

    int total = H * W;
    size_t words = words_for_bits(total);

    uint32_t *h_cur = (uint32_t*)calloc(words, sizeof(uint32_t));
    uint32_t *h_next = (uint32_t*)calloc(words, sizeof(uint32_t));
    if (!h_cur || !h_next) { perror("alloc"); return 1; }

    srand((unsigned)time(NULL));
    for (int i = 0; i < total; ++i) {
        if (rand() & 1) set_bit_host(h_cur, i, 1);
    }

    // Limiting printing for large boards during performance tests
    if (H <= 50 && W <= 50) {
        printf("Initial board:\n");
        print_board_bits(h_cur, H, W);
    }

    uint32_t *d_cur = NULL, *d_next = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_cur, words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc((void**)&d_next, words * sizeof(uint32_t)));

    // Create CUDA events for timing
    cudaEvent_t start, stop, compute_start, compute_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&compute_start);
    cudaEventCreate(&compute_stop);

    // --- TOTAL TIMING START (Includes H2D copy) ---
    cudaEventRecord(start);

    CHECK_CUDA(cudaMemcpy(d_cur, h_cur, words * sizeof(uint32_t), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    // --- COMPUTE TIMING START ---
    cudaEventRecord(compute_start);

    for (int s = 0; s < steps; ++s) {
        CHECK_CUDA(cudaMemset(d_next, 0, words * sizeof(uint32_t)));
        step_kernel<<<blocks, threads>>>(d_cur, d_next, H, W, total);
        
        // No need for cudaDeviceSynchronize inside loop unless debugging or swapping requires host intervention
        // Pointers swap logic is host-side only, so it's instant, but we are swapping device pointers.
        uint32_t *tmp = d_cur; d_cur = d_next; d_next = tmp;
    }
    
    // --- COMPUTE TIMING STOP ---
    cudaEventRecord(compute_stop);

    CHECK_CUDA(cudaMemcpy(h_cur, d_cur, words * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // --- TOTAL TIMING STOP ---
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_milliseconds = 0;
    float compute_milliseconds = 0;
    cudaEventElapsedTime(&total_milliseconds, start, stop);
    cudaEventElapsedTime(&compute_milliseconds, compute_start, compute_stop);

    if (H <= 50 && W <= 50) {
        printf("\nAfter %d steps:\n", steps);
        print_board_bits(h_cur, H, W);
    }

    printf("\n------------------------------------------------\n");
    printf("GPU CUDA Execution Summary:\n");
    printf("Grid Size: %dx%d\n", H, W);
    printf("Steps: %d\n", steps);
    printf("Total Time (incl. transfer): %.3f ms\n", total_milliseconds);
    printf("Compute Time (Kernel only):  %.3f ms\n", compute_milliseconds);
    printf("Transfer Overhead:           %.3f ms\n", total_milliseconds - compute_milliseconds);
    printf("Avg Kernel Time per Step:    %.3f ms\n", compute_milliseconds / steps);
    printf("------------------------------------------------\n");

    free(h_cur);
    free(h_next);
    cudaFree(d_cur);
    cudaFree(d_next);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(compute_start);
    cudaEventDestroy(compute_stop);
    
    return 0;
}