#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <cuda_runtime.h>

// Makro do sprawdzania błędów CUDA
#define CHECK_CUDA(err) do { cudaError_t e = (err); if (e != cudaSuccess) { \
    fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

// Rozmiar bloku (kafla) - 16x16 wątków
#define BLOCK_DIM 16

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

// Funkcja pomocnicza device do odczytu bitu z pamięci globalnej
__device__ inline int get_bit_device(const uint32_t *arr, int r, int c, int W) {
    int idx = r * W + c;
    size_t word = idx / 32;
    int bit = idx % 32;
    return (arr[word] >> bit) & 1u;
}

// --- CUDA kernel z pamięcią współdzieloną ---
__global__ void step_kernel_shared(const uint32_t *cur_bits, uint32_t *next_bits, int H, int W) {
    // Shared memory: Kafelek + marginesy (Halo) dla sąsiadów
    __shared__ uint8_t tile[BLOCK_DIM + 2][BLOCK_DIM + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Globalne współrzędne komórki, za którą odpowiada ten wątek
    int out_c = blockIdx.x * blockDim.x + tx;
    int out_r = blockIdx.y * blockDim.y + ty;

    // --- FAZA 1: Ładowanie danych do Shared Memory ---
    // Pobranie skompresowanego słowa bitowego i wyłuskanie odpowiedniego bitu
    int tile_h = BLOCK_DIM + 2;
    int tile_w = BLOCK_DIM + 2;
    int num_threads = BLOCK_DIM * BLOCK_DIM;
    int tid = ty * BLOCK_DIM + tx;

    // Pętla po wszystkich elementach kafelka (wątki dzielą się pracą)
    for (int i = tid; i < tile_h * tile_w; i += num_threads) {
        int ly = i / tile_w; // lokalny wiersz w tile
        int lx = i % tile_w; // lokalna kolumna w tile
        
        // Mapowanie lokalnych współrzędnych tile na globalne współrzędne planszy
        // Odejmujemy 1, aby uwzględnić lewy/górny margines halo
        int global_r = (blockIdx.y * blockDim.y + ly - 1);
        int global_c = (blockIdx.x * blockDim.x + lx - 1);

        // Obsługa periodycznych warunków brzegowych (torus)
        global_r = (global_r % H + H) % H;
        global_c = (global_c % W + W) % W;

        // Odczyt bitowy z global memory, zapis bajtowy do shared memory
        tile[ly][lx] = get_bit_device(cur_bits, global_r, global_c, W);
    }

    __syncthreads(); // Bariera: czekamy aż cały kafelek zostanie załadowany

    // --- FAZA 2: Obliczenia ---
    // Jeśli wątek jest poza faktycznym wymiarem planszy, kończymy
    if (out_c >= W || out_r >= H) return;

    // Współrzędne wątku wewnątrz kafelka (przesunięte o 1 przez margines)
    int my_ly = ty + 1;
    int my_lx = tx + 1;

    int live = 0;
    // Sumowanie sąsiadów - teraz czytamy z szybkiej pamięci shared (bajty)
    live += tile[my_ly - 1][my_lx - 1];
    live += tile[my_ly - 1][my_lx    ];
    live += tile[my_ly - 1][my_lx + 1];
    live += tile[my_ly    ][my_lx - 1];
    live += tile[my_ly    ][my_lx + 1];
    live += tile[my_ly + 1][my_lx - 1];
    live += tile[my_ly + 1][my_lx    ];
    live += tile[my_ly + 1][my_lx + 1];

    int state = tile[my_ly][my_lx];
    int new_state = (state && (live == 2 || live == 3)) || (!state && live == 3);

    // --- FAZA 3: Zapis wyników ---
    if (new_state) {
        int idx_global = out_r * W + out_c;
        size_t w_out = idx_global / 32;
        int b_out = idx_global % 32;
        
        // Zapis bitowy do pamięci globalnej.
        // Konieczny atomicOr, ponieważ wiele wątków może pisać do tego samego słowa uint32_t.
        atomicOr(&next_bits[w_out], (1u << b_out));
    }
}

void print_board_bits(const uint32_t *bits, int H, int W) {
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

    int H = atoi(argv[1]);
    int W = atoi(argv[2]);
    int steps = atoi(argv[3]);

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Running on GPU: %s\n", prop.name);

    int total = H * W;
    size_t words = words_for_bits(total);

    uint32_t *h_cur = (uint32_t*)calloc(words, sizeof(uint32_t));
    uint32_t *h_next = (uint32_t*)calloc(words, sizeof(uint32_t));
    if (!h_cur || !h_next) { perror("alloc"); return 1; }

    srand((unsigned)time(NULL));
    for (int i = 0; i < total; ++i) {
        if (rand() & 1) set_bit_host(h_cur, i, 1);
    }

    if (H <= 50 && W <= 50) {
        printf("Poczatkowa plansza:\n");
        print_board_bits(h_cur, H, W);
    } else {
        printf("Plansza zbyt duza, pomijam wizualizacje...\n");
    }

    uint32_t *d_cur = NULL, *d_next = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_cur, words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc((void**)&d_next, words * sizeof(uint32_t)));

    cudaEvent_t start, stop, compute_start, compute_stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventCreate(&compute_start); cudaEventCreate(&compute_stop);

    cudaEventRecord(start);
    CHECK_CUDA(cudaMemcpy(d_cur, h_cur, words * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Konfiguracja kernela 2D
    dim3 threads(BLOCK_DIM, BLOCK_DIM); // 16x16 = 256 wątków
    dim3 blocks((W + BLOCK_DIM - 1) / BLOCK_DIM, (H + BLOCK_DIM - 1) / BLOCK_DIM);

    cudaEventRecord(compute_start);

    for (int s = 0; s < steps; ++s) {
        // Zerowanie bufora wyjściowego jest konieczne przy użyciu atomicOr
        CHECK_CUDA(cudaMemset(d_next, 0, words * sizeof(uint32_t)));
        
        step_kernel_shared<<<blocks, threads>>>(d_cur, d_next, H, W);
        
        // Swap pointers
        uint32_t *tmp = d_cur; d_cur = d_next; d_next = tmp;
    }
    
    cudaEventRecord(compute_stop);
    CHECK_CUDA(cudaMemcpy(h_cur, d_cur, words * sizeof(uint32_t), cudaMemcpyDeviceToHost));

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
    printf("GPU CUDA z pamiiecia wspoldzielona:\n");
    printf("Rozmiar siatki: %dx%d\n", H, W);
    printf("Liczba krokow: %d\n", steps);
    printf("Czas obliczen: %.3f ms\n", compute_milliseconds);
    printf("Sredni czas na krok: %.3f ms\n", compute_milliseconds / steps);
    printf("------------------------------------------------\n");

    free(h_cur); free(h_next);
    cudaFree(d_cur); cudaFree(d_next);
    return 0;
}