#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// TODO: Add OpenMP parallelization for performance improvement
// TODO: Add MPI support for distributed computing

static inline int idx(int r, int c, int W) { return r * W + c; }

void step(const unsigned char *cur, unsigned char *next, int H, int W)
{
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < H; ++r)
    {
        for (int c = 0; c < W; ++c)
        {
            int live = 0;
            for (int dr = -1; dr <= 1; ++dr)
            {
                for (int dc = -1; dc <= 1; ++dc)
                {
                    if (dr == 0 && dc == 0)
                        continue;
                    int rr = (r + dr + H) % H;
                    int cc = (c + dc + W) % W;
                    live += cur[idx(rr, cc, W)];
                }
            }
            unsigned char state = cur[idx(r, c, W)];
            next[idx(r, c, W)] = (state && (live == 2 || live == 3)) || (!state && live == 3);
        }
    }
}

void print_board(const unsigned char *board, int H, int W)
{
    // Limiting printing for large boards during performance tests
    if (H > 50 || W > 50) return;

    for (int r = 0; r < H; ++r)
    {
        for (int c = 0; c < W; ++c)
        {
            putchar(board[idx(r, c, W)] ? 'O' : '.');
        }
        putchar('\n');
    }
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("Usage: %s H W steps\n", argv[0]);
        return 1;
    }
    int H = atoi(argv[1]);
    int W = atoi(argv[2]);
    int steps = atoi(argv[3]);

    unsigned char *cur = malloc(H * W);
    unsigned char *next = malloc(H * W);
    if (!cur || !next)
    {
        perror("malloc");
        return 1;
    }

    srand(time(NULL));
    for (int i = 0; i < H * W; ++i)
        cur[i] = rand() & 1;

    if (H <= 50 && W <= 50) {
        printf("Initial board:\n");
        print_board(cur, H, W);
    } else {
        printf("Board too large to print, skipping visualization...\n");
    }

    printf("OpenMP threads: %d\n", omp_get_max_threads());

    // --- TIME MEASUREMENT START ---
    double start = omp_get_wtime();

    for (int s = 0; s < steps; ++s)
    {
        step(cur, next, H, W);
        unsigned char *tmp = cur;
        cur = next;
        next = tmp;
    }

    double end = omp_get_wtime();
    // --- TIME MEASUREMENT STOP ---

    double time_spent = end - start;

    if (H <= 50 && W <= 50) {
        printf("\nAfter %d steps:\n", steps);
        print_board(cur, H, W);
    }

    printf("\n------------------------------------------------\n");
    printf("CPU Execution Summary:\n");
    printf("Grid Size: %dx%d\n", H, W);
    printf("Steps: %d\n", steps);
    printf("Total Time: %.6f seconds\n", time_spent);
    printf("Avg Time per Step: %.6f seconds\n", time_spent / steps);
    printf("------------------------------------------------\n");

    free(cur);
    free(next);
    return 0;
}