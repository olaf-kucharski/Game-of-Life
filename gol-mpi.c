#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

/* MPI Gra w Życie z podziałem planszy na fragmenty (domain decomposition)
 * Każdy proces MPI obsługuje pasek wierszy planszy.
 * Po każdej iteracji wątki wymieniają krawędzie (halo exchange).
 */

static inline int idx(int r, int c, int W) { return r * W + c; }

/* Oblicza ile żywych sąsiadów ma komórka (r, c).
 * Para parametrów (cur, W, H) definiuje lokalny fragment planszy.
 * Warunki brzegowe są periodyczne (torus).
 */
int count_live_neighbors(const unsigned char *cur, int r, int c, int W, int H) {
    int live = 0;
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) continue;
            int rr = (r + dr + H) % H;
            int cc = (c + dc + W) % W;
            live += cur[idx(rr, cc, W)];
        }
    }
    return live;
}

/* Wykona jeden krok gry na lokalnym fragmencie (dla procesu rank).
 * local_cur: bieżący stan lokalnego fragmentu (z marginesami halo z sąsiadów)
 * local_next: wynikowy stan lokalnego fragmentu
 * local_H: liczba wierszy lokalnego fragmentu (bez marginesów halo)
 * W: liczba kolumn (wspólna dla całej planszy)
 * 
 * local_cur ma wymiary (local_H + 2) x W
 * Wiersze 0 i local_H+1 to marginesy halo
 * Wiersze 1..local_H to część lokalnego fragmentu
 */
void step_local(const unsigned char *local_cur, unsigned char *local_next,
                int local_H, int W) {
    for (int r = 1; r <= local_H; ++r) {
        for (int c = 0; c < W; ++c) {
            int live = 0;
            // Sumowanie 8 sąsiadów (indeks dr, dc od -1 do 1, z wyłączeniem (0,0))
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    int rr = r + dr;      // wiersz w local_cur (może być 0, 1..local_H, lub local_H+1)
                    int cc = (c + dc + W) % W;  // kolumna z periodycznym warunkiem brzegowym
                    live += local_cur[idx(rr, cc, W)];
                }
            }
            unsigned char state = local_cur[idx(r, c, W)];
            local_next[idx(r, c, W)] = 
                (state && (live == 2 || live == 3)) || (!state && live == 3);
        }
    }
}

/* Drukuje lokalny fragment planszy (bez marginesów halo) */
void print_local_board(const unsigned char *board, int local_H, int W, int rank) {
    printf("Rank %d:\n", rank);
    for (int r = 1; r <= local_H; ++r) {
        for (int c = 0; c < W; ++c) {
            putchar(board[idx(r, c, W)] ? 'O' : '.');
        }
        putchar('\n');
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 5) {
        if (rank == 0) {
            printf("Usage: %s H W steps num_procs\n", argv[0]);
            printf("  H: wysokość planszy\n");
            printf("  W: szerokość planszy\n");
            printf("  steps: liczba iteracji\n");
            printf("  num_procs: liczba procesów MPI (powinno być równe liczbie procesów)\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    int H = atoi(argv[1]);
    int W = atoi(argv[2]);
    int steps = atoi(argv[3]);
    int num_procs = atoi(argv[4]);
    
    if (num_procs != size) {
        if (rank == 0) {
            fprintf(stderr, "Blad: num_procs (%d) != liczba procesów (%d)\n", num_procs, size);
        }
        MPI_Finalize();
        return 1;
    }
    
    /* Podział wierszy między procesy */
    int local_H = H / size;
    int remainder = H % size;
    if (rank < remainder) {
        local_H++;
    }
    
    /* Globalna pozycja startowa tego procesu */
    int start_row = 0;
    for (int i = 0; i < rank; ++i) {
        int tmp_H = H / size;
        if (i < remainder) tmp_H++;
        start_row += tmp_H;
    }
    
    /* Alokacja lokalnych buforów
     * Wymiary: (local_H + 2) x W, żeby zmieścić marginesy halo
     */
    unsigned char *local_cur = (unsigned char*)calloc((local_H + 2) * W, sizeof(unsigned char));
    unsigned char *local_next = (unsigned char*)calloc((local_H + 2) * W, sizeof(unsigned char));
    if (!local_cur || !local_next) {
        perror("malloc");
        MPI_Finalize();
        return 1;
    }
    
    /* Inicjalizacja losowego stanu dla tego procesu
     * (inicjalizujemy tylko część lokalną, bez marginesów)
     */
    srand(time(NULL) + rank);
    for (int r = 1; r <= local_H; ++r) {
        for (int c = 0; c < W; ++c) {
            local_cur[idx(r, c, W)] = rand() & 1;
        }
    }
    
    /* Drukowanie początkowego stanu (tylko dla małych plansz i procesu 0) */
    if (rank == 0 && H <= 50 && W <= 50) {
        printf("Poczatkowa plansza:\n");
        for (int r = 1; r <= local_H; ++r) {
            for (int c = 0; c < W; ++c) {
                putchar(local_cur[idx(r, c, W)] ? 'O' : '.');
            }
            putchar('\n');
        }
    }
    
    /* Barrier do synchronizacji przed pomiarem czasu */
    MPI_Barrier(MPI_COMM_WORLD);
    
    /* Pomiar czasu */
    double start_time = MPI_Wtime();
    
    /* Główna pętla iteracji */
    for (int step = 0; step < steps; ++step) {
        /* --- Wymiana krawędzi (halo exchange) ---
         * Każdy proces wysyła górny wiersz (r=1) do poprzednika
         * i dolny wiersz (r=local_H) do następnika,
         * i otrzymuje od nich marginesy halo.
         * Korzystamy z non-blocking MPI aby uniknąć deadlocka.
         */
        
        MPI_Request req[4];
        int nreq = 0;
        
        int prev_rank = (rank - 1 + size) % size;
        int next_rank = (rank + 1) % size;
        
        /* Wysłanie góry (wiersz 1), otrzymanie góry (wiersz 0) */
        MPI_Isend(&local_cur[idx(1, 0, W)], W, MPI_UNSIGNED_CHAR, prev_rank, 10, MPI_COMM_WORLD, &req[nreq++]);
        MPI_Irecv(&local_cur[idx(0, 0, W)], W, MPI_UNSIGNED_CHAR, prev_rank, 11, MPI_COMM_WORLD, &req[nreq++]);
        
        /* Wysłanie dołu (wiersz local_H), otrzymanie dołu (wiersz local_H+1) */
        MPI_Isend(&local_cur[idx(local_H, 0, W)], W, MPI_UNSIGNED_CHAR, next_rank, 11, MPI_COMM_WORLD, &req[nreq++]);
        MPI_Irecv(&local_cur[idx(local_H + 1, 0, W)], W, MPI_UNSIGNED_CHAR, next_rank, 10, MPI_COMM_WORLD, &req[nreq++]);
        
        /* Oczekiwanie na wszystkie komunikaty */
        MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
        
        /* --- Obliczenie lokalnego kroku --- */
        step_local(local_cur, local_next, local_H, W);
        
        /* Zamiana wskaźników */
        unsigned char *tmp = local_cur;
        local_cur = local_next;
        local_next = tmp;
    }
    
    /* Koniec pomiaru czasu */
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double time_spent = end_time - start_time;
    
    /* Drukowanie końcowego stanu (tylko dla małych plansz i procesu 0) */
    if (rank == 0 && H <= 50 && W <= 50) {
        printf("\nAfter %d steps:\n", steps);
        for (int r = 1; r <= local_H; ++r) {
            for (int c = 0; c < W; ++c) {
                putchar(local_cur[idx(r, c, W)] ? 'O' : '.');
            }
            putchar('\n');
        }
    }
    
    /* Zbieranie czasu z wszystkich procesów */
    double max_time = 0;
    MPI_Reduce(&time_spent, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    /* Raport (tylko rank 0) */
    if (rank == 0) {
        printf("\n------------------------------------------------\n");
        printf("MPI Gra w Zycie - Podsumowanie wykonania:\n");
        printf("Rozmiar planszy: %d x %d\n", H, W);
        printf("Liczba procesów MPI: %d\n", size);
        printf("Liczba kroków: %d\n", steps);
        printf("Maksymalny czas (z wszystkich procesów): %.3f ms\n", max_time * 1000.0);
        printf("Średni czas na krok: %.3f ms\n", (max_time * 1000.0) / steps);
        printf("------------------------------------------------\n");
    }
    
    free(local_cur);
    free(local_next);
    
    MPI_Finalize();
    return 0;
}
