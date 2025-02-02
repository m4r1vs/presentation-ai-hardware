#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// Matrix size and number of threads
#define N 1000
#define NUM_THREADS 1

// Structure to pass arguments to each thread
typedef struct {
  int start_row;
  int end_row;
  float *A;
  float *B;
  float *C;
  int n;
} ThreadArgs;

// Thread function to compute a subset of rows in matrix C
void *matrixMulThread(void *args) {
  ThreadArgs *ta = (ThreadArgs *)args;
  for (int row = ta->start_row; row <= ta->end_row; ++row) {
    for (int col = 0; col < ta->n; ++col) {
      float value = 0.0f;
      for (int k = 0; k < ta->n; ++k) {
        value += ta->A[row * ta->n + k] * ta->B[k * ta->n + col];
      }
      ta->C[row * ta->n + col] = value;
    }
  }
  return NULL;
}

int main() {
  size_t size = N * N * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);

  // Initialize matrices with random values
  for (size_t i = 0; i < N * N; ++i) {
    h_A[i] = (float)(rand()) / RAND_MAX;
    h_B[i] = (float)(rand()) / RAND_MAX;
  }

  int iterations = 100;

  for (int iter = 0; iter < iterations; ++iter) {
    pthread_t threads[NUM_THREADS];
    ThreadArgs args[NUM_THREADS];

    int rows_per_thread = N / NUM_THREADS;
    int remainder = N % NUM_THREADS;
    int current_row = 0;

    // Distribute rows among threads, accounting for remainder
    for (int i = 0; i < NUM_THREADS; ++i) {
      int chunk = rows_per_thread + (i < remainder ? 1 : 0);
      args[i].start_row = current_row;
      args[i].end_row = current_row + chunk - 1;
      args[i].A = h_A;
      args[i].B = h_B;
      args[i].C = h_C;
      args[i].n = N;

      current_row += chunk;

      pthread_create(&threads[i], NULL, matrixMulThread, &args[i]);
    }

    // Wait for all threads to complete
    for (int i = 0; i < NUM_THREADS; ++i) {
      pthread_join(threads[i], NULL);
    }
  }

  // Print a portion of the result for verification
  printf("Result matrix (partial):\n");
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      printf("%f ", h_C[i * N + j]);
    }
    printf("\n");
  }

  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
