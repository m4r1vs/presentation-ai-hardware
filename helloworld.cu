#include <cuda_runtime.h>
#include <iostream>

// Matrix size
#define N 100

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float *A, float *B, float *C, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    float value = 0.0f;
    for (int k = 0; k < n; k++) {
      value += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = value;
  }
}

int main() {
  // Matrix dimensions
  int size = N * N * sizeof(float);

  // Allocate memory for host matrices
  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);

  // Initialize host matrices with random values
  for (int i = 0; i < N * N; i++) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Allocate memory for device matrices
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  // Copy host matrices to device
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Define block and grid dimensions
  dim3 blockDim(16, 16);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
               (N + blockDim.y - 1) / blockDim.y);

  // Number of iterations
  int iterations = 1000000; // Change this value to run the multiplication multiple times

  // Perform matrix multiplication n times
  for (int i = 0; i < iterations; i++) {
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); // Ensure kernel finishes before the next iteration
  }

  // Copy result back to host
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // Print a portion of the result for verification
  std::cout << "Result matrix (partial):" << std::endl;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      std::cout << h_C[i * N + j] << " ";
    }
    std::cout << std::endl;
  }

  // Free device and host memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
