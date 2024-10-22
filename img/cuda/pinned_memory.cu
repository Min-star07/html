#include <cstdio>
#include <cuda_runtime.h>

// CUDA kernel for vector addition using pinned memory
__global__ void vectorAddPinned(int *c, int *a, int *b, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int N = 100;
    size_t size = N * sizeof(int);

    // Allocate pinned host memory
    int *h_a, *h_b, *h_c;
    cudaHostAlloc((void**)&h_a, size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_b, size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_c, size, cudaHostAllocDefault);

    // Initialize host arrays with values
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Device memory pointers
    int *d_a, *d_b, *d_c;

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data from pinned host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;  // Ensure N elements are processed
    vectorAddPinned<<<numBlocks, blockSize>>>(d_c, d_a, d_b, N);
    
    // Copy result from device to pinned host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free pinned host memory
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    return 0;
}
