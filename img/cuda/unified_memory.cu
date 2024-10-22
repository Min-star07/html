#include <cstdio>
#include <cuda_runtime.h>

// CUDA kernel for vector addition using unified memory
__global__ void vectorAddUnified(int *c, int *a, int *b, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int N = 100;
    size_t size = N * sizeof(int);

    // Unified memory allocation
    int *a, *b, *c;
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    // Initialize host arrays with values
    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;  // Ensure N elements are processed
    vectorAddUnified<<<numBlocks, blockSize>>>(c, a, b, N);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Print results
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // Free unified memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
