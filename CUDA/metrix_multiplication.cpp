// %%writefile asif.cu  // Uncomment this line when using Google Colab to save file

#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// CUDA kernel to perform matrix multiplication
// Each thread computes matrix multiplication for one pair of matrices
__global__ void matrixMultiply(float* d_A, float* d_B, float* d_C, int M, int N, int P, int offset) {
    int matrixIdx = threadIdx.x + offset;  // Calculate which matrix to process

    // Pointers to individual matrices
    float* a = d_A + matrixIdx * M * N;  // Pointer to A[matrixIdx]
    float* b = d_B + matrixIdx * N * P;  // Pointer to B[matrixIdx]
    float* c = d_C + matrixIdx * M * P;  // Pointer to C[matrixIdx]

    // Standard matrix multiplication: C = A * B
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int l = 0; l < P; l++) {
                // Row-Major addressing: c[i][l] += a[i][j] * b[j][l]
                c[i * P + l] += a[i * N + j] * b[j * P + l];
            }
        }
    }
}

int main(int argc, char* argv[]) {

    // Read arguments from command line
    int maxThreadsPerBatch = atoi(argv[1]); // Maximum number of threads (how many matrices to multiply at once)
    int totalMatrices = atoi(argv[2]);       // Total number of matrix multiplications to perform

    // Dimensions for matrices
    int M = 400, N = 400, P = 400; // A[M][N], B[N][P], result C[M][P]

    // Calculate sizes for memory allocations
    int sizeA = M * N * totalMatrices; // Total size for all matrices A
    int sizeB = N * P * totalMatrices; // Total size for all matrices B
    int sizeC = M * P * totalMatrices; // Total size for all result matrices C

    // Allocate host (CPU) memory
    float* h_A = new float[sizeA];
    float* h_B = new float[sizeB];
    float* h_C = new float[sizeC];

    // Allocate device (GPU) memory
    float* d_A;
    cudaMalloc(&d_A, sizeA * sizeof(float));
    float* d_B;
    cudaMalloc(&d_B, sizeB * sizeof(float));
    float* d_C;
    cudaMalloc(&d_C, sizeC * sizeof(float));

    // Initialize matrices A and B with random data
    for (int i = 0; i < sizeA; i++) {
        h_A[i] = rand();  // Fill A with random values
    }
    for (int i = 0; i < sizeB; i++) {
        h_B[i] = rand();  // Fill B with random values
    }

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB * sizeof(float), cudaMemcpyHostToDevice);

    // Launch CUDA kernel in batches
    int matricesRemaining = totalMatrices; // How many matrices left to multiply
    int offset = 0;                        // Offset for next batch

    while (matricesRemaining > 0) {
        int batchSize = min(matricesRemaining, maxThreadsPerBatch);

        // Launch kernel: 1 block, batchSize threads
        matrixMultiply<<<1, batchSize>>>(d_A, d_B, d_C, M, N, P, offset);
        cudaDeviceSynchronize(); // Wait for GPU to finish

        matricesRemaining -= batchSize;  // Decrease remaining matrices
        offset += batchSize;             // Move offset
    }

    // Copy result matrices back from device to host
    cudaMemcpy(h_C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "All operations completed successfully." << endl;

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
