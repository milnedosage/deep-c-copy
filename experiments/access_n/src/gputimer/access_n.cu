#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define MILLI_TO_NANO 1000000

__global__
void access(float* A, long long n) {
    A[0] = A[n-1];
}

int main(int argc, char *argv[]) {
    // Track synchronous and asynchronous errors
    cudaError_t syncErr, asyncErr;

    // Get length of array
    long long n = strtol(argv[1], NULL, 10); // is long long to avoid overflow
    assert(n>=2);

    // Allocate memory, accessible by CPU and GPU
    float *A;
    cudaMallocManaged(&A, n * sizeof(float));
    syncErr = cudaGetLastError();
    if (syncErr != cudaSuccess) printf("syncErr: %s\n", cudaGetErrorString(syncErr));

    // Set last element of array
    float last = -84.845;
    A[n-1] = last;

    // gpu timer using cuda events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Time taken to run GPU kernel with 1 block and 1 thread in it
    cudaEventRecord(start);
    access<<<1, 1>>>(A, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    asyncErr = cudaDeviceSynchronize();
    if (asyncErr != cudaSuccess) printf("syncErr: %s\n", cudaGetErrorString(asyncErr));

    // Print timings
    float milliseconds = 0;
    double nanoseconds = 0;
    // cuda returns milliseconds we convert to nanoseconds
    cudaEventElapsedTime(&milliseconds, start, stop);
    nanoseconds = milliseconds * MILLI_TO_NANO;
    printf("%lld,%f\n", n, nanoseconds);

    // Check successfully changed first element
    assert(A[0] == last);

    // Free memory
    cudaFree(A);
}
