#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define S_TO_NSEC 1000000000

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

    // Time taken to run GPU kernel with 1 block and 1 thread in it
    struct timespec start;
    struct timespec end;

    clock_gettime(CLOCK_MONOTONIC, &start);
    access<<<1, 1>>>(A, n);
    asyncErr = cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    if (asyncErr != cudaSuccess) printf("syncErr: %s\n", cudaGetErrorString(asyncErr));

    // Print timings
    time_t sec = end.tv_sec - start.tv_sec;
    long nsec = end.tv_nsec - start.tv_nsec;
    printf("%lld,%ld\n", n, sec * S_TO_NSEC + nsec);

    // Check successfully changed first element
    assert(A[0] == last);

    // Free memory
    cudaFree(A);
}
