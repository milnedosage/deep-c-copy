// WarpSpreadCopy

// 1 param: n
// Given n the number of items
// bulids an n + 1 array
// also defined NUM_THREADS
// so calculate segment_size = n / NUM_THREADS
// initialises array with defined INIT_SEED value spread over array at the end of each segment
// the test is to copy the spread values into the 1st NUM_THREADS indexes in the 1st segment
// and to time how long it takes the kernel to complete
// initialisation is not included in the timing

// Note minimum size of N for compact to work is NUM_THREADS * 2
// in this case NUM_THREADS = 32 so minimum is 64
// when segment_size is < 32 then origianal spread values will be overwritten
// but as long as n >= 64 this should not be a problem
// it should be a small enough value to see the cache effects

// using n + 1 sized array so that the end of a segment is ((i + 1) * segment_size)
// the 0 index is unused, the n index is valid and is the last
// otherwise the end of a segment would be (((i + 1) * segment_size) - 1)
// the start of a segment is ((i * segment_size) + 1)
// would normally be (i * segment_size)
// but this experiment uses only the end of a segment
// so may be slightly better for the end formula to be simpler


#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
// #include <time.h>
// #include <math.h>

#define NUM_BLOCKS (1)
#define NUM_THREADS (32)

// random constant seed value
// for verification purposes
// const float seed = 3.257;
#define INIT_SEED (3.257)

#define MILLI_TO_NANO (1000000)


__global__
void zero(float* A, long long n, long long segment_size) {
  // zero the start and spread of the array
  int i = threadIdx.x + 1;
  if (i <= n) {
    A[i] = 0;
    A[i * segment_size] = 0;
  }
}

__global__
void init(float* A, long long n, long long segment_size, float seed) {
  // initalise the array spreading the seed

  // if n is too small such that segment_size < NUM_THREADS
  // then these i and this index below will overlap
  // meaning a larger i may errase an earlier seeded entry
  // so skip A[i] = 0; made separate zero kernel
  int i = threadIdx.x + 1;
  if (i <= n) {
    A[i * segment_size] = (i * seed);
  }
}

__global__
void compact(float* A, long long n, long long segment_size) {
  // compact the spread values to the front
  int i = threadIdx.x + 1;
  if (i <= n) {
    A[i] = A[i * segment_size];
  }
}

__global__
void verify(float* A, long long n, float seed) {
  // verify that the values were compacted
  // compare floats, check difference between the float values is small
  int i = threadIdx.x + 1;
  if (i <= n) {
    float dif = (A[i] - (i * seed));
    assert(dif < 0.001);
    assert(dif > -0.001);
  }
}

int main(int argc, char *argv[]) {
    // note cuda errors
    cudaError_t cudaErr;

    // read parm n
    long long n = strtol(argv[1], NULL, 10);
    // assert that n >= 64
    assert(n >= 64);
    // assert that n is divisible by NUM_THREADS
    assert((n % NUM_THREADS) == 0);

    int deviceId;
    cudaGetDevice(&deviceId);

    // Allocate memory, accessible by CPU and GPU
    float *A;
    int size = ((n + 1) * sizeof(float));
    cudaMallocManaged(&A, size);

    cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) printf("cudaErr: %s\n", cudaGetErrorString(cudaErr));

    // int numberOfSMs; // not needed yet
    // cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    // prefetch to GPU
    cudaMemPrefetchAsync(A, size, deviceId);

    long long segment_size = n / NUM_THREADS;
    // assert that segment_size >= 2
    // assert that segment_size >= NUM_THREADS
    // assert(segment_size >= 2);

    // zero the start and spread of the array
    zero<<<NUM_BLOCKS, NUM_THREADS>>>(A, n, segment_size);

    // initalise the array spreading the seed
    init<<<NUM_BLOCKS, NUM_THREADS>>>(A, n, segment_size, INIT_SEED);

    cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) printf("cudaErr: %s\n", cudaGetErrorString(cudaErr));

    // gpu timer using cuda events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ensure all memory is loaded zeroed and initialised
    cudaErr = cudaDeviceSynchronize();
    if (cudaErr != cudaSuccess) printf("cudaErr: %s\n", cudaGetErrorString(cudaErr));

    // Time the compact GPU kernel only
    cudaEventRecord(start);
    compact<<<NUM_BLOCKS, NUM_THREADS>>>(A, n, segment_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // verify that the values were compacted
    verify<<<NUM_BLOCKS, NUM_THREADS>>>(A, n, INIT_SEED);

    cudaErr = cudaDeviceSynchronize();
    if (cudaErr != cudaSuccess) printf("cudaErr: %s\n", cudaGetErrorString(cudaErr));

    // Print cuda event GPU time
    float milliseconds = 0;
    double nanoseconds = 0;

    // cuda returns milliseconds we convert to nanoseconds
    // to be consistent with other experiments
    cudaEventElapsedTime(&milliseconds, start, stop);
    nanoseconds = milliseconds * MILLI_TO_NANO;

    printf("%lld,%f\n", n, nanoseconds);

    // Free cuda events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free memory
    cudaFree(A);

    return 0;
}
