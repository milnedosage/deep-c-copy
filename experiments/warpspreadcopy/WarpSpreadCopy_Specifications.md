WarpSpreadCopy Specifications
Chris: Yes your understanding of my request is solid

Has 1 parameter, N, the number of float items in the array

Use 1 Block with 1 warp ( 1 warp = 32 threads)

Assert that N is divisible by the number of threads, 32.

Constants
Block_size = 1
number_threads = 32
constant random float SEED value
Want SEED * 32 <= 255,  gives SEED < 7.96875
When reused for later experiments these constants may be changed

Overview

There will be N sized array that is uninitialised
by C++ default has random undefined data, but conceptually its empty or 0
Will initialise the 1st 32 entries to zero.
Will initialise the last entry of each of the 32 segments of the array with (i * SEED)
So it will be a sparse array, the 32 values are evenly distributed, spread across the array.

The test is to compact these values to the front of the array.
each ith thread will copy the last entry of its allocated segment into the ith entry in (the 1st segment) the array.

Example for 3 threads with N = 15
segment_size = 15/3 = 5
thread_start_pos values = 1, 6, 11 unused
thread_end_pos values = 5, 10, 15
kernel function will copy the values at positions 5, 10, 15 into positions 1,2,3.

Excluding initialisation and verification all experiment work done is a single kernel function.

Use Nvidia Cuda GPU time and report the time that kernel takes to complete.

After, an untimed verification test will check that that the 1st 32 entries have the expected (i * SEED) values
Important that verification is not included in the timing.

From Cuda slides, thread_step is when due to large N and small number of total threads, want each thread to do work on multiple segments.
Normally very good practice.
you calculate the index then do work then add thread_step and repeat while < N
ie thread 1 does segment 1 and 201 and 401, thread_step=200
This is not required in this experiment, each thread should only read from 1 allocated segment.
Do not use thread_steps.

One thread of the kernel will only do 1 copy operation.
The kernel overall will only do 32 copy operations.


This experiment was originally conceived to sum each initialised value into the 1st entry, but this added synchronisation delays,
this new idea is better for what we need timing memory movement.

Experimental Testing Details
Chris: Details of the test loops left to the team

Test with multiple N values
Start with minimum N value of 64
No maximum N value set
No step size set, but some multiple of 32.


Implementation Specific Technical Formulas
Chris: Derivation and verification of these formulas left to the team

Required memory_size
Define an N + 1 sized global array
Use Cuda Managed memory to be consistent with previous experiments
size = ((N + 1) * sizeof(float));
calculate segments and index positions
Multiple smaller arrays are not necessary
Using global memory, don't need to use shared memory.
Shared memory is a specific technical term for memory shared between threads of the same block.

An N + 1 sized array helps simplify the thread_end_pos formula
the 0 index is unused, the 1st position is index 1, the last position is N.
Since this experiment uses only the end of a segment
It may be slightly better for the end formula to be simpler

The length of a threads segment
segment_size = N / number_threads;

thread_start_pos = (i * segment_size) + 1;
thread_end_pos = ((i + 1) * segment_size);

Each thread accessing a separate segment of the global array
Specifically each thread only reading the end item of its allocated segment
called thread_end_pos and also the writing the ith entry in (the 1st segment) the array.

Is initialised using GPU in separate kernel (not a strict requirement)

Written after clarification discussion

Oliver suggested I do this task, Thanks Oliver
