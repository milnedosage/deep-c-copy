# warpspreadcopy

WarpSpreadCopy is an experiment to run on 1 block and 1 warp (a warp is 32 threads)

num_threads = 32 was specified

It accepts 1 parameter N the size of the float array required to test with.

usage:
warpspreadcopy <N>

example:
warpspreadcopy 64000000
note the parameter is positional not named

validity rules for N:
must be divisible by num_threads (=32)
minimum is num_threads * 2 (=64)

there are assertions enforcing these rules,
it is designed and documented with these rules.

Want N within the memory capacity (/4) of the GPU
since a float uses 4 bytes

Given N it calculates
segment_size = N / num_threads
a segment is a logical division of the array into equal pieces
1 segment for each thread.

The task of this experiment is to create the array of size N floats
consider the array is conceptually empty, doesn't need each entry to be explicitly zero.
initialise the last entry of each segment of the array with a known value.

Each (i) thread to copy the last entry of its segment of the array into the (i) entry at the start of the array.

It will report how long it took to complete.

It will verify that values at the start of the array are as expected after the copy.

A trivial example to demonstrate
Example for 3 threads with N = 15
Array entries (1 to 15)
segment_size = 15/3 = 5
thread_start_pos values = (1, 6, 11) unused
thread_end_pos values = (5, 10, 15)
kernel function will copy the values at positions 5, 10, 15 into positions 1,2,3 respectively.


There is bounds checking that i <= N before accessing i
i should be < N as i is < nun_threads
Note that entries 1 to N are valid

Not checking that i * segment_size is < N
as the point is for the test to be as fast as possible
and should not be necessary given stated rules.

for technical specifications see [specifcations](./WarpSpreadCopy_Specifications.md)
This readme is simplified from the technical specifications
Where this readme appears different then reference the technical specifications

## Usage

IDEALLY, THIS SHOULD JUST BE RUNNING A BASH SCRIPT.

```
./run_experiment.sh
```

But right now, we need to go into the source directories and manually run the python scripts and manually open up the Jupyter notebook. 

FINISH THIS

## Discussion

*FROM ACCESS N, THIS NEEDS TO BE REWRITTEN FOR WARPSPREADCOPY*

Suppose we hypothesise that the array is contiguously loaded into the GPU cache from beginning to end. Let the size of the GPU cache be X MB.

If the size of the array is smaller than X MB, then it entirely fits within the GPU cache. If it is larger than X MB, then the end of the array is truncated and not loaded into the GPU cache.

We theorise that if our hypothesis is correct, we should see a sharp increase in access times when the size of the array crosses becomes larger than X MB. This is because the GPU must fetch the memory from a more distanced hardware component (e.g: RAM), so the access time to the end of the array dominates the rest of the access time.