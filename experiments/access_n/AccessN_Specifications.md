## AccessN Specifications

Has 1 parameter, N, the number of float items in the array

Use 1 Block with 1 thread

Constants
Block_size = 1
number_threads = 1
constant random float SEED value

### Overview

Required memory_size
There will be N sized array that is uninitialised
by C++ default has random undefined data, but conceptually its empty or 0
Will initialise the 1st entry to zero.
Will initialise the last entry of the array with SEED.

The test is for the thread to simply copy the last entry into the 1st entry of the array.

Excluding initialisation and verification all experiment work done is a single kernel function.

After, an untimed verification test will check that that the 1st entry has the expected SEED value
Important that verification is not included in the timing.

### Experimental Testing Details

Test with multiple N values
Start with minimum N value of 2
No maximum N value set
No step size set
