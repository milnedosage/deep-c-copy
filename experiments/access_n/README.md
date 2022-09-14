# access_n

access_n is an experiment aiming to find out how memory is loaded into the GPU. A graphical overview of the experiment is shown below:

![An overview of `access_n`](access_n_overview.png)

We dynamically allocate an array in global memory (memory accessible to both the CPU and GPU), and initialise the last four bytes of the array to a randomly picked (but constant) number. We then load the array onto the GPU, then time how long it takes for the program to copy the last four bytes of the array into the first four bytes of the array.

## Usage

**NOTE:**

* The max array size we used for A100 is `10000000000`, and for A6000, `12500000000`.

1. Run `chmod u+x fix_permission.sh && ./fix_permission.sh` to fix any potential permission issue.

2. Activate the virtual environment. `. ../../env/bin/activate`

3. Run the experiment and follow the instructions by running `./run_experiment.sh`.


## Discussion

Suppose we hypothesise that the array is contiguously loaded into the GPU cache from beginning to end. Let the size of the GPU cache be X MB.

If the size of the array is smaller than X MB, then it entirely fits within the GPU cache. If it is larger than X MB, then the end of the array is truncated and not loaded into the GPU cache.

We theorise that if our hypothesis is correct, we should see a sharp increase in access times when the size of the array crosses becomes larger than X MB. This is because the GPU must fetch the memory from a more distanced hardware component (e.g: RAM), so the access time to the end of the array dominates the rest of the access time.

## Technical Specifications
[Access N Specifications](AccessN_Specifications.md)
