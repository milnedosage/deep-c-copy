# run warpspreadcopy

# Note that minimum n test value is 64
# and Step size should be divisible by num_threads (ie 32)

# calclate Step size that is divisible by num_threads
# and ensure is at least 32 not 0

# a step of 0 would result in an infinite loop
# from test seems that bash detects step 0 and sets step to 1

# THought that eval not needed in for loop as it unnecessaily expands {1..7..2} into 1 3 5 7
# But from testing it really didn't work without it, need it or a better method

# Crash if there is any errors
set -e

# Compile
make > /dev/null

# Setup variables
NUM_SAMPLES=30
NUM_ITERS=100
MAX_ARRAY_SIZE=$1
NUM_THREADS=32
MIN_SIZE=64

STEP_SIZE1=$((MAX_ARRAY_SIZE / NUM_ITERS))
STEP_SIZEM=$((STEP_SIZE1 % NUM_THREADS))
STEP_SIZE=$((STEP_SIZE1 - STEP_SIZEM))
if [[ $STEP_SIZE == 0 ]]; then
  STEP_SIZE=$((NUM_THREADS))
fi

# add csv header
echo "n,duration_ns"

# sample 5 times
for s in $(eval echo {1..$NUM_SAMPLES})
do
	for i in $(eval echo {$MIN_SIZE..$MAX_ARRAY_SIZE..$STEP_SIZE})
	do
	    ./warpspreadcopy.exe $i
	done
done
