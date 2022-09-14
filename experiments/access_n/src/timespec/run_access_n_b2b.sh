#!/bin/bash

MIN_ARGC=1

if [ $# -lt $MIN_ARGC ]
then
  echo "Usage: $0 [max_array_size]"
  exit 1
fi

# Getting compile environment
GIT_HASH=$(cat .COMPILE_ENV)
if [ $? -ne 0 ]
then
  GIT_HASH='Unavailable'
fi

# Crash if there is any errors
set -e
# Setup variables
NUM_SAMPLES=30
NUM_ITERS=100
MAX_ARRAY_SIZE=$1
STEP_SIZE=$(( $MAX_ARRAY_SIZE/$NUM_ITERS ))

# Add addtional info
echo "git_hash=$GIT_HASH,max_arr_sz=$MAX_ARRAY_SIZE"
# add csv header
echo "n,duration_ns"

# sample 5 times
for i in $( eval echo {2..$MAX_ARRAY_SIZE..$STEP_SIZE} )
do
	for _ in $( eval echo {1..$NUM_SAMPLES} )
	do
	    ./access_n.exe $i
	done
done
