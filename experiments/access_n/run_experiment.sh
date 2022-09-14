#!/bin/sh

EXPERIMENTS_AVAILABLE=$(ls -l src | grep '^d' | cut -d ' ' -f 9 | xargs)
echo "Choose experiment to run. Options: $EXPERIMENTS_AVAILABLE"
read EXPERIMENT

set -e
cd "src/$EXPERIMENT"

echo "GPU/s available:"
nvidia-smi -L
echo "Choose GPU name for results directory. Options: A6000 A100 RTX3090"
read GPU

if [ $GPU != "A6000" ] && [ $GPU != "A100" ] && [ $GPU != "RTX3090" ]
then
  echo 'Invalid GPU specified'
  exit 1
fi

echo "Enter max array size to run experiment on (MAX_ARR_SZ = (Amount of VRAM in GB) * 1,000,000,000 / 4)."
read MAX_ARR_SZ

if [ $MAX_ARR_SZ -lt 2 ] # The value is from the assert statement in the code
then
  echo 'Invalid max array size. Must be at least than 2'
  exit 1
fi

./compile.sh
echo "Running experiment... This might take a while..."
./run_access_n.sh "$MAX_ARR_SZ" > "../../results/$GPU/$EXPERIMENT.csv"
./run_access_n_b2b.sh "$MAX_ARR_SZ" > "../../results/$GPU/$EXPERIMENT""_b2b.csv"
jupyter notebook ../../analysis/analysis.ipynb --ip=0.0.0.0
