#!/bin/bash

d=$(dirname "$0")

sudo apt update
sudo apt install build-essential

#sudo apt-key del 7fa2af80
#sudo apt-get install nvidia-cuda-toolkit --yes

chmod u+x $d/create_env.sh

$d/create_env.sh
