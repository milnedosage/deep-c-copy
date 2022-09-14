# USYD-07-A - Strong Memory Deep C (Advanced)

## Introduction

More and more companies around the world are investing more money into AI/DL applications. The Cerebras WSE-2 is currently the largest and fastest AI accelerator but also comes with a unit price of around $2,000,000 USD. Nvidia's A100 is a lot smaller but only $200,000 USD by comparsion. Our project aims to profile the memory access patterns in Nvidia's A100 and A6000 GPUs and discover how their drivers work underneath the surface. 
With insight into this low level knowledge, ML hardware specialists can use our research to optimise GPU memory access speeds to the hardware's absolute limits. Discovering how data gets loaded onto the GPU has profound impacts on training ML models, which heavily rely on the card's tensor cores to quickly map-reduce matrices. The discovery of an atomic speedup in memory access will lead to a significant increase in ML model training speeds. 
This will ultimately result in lower operating costs and compute times to execute algorithms on cheaper hardware as compared to more expensive hardware. In summary, this project aims to optimise memory access on lower-end hardware to replicate the results as if it was run on faster but more expensive hardware.

To conduct our research, we developed this codebase to automate the process of running experiments and performing IDA on the experimental data. This way, we can quickly convene to discuss the experimental results and iterate the next experiment without the need for extra human work.
We currently have two experiments which are "Access-N" and "WarpSpreadCopy". These experiments interact with the memory differently and aim to provide us with an understanding on how Nvidia's memory architecture works. For more information on each experiment, please read the README.md file of each individual experiment within the experiments folder. 

## Setup and Usage

1. You will need to install Nvidia CUDA Toolkit by following the instruction [here](https://developer.nvidia.com/cuda-downloads) if the instance that you are running on does not have tools such as `nvcc` available.

2. Run `chmod u+x setup.sh` 

3. Run `./setup.sh` once you have installed CUDA toolkit.

4. Ensure that you're on a Strong Compute Lambda Labs Ubuntu instance with an available A100 or an A6000. To check, run `nvidia-smi` to check the available GPU.

## Next steps

Go into the `experiments` directory and choose the experiment to run (only `access_n` for now) and follow the README in the applicable experiment.

