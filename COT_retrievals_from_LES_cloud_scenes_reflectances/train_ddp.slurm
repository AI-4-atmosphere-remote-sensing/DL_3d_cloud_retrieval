#!/bin/bash
#SBATCH --job-name=4cot
#SBATCH --output=4slurm.out
#SBATCH --error=4slurm.err
#SBATCH --partition=gpu2018
#SBATCH --qos=high_mem
#SBATCH --time=25:30:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4

nvidia-smi

export CUDA_LAUNCH_BLOCKING=1

while true;
do nvidia-smi --query-gpu=timestamp,name,pci.bus_id,pstate,
driver_version,temperature.gpu,utilization.gpu,
utilization.memory,memory.total,memory.free,
memory.used --format=csv >> gpu_utillization.log; sleep 1;
done

torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py  --model_name okamura --batch_size 128 --lr 0.01 --world_size 4

