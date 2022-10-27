#!/bin/bash
device=$1
model=$2
for batch in 128
do
    for lr in 0.1 0.01
    do
        echo "Training with Learning Rate $lr now"
        python main.py --device $device --model_name $model --batch_size $batch --lr $lr
    done
done