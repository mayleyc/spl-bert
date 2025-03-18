#!/bin/bash
device=0
mkdir -p overparam_logs
for num_reps in 2 4 8
do
    for gates in 2 4 
    do
        for S in 2 4
        do
            CUDA_VISIBLE_DEVICES=2 python -u train_cub.py --dataset cub_mini --seed 0 --num_reps $num_reps --S $S --gates $gates --lr 1e-3 --batch-size 128 > overparam_logs/cub_mini.$num_reps.$gates.$S.txt
        done
    done
done
