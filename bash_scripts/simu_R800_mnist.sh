#!/bin/bash

range=800
for val_K in 1 2 5 10 20
do
    python main_fed.py --dataset mnist --total_UE 500 --active_UE $val_K --model Mnist_oldMLP --iid dirichlet --round 1000 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "solve_opti_loss_size2"  --name "opti_R$range" --vanish 1 --seed 1 --wireless_seed 3 --cell_radius $range
done

for val_K in 1 2 5 10 20
do
    python main_fed.py --dataset mnist --total_UE 500 --active_UE $val_K --model Mnist_oldMLP --iid niid --round 1000 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "solve_opti_loss_size2"  --name "opti_R$range" --vanish 1 --seed 1 --wireless_seed 3 --cell_radius $range
done


for val_K in 1 2 5 10 20
do
    python main_fed.py --dataset mnist --total_UE 500 --active_UE $val_K --model Mnist_oldMLP --iid dirichlet --round 1000 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "weighted_random"  --name "opti_R$range" --vanish 1 --seed 1 --wireless_seed 3 --cell_radius $range
done

for val_K in 1 2 5 10 20
do
    python main_fed.py --dataset mnist --total_UE 500 --active_UE $val_K --model Mnist_oldMLP --iid niid --round 1000 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "weighted_random"  --name "WR_R$range" --vanish 1 --seed 1 --wireless_seed 3 --cell_radius $range
done

