#!/bin/bash

range=800
for val_K in 1 2 5 10 20
do
    python main_fed.py --dataset cifar --total_UE 100 --active_UE $val_K --model vgg --iid dirichlet --round 1000 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "solve_opti_loss_size2"  --name "opti_R$range" --vanish 1 --seed 1 --wireless_seed 3 --later_weights_coef 10 --weight_normalization median --cell_radius $range
done

#range=800
#for val_K in 1 2 5 10 20
#do
 #   python main_fed.py --dataset cifar --total_UE 100 --active_UE $val_K --model vgg --iid dirichlet --round 1000 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "solve_opti_loss_size2"  --name "opti_R$range" --Pname "opti2_R${range}_K${val_K}" --vanish 1 --seed 1 --wireless_seed 3 --later_weights_coef 10 --cell_radius $range
#done
