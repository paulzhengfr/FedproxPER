#!/bin/bash

{
python main_fed.py --dataset cifar100 --total_UE 100 --active_UE 10 --model vgg --iid dirichlet --round 800 --gpu 0 --optimizer fedprox --mu 1 --lr 0.01 --selection weighted_random  --name 'hyperparam' --vanish 1 --seed 1 --wireless_seed 3 --scenario woPER
}&
{
python main_fed.py --dataset cifar100 --total_UE 100 --active_UE 10 --model vgg --iid dirichlet --round 800 --gpu 0 --optimizer fedprox --mu 1 --lr 0.004 --selection weighted_random  --name 'hyperparam' --vanish 1 --seed 1 --wireless_seed 3 --scenario woPER
}
wait
# python main_fed.py --dataset cifar100 --total_UE 100 --active_UE 10 --model vgg --iid dirichlet --round 800 --gpu 0 --optimizer fedprox --mu 1 --lr 0.0004 --selection weighted_random  --name 'hyperparam' --vanish 1 --seed 1 --wireless_seed 3 --scenario woPER
