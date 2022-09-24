#!/bin/bash


python main_fed.py --dataset shakespeare --total_UE 142 --active_UE 10 --model LSTMShakespeare --iid dirichlet --round 50 --gpu 0 --optimizer fedprox --mu 1 --lr 0.01 --selection weighted_random  --name 'hyperparam' --Pname 'mu1' --vanish 1 --seed 1 --wireless_seed 3 --scenario woPER

python main_fed.py --dataset shakespeare --total_UE 142 --active_UE 10 --model LSTMShakespeare --iid dirichlet --round 50 --gpu 0 --optimizer fedprox --mu 0.1 --lr 0.01 --selection weighted_random  --name 'hyperparam' --Pname 'mu0.1' --vanish 1 --seed 1 --wireless_seed 3 --scenario woPER

python main_fed.py --dataset shakespeare --total_UE 142 --active_UE 10 --model LSTMShakespeare --iid dirichlet --round 50 --gpu 0 --optimizer fedprox --mu 0.01 --lr 0.01 --selection weighted_random  --name 'hyperparam' --Pname 'mu0.01' --vanish 1 --seed 1 --wireless_seed 3 --scenario woPER

python main_fed.py --dataset shakespeare --total_UE 142 --active_UE 10 --model LSTMShakespeare --iid dirichlet --round 50 --gpu 0 --optimizer fedprox --mu 0.001 --lr 0.01 --selection weighted_random  --name 'hyperparam' --Pname 'mu0.001' --vanish 1 --seed 1 --wireless_seed 3 --scenario woPER

