#!/bin/bash

function max4 {
   while [ `jobs | wc -l` -ge 5 ] # tune the number of parallel process
   do
      wait -n
   done
}

UE=500
local_UE=10

for data_distr in "label_separated-a" "label_separated-b"
do
for curvy in -10 0 10
do

{
    python3 main_fed.py --dataset "mnist" --total_UE $UE --active_UE $local_UE --model "Mnist_oldMLP" \
     --iid "niid" --round 1500 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "solve_opti_loss_size2"  \
     --Pname "nOp_M_${curvy}_${data_distr}" --seed 3 --wireless_seed 1 \
     --curvy $curvy --data_distr_scenarios $data_distr --eta_init 2
	}& max4; 
   
done
{
    python3 main_fed.py --dataset "mnist" --total_UE $UE --active_UE $local_UE --model "Mnist_oldMLP" \
     --iid "niid" --round 1500 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "weighted_random"  \
     --Pname "nWR_${data_distr}" --seed 3 --wireless_seed 1 \
     --data_distr_scenarios $data_distr
	}& max4; 
{
    python3 main_fed.py --dataset "mnist" --total_UE $UE --active_UE $local_UE --model "Mnist_oldMLP" \
     --iid "niid" --round 1500 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "solve_opti_loss_size2"  \
     --Pname "nOp_no_later_${data_distr}" --seed 3 --wireless_seed 1 \
     --data_distr_scenarios $data_distr --no_later --eta_init 2
	}& max4; 
done
wait