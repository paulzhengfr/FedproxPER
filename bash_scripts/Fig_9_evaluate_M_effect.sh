#!/bin/bash

function max4 {
   while [ `jobs | wc -l` -ge 3 ] # tune the number of parallel process
   do
      wait -n
   done
}

#model=Mnist_oldMLP
#model=LeNet

UE=500
local_UE=10

## For Cifar10
lr=0.01
## For MNIST
lr=0.1

for curvy in -1000 -10 0 10 1000
do
for seed_id in 1 2 3
do
{
    python3 main_fed.py --dataset "cifar" --total_UE $UE --active_UE $local_UE --model "vgg" \
     --iid "dirichlet" --round 400 --gpu 0 --optimizer fedprox --mu 1 --lr 0.01 --selection "solve_opti_loss_size2" \
     --Pname "dOp_M_${curvy}_${seed_id}" --seed $seed_id --wireless_seed $seed_id \
     --curvy $curvy
}& max4;{
    python3 main_fed.py --dataset "cifar" --total_UE $UE --active_UE $local_UE --model "vgg" \
     --iid "niid" --round 800 --gpu 0 --optimizer fedprox --mu 1 --lr 0.01 --selection "solve_opti_loss_size2"  \
     --Pname "nOp_M_${curvy}_${seed_id}" --seed $seed_id --wireless_seed $seed_id \
     --curvy $curvy
     	}& max4;
   #    {
   #  python3 main_fed.py --dataset "mnist" --total_UE $UE --active_UE $local_UE --model "Mnist_oldMLP" \
   #   --iid "dirichlet" --round 300 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "solve_opti_loss_size2" \
   #   --Pname "dOp_M_${curvy}_${seed_id}" --seed $seed_id --wireless_seed $seed_id \
   #   --curvy $curvy
   #    } &max4;
   #  {
   #  python3 main_fed.py --dataset "mnist" --total_UE $UE --active_UE $local_UE --model "Mnist_oldMLP" \
   #   --iid "niid" --round 600 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "solve_opti_loss_size2"  \
   #   --Pname "nOp_M_${curvy}_${seed_id}" --seed $seed_id --wireless_seed $seed_id \
   #   --curvy $curvy
	# }& max4; 
done 
done
wait