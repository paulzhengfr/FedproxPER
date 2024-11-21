#!/bin/bash

function max4 {
   while [ `jobs | wc -l` -ge 5 ] # tune the number of parallel process
   do
      wait -n
   done
}



UE=500
local_UE=10


## For MNIST
lr=0.1
dataset="mnist"
model="Mnist_oldMLP"

## For Cifar10
# lr=0.01
# dataset="cifar"
# model="vgg"

{
python3 main_fed.py --dataset $dataset --total_UE $UE --active_UE $local_UE --model $model \
    --iid "dirichlet" --round 300 --gpu 0 --optimizer fedprox --mu 1 --lr $lr --selection "solve_opti_AoU" \
    --Pname "dAoU_EVRYX" --seed 3 --wireless_seed 1 
}& max4 ; {
python3 main_fed.py --dataset $dataset --total_UE $UE --active_UE $local_UE --model $model \
    --iid "dirichlet" --round 300 --gpu 0 --optimizer fedprox --mu 1 --lr $lr --selection "solve_opti_AoU" \
    --Pname "dAoU_pure" --seed 3 --wireless_seed 1 --opti "cstWObj" --no_later
}& max4 ; {
python3 main_fed.py --dataset $dataset --total_UE $UE --active_UE $local_UE --model $model \
    --iid "dirichlet" --round 300 --gpu 0 --optimizer fedprox --mu 1 --lr $lr --selection "solve_opti_AoU" \
    --Pname "dAoU_nolater" --seed 3 --wireless_seed 1 --no_later
}& max4 ; {
python3 main_fed.py --dataset $dataset --total_UE $UE --active_UE $local_UE --model $model \
    --iid "dirichlet" --round 300 --gpu 0 --optimizer fedprox --mu 1 --lr $lr --selection "solve_opti_AoU" \
    --Pname "dAoU_nochannel" --seed 3 --wireless_seed 1 --opti "cstWObj"
}& max4 ; {

python3 main_fed.py --dataset $dataset --total_UE $UE --active_UE $local_UE --model $model \
    --iid "niid" --round 600 --gpu 0 --optimizer fedprox --mu 1 --lr $lr --selection "solve_opti_AoU" \
    --Pname "nAoU_EVRYX" --seed 3 --wireless_seed 1 
}& max4 ; {
python3 main_fed.py --dataset $dataset --total_UE $UE --active_UE $local_UE --model $model \
    --iid "niid" --round 600 --gpu 0 --optimizer fedprox --mu 1 --lr $lr --selection "solve_opti_AoU" \
    --Pname "nAoU_pure" --seed 3 --wireless_seed 1 --opti "cstWObj" --no_later
}& max4 ; {
python3 main_fed.py --dataset $dataset --total_UE $UE --active_UE $local_UE --model $model \
    --iid "niid" --round 600 --gpu 0 --optimizer fedprox --mu 1 --lr $lr --selection "solve_opti_AoU" \
    --Pname "nAoU_nolater" --seed 3 --wireless_seed 1 --no_later
}& max4 ; {
python3 main_fed.py --dataset $dataset --total_UE $UE --active_UE $local_UE --model $model \
    --iid "niid" --round 600 --gpu 0 --optimizer fedprox --mu 1 --lr $lr --selection "solve_opti_AoU" \
    --Pname "nAoU_nochannel" --seed 3 --wireless_seed 1 --opti "cstWObj"
}
wait

