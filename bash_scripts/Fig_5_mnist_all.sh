#!/bin/bash

# limit the number of parallel process
function max4 {
   while [ `jobs | wc -l` -ge 5 ] 
   do
      wait -n
   done
}

model=Mnist_oldMLP
#model=LeNet
lr=0.1
UE="500"
nb_rounds_d=1000
nb_rounds_n=1000
for seed_id in 0 1 2
do
    seed_oth=$((3-seed_id))
    seed_wireless=$((1+seed_id))
    # echo $seed_oth
    # echo $seed_wireless

for local_UE in 1 2 5 10 20 30
do 
for cell_R in 600 800 1000
do
# dirichlet WR
{
    python3 main_fed.py --dataset "mnist" --total_UE $UE --active_UE $local_UE --model $model \
     --iid "dirichlet" --round $nb_rounds_d --gpu 0 --optimizer "fedprox" --mu 1 --lr $lr --selection "weighted_random" \
     --name "dWR_lUE_${local_UE}_cellR_${cell_R}" --Pname "dWR_lUE_${local_UE}_cellR_${cell_R}_s_${seed_id}" --seed $seed_oth --wireless_seed $seed_wireless \
     --cell_radius $cell_R
}& max4;{ 
    # niid WR
    python3 main_fed.py --dataset "mnist" --total_UE $UE --active_UE $local_UE --model $model \
     --iid "niid" --round $nb_rounds_n --gpu 0 --optimizer "fedprox" --mu 1 --lr $lr --selection "weighted_random"  \
     --name "nWR_lUE_${local_UE}_cellR_${cell_R}" --Pname "nWR_lUE_${local_UE}_cellR_${cell_R}_s_${seed_id}" --seed $seed_oth --wireless_seed $seed_wireless \
     --cell_radius $cell_R
}& max4;{
    # dirichlet our method
    python3 main_fed.py --dataset "mnist" --total_UE $UE --active_UE $local_UE --model $model \
     --iid "dirichlet" --round $nb_rounds_d --gpu 0 --optimizer fedprox --mu 1 --lr $lr --selection "solve_opti_loss_size2" \
     --name "dOp_lUE_${local_UE}_cellR_${cell_R}" --Pname "dOp_lUE_${local_UE}_cellR_${cell_R}_s_${seed_id}" --seed $seed_oth --wireless_seed $seed_wireless  \
     --cell_radius $cell_R
}& max4;{
    # dirichlet our method
    python3 main_fed.py --dataset "mnist" --total_UE $UE --active_UE $local_UE --model $model \
     --iid "niid" --round $nb_rounds_n --gpu 0 --optimizer fedprox --mu 1 --lr $lr --selection "solve_opti_loss_size2"  \
     --name "nOp_lUE_${local_UE}_cellR_${cell_R}" --Pname "nOp_lUE_${local_UE}_cellR_${cell_R}_s_${seed_id}" --seed $seed_oth --wireless_seed $seed_wireless  \
     --cell_radius $cell_R
	}& max4;
done 
done
done
# wait