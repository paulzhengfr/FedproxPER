#!/bin/bash
function max4 {
   while [ `jobs | wc -l` -ge 6 ] # tune the number of parallel process
   do
      wait -n
   done
}

for seed_id in 0 1 2
do
    seed_oth=$((3-seed_id))
    seed_wireless=$((1+seed_id))
{
python main_fed.py --dataset mnist --total_UE 500 --active_UE 10 --model Mnist_oldMLP --iid dirichlet --round 400 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "solve_opti_loss_size2"  --name 'opti_fc' --Pname "opti_s${seed_id}" --seed $seed_oth --wireless_seed $seed_wireless
}& max4 ; {
python main_fed.py --dataset mnist --total_UE 500 --active_UE 10 --model Mnist_oldMLP --iid dirichlet --round 400 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "weighted_random"  --name 'WR_fc' --Pname "Sal_WR_s${seed_id}" --seed $seed_oth --wireless_seed $seed_wireless
}& max4 ; {
python main_fed.py --dataset mnist --total_UE 500 --active_UE 10 --model Mnist_oldMLP --iid dirichlet --round 400 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "best_loss"  --name 'BL_fc' --Pname "BL_s${seed_id}" --seed 1 --seed $seed_oth --wireless_seed $seed_wireless
}& max4 ; {
python main_fed.py --dataset mnist --total_UE 500 --active_UE 10 --model Mnist_oldMLP --iid dirichlet --round 400 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "uni_random"  --name 'UR_fc' --Pname "UR_s${seed_id}" --seed $seed_oth --wireless_seed $seed_wireless
}& max4 ; {
python main_fed.py --dataset mnist --total_UE 500 --active_UE 10 --model Mnist_oldMLP --iid dirichlet --round 400 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "best_channel"  --name 'BC_fc' --Pname "BC_s${seed_id}" --seed $seed_oth --wireless_seed $seed_wireless
}& max4 ; {
python main_fed.py --dataset mnist --total_UE 500 --active_UE 10 --model Mnist_oldMLP --iid dirichlet --round 400 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "solve_opti_loss_size2"  --name 'opti_fc' --Pname "opti_s${seed_id}" --seed $seed_oth --wireless_seed $seed_wireless
}& max4 ; {
python main_fed.py --dataset mnist --total_UE 500 --active_UE 10 --model Mnist_oldMLP --iid dirichlet --round 400 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "weighted_random"  --name 'WR_fc' --Pname "Sal_WR_s${seed_id}" --seed $seed_oth --wireless_seed $seed_wireless
}& max4 ; {
    python main_fed.py --dataset mnist --total_UE 500 --active_UE 10 --model Mnist_oldMLP --iid dirichlet --round 400 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "best_loss"  --name 'BL_fc' --Pname "BL_s${seed_id}" --seed 1 --seed $seed_oth --wireless_seed $seed_wireless

}& max4 ; {
    python main_fed.py --dataset mnist --total_UE 500 --active_UE 10 --model Mnist_oldMLP --iid dirichlet --round 400 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "uni_random"  --name 'UR_fc' --Pname "UR_s${seed_id}" --seed $seed_oth --wireless_seed $seed_wireless
}& max4 ; {
python main_fed.py --dataset mnist --total_UE 500 --active_UE 10 --model Mnist_oldMLP --iid dirichlet --round 400 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "best_channel"  --name 'BC_fc' --Pname "BC_s${seed_id}" --seed $seed_oth --wireless_seed $seed_wireless
}

{
python main_fed.py --dataset mnist --total_UE 500 --active_UE 10 --model Mnist_oldMLP --iid niid --round 800 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "solve_opti_loss_size2"  --name 'opti_fc' --Pname "opti_s${seed_id}" --seed $seed_oth --wireless_seed $seed_wireless
}& max4 ; {
python main_fed.py --dataset mnist --total_UE 500 --active_UE 10 --model Mnist_oldMLP --iid niid --round 800 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "weighted_random"  --name 'WR_fc' --Pname "Sal_WR_s${seed_id}" --seed $seed_oth --wireless_seed $seed_wireless
}& max4 ; {
python main_fed.py --dataset mnist --total_UE 500 --active_UE 10 --model Mnist_oldMLP --iid niid --round 800 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "best_loss"  --name 'BL_fc' --Pname "BL_s${seed_id}" --seed 1 --seed $seed_oth --wireless_seed $seed_wireless
}& max4 ; {
python main_fed.py --dataset mnist --total_UE 500 --active_UE 10 --model Mnist_oldMLP --iid niid --round 800 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "uni_random"  --name 'UR_fc' --Pname "UR_s${seed_id}" --seed $seed_oth --wireless_seed $seed_wireless
}& max4 ; {
python main_fed.py --dataset mnist --total_UE 500 --active_UE 10 --model Mnist_oldMLP --iid niid --round 800 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "best_channel"  --name 'BC_fc' --Pname "BC_s${seed_id}" --seed $seed_oth --wireless_seed $seed_wireless
}& max4 ; {
python main_fed.py --dataset mnist --total_UE 500 --active_UE 10 --model Mnist_oldMLP --iid niid --round 800 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "solve_opti_loss_size2"  --name 'opti_fc' --Pname "opti_s${seed_id}" --seed $seed_oth --wireless_seed $seed_wireless
}& max4 ; {
python main_fed.py --dataset mnist --total_UE 500 --active_UE 10 --model Mnist_oldMLP --iid niid --round 800 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "weighted_random"  --name 'WR_fc' --Pname "Sal_WR_s${seed_id}" --seed $seed_oth --wireless_seed $seed_wireless
}& max4 ; {
    python main_fed.py --dataset mnist --total_UE 500 --active_UE 10 --model Mnist_oldMLP --iid niid --round 800 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "best_loss"  --name 'BL_fc' --Pname "BL_s${seed_id}" --seed 1 --seed $seed_oth --wireless_seed $seed_wireless

}& max4 ; {
    python main_fed.py --dataset mnist --total_UE 500 --active_UE 10 --model Mnist_oldMLP --iid niid --round 800 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "uni_random"  --name 'UR_fc' --Pname "UR_s${seed_id}" --seed $seed_oth --wireless_seed $seed_wireless
}& max4 ; {
python main_fed.py --dataset mnist --total_UE 500 --active_UE 10 --model Mnist_oldMLP --iid niid --round 800 --gpu 0 --optimizer fedprox --mu 1 --lr 0.1 --selection "best_channel"  --name 'BC_fc' --Pname "BC_s${seed_id}" --seed $seed_oth --wireless_seed $seed_wireless
}

done
wait