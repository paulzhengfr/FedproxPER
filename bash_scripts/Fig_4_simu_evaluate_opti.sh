#!/bin/bash



for K in 3 5 7 9
do

    python3 main_evaluate_solveopti.py --K $K --Sl 0.1 --Su 4.1 --nseed 20
    python3 main_evaluate_solveopti.py --K $K --Sl 1.1 --Su 3.1 --nseed 20
    python3 main_evaluate_solveopti.py --K $K --Sl 1.6 --Su 2.6 --nseed 20

done