# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:27:02 2021

@author: Paul Zheng
"""
import time
import numpy as np

from selections.function_LR import opti_LR
from selections.function_user_selection import user_selection_opti
from selections.function_user_selection_allCombinations import choose_comb, solve_opti_all, solve_opti_wrt_P
import argparse

# %%
parser = argparse.ArgumentParser(description='Evaluate solve opti')
parser.add_argument('--K',
                    default=3,
                    type=int,
                    help='parameters for non IID data distribution')

parser.add_argument('--Sl',
                    default=1.1,
                    type=float,
                    help='parameters for non IID data distribution')
parser.add_argument('--Su',
                    default=3.1,
                    type=float,
                    help='parameters for non IID data distribution')
parser.add_argument('--nseed',
                    default=5,
                    type=int,
                    help='parameters for non IID data distribution')
args = parser.parse_args()

print(f"Simulations with K {args.K}, Sl {args.Sl}")

# %% initialization
def dB2power(x):
    return np.exp(x / 10 * np.log(10))


N = 10
K = args.K
wireless_arg = {
    'radius': 1000,
    'ampli': 15,
    'N0': dB2power(-150),
    'B': 1e6,
    'm': dB2power(0.023),
    'M': 16,
    'Mprime': 15,
    'E_max': 60,  # mJ
    'Tslot': 1.3,
    'sigma': 1,  # 1
    'freq': 2400,  # Mhz
    'P_max': 10,  # mW
    'alpha': 0.1,
    'beta': 0.001,
    'kappa':10**(-28),
    'freq_comp': 2*10**9,
    'C': 2*10**4
}

P_max = wireless_arg['P_max']
E_max = wireless_arg['E_max']
T = wireless_arg['Tslot']
P_sum = E_max / T

const_alpha = wireless_arg['N0'] * wireless_arg['B'] / wireless_arg['m']
wireless_arg['theta'] = wireless_arg['kappa']*wireless_arg['freq_comp']**2 *wireless_arg['C']*20 *60000
def f(x, alpha, h_i, S_i, data_size, const, later_weights,P_max, incr,decr):
    vec = S_i * np.exp(-alpha / h_i / x) + later_weights
    return np.sum(vec[x>0])

def verify_constraints(sol, K, P_max,alpha, h_i, P_sum, const, data_size):
    verified = True
    if np.sum(sol>0) > K:
        verified = False
        print("K not feasible")
    if np.sum(sol>P_max) >= 1:
        verified = False
        print("P_max not feasible")
    Pth = alpha/h_i/2
    if np.sum(sol[sol>0]< Pth[sol>0]) >= 1:
        print("P_th not feasible")
        verified =False
    if np.sum(sol[sol>0] + data_size[sol>0]*const)>P_sum:
        verified = False
        print("P_sum not feasible")
    return verified


# %%
def generate_random_S_and_h(seed, N, wireless_arg, args):
    np.random.seed(seed)
    wireless_arg['distance'] = np.sqrt(np.random.uniform(1, wireless_arg['radius'] ** 2, N))
    FSPL = 20 * np.log10(wireless_arg['distance']) + 20 * np.log10(wireless_arg['freq']) - 27.55
    wireless_arg['FSPL'] = dB2power(FSPL)

    wireless_arg['P_sum'] = wireless_arg['E_max'] / wireless_arg['Tslot']
    o_avg = wireless_arg['sigma'] * 2  # Rayleigh distribution mean
    wireless_arg['h_avg'] = o_avg / wireless_arg['FSPL']
    h_i = wireless_arg['h_avg']

    S_i = np.random.uniform(low=args.Sl, high=args.Su, size=N)
    return h_i, S_i


def evaluate(Nvec, K, nb_seed, wireless_arg, P_max, P_sum, args):
    P_max = wireless_arg['P_max']
    E_max = wireless_arg['E_max']
    T = wireless_arg['Tslot']
    P_sum = E_max / T

    const_alpha = wireless_arg['N0'] * wireless_arg['B'] / wireless_arg['m']

    Nmax = len(Nvec)
    vecLR = np.zeros((nb_seed, Nmax))
    vecTrue = np.zeros((nb_seed, Nmax))
    durationLR = np.zeros((nb_seed, Nmax))
    durationTrue = np.zeros((nb_seed, Nmax))
    
    Tslot = 1.3
    theta = wireless_arg['kappa']*wireless_arg['freq_comp']**2 *wireless_arg['C']*20 *60000
    const = theta / Tslot
    for idN, N in enumerate(Nvec):
        np.random.seed(idN)
        data_size = np.random.power(4, N)
        data_size = data_size / np.sum(data_size)
        later_weights = np.random.rand(N)
        list2choose = list(range(N))
        all_comb = choose_comb(list2choose, K)
        if K > N:
            all_comb = choose_comb(list2choose, N)
        for seed in range(nb_seed):
            h_i, S_i = generate_random_S_and_h(seed, N, wireless_arg, args)
            K_effective = K
            # True computation
            if K > N:
                K_effective = N
                # active_clients, power_active, run_time = solve_opti_wrt_P(weights_uni, h_avg_p, N0,B, m, args.active_UE, wireless_arg['alpha']  , wireless_arg['beta'] , P_max,P_sum, 'P1',data_size_p, theta, Tslot, later_weights_p)
            active_clients, power_active, run_time = solve_opti_all(all_comb, list2choose, S_i, h_i,
                                                                    wireless_arg['N0'], wireless_arg['B'],
                                                                    wireless_arg['m'], K_effective, wireless_arg['alpha'],
                                                                    wireless_arg['beta'], P_max, P_sum, 'LR',data_size, theta, Tslot, later_weights) # 'test_convex_ext'
            P_true = np.zeros(len(h_i))
            P_true[active_clients] = power_active
            true_opt = f(P_true, const_alpha, h_i, S_i, data_size, const, later_weights,P_max,0,0)
            if not verify_constraints(P_true, K_effective, P_max,const_alpha, h_i, P_sum, const, data_size):
                print(f"For N {N} seed {seed}, the naif method solution isn't feasible")
            vecTrue[seed, idN] = true_opt
            durationTrue[seed, idN] = run_time

            # LR computation
            start = time.time()
            _, P_LR,_ ,_= opti_LR(K_effective, const_alpha, h_i, S_i, P_max, P_sum, data_size,const,
                                                 later_weights, 1,1)
            if not verify_constraints(P_true, K_effective, P_max,const_alpha, h_i, P_sum, const, data_size):
                print(f"For N {N} seed {seed}, the LR method solution isn't feasible")
            end = time.time()
            LR_opt = f(P_LR, const_alpha, h_i, S_i, data_size, const, later_weights,P_max,0,0)
            
            vecLR[seed, idN] = LR_opt
            durationLR[seed, idN] = end - start
            
            
    print("-----------True Results---------------")
    print(vecTrue)
    print("------------LR Results-----------------")
    print(vecLR)
   
    return vecLR, durationLR, vecTrue, durationTrue

Nvec = [3,4, 5,7,9, 10,12,15]
# Nvec = [3,4, 5,7,9, 10,12,15]
# Nvec = [3,5,7,10]
# Nvec = [3,  5,  8]
nb_seed = args.nseed
res = evaluate(Nvec, K, nb_seed, wireless_arg, P_max, P_sum, args)
file_name = 'python_res/res_K{}_{}_{}_compl3.npy'.format(K, args.Sl, args.Su)
with open(file_name, 'wb') as f:
    np.save(f, res[0])
    np.save(f, res[1])
    np.save(f, res[2])
    np.save(f, res[3])

