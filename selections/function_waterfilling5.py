# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 11:07:13 2021

@author: Paul Zheng
"""

# Waterfilling for Problem with sum of three terms
import numpy as np
from selections.function_user_selection_allCombinations import solve_opti_wrt_P
import copy
#%% initialization
# def dB2power(x):
#     return np.exp(x/10*np.log(10))

# N = 100
# K = 10
# wireless_arg = {
#     'radius':1000,
#     'ampli': 15,
#     'N0':dB2power(-150),
#     'B': 1e6,
#     'm': dB2power(0.023),
#     'M' : 16,
#     'Mprime': 15,
#     'E_max':60, #mJ
#     'Tslot': 1.3,
#     'sigma':1, #1
#     'freq' : 2400, #Mhz
#     'P_max': 10, #mW
#     'alpha':0.1,
#     'beta': 0.001,
#     'kappa':10**(-28),
#     'freq_comp': 2*10**9,
#     'C': 2*10**4
#     }

# data_size = np.ones(N) * 600000/N
# data_size[16] = 16000
# data_size[10] = 2000
# data_size[8] = 2000
# wireless_arg['theta'] = wireless_arg['kappa']*wireless_arg['freq_comp']**2 *wireless_arg['C']*20
# seed = 3
# np.random.seed(seed)
# wireless_arg['distance']  = np.sqrt(np.random.uniform(1, wireless_arg['radius']  ** 2, N))
# FSPL = 20 * np.log10(wireless_arg['distance']) + 20 * np.log10(wireless_arg['freq']) - 27.55
# wireless_arg['FSPL'] = dB2power(FSPL)

# wireless_arg['P_sum'] = wireless_arg['E_max'] / wireless_arg['Tslot']
# o_avg = wireless_arg['sigma']   * 2 # Rayleigh distribution mean
# wireless_arg['h_avg'] = o_avg  / wireless_arg['FSPL']
# training_round = 0
# np.random.seed(seed  + 12345 + training_round)
# o_i = wireless_arg['sigma']   * (np.square(np.random.randn(N)) + np.square(np.random.randn(N))) # Rayleigh distribution
# h_i = o_i  / wireless_arg['FSPL']
# h_i = wireless_arg['h_avg']
# # h_i = np.ones(N)*np.mean(h_i)


# # S_i = np.random.randint(0, high=1000, size=N) / 10
# S_i = np.ones(N) * 0.01
# P_max = wireless_arg['P_max']
# E_max =  wireless_arg['E_max']
# T = wireless_arg['Tslot']
# P_sum = E_max / T


# const_alpha = wireless_arg['N0'] * wireless_arg['B']
# later_weights = np.ones(N)
# later_weights[1:30] =10
# # later_weights[96] = 15
# const = wireless_arg['theta'] / wireless_arg['Tslot']
#%% Define functions


def f(x, alpha, h_i, S_i, data_size, const, later_weights,P_max, incr,decr):
    return np.sum(S_i * np.exp(-alpha / h_i / (x-const*data_size)) + (x-const*data_size)/P_max *10* later_weights)
def f_val(alpha, h_i, S_i, data_size, const, later_weights,P_max, incr,decr):
    return S_i * np.exp(-alpha / h_i / (P_max-const*data_size)) + later_weights

def fgrad(x, alpha, h_i, S_i, data_size, const, later_weights,P_max, incr,decr):
    Pth = alpha / 2 / h_i
    Qth = Pth + const*data_size
    const_pts = S_i  *alpha / h_i / np.square(Pth) * np.exp(-alpha/h_i/Pth)
    if isinstance(x, int) or isinstance(x, float) or (len(x) == 1 and len(h_i) == 1) :
        if x < Qth:
            return (x - Qth)**2 + const_pts+1/P_max * later_weights
        return S_i  *alpha / h_i / (x-const*data_size)**2 * np.exp(-alpha/h_i/(x-const*data_size))+10/P_max * later_weights
    val = (S_i  *alpha / h_i / np.square(x-const*data_size) * np.exp(-alpha/h_i/(x-const*data_size))+
           10/P_max * later_weights)
    val[x<Qth] = np.square(x[x<Qth] - Pth[x<Qth]) + const_pts[x<Qth]+ 10/P_max * later_weights[x<Qth]
    return val

def fgrad_nolater(x, alpha, h_i, S_i, data_size, const, later_weights,P_max, incr,decr):
    Pth = alpha / 2 / h_i
    Qth = Pth + const*data_size
    const_pts = S_i  *alpha / h_i / np.square(Pth) * np.exp(-alpha/h_i/Pth)
    if isinstance(x, int) or isinstance(x, float) or (len(x) == 1 and len(h_i) == 1) :
        if x < Qth:
            return (x - Qth)**2 + const_pts+1/P_max * later_weights
        return S_i  *alpha / h_i / (x-const*data_size)**2 * np.exp(-alpha/h_i/(x-const*data_size))
    val = (S_i  *alpha / h_i / np.square(x-const*data_size) * np.exp(-alpha/h_i/(x-const*data_size)))
    val[x<Qth] = np.square(x[x<Qth] - Pth[x<Qth]) + const_pts[x<Qth]
    return val
# x1 =  np.ones(N)#np.square(np.random.randn(N))
# f(x1, const_alpha, h_i,S_i)

#%% Utility
def find_maxi_fderivatives(alpha, h_i, S_i, data_size, const, later_weights,P_max, incr,decr):
    Pth = alpha / 2 / h_i
    Qth = Pth + const*data_size
    return (Qth, fgrad(Qth, alpha, h_i,S_i, data_size, const, later_weights,P_max, incr,decr), fgrad_nolater(Qth, alpha, h_i,S_i, data_size, const, later_weights,P_max, incr,decr))


# def f(x, alpha, h_i, S_i, data_size, const, later_weights,P_max, incr,decr):
#     return np.sum(np.power(S_i * np.exp(-alpha / h_i / (x-const*data_size)), decr) * np.power(later_weights, incr))
# def f_val(alpha,h_i, S_i, data_size, const, later_weights, P_max, incr,decr):
#     return S_i**decr  * np.exp(-alpha * decr / h_i/(P_max-const*data_size)) * np.power(later_weights, incr)

# def fgrad(x, alpha_orig, h_i, S_i_orig, data_size, const, later_weights,P_max, incr,decr):
#     S_i = np.power(S_i_orig, decr)
#     alpha = alpha_orig ** decr

#     Pth = alpha / 2 / h_i
#     Qth = Pth + const*data_size

#     const_pts = S_i  *alpha / h_i / np.square(Pth) * np.exp(-alpha/h_i/Pth)*np.power(later_weights, incr)
#     if isinstance(x, int) or isinstance(x, float) or (len(x) == 1 and len(h_i) == 1) :
#         if x < Qth:
#             return (x - Qth)**2 + const_pts
#         return S_i  *alpha / h_i / (x-const*data_size)**2 * np.exp(-alpha/h_i/(x-const*data_size))*np.power(later_weights, incr)
#     val = S_i  *alpha / h_i / np.square(x-const*data_size) * np.exp(-alpha/h_i/(x-const*data_size)) * np.power(later_weights, incr)
#     val[x<Qth] = np.square(x[x<Qth] - Pth[x<Qth]) + const_pts[x<Qth]
#     return val


# x1 =  np.ones(N)#np.square(np.random.randn(N))
# f(x1, const_alpha, h_i,S_i)

#%% Utility
# def find_maxi_fderivatives(alpha, h_i, S_i, data_size, const, later_weights,P_max, incr,decr):
#     Pth = alpha / 2 / h_i
#     Qth = Pth + const*data_size
#     return (Qth, fgrad(Qth, alpha, h_i,S_i, data_size, const, later_weights,P_max, incr,decr))

# def find_maxi_fderivatives(alpha_orig, h_i, S_i_orig, data_size, const, later_weights,P_max, incr,decr):
#     S_i = np.power(S_i_orig, decr)
#     alpha = alpha_orig ** decr
#     Pth = alpha / 2 / h_i
#     Qth = Pth + const*data_size
#     return (Qth, fgrad(Qth, alpha, h_i,S_i, data_size, const, later_weights,P_max, incr,decr))


#%%

# Given mu find corresponding P_k
# err_p = 2**(-40)
def find_Q_Given_mu(mu,  alpha, h_i, S_i,err, P_max, data_size, const, later_weights, incr,decr):
    Q_min, val,_ = find_maxi_fderivatives(alpha, h_i, S_i, data_size, const, later_weights,P_max, incr,decr)
    Q_max = P_max + const*data_size
    N = len(Q_min)
    if np.sum(val > mu) == 0:
        P_k = np.zeros(N)
        # P_k[bestK] = P_max
        # mu2decrease = True
        return P_k#, mu2increase
    ind2search = (val > mu)
    h_i_p = h_i[ind2search]
    S_i_p = S_i[ind2search]
    later_weights_p = later_weights[ind2search]
    data_size_p = data_size[ind2search]
    Q_k = (Q_min + Q_max)/2
    Q_k = Q_k[ind2search]
    Q_maxVn = Q_max[ind2search]
    Q_minVn = Q_min[ind2search]
    fprime = fgrad(Q_k, alpha, h_i_p, S_i_p, data_size_p, const, later_weights_p, P_max, incr,decr)
    iter_max = 10000
    n_iter = 0
    eps = 1
    while np.max(np.abs(fprime - mu)) > err and n_iter<iter_max and eps > 1e-15:
        n_iter += 1
        if len(fprime)>1:
            Q_maxVn[fprime < mu] = (Q_minVn[fprime < mu] + Q_maxVn[fprime < mu])/2
            Q_minVn[fprime >= mu] = (Q_minVn[fprime >= mu] + Q_maxVn[fprime >=  mu])/2
        else:
            if fprime < mu:
                Q_maxVn = (Q_maxVn + Q_minVn)/2
            else:
                Q_minVn = (Q_maxVn + Q_minVn)/2
        eps = np.max(np.abs(Q_k - (Q_maxVn + Q_minVn)/2))
        Q_k = (Q_maxVn + Q_minVn)/2
        fprime = fgrad(Q_k, alpha, h_i_p, S_i_p, data_size_p, const,later_weights_p, P_max, incr,decr)
    if n_iter == iter_max:
        print('find Q max iteration attained')
    Qvec = np.zeros(N)
    Qvec[ind2search] = Q_k
    # if np.sum(ind2search) < K:
    #     lala = np.argsort(-val[val<mu])
    #     ind = np.nonzero(val<mu)[0]
    #     ind2add = ind[lala[0:K - np.sum(ind2search)]]
    #     Pvec[ind2add] = P_max
    #     mu2decrease = True
        # print('yo')
    # elif np.sum(ind2search) > K:
    #     ind2del = np.argsort(-Pvec)[:np.sum(ind2search)-K]
    #     Pvec[ind2del] = Pvec[ind2del] * 0
    # if np.sum(ind2search) > K:
    #     np.sum(ind2search) > K:
    #     ind2del = np.argsort(-Pvec)[:np.sum(ind2search)-K]
    #     Pvec[ind2del] = Pvec[ind2del] * 0
    return Qvec

# find mu
# err_mu = 2**(-40)
def find_mu(err_mu,err_p, alpha, h_i, S_i, P_max, P_sum,data_size, const, later_weights, incr,decr):
    mu_min = 1e-15
    Q_min, val, _=  find_maxi_fderivatives(alpha, h_i, S_i,data_size, const, later_weights,P_max, incr,decr)

    mu_max = np.max(val)*2
    mu = (mu_min + mu_max) /2
    Qvec = find_Q_Given_mu(mu, alpha, h_i, S_i,err_p, P_max,data_size, const, later_weights, incr,decr)

    iter_max = 1000
    n_iter = 0
    while np.abs(np.sum(Qvec) - P_sum) > err_mu and n_iter < iter_max:
        if np.sum(Qvec) < P_sum: #or (np.sum(Pvec) > P_sum and mu2decrease):
            # mu_max = 10**((np.log10(mu_min) + np.log10(mu_max)) / 2)
            mu_max = (mu_min + mu_max)/2
            # print('la')
        else:
            # print('yo')
            # mu_min = 10**((np.log10(mu_min) + np.log10(mu_max)) / 2)
            mu_min = (mu_min + mu_max)/2
        # mu = 10**((np.log10(mu_min) + np.log10(mu_max)) / 2)
        mu = (mu_min + mu_max)/2
        Qvec  = find_Q_Given_mu(mu, alpha, h_i, S_i,err_p, P_max,data_size, const, later_weights, incr,decr)
        n_iter += 1
        # if n_iter % 500 == 0:
        #     print('Iteration ', n_iter, 'mu ', mu)

    #if n_iter == iter_max:
        #print('find mu max iteration attained')
    return Qvec, mu
# err_p = 2**(-40)
# err_mu = 2**(-40)
# P_opt, la = find_mu(err_mu,err_p, const_alpha, h_i, S_i, P_max, P_sum,data_size, const, later_weights)
# Q_opt = np.copy(P_opt)
# P_opt[Q_opt > 1e-10] = Q_opt[Q_opt > 1e-10] - const * data_size[Q_opt > 1e-10]
# active_users = (P_opt > 1e-8)
# sum_gap = P_sum - np.sum(Q_opt[active_users])
# print('-----------------------------------------------')
# print('sum gap dB', np.log10(np.abs(sum_gap)))
# # verify derivative
# deriv = fgrad(Q_opt[P_opt > 1e-8], const_alpha, h_i[P_opt > 1e-8], S_i[P_opt > 1e-8],
#               data_size[P_opt > 1e-8], const, later_weights[P_opt > 1e-8],P_max)
# print('derivative diff', np.max(np.abs(deriv - la)))
# print('nb of nonzero value', np.sum(P_opt > 1e-8))
# print('Power chosen', P_opt[P_opt > 1e-8])
# print('S_i', S_i[P_opt > 1e-8])
# nb_active = len(P_opt[P_opt > 1e-10])
# print('largest S_i values indices', np.argsort(-S_i)[0:nb_active])
# print('largest h_i values indices', np.argsort(-h_i)[0:nb_active])
# print('ordered indices P_opt', np.argsort(-P_opt)[0:nb_active])
# print('choosen indices', np.argsort(-P_opt)[0:nb_active])
# print('is in the best h_i', np.isin(np.argsort(-P_opt)[0:K], np.argsort(-h_i)[0:nb_active]))
# print('is in the best S_i', np.isin(np.argsort(-P_opt)[0:K], np.argsort(-S_i)[0:nb_active]))
#%% Reduce the previous size
def opti_WF( K, const_alpha, h_i, S_i, P_max, P_sum,data_size, const, later_weights, incr,decr):
    # Q_min, val =  find_maxi_fderivatives(const_alpha, h_i, S_i,data_size, const, later_weights,P_max, incr,decr)
    # val = f_val(const_alpha, h_i, S_i, data_size, const, later_weights,P_max, incr,decr)
    Q_min, val, val_nolater =  find_maxi_fderivatives(const_alpha, h_i, S_i,data_size, const, later_weights,P_max, incr,decr)
    # print('mean', np.mean(val_nolater), ' variance', np.std(val_nolater))
    active_clients = np.argsort(-val)[0:K]
    h_avg_p = copy.deepcopy(h_i[active_clients])
    data_size_p = copy.deepcopy(data_size[active_clients])
    later_weights_p = copy.deepcopy(later_weights[active_clients])
    weights_uni = np.ones(K)
    power_allocated = np.zeros(len(h_i))
    # print("length weight", len(weights_uni))
    # print("length h_i", len(h_i))
    # print("length h_avg", len(h_avg_p))
    _, power_allocated[active_clients] = solve_opti_wrt_P(weights_uni, h_avg_p, const_alpha,1, 1, K, 1 , 1 , P_max,P_sum, 'P1',data_size_p,const,1, later_weights_p)
    
    la = 1
    message = ''
    return power_allocated, la, message


def opti_WF_orig( K, const_alpha, h_i, S_i, P_max, P_sum,data_size, const, later_weights, incr,decr):
    err_mu = 2**(-30)
    err_p = 2**(-30)
    print(f_val(const_alpha,h_i,S_i,data_size,const,later_weights, P_max, incr,decr))
    Q_min, val =  find_maxi_fderivatives(const_alpha, h_i, S_i,data_size, const, later_weights,P_max, incr,decr)
    print(Q_min)
    print(val)
    N = len(h_i)


    Q_opt, la = find_mu(err_mu,err_p, const_alpha, h_i, S_i, P_max, P_sum,data_size, const, later_weights, incr,decr)
    active = (Q_opt > 1e-12)
    P_opt = np.zeros(N)
    P_opt[active] = Q_opt[active]  - const*data_size[active]

    indvec = np.nonzero(active)[0]
    nb_active = np.sum(active)
    message = ""
    if nb_active == 0:
        print("=======================================\n")
        print("user selection gives no users, something is wrong, best channel is carried")
        message = "user selection gives no users, something is wrong, best channel is carried"
        S_uni = np.ones(K)
        active_clients =  np.argsort(-h_i)[0:K]
        Q_opt, la = find_mu(err_mu,err_p, const_alpha, h_i[active_clients], S_uni, P_max, P_sum,
                            data_size[active_clients], const, later_weights, incr,decr)
        P_opt = np.zeros(N)
        P_opt[active_clients] = Q_opt[active_clients]  - const*data_size[active_clients]
        return P_opt, la, message
    if nb_active <= K:
        return P_opt, la, message
    while nb_active > K:
        #print('nb of active users', nb_active)
        nb2reduce = 1
        if nb_active - K >= 20:
            nb2reduce = 10
        elif nb_active - K >= 10:
            nb2reduce = 3
        # Sort by val
        importance = np.argsort(val[indvec])
        vec2reduce = importance[:nb2reduce]
        old_indvec = indvec
        indvec = np.setdiff1d(indvec, indvec[vec2reduce])

        Q_old = Q_opt
        #
        Q_opt, la = find_mu(err_mu,err_p,  const_alpha, h_i[indvec], S_i[indvec],
                            P_max, P_sum,data_size[indvec], const, later_weights[indvec], incr,decr)
        active = (Q_opt > 1e-8)
        nb_active = np.sum(active)
    if nb_active < K:
        Q_opt = Q_old
        indvec = old_indvec
        P_opt = Q_opt - const*data_size
        print('haha')
    Pvec = np.zeros(N)
    Pvec[indvec] = Q_opt - const * data_size[indvec]
    return Pvec, la, message
# P_opt, la,yo = opti_WF( K, const_alpha, h_i, S_i, P_max, P_sum,data_size, const,later_weights)
# active = P_opt>1e-10
# Q_opt = np.zeros(N)
# Q_opt[active] = P_opt[active] + const * data_size[active]
# print('-----------------------------------------------')
# print('sum gap dB', np.log10(np.abs(np.sum(Q_opt) - P_sum)))
# # verify derivative
# deriv = fgrad(Q_opt[P_opt > 1e-8], const_alpha, h_i[P_opt > 1e-8], S_i[P_opt > 1e-8],data_size[P_opt > 1e-8], const,
#               later_weights[P_opt>1e-8], P_max)
# print('derivative diff', np.max(np.abs(deriv - la)))
# print('nb of nonzero value', np.sum(P_opt > 1e-8))
# print('Power chosen', P_opt[P_opt > 1e-8])
# print('S_i', S_i[P_opt > 1e-8])
# print('largest S_i values indices', np.argsort(-S_i)[0:K])
# print('largest h_i values indices', np.argsort(-h_i)[0:K])
# print('choosen indices', np.argsort(-P_opt)[0:K])
# print('is in the best h_i', np.isin(np.argsort(-P_opt)[0:K], np.argsort(-h_i)[0:K]))
# print('is in the best S_i', np.isin(np.argsort(-P_opt)[0:K], np.argsort(-S_i)[0:K]))
#%%

# import matplotlib.pyplot as plt
# Npts= 10000
# PplotVn = np.linspace(0, P_max, Npts)
# d1 = np.zeros(Npts)
# d2 = np.zeros(Npts)
# d3 = np.zeros(Npts)
# for idp in range(Npts):
#     d1[idp] = fgrad(PplotVn[idp], const_alpha, h_i[np.argsort(-P_opt)[0]], S_i[np.argsort(-P_opt)[0]])
#     d2[idp] = fgrad(PplotVn[idp], const_alpha, h_i[np.argsort(-P_opt)[1]], S_i[np.argsort(-P_opt)[1]])
#     d3[idp] = fgrad(PplotVn[idp], const_alpha, h_i[np.argsort(-P_opt)[2]], S_i[np.argsort(-P_opt)[2]])
# muVn = la* np.ones(Npts)
# plt.plot(PplotVn, d1)
# plt.plot(PplotVn, d2)
# plt.plot(PplotVn, d3)
# plt.plot(PplotVn, muVn)
# plt.show()
#%%
def calc_fprime_inv(mu,  alpha, h_i, S_i, P_max):
    err_p = 2**(-30)
    p_min = -10
    p_max = P_max * 2
    N = len(h_i)
    P_k = np.ones(N) * (p_min + p_max) / 2
    fprime = fgrad(P_k, alpha, h_i, S_i)
    P_maxVn = np.ones(N) * p_max
    P_minVn = np.ones(N) * p_min
    iter_max = 2000
    n_iter = 0
    eps = 2
    while np.max(np.abs(fprime - mu)) > err_p and n_iter<iter_max and eps > 1e-15:
        n_iter += 1
        if len(fprime)>1:
            P_maxVn[fprime < mu] = (P_minVn[fprime < mu] + P_maxVn[fprime < mu])/2
            P_minVn[fprime >= mu] = (P_minVn[fprime >= mu] + P_maxVn[fprime >=  mu])/2
        else:
            if fprime < mu:
                P_maxVn = (P_maxVn + P_minVn)/2
            else:
                P_minVn = (P_maxVn + P_minVn)/2
        eps = np.max(np.abs(P_k - (P_maxVn + P_minVn)/2))
        P_k = (P_maxVn + P_minVn)/2
        fprime = fgrad(P_k, alpha, h_i, S_i)
    if n_iter == iter_max:
        print('find P max iteration attained')
    return P_k


def calc_eq_44(mu, i, order, alpha, h_i, S_i, P_max):
    N = len(h_i)
    P_min,fprime_min = find_maxi_fderivatives(alpha, h_i, S_i)
    Pvec = np.ones(N)
    kvec = np.array(list(range(1, N+1)))
    Pvec[order[kvec<=i]] = np.ones(np.sum(kvec<=i)) * P_max
    ind2= fprime_min[order] > mu
    ind2[:i-1] = False
    ind2calc = order[ind2]
    Pvec[ind2calc] = calc_fprime_inv(mu,  alpha, h_i[ind2calc], S_i[ind2calc], P_max)
    ind2 = fprime_min[order] <= mu
    ind2[:i-1] = False
    Pvec[order[ind2]] = np.zeros(np.sum(ind2))
    return Pvec
def find_mu_given_Ik(active, alpha, h_i, S_i, P_max, P_sum):
    assert K * P_max > P_sum
    N = len(h_i)
    err_mu = 2^(-30)
    mu_min = 1e-15
    P_min, val =  find_maxi_fderivatives(alpha, h_i, S_i)
    mu_max = np.max(val)*2
    mu_max = 10000
    mu = (mu_min + mu_max) /2
    Pvec = calc_fprime_inv(mu,  alpha, h_i[active], S_i[active], P_max)

    iter_max = 5000
    n_iter = 0
    while np.abs(np.sum(Pvec) - P_sum) > err_mu and n_iter < iter_max:
        if np.sum(Pvec) < P_sum:
            mu_max = (mu_min + mu_max)/2
        else:
            mu_min = (mu_min + mu_max)/2
        mu = (mu_min + mu_max)/2
        Pvec = calc_fprime_inv(mu,  alpha, h_i[active], S_i[active], P_max)
        n_iter += 1
        if n_iter % 200 == 0:
            print('Iteration ', n_iter, 'mu ', mu)

    if n_iter == iter_max:
        print('find mu max iteration attained')
    return Pvec, mu
def algo4( alpha, h_i, S_i, P_max, P_sum):
    N = len(h_i)
    indvec = np.ones(N)
    active = np.nonzero(indvec)
    p_k, mu  = find_mu_given_Ik(active, alpha, h_i, S_i, P_max, P_sum)
    Pth = alpha / 2 / h_i
    iter_max = 1000
    n_iter = 0
    stop_bool = False
    while len(np.nonzero(p_k<Pth)[0]) > 0 and n_iter < iter_max and (not stop_bool):
        n_iter += 1
        inactive = np.nonzero(p_k<Pth)[0]
        indvec[inactive] = 0
        if np.sum(indvec[inactive]) == 0:
            stop_bool = True
        active = np.nonzero(indvec)

        p_k[active], mu = find_mu_given_Ik(active, alpha, h_i, S_i, P_max, P_sum)
    return p_k
def algo8(K, alpha, h_i, S_i, P_max, P_sum):
    # calculate fprime(P_max)and order.
    N = len(h_i)
    fprime_max = fgrad(np.ones(N) * P_max, alpha, h_i, S_i)
    order = np.argsort(-fprime_max)

    #
    i = 1
    el_i = order[i-1]
    mu = fgrad(P_max, alpha, h_i[el_i], S_i[el_i])
    Pk = calc_eq_44(mu, i, order, alpha, h_i, S_i, P_max)
    while np.sum(Pk) < P_sum and i < N:
        i = i + 1
        el_i = order[i-1]
        mu = fgrad(P_max, alpha, h_i[el_i], S_i[el_i])
        Pk = calc_eq_44(mu, i, order, alpha, h_i, S_i, P_max)

        print('Algo 8 iteration'. i)
        print('------------------------------------------------')
    if np.sum(Pk) > P_sum:
        ind2choose = order[i-1:N]
        P_sum_p = P_sum - P_max * i
        Pk[ind2choose] = algo4( alpha, h_i[ind2choose], S_i[ind2choose], P_max, P_sum_p)
    return Pk, mu


# P_opt, la = algo8(K, const_alpha, h_i, S_i, P_max, P_sum)
# print('-----------------------------------------------')
# print('sum gap dB', np.log10(np.abs(np.sum(P_opt) - P_sum)))
# # verify derivative
# deriv = fgrad(P_opt[P_opt > 1e-8], const_alpha, h_i[P_opt > 1e-8], S_i[P_opt > 1e-8])
# print('derivative diff', np.max(np.abs(deriv - la)))
# print('nb of nonzero value', np.sum(P_opt > 1e-8))
# print('Power chosen', P_opt[P_opt > 1e-8])
# print('S_i', S_i[P_opt > 1e-8])
# K = len(P_opt[P_opt > 1e-8])
# print('largest S_i values indices', np.argsort(-S_i)[0:K])
# print('largest h_i values indices', np.argsort(-h_i)[0:K])
# print('choosen indices', np.argsort(-P_opt)[0:K])
# print('is in the best h_i', np.isin(np.argsort(-P_opt)[0:K], np.argsort(-h_i)[0:K]))
# print('is in the best S_i', np.isin(np.argsort(-P_opt)[0:K], np.argsort(-S_i)[0:K]))
