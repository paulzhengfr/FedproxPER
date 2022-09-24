# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 15:51:40 2021

@author: Paul Zheng
"""
import numpy as np
import scipy.optimize as opt
from math import comb
from selections.function_user_selection_allCombinations import solve_opti_wrt_P

#%%
def naif_power_allocation(P_max, P_minVn, P_sum, theta, Tslot, data_size, m, B, N0, h_i):
    actual_Psum = P_sum - np.sum(theta / Tslot * data_size)
    K = len(data_size)
    val = min(actual_Psum / K, P_max)
    P_min =m* B*N0 /h_i /2
    res = np.ones(K)* val
    res[res<P_min] = 0
    return res


#%% Use scipy.optimize.fmin_slsqp sequential quadratic programming

# x= np.zeros(2*N+1)
def initialization(N, K, M, Mprime, m, h_i,N0,P_max,P_sum):
    a0 = np.ones(N) * K / N**2
    eps = (M+Mprime) / (M-1) / Mprime
    #P0 = m / h_i *  N0 * (1+eps)
    P0 = min(P_max, P_sum * K/ N)
    if np.sum(P0 >= P_max) > 0 or np.sum(P0) >= P_sum:
        print('Some constraints cannot be satisfied initially.')
    x0 = np.ones(2*N+1)
    x0[0:N] = np.power(a0, 1/(Mprime+2))
    x0[N:2*N] = P0 ** (-1/(1+Mprime))
    x0[2*N] = K
    return x0
# x0 = initialization(N, K, M)

def objective_func(x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv):
    N = int(len(x) / 2)
    a_Vn = x[0:N]
    b_Vn = x[N:2*N]
    t = x[2*N]
    return -1*np.sum(np.power(a_Vn, 1/M) * weights *  np.exp(- m * N0 / h_i * np.power(b_Vn, 1+Mprime)))

def gradient_obj(x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv):
    N = int(len(x) / 2)
    a_Vn = x[0:N]
    b_Vn = x[N:2*N]
    grad = np.zeros(2*N+1)
    grad[0:N] = weights / M * np.power(x[0:N], 1/M-1) *  np.exp(- m * N0 / h_i * np.power(b_Vn, 1+M))
    grad[N:2*N] = -m * N0 / h_i *(1+Mprime) * np.power(x[N:2*N], Mprime)* np.power(x[0:N], 1/M) * weights  *  np.exp(- m * N0 / h_i * np.power(b_Vn, 1+Mprime))
    grad[2*N] = 0
    return -1*grad


# inequality constraints

def make_const_a_l_bound(i,x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv):
    def f(x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv):
        return x[i]
    return f
# const_a_l_bound = [make_const_a_l_bound(i,x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Ck_inv)
#                    for i in range(N)]

# def const_a_l_bound(x,  weights, h_i, N0, m, K, alpha, beta, P_max,M):
#     N = int(len(x) / 2)
#     return min(x[0:N])

def make_const_a_u_bound(i,x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv):
    def f(x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv):
        return 1-x[i]
    return f
# const_a_u_bound = [make_const_a_u_bound(i,x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Ck_inv)
#                    for i in range(N)]

# def const_a_u_bound(x,  weights, h_i, N0, m, K, alpha, beta, P_max,M):
#     N = int(len(x) / 2)
#     return 1 - max(x[0:N])
def const_a_sum(x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv):
    N = int(len(x) / 2)
    t = x[2*N]
    return  -np.sum(np.power(x[0:N], Mprime+2)) + t
# const_a_sum(x0,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Ck_inv)

def const_t_K(x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv):
    N = int(len(x) / 2)
    return K - x[2*N]

def fun_C(beta_k,k, h_i,N0,m, K, Mprime):
    success_prob =  np.exp(- m * N0 /  h_i[k] * beta_k**(Mprime+1))
    C_k = 0
    for _d in range(1, K+1):
        C_k += K / _d * comb(K, _d) * (1 - success_prob)**(K-_d)
    return C_k
def calculate_Ckinv(weights, h_i, N0, m, K, alpha, beta, P_max,M): #Mprime
    N = len(h_i)
    sum_max = 0
    for d in range(1,K+1):
        sum_max += 1 / d * comb(K, d)
    sum_max = sum_max * K # upper bound of the inverse argument
    #print(sum_max)
    # the inverse argument to calculate
    val_inv =  K + alpha / 2 / beta - np.sqrt(K * (K + alpha / beta))
    def fun_C(beta_k,k, h_i,N0,m, K, M):
        success_prob =  np.exp(- m * N0 /  h_i[k] * beta_k**(M+1))
        C_k = 0
        for _d in range(1, K+1):
            C_k += K / _d * comb(K, _d) * (1 - success_prob)**(K-_d)
        return C_k

    def diff(x,a, k, h_i,N0,m, K, M):
        yt = fun_C(x,k, h_i,N0,m, K, M)
        return (yt - a)**2
    res = []
    for i in range(N):
        yoyo = opt.minimize(diff, 1, args = (val_inv,i, h_i,N0,m, K,M),
                            method = 'Nelder-Mead', tol = 1e-10)
        res.append(yoyo)
    Ck_inv = [res[i].x[0] for i in range(N)]
    Ck_res = [fun_C(Ck_inv[i],i, h_i,N0,m, K, M) for i in range(N)]
    Ck_res = np.array(Ck_res)
    ok_bool = True
    if np.sum(np.square(Ck_res - sum_max)) > 1e-10*N and np.sum(np.square(Ck_res - val_inv)) > 1e-10*N:
        ok_bool = False
    return Ck_inv, ok_bool, Ck_res

# Ck_inv, ok_bool, Ck_res = calculate_Ckinv(weights, h_i, N0, m, K, alpha, beta, P_max,M)
# if not ok_bool:
#     print('inverse of the function C_k falsely calculated')
def make_const_rho(i, x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv):
    def f(x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv):
        N = int(len(x)/2)
        return Ck_inv[i] - x[N+i]
    return f
# const_rho = [make_const_rho(i,x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Ck_inv)
#                    for i in range(N)]
# def const_rho(x,  weights, h_i, N0, m, K, alpha, beta, P_max,M):
#     N = int(len(x)/2)
#     success_prob =  np.exp(- m * N0 / (P_i * h_i))
#     C_k = np.zeros(N)
#     for ind_k, prob in enumerate(success_prob):
#         for _d in range(1, K+1):
#             C_k[ind_k] = K / _d * comb(K, _d) * (1 - prob)**(K-_d) + C_k[ind_k]
#     return K + alpha / 2 / beta - np.sqrt(K * (K + alpha / beta)) - max(C_k)

def make_const_snr(i, x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv):
    def f(x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv):
        N = int(len(x)/2)
        eps = (M+Mprime) / (M-1) / Mprime
        val = (h_i[i] / N0 / (1+eps) / m)**(1/(1+Mprime))
        return val - x[N+i]
    return f
# const_snr = [make_const_snr(i,x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Ck_inv)
#                    for i in range(N)]
# def const_snr(x,  weights, h_i, N0, m, K, alpha, beta, P_max,M,Ck_inv):
#     N = int(len(x) / 2)
#     P_Vn = x[N:2*N]
#     return min(P_Vn * h_i / B / N0) - m
def make_const_betamin(i, x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv):
    def f(x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv):
        N = int(len(x)/2)
        beta_min = 1 / P_max **(1/(1+Mprime))
        return x[N+i] - beta_min
    return f
# const_betamin = [make_const_betamin(i,x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Ck_inv)
#                    for i in range(N)]
# def const_P_l_bound(x,  weights, h_i, N0, m, K, alpha, beta, P_max,M,Ck_inv):
#     N = int(len(x) / 2)
#     return min(x[N:2*N])
# def const_P_u_bound(x,  weights, h_i, N0, m, K, alpha, beta, P_max,M,Ck_inv):
#     N = int(len(x) / 2)
#     return P_max - max(x[N:2*N])
# def const_P_sum(x):
#     return Psum_max - np.sum(x[N:2*N])
def const_sum_prod(x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv):
    N = int(len(x)/2)
    sum_prod = np.sum(np.power(x[0:N], Mprime+2) / np.power(x[N:2*N], Mprime+1))
    return P_sum - sum_prod
# Define list of constraints
# ieqcons = [const_a_l_bound, const_a_u_bound,const_a_sum, const_t_K, const_rho, const_snr,
#            const_betamin, const_sum_prod] #, const_P_sum
# ieqcons_flat = []
# for _id,  list_cons in enumerate(ieqcons):
#     if isinstance(list_cons, list):
#         for cons in list_cons:
#             ieqcons_flat.append(cons)
#     elif list_cons is None:
#         print(_id)
#     else:
#         ieqcons_flat.append(list_cons)
# verify = [(nb,const(x0,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Ck_inv)>=0) for nb, const in enumerate(ieqcons_flat)]
# verify_bool = [const(x0,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Ck_inv)>=0 for const in ieqcons_flat]
# verify_bool = np.array(verify_bool)
# if np.sum(verify_bool == False) != 0:
#     print("Some constraints are not verified in initialized value")
#%%
# Run optimziation
# yoyo = opt.fmin_slsqp(objective_func, x0, ieqcons=ieqcons_flat,
#                               bounds=(), fprime=gradient_obj, fprime_eqcons=None,
#                               fprime_ieqcons=None,
#                               args=(weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Ck_inv),
#                               iter=1000, acc=1e-6,
#                               iprint=2, disp=None, full_output=0, epsilon = 1e-6,
#                               callback=None)


#%% Use scipy.optimize

# var_bounds = [(0,1) for _ in range(N)] + [(1/P_max**(1/(1+M)), 1e16) for _ in range(N)] + [0,K]
# Define list of dict of constraints

# const_list = []
# for _, cons in enumerate(ieqcons_flat):
#     const_dict = dict()
#     const_dict['type'] = 'ineq'
#     const_dict['fun'] = cons
#     const_dict['args'] = (weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Ck_inv)
#     const_list.append(const_dict)
#%%

# Compute the hessian
# def hessian(x, weights, h_i, N0, m, K, alpha, beta, P_max):
#     N = int(len(x)/2)
#     hess_mat = np.zeros((2*N,2*N))
#     diag_aa = - weights / 4 / np.sqrt(np.power(x[0:N],3)) *  np.exp(-m * N0 / x[N:2*N] / h_i)
#     diag_PP = m * N0 / h_i * np.sqrt(x[0:N]) * weights / np.power(x[N:2*N],4) *\
#         np.exp(-m * N0 / x[N:2*N] / h_i) * (-2 * x[N:2*N] + m * N0 / h_i)
#     diag_aP = weights * m * N0 / h_i / 2 / np.sqrt(x[0:N]) / np.square(x[N:2*N]) * np.exp(-m * N0 / x[N:2*N] / h_i)
#     hess_mat[0:N, 0:N] = np.diag(diag_aa)
#     hess_mat[N:2*N, N:2*N] = np.diag(diag_PP)
#     hess_mat[0:N, N:2*N] = np.diag(diag_aP)
#     hess_mat[N:2*N, 0:N] = np.diag(diag_aP)
#     linear_opt = LinearOperator((2*N,2*N), matvec = lambda z: np.matmul(hess_mat,z))

#     return sp.csr_matrix(-hess_mat)

#%%
# x0 = lala
# yoyo = opt.minimize(objective_func, x0,
#                     args=(weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Ck_inv),
#                     method='trust-constr', jac=gradient_obj,
#                     hess=None, constraints=const_list, bounds = None, #var_bounds,
#                     options={'gtol':1e-8, 'xtol': 1e-6,
#                              'maxiter': 50000, 'verbose':2})


#%% Assign the relaxed solution to integer solution.
# opt_sol = yoyo.x
# N = int(len(opt_sol)/2)
# a_int = np.zeros(N)
# P_int = np.zeros(N)
# b_int = np.zeros(N)
# ind_Vn = [x for x in range(N)]
# id_max= None
# assign_Vn = np.zeros(K)

# for _k in range(K):
#     ind_Vn = [x for x in ind_Vn if x!=id_max]
#     id_max = ind_Vn[np.argmax(opt_sol[ind_Vn])]
#     assign_Vn[_k] = id_max
#     a_int[id_max] = 1
#     b_int[id_max] = opt_sol[N+id_max]
#     P_int[id_max] = 1 / b_int[id_max] ** (1+M)

# power_scale = max(min(P_sum / np.sum(P_int), P_max/np.max(P_int)), 1)

# P_int = power_scale * P_int



def user_selection_opti(K,M, Mprime, weights, h_i, N0, m, alpha, beta, P_max, P_sum):
    N = len(h_i)
    Ck_inv = np.zeros(N)
    x= np.zeros(2*N+1)
    x0 = initialization(N, K, M,Mprime, m, h_i,N0,P_max,P_sum)
    # define constraints
    const_a_l_bound = [make_const_a_l_bound(i,x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv)
                       for i in range(N)]
    const_a_u_bound = [make_const_a_u_bound(i,x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv)
                       for i in range(N)]
    const_a_sum(x0,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv)
    Ck_inv, ok_bool, Ck_res = calculate_Ckinv(weights, h_i, N0, m, K, alpha, beta, P_max,Mprime)
    if not ok_bool:
        print('inverse of the function C_k falsely calculated')
    # const_rho = [make_const_rho(i,x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Ck_inv)
    #                 for i in range(N)]
    const_snr = [make_const_snr(i,x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv)
                   for i in range(N)]
    const_betamin = [make_const_betamin(i,x,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv)
                   for i in range(N)]
    ieqcons = [const_a_l_bound, const_a_u_bound,const_a_sum, const_t_K,const_snr,#const_rho, #const_snr, # remove constraint of snr for now.
               const_betamin, const_sum_prod] #, const_P_sum
    ieqcons_flat = []
    for _id,  list_cons in enumerate(ieqcons):
        if isinstance(list_cons, list):
            for cons in list_cons:
                ieqcons_flat.append(cons)
        elif list_cons is None:
            print(_id)
        else:
            ieqcons_flat.append(list_cons)
    verify = [(nb,const(x0,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv)>=0) for nb, const in enumerate(ieqcons_flat)]
    verify_bool = [const(x0,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv)>=0 for const in ieqcons_flat]
    verify_bool = np.array(verify_bool)
    # if np.sum(verify_bool == False) != 0:
        # print("Some constraints are not verified in initialized value")
        # print(verify_bool)
    const_list = []
    for _, cons in enumerate(ieqcons_flat):
        const_dict = dict()
        const_dict['type'] = 'ineq'
        const_dict['fun'] = cons
        const_dict['args'] = (weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv)
        const_list.append(const_dict)
    try:
        yoyo = opt.minimize(objective_func, x0,
                            args=(weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Mprime,Ck_inv),
                            method='trust-constr', jac=gradient_obj,
                            hess=None, constraints=const_list, bounds = None, #var_bounds,
                            options={'gtol':1e-5, 'xtol': 1e-4,
                                     'maxiter': 5000, 'verbose':0})
    except:
        a_opt = np.argsort(-weights)[0:K]
        a_int = np.zeros(N)
        a_int[a_opt] = np.ones(K)
        P_int = np.zeros(N)
        h_i_p = h_i[a_opt]
        weights_p = weights[a_opt]
        _, P_int[a_opt] = solve_opti_wrt_P(weights_p, h_i_p, N0,1, m, K, alpha, beta, P_max,P_sum)
        output_message = 'Errors during optimization algorithm, better weights is taken.'
        opt_sol = 10
        run_time = 1
        return a_int, P_int, output_message, opt_sol, run_time

    opt_sol = yoyo.x
    a_int = np.zeros(N)
    P_int = np.zeros(N)
    b_int = np.zeros(N)
    ind_Vn = [x for x in range(N)]
    id_max= None
    assign_Vn = np.zeros(K)

    for _k in range(K):
        ind_Vn = [x for x in ind_Vn if x!=id_max]
        id_max = ind_Vn[np.argmax(opt_sol[ind_Vn])]
        assign_Vn[_k] = id_max
        a_int[id_max] = 1
        b_int[id_max] = opt_sol[N+id_max]
        P_int[id_max] = 1 / b_int[id_max] ** (1+M)

    power_scale = max(min(P_sum / np.sum(P_int), P_max/np.max(P_int)), 1)

    P_int = power_scale * P_int

    output_message = "{} the number of iterations: {}, optimality: {:.3f}, constraint violation: {:.3f}, execution time:{:.3f}".format(yoyo.message, yoyo.nit, yoyo.optimality, yoyo.constr_violation, yoyo.execution_time)
    return a_int, P_int, output_message, opt_sol, yoyo.execution_time

#%% Settings
# N = 15
# K = 3
# radius = 10
# ampli = 15
# N0 = 1
# B = 1 # bandwidth
# m = 0.5
# np.random.seed(1)
# distance = np.sqrt(np.random.uniform(0, radius ** 2, N))
# # np.random.seed(2)
# o_i = ampli * (np.square(np.random.randn(N)) + np.square(np.random.randn(N))) # Rayleigh distribution
# # h_i = o_i  / np.square(distance)
# h_i = o_i / ampli * 1
# # for seed 0 h_i[0,4 and 8]too small
# # h_i[0] = 0.3
# # h_i[4] = 0.2
# # h_i[8] = 0.2
# # P_i = np.ones(N)
# # snr = P_i * h_i / N0
# # transmission_rate = np.log2(1 + snr)
# # proba_PER = 1 - np.exp(-m / snr)
# weights = np.ones(N)
# # power = np.ones(N)

# # error_prob = np.random.rand(N) / 2

# #
# P_max = 5
# P_sum = 8
# alpha = 0.1
# beta = 0.001

# M = 5
# eps = 2/(M-1)
# a, P, output, opt_sol = user_selection_opti(K,M, weights, h_i, N0, m, alpha, beta, P_max, P_sum)

# alphaVn = opt_sol[1:N]
# betaVn = opt_sol[N:2*N]
# Ck_inv, ok_bool, Ck_res = calculate_Ckinv(weights, h_i, N0, m, K, alpha, beta, P_max,M)
# const_rho = [make_const_rho(i,opt_sol,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Ck_inv)
#                 for i in range(N)]
# res_const_rho = [fun(opt_sol,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Ck_inv) for fun in const_rho]

# const_snr = [make_const_snr(i,opt_sol,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Ck_inv)
#                     for i in range(N)]
# res_const_snr = [fun(opt_sol,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Ck_inv) for fun in const_snr]
# const_betamin = [make_const_betamin(i,opt_sol,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Ck_inv)
#                     for i in range(N)]
# res_const_betamin = [fun(opt_sol,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Ck_inv) for fun in const_betamin]
# res_const_sum_prod = const_sum_prod(opt_sol,  weights, h_i, N0, m, K, alpha, beta, P_max,P_sum,M,Ck_inv)
# print('constraint rho', res_const_rho,'\n')
# print('constraint betamin', res_const_betamin,'\n')
# print('constraint sum_prod', res_const_sum_prod,'\n')
# print('constraint snr', res_const_snr)
#%% Later for more complex problems
# id_max = None
# nb_assigned = 0
# constraints_verified = False
# nest_trial = []

# def check_intPb_const(aVn, PVn, ):
    # only check if the product is smaller than the sum of power


# while nb_assigned < K:
#     ind_Vn = [x for x in ind_Vn if x!=id_max]
#     id_max = ind_Vn[np.argmax(opt_sol[ind_Vn])]
#     print(id_max)
#     constraints_verified = True
#     # if id_max in [3,2,7,1,10,11,13,0,5,8,4,9, 12] and nb_assigned == 2:
#     #     constraints_verified = False
#     if constraints_verified:
#         a_int[id_max] = 1
#         b_int[id_max] = opt_sol[N+id_max]
#         P_int[id_max] = 1 / b_int[id_max] ** (1+M)
#         assign_Vn[nb_assigned] = id_max

#         if len(nest_trial) >= nb_assigned+1:
#             nest_trial[nb_assigned].append(id_max)
#         else:
#             nest_trial.append([id_max])
#         nb_assigned += 1
#     elif len(ind_Vn) == 1: # all vectors are checked and the last one doesn't either verify the constraint
#         nb_assigned += -1
#         i_pre_assigned = int(assign_Vn[nb_assigned])
#         nest_trial[nb_assigned + 1] = []
#         nest_trial_flat = [j for i in nest_trial for j in i]
#         assign_Vn[nb_assigned] = 0
#         a_int[i_pre_assigned] = 0
#         b_int[i_pre_assigned] = 0
#         P_int[i_pre_assigned] = 0
#         id_max = None
#         ind_Vn = [i for i in range(N) if i not in nest_trial_flat]
#     else:
#         if len(nest_trial) >= nb_assigned+1:
#             nest_trial[nb_assigned].append(id_max)
#         elif len(nest_trial) < nb_assigned+1:
#             nest_trial.append([id_max])
