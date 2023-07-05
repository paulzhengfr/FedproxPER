''' 
Exact optimization method by going over all client selection.
'''


import numpy as np
import scipy.optimize as opt
import time
from selections.function_LR import opti_LR_for_naif
#%% Parameters
def dB2power(x):
    return np.exp(x/10*np.log(10))

# N = 30
# K = 10
# alpha = 0.1
# beta = 0.001
#
# # User distributions
# radius = 1000 # m
# diff = np.zeros(100)
# diff_obj = np.zeros(100)
# # for _seed in range(10):
# _seed = 1
# print('iteration {} '.format(_seed))
# np.random.seed(_seed)
# distance = np.sqrt(np.random.uniform(1, radius ** 2, N))
#
# # Users channel condition
# N0 = dB2power(-150) #* 1e-3 #dBm/Hz
# B = 1e6 # Hz
# m = dB2power(0.023) #dB
# P_max = 10 #0.01
# sigma = 1
# o_i = sigma* (np.square(np.random.randn(N)) + np.square(np.random.randn(N))) # Rayleigh distribution
# # h_i = o_i  / np.square(distance)
# freq = 2400 #Mhz
# FSPL = 20 * np.log10(distance) + 20 * np.log10(freq) - 27.55
# h_i = o_i / dB2power(FSPL)
# snr_max = P_max* h_i / (B*N0)
# D_i = B*N0/h_i
#
# E_max = 0.003
# T = 0.005
# P_sum = E_max / T * 1e2
# weights = np.ones(N)
# x= np.zeros(N)


#%% Define the optimization problem to solve for each combination.
def objective_func(x,  weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights):
    D_i = m*N0*B/h_i
    return -1 * np.sum(weights * np.exp(-D_i/x) + later_weights)
def gradient_obj(x,  weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights):
    D_i = m*N0*B/h_i
    grad = -1 * weights * (m*N0*B/np.square(x)/h_i) * np.exp(-D_i/x)
    return grad
def objective_func_conc(x,  weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights):
    D_i = m*N0*B/h_i
    Pth = D_i/2
    ind_r = x>=Pth
    ind_l = x<Pth
    res = weights * np.exp(-D_i/x) + later_weights
    if len(res) == 0:
        print("vector evaluate is zeros??")
        return 0
    if len(res) == 1:
        if x>= Pth:
            return -1 * res
        return -1 * (weights * np.exp(-D_i /Pth) * (D_i /np.square(Pth ) * (x  - Pth) + 1) + later_weights)
    res[ind_l] = weights[ind_l]  * np.exp(-D_i[ind_l] /Pth[ind_l] ) * (D_i[ind_l] /np.square(Pth[ind_l] ) * (x[ind_l]  - Pth[ind_l] ) + 1) + later_weights[ind_l] 
    return -1 * np.sum(res)
def gradient_obj_conc(x,  weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights):
    D_i = m*N0*B/h_i
    grad = -1 * weights * (m*N0*B/np.square(x)/h_i) * np.exp(-D_i/x)
    Pth = D_i/2
    ind_l = x<Pth
    if len(grad) == 0:
        print("vector evaluate is zeros??")
        return 0
    if len(grad) == 1:
        if x >= Pth:
            return grad
        return -1 * weights * (m*N0*B/np.square(Pth)/h_i) * np.exp(-D_i/Pth)
    grad[ind_l] = -1 * weights[ind_l] * (m*N0*B/np.square(Pth[ind_l])/h_i[ind_l]) * np.exp(-D_i[ind_l]/Pth[ind_l])
    return grad


def objective_func3(x,  weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights):
    D_i = m*N0*B/h_i
    return -1 * np.sum(weights * np.exp(-D_i/x)) - np.sum(later_weights)
def gradient_obj3(x,  weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights):
    D_i = m*N0*B/h_i
    grad = -1 * weights * (m*N0*B/np.square(x)/h_i) * np.exp(-D_i/x)
    return grad
# def objective_func4(x,  weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights):
#     D_i = m*N0*B/h_i
#     return -1 * np.sum(weights * np.exp(-D_i/x)) - np.sum(later_weights * x) / P_max
# def gradient_obj4(x,  weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights):
#     D_i = m*N0*B/h_i
#     grad = -1 * weights * (m*N0*B/np.square(x)/h_i) * np.exp(-D_i/x) - later_weights / P_max
#     return grad
def objective_func4(x,  weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights):
    D_i = m*N0*B/h_i
    return -1 * np.sum(weights * np.exp(-D_i/x)) - np.sum(later_weights)
def gradient_obj4(x,  weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights):
    D_i = m*N0*B/h_i
    grad = -1 * weights * (m*N0*B/np.square(x)/h_i) * np.exp(-D_i/x)
    return grad

def objective_func2(x,  weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot):
    gamma = m*N0*B/2/h_i
    return -np.sum(weights * np.amin(np.stack((np.exp(-m*N0*B/x/h_i), 2/gamma*np.exp(-2) * (x-gamma) + np.exp(-2)),1), axis = 1))
def gradient_obj2(x,  weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot):
    gamma = m*N0*B/2/h_i
    grad = weights * (m*N0*B/np.square(x)/h_i) * np.exp(-m*N0*B/x/h_i)
    grad[x<gamma] = 2/gamma[x<gamma] * np.exp(-2)
    return -grad

def make_const_P_l(i, x, weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights):
    def f(x, weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights):
        return x[i]- m* B*N0 /h_i /2 
    return f

def make_const_P_u(i, x, weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights):
    def f(x, weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights):
        return P_max - x[i]
    return f

def const_P_sum(x, weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights):
    return P_sum - np.sum(x) - theta/Tslot * np.sum(data_size)

def flat_constraints(cons_list):
    list_flat = []
    for _id,  list_cons in enumerate(cons_list):
        if isinstance(list_cons, list):
            for cons in list_cons:
                list_flat.append(cons)
        elif list_cons is None:
            print(_id)
        else:
            list_flat.append(list_cons)
    return list_flat

def initialization(N, P_max, P_sum):
    bound = min(P_max, P_sum / N)* 0.8
    x0 = np.ones(N) * bound
    return x0





def solve_opti_wrt_P( weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum, opti_pb,data_size, theta, Tslot, later_weights):
    K = int(K)
    x0= initialization(int(K), P_max, P_sum)
    x = x0
    const_P_l = [make_const_P_l(i, x, weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights)
                        for i in range(K)]
    const_P_u = [make_const_P_u(i, x, weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights)
                        for i in range(K)]
    ieqcons = [const_P_l,const_P_u, const_P_sum]
    ieqcons_flat = flat_constraints(ieqcons)
    const_list = []
    for _, cons in enumerate(ieqcons_flat):
        const_dict = dict()
        const_dict['type'] = 'ineq'
        const_dict['fun'] = cons
        const_dict['args'] = (weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights)
        const_list.append(const_dict)
    if opti_pb == 'P1':
        yoyo = opt.minimize(objective_func, x0,
                        args=(weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights),
                        method='trust-constr', jac=gradient_obj,
                        hess=None, constraints=const_list, bounds = None, #var_bounds,
                        options={'gtol':1e-7, 'xtol': 1e-7,
                                  'maxiter': 5000, 'verbose':0})
    elif opti_pb == 'P2':

        yoyo = opt.minimize(objective_func2, x0,
                        args=(weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights),
                        method='trust-constr', jac=gradient_obj2,
                        hess=None, constraints=const_list, bounds = None, #var_bounds,
                        options={'gtol':1e-7, 'xtol': 1e-7,
                                  'maxiter': 5000, 'verbose':0})

    elif opti_pb == 'P1_3':
        # print("Start")
        yoyo = opt.minimize(objective_func3, x0,
                        args=(weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights),
                        method='trust-constr', jac=gradient_obj3,
                        hess=None, constraints=const_list, bounds = None, #var_bounds,
                        options={'gtol':1e-7, 'xtol': 1e-7,
                                  'maxiter': 5000, 'verbose':0})
        # print("End")
    elif opti_pb == 'P1_4':
        yoyo = opt.minimize(objective_func4, x0,
                        args=(weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights),
                        method='trust-constr', jac=gradient_obj4,
                        hess=None, constraints=const_list, bounds = None, #var_bounds,
                        options={'gtol':1e-7, 'xtol': 1e-7,
                                  'maxiter': 5000, 'verbose':0})
    elif opti_pb == 'test_convex_ext':
        yoyo = opt.minimize(objective_func_conc, x0,
                        args=(weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights),
                        method='trust-constr', jac=gradient_obj_conc,
                        hess=None, constraints=const_list, bounds = None, #var_bounds,
                        options={'gtol':1e-7, 'xtol': 1e-7,
                                  'maxiter': 5000, 'verbose':0})
    
    else:
        print('error')
        yoyo = opt.minimize(objective_func, x0,
                        args=(weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum,data_size, theta, Tslot, later_weights),
                        method='trust-constr', jac=gradient_obj,
                        hess=None, constraints=const_list, bounds = None, #var_bounds,
                        options={'gtol':1e-7, 'xtol': 1e-7,
                                  'maxiter': 5000, 'verbose':0})
    return yoyo.fun, yoyo.x


#%%
def choose_iter(elements, length):
    for i in range(len(elements)):
        if length == 1:
            yield (elements[i],)
        else:
            for next in choose_iter(elements[i+1:len(elements)], length-1):
                yield (elements[i],) + next
def choose_comb(l, k):
    return list(choose_iter(l, k))

def tuple_ind2accessed_list(tuple_ind, list2access):
    _comb_list = np.array([i for i in tuple_ind])
    accessed_mapping = map(list2access.__getitem__, _comb_list)
    accessed_list = list(accessed_mapping)
    return accessed_list


# list2choose = list(range(N))
# all_comb = choose_comb(list2choose, K)
def solve_opti_all(all_comb, list2choose,  weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum, opti_pb,data_size, theta, Tslot, later_weights):
    len_comb = len(all_comb)
    N = len(h_i)
    obj_values = np.zeros(len_comb)
    tic = time.perf_counter()
    const_alpha = N0 * B /m 
    const = theta / Tslot
    for _id, _comb in enumerate(all_comb):
        # if _id <= 5:
        #     continue
        #print('iteration ',_id, _comb)
        # _id = 1
        # _comb = all_comb[_id]
        accessed_list = tuple_ind2accessed_list(_comb, list2choose)
        # solve the optimization problem wrt P. and save the results and the objective values.
        h_i_p = h_i[accessed_list]
        weights_p = weights[accessed_list]
        later_weights_p = later_weights[accessed_list]
        data_size_p = data_size[accessed_list]
        
        if opti_pb == "LR":
            
            _, _, _, val  = opti_LR_for_naif(const_alpha, h_i_p, weights_p, P_max, P_sum,data_size_p, const, later_weights_p, 1,1)
            obj_values[_id] = -val
            # print(f"length of yo {len(yo)}" )
        else:
            # print("heiy")
            try:
                obj_values[_id], _ = solve_opti_wrt_P(weights_p, h_i_p, N0,B, m, K, alpha, beta, P_max,P_sum, opti_pb,data_size_p, theta, Tslot, later_weights_p)
            except Exception as e:
                obj_values[_id] = 1000
                print(e)
                print('errors when solving problem with regard to P for combination', _comb)
            #print('iteration ',_id, _comb, ' objective values ', obj_values[_id])

    id_max = np.argmin(obj_values)
    # print('maximum of index ', id_max, 'objective value', obj_values[id_max], 'other objective values' , obj_values)

    opt_a = tuple_ind2accessed_list(all_comb[id_max], list2choose)
    #print('comb chosen ', all_comb[id_max], 'comb list', opt_a)
    h_i_p = h_i[opt_a]
    weights_p = weights[opt_a]
    later_weights_p = later_weights[opt_a]
    data_size_p = data_size[opt_a]
    if opti_pb == "LR":
            _, opt_P, _, _ = opti_LR_for_naif( const_alpha, h_i_p, weights_p, P_max, P_sum,data_size_p, const, later_weights_p, 1,1)
    else:
        _, opt_P = solve_opti_wrt_P(weights_p, h_i_p, N0,B, m, K, alpha, beta, P_max,P_sum, opti_pb,data_size_p, theta, Tslot, later_weights_p)
    toc = time.perf_counter()
    #print(f"The whole algo runs for {toc - tic:0.4f} seconds which is {(toc - tic)//60:0.0f} minutes and {(toc - tic)%60:0.2f} seconds")
    return opt_a, opt_P, toc- tic



#%% Testing wrt snr (equal weights)
# later_weights = np.random.rand(N)
# data_size = np.random.rand(N)
# theta = 0.001
# Tslot = 1.3
# # a,b = solve_opti_wrt_P( weights, h_i, N0,B, m, 15, alpha, beta, P_max,P_sum, 'P1_3',data_size, theta, Tslot, later_weights)
# a_1, P_1, time = solve_opti_all(all_comb, list2choose,  weights, h_i, N0,B, m, K, alpha, beta, P_max,P_sum, 'P1_3',data_size, theta, Tslot, later_weights)
