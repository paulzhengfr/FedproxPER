#Packages
import numpy as np


def f(x, alpha, h_i, S_i, data_size, const, later_weights,P_max ):
    # Objective
    vec = S_i * np.exp(-alpha / h_i / x) + later_weights
    return np.sum(vec[x>0])


def f_val(x, alpha, h_i, S_i, data_size, const, later_weights,P_max ):
    return S_i * np.exp(-alpha / h_i / x) + later_weights
def fgrad(x, alpha, h_i, S_i, data_size, const, later_weights,P_max ):
    # Objective gradient
    val = S_i  *alpha / h_i / np.square(x) * np.exp(-alpha/h_i/x)
    return val
def fgrad_max(alpha, h_i, S_i, data_size, const, later_weights,P_max ):
    # Objective gradient max.
    h_i_obj = h_i
    Pth = alpha / 2 / h_i
    val_max = S_i  *alpha / h_i_obj / np.square(Pth) * np.exp(-alpha/h_i_obj/Pth)
    return  Pth, val_max
def fgrad_min(alpha, h_i, S_i, data_size, const, later_weights,P_max ):
    PVn = P_max * np.ones(h_i)
    val_min = fgrad(PVn, alpha, h_i, S_i, data_size, const, later_weights,P_max )
    return  PVn, val_min

def fgradgrad(x, alpha, h_i, S_i, data_size, const, later_weights,P_max , lambda_LR):
    return S_i * alpha / h_i / np.power(x,3) * np.exp(-alpha/h_i/x) * (alpha / h_i / x - 2)



## Associated function with lagrangian relaxation with lambda.
def f_val_lambda(x, alpha, h_i, S_i, data_size, const, later_weights,P_max , lambda_LR):
    return f_val(x, alpha, h_i, S_i, data_size, const, later_weights,P_max ) - lambda_LR * (x + const*data_size)
def f_lambda(x, alpha, h_i, S_i, data_size, const, later_weights,P_max, P_sum , lambda_LR):
    vec = f_val_lambda(x, alpha, h_i, S_i, data_size, const, later_weights,P_max ,lambda_LR)
    return np.sum(vec[x>0]) + lambda_LR * P_sum
def fgrad_lambda(x, alpha, h_i, S_i, data_size, const, later_weights,P_max , lambda_LR):
    return fgrad(x, alpha, h_i, S_i, data_size, const, later_weights,P_max ) - lambda_LR 








def hard_constraint(x, const, data_size, P_sum):
    vec = x + const*data_size
    return P_sum - np.sum(vec[x>0])
def hard_constraint_naif(x, const, data_size, P_sum):
    vec = x + const*data_size
    return P_sum - np.sum(vec)

def find_inverse_lambda(alpha, h_i, S_i, data_size, const, later_weights,P_max , index, lambdaL):
    # bisection to find the Lambert W function solution
    tol = 1e-15
    Nitermax = 1000
    h_i_p = h_i[index]
    S_i_p = S_i[index]
    data_size_p = data_size[index]
    later_weights_p = later_weights[index]
    # print(h_i_p)
    Pth = alpha / 2 / h_i[index]
    xl = Pth # val(x) > 0 
    xu = P_max*np.ones(len(Pth)) # val(x) < 0
    # xmid = (xl + xu) / 2
    for i in range(Nitermax):
        xmid = (xl + xu) / 2
        val_mid = fgrad_lambda(xmid, alpha, h_i_p, S_i_p, data_size_p, const, later_weights_p,P_max , lambdaL)
        if np.sum(np.abs(val_mid)) <= tol:
            # print(i)
            return xmid, val_mid
        xu[val_mid<0] = xmid[val_mid<0]
        xl[val_mid>0] = xmid[val_mid>0]
    return xmid, fgrad_lambda(xmid, alpha, h_i_p, S_i_p, data_size_p, const, later_weights_p,P_max , lambdaL)


def solve_pb_without_selection(alpha, h_i, S_i, P_max, P_sum,data_size, const, later_weights , lambda_LR):
    # Solve the Lagrangian relaxation problem P_lambda
    if lambda_LR <= 0:
        sol = np.ones(len(h_i)) * P_max
        return sol,  f_val_lambda(sol, alpha, h_i, S_i, data_size, const, later_weights,P_max , lambda_LR)
    
    x_max =  np.ones(len(h_i)) * P_max
    P_res = np.zeros(len(h_i))
    
    ## Assign edge value cases.
    Pth, fderiv_max = fgrad_max(alpha, h_i, S_i, data_size, const, later_weights,P_max )
    P_res[fderiv_max <= lambda_LR] = Pth[fderiv_max <= lambda_LR]

    h_i_obj = h_i
    fderiv_Pmax = fgrad(x_max, alpha, h_i_obj, S_i, data_size, const, later_weights,P_max )
    P_res[fderiv_Pmax >= lambda_LR] = P_max
    ind_OK = np.logical_or(fderiv_max <= lambda_LR, fderiv_Pmax >= lambda_LR)
    # print("number of index ok", np.sum(ind_OK))
    ind_remain = np.logical_not(ind_OK)
    if np.sum(ind_remain) == 0: # if each element has been take care of.
        return P_res, f_val_lambda(P_res, alpha, h_i_obj, S_i, data_size, const, later_weights,P_max , lambda_LR)
    
    ## Assign rest of non-edge users.
    P_bar, _ = find_inverse_lambda(alpha,h_i_obj, S_i, data_size, const, later_weights,P_max , ind_remain, lambda_LR)
    yoyo = np.ones(len(P_bar))*P_max
    min_Pmax = np.amin(np.stack((P_bar, np.ones(len(P_bar))*P_max)),axis= 0)
    max_Pth = np.amax(np.stack((min_Pmax, Pth[ind_remain])),axis= 0)
    P_res[ind_remain] = max_Pth
    return P_res, f_val_lambda(P_res, alpha, h_i_obj, S_i, data_size, const, later_weights,P_max , lambda_LR)


def algo_LR1( K, const_alpha, h_i, S_i, P_max, P_sum,data_size, const, later_weights , lambda_LR):
    # Calulate the solution given a value of lambda
    P_k, obj_LR_k = solve_pb_without_selection(const_alpha, h_i, S_i, P_max, P_sum,data_size, const, later_weights , lambda_LR)
    P_res = np.zeros(len(h_i))
    bestK = np.argsort(-obj_LR_k)[0:K]
    # print(bestK)
    P_res[bestK] = P_k[bestK]
    h_i_obj = h_i
    obj_res = f_lambda(P_res,const_alpha, h_i_obj, S_i, data_size, const, later_weights,P_max, P_sum, lambda_LR)
    obj_primal = f(P_res,const_alpha,h_i_obj, S_i, data_size, const, later_weights,P_max )
    return P_res, obj_res, obj_primal

def algo_LR1_for_naif(const_alpha, h_i, S_i, P_max, P_sum,data_size, const, later_weights , lambda_LR):
    P_k, obj_LR_k = solve_pb_without_selection(const_alpha, h_i, S_i, P_max, P_sum,data_size, const, later_weights , lambda_LR)
    obj_res = f_lambda(P_k,const_alpha, h_i, S_i, data_size, const, later_weights,P_max, P_sum,lambda_LR)
    return P_k, obj_res, f(P_k, const_alpha, h_i, S_i, data_size, const, later_weights,P_max)


# def initialization(N, K, P_max, P_sum, alpha, h_i):
#     P_res = np.zeros(N)
#     Pth = alpha / 2 / h_i
#     bound = min(P_max, P_sum / K)* 0.95
#     ind_okay = Pth < bound
#     P_res[ind_okay[0:K]] = bound
#     return P_res

def opti_LR( K, const_alpha, h_i, S_i, P_max, P_sum,data_size, const, later_weights ):
    ## Full LR algorithm with bisection on lambda.
    N = len(h_i)
    # P_init = initialization(N, K, P_max, P_sum, const_alpha, h_i)
    lam_feas_min = 10000
    lam_unfeas_max = 0
    lam_mid = (lam_feas_min + lam_unfeas_max) / 2
    for i in range(1000):
        P_lam, obj_lam, obj_prim = algo_LR1( K, const_alpha, h_i, S_i, P_max, P_sum,
                                            data_size, const, later_weights , lam_mid)
        # print('iter', i, ' objec LR val ', obj_lam, ' objec prim val ', obj_prim,)
        if hard_constraint(P_lam, const, data_size, P_sum) >= 0: # feasible
            lam_feas_min = lam_mid
        else:
            lam_unfeas_max = lam_mid
        lam_mid = (lam_feas_min + lam_unfeas_max) / 2
        diff = np.abs(lam_feas_min - lam_unfeas_max)
        if np.abs(diff) < 1e-12:
            P_lam, obj_lam, obj_primal = algo_LR1( K, const_alpha, h_i, S_i, P_max, P_sum,data_size, const, later_weights , lam_feas_min)
            # print(f"For N {N} and K {K}, optimal objective attained is {obj_primal}")
            break
    
    return lam_feas_min, P_lam, obj_lam, obj_primal



def opti_LR_for_naif(const_alpha, h_i, S_i, P_max, P_sum,data_size, const, later_weights ):
    N = len(h_i)
    # P_init = initialization(N, K, P_max, P_sum, const_alpha, h_i)
    lam_feas_min = 20000
    lam_unfeas_max = 0
    lam_mid = (lam_feas_min + lam_unfeas_max) / 2
    for i in range(1000):
        P_lam, obj_lam, obj_prim = algo_LR1_for_naif( const_alpha, h_i, S_i, P_max, P_sum,
                                            data_size, const, later_weights , lam_mid)
        # print('iter', i, ' objec LR val ', obj_lam, ' objec prim val ', obj_prim,)
        if hard_constraint_naif(P_lam, const, data_size, P_sum) >= 0: # feasible
            lam_feas_min = lam_mid
        else:
            lam_unfeas_max = lam_mid
        lam_mid = (lam_feas_min + lam_unfeas_max) / 2
        diff = np.abs(lam_feas_min - lam_unfeas_max)
        if np.abs(diff) < 1e-15:
            P_lam, obj_lam, obj_prim = algo_LR1_for_naif(const_alpha, h_i, S_i, P_max, P_sum,data_size, const, later_weights , lam_feas_min)
            # print(f"For N {N} and K {K}, optimal lamb is {obj_lam},  objective attained is {obj_prim}")
            break
    
    return lam_feas_min, P_lam, obj_lam, obj_prim

