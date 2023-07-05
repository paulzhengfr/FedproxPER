import numpy as np
from scipy.optimize import milp
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds

def solve_opti_pb_wConstantObj(K, etaVn, Psum, Pmax, PminVn, enjSelectVn):
    ## When the objective of the optimization problem is constant with regard to power, this function should be applied.
    
    N = len(etaVn)
    
    ## Transform all linear constraints to matrix formats to fit to scipy.optimize
    Amin = np.concatenate((np.diag(PminVn), -np.eye(N)), axis = 1)
    Amax = np.concatenate((Pmax * np.eye(N), -np.eye(N)), axis = 1)
    ind_aVn = np.concatenate((np.ones(N), np.zeros(N)))
    # ind_bVn = np.concatenate((np.zeros(N), np.ones(N)))
    ind_bVn = np.concatenate((enjSelectVn, np.ones(N)))
    consAmin = LinearConstraint(Amin, np.full_like(np.ones(N), -np.inf), np.zeros(N))
    consAmax = LinearConstraint(Amax, np.zeros(N), np.full_like(np.ones(N), np.inf))
    consK = LinearConstraint(ind_aVn, K, K)
    consPsum = LinearConstraint(ind_bVn, -np.inf, Psum)
    obj_cost = np.concatenate((-etaVn, np.zeros(N)))

    var_lb = np.concatenate((np.zeros(N), np.full_like(np.ones(N), -np.inf)))
    var_ub = np.concatenate((np.ones(N), np.full_like(np.ones(N), np.inf)))                     
    bound_var = Bounds(var_lb, var_ub)
    integrality = np.concatenate((np.ones(N), np.zeros(N)))

    res = milp(c = obj_cost, constraints = [consAmin, consAmax, consK, consPsum], bounds= bound_var, integrality= integrality)
    user_selected = np.where(res.x[:N])
    user_selected = user_selected[0]
    indbVn = np.where(res.x[N:])
    indbVn = indbVn[0]
    return res.x[N:], res.success

def solve_cstObj_pb(K, alpha, h_avg, weights_to_use, P_max, P_sum,data_size,const, later_weights):
    etaVn = weights_to_use + later_weights
    enjSelectVn = const * data_size
    PminVn = alpha / 2/ h_avg
    # print("etaVn values", etaVn)
    powerVn, success_bool = solve_opti_pb_wConstantObj(K, etaVn, P_sum, P_max, PminVn, enjSelectVn)
    assert success_bool
    return powerVn