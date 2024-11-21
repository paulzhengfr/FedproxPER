# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 17:57:33 2021

@author: Paul Zheng
"""

import cvxpy as cp
import numpy as np
#%%Salehi
# N = 500
# K = 10

# x = cp.Variable(N)
# constraints = [cp.sum(x) <= K, x>=0]
# S_i = np.random.randint(0, high=1000, size=N) / 10
# obj = cp.Minimize(cp.sum(cp.multiply(S_i, cp.inv_pos(x))))

# prob = cp.Problem(obj, constraints)
# prob.solve()  # Returns the optimal value.
# print("status:", prob.status)
# print("optimal value", prob.value)
# print("optimal var", x.value)
# print("sum", np.sum(x.value))
# print("K best S_i", np.argsort(-S_i)[:K])
# print("K best opt", np.argsort(-x.value)[:K])
# print("K worst opt", np.argsort(x.value)[:K])

#%% function to import
# def CS_salehi(N,K, p_k, U_k):
#     x = cp.Variable(N)
#     constraints = [cp.sum(x) <= K, x>=0]
#     vec = p_k / U_k
#     obj = cp.Minimize(cp.sum(cp.multiply(vec, cp.inv_pos(x))))
#
#     prob = cp.Problem(obj, constraints)
#     prob.solve()
#     return x.value
def CS_salehi(N,K, p_k, U_k):
    x = cp.Variable(N)
    constraints = [cp.sum(cp.exp(x)) <= K]
    vec = p_k / U_k
    obj = cp.Minimize(cp.sum(cp.multiply(vec, cp.exp(-x))))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver="SCS")
    yk = x.value
    return np.exp(yk)
