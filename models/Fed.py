#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAggregation(w, w_old, data_weight, active_clients, args, wireless_arg):

    weight_i = calc_weightAggreg(data_weight, active_clients, args, wireless_arg) # weight_i already normalized
    weight_i = torch.from_numpy(weight_i).to(args.device)
    if len(w) == 0:
        print("active clients when len w = 0", active_clients)
        return w_old
    if len(w) == 1:
        w_avg = copy.deepcopy(w[0])
        for k in w[0].keys():
            if w_avg[k].dtype == torch.long:
                continue
            w_avg[k] = weight_i * w_avg[k]
            w_avg[k] = w_avg[k] + (1 - weight_i) * w_old[k]
        return w_avg
    w_avg = copy.deepcopy(w[0])
    sum_weight = copy.deepcopy(torch.sum(weight_i))
    for k in w_avg.keys():
        if w_avg[k].dtype == torch.long:
            continue
        w_avg[k] = weight_i[0] * w_avg[k]
        for i in range(1, len(w)):
            w_avg[k] += weight_i[i] * w[i][k]
        w_avg[k] = w_avg[k] + (1-sum_weight) * w_old[k]
    return w_avg


def calc_weightAggreg(data_weight, active_clients, args, wireless_arg):
    if args.aggregation == 'oneKmomentum' and args.scenario == 'PER':
        ratio_aggreg =  args.total_UE / args.active_UE / wireless_arg['success prob'][active_clients]
    elif args.aggregation == 'oneK' or args.scenario =="woPER":
        ratio_aggreg =  args.total_UE / args.active_UE
        # print("oneK activated")
    elif args.aggregation =='oneN':
        ratio_aggreg= 1
    else:
        print("================================\n")
        print("args aggregation erroneous\n")
        print("================================\n")
        ratio_aggreg= 1
    if args.selection == 'uni_random' or args.selection == 'best_loss' or args.selection == 'best_channel' or args.selection == 'best_channel_ratio' or args.selection == 'best_exact_loss':
        weight_c = copy.deepcopy(data_weight[active_clients]) *ratio_aggreg
        

    elif args.selection == 'solve_opti_loss_size' or args.selection == 'solve_opti_loss_size2' or args.selection == 'solve_opti_size' or args.selection == 'weighted_random' or args.selection == 'solve_opti_loss_size4' or args.selection == 'best_datasize_success_rate':
        weight_c = np.ones(len(active_clients)) * 1/args.total_UE * ratio_aggreg
        if args.selection == 'solve_opti_loss_size2' and (args.formulation == 'salehi' or args.formulation == 'log-salehi'):
            weight_c =  1/args.total_UE * ratio_aggreg / copy.deepcopy(wireless_arg['salehi_weight_sampling'][active_clients])
    elif args.selection == 'solve_opti_loss' or args.selection == 'solve_opti_loss_size3':
        weight_c = copy.deepcopy(data_weight[active_clients]) * ratio_aggreg
        if args.selection == 'solve_opti_loss_size3' and (args.formulation == 'salehi' or args.formulation == 'log-salehi'):
            weight_c  = weight_c  * copy.deepcopy(wireless_arg['salehi_weight_sampling'][active_clients])
    elif args.selection == 'salehi':
        weight_c = copy.deepcopy(data_weight[active_clients]) * ratio_aggreg / copy.deepcopy(wireless_arg['salehi_weight_sampling'][active_clients])

    return weight_c
