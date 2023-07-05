#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--round', type=int, default=10, help="rounds of training")
    parser.add_argument('--total_UE', type=int, default=100, help="number of users: N")
    #parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--active_UE', type=int, default=10, help="number of activate clients: K")
    parser.add_argument('--local_ep', type=int, default=20, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=256, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--mu', type=float, default=0.01, help="fedprox parameter")
    parser.add_argument('--momentum', type=float, default=0, help="SGD momentum (default: 0)")
    parser.add_argument('--weight_decay', type=float, default=0, help="SGD weight_decay (default: 0)")
    #parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--optimizer', type=str, default='fedavg', help='optimizer: fedavg, fedprox')
    parser.add_argument('--model', type=str, default='mlp', help='model name: mlp, cnn, vgg, Mnist_oldMLP,synth_Net') #
    #parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    # parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
    #                     help='comma-separated kernel size to use for convolution')
    #parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    #parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    #parser.add_argument('--max_pool', type=str, default='True',
                        # help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--print_freq',
                        default=32,
                        type = int,
                        help='print frequency')
    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset:mnist, cifar, synthetic1,05,0,iid")
    parser.add_argument('--iid', type=str, default='iid', help='iid, niid, dirichlet') # iid seems to have some problem
    parser.add_argument('--alpha', type=float, default=0.5, help='dirichlet distribution parameter')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges for CNN")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    #parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--save',default=True,help='save the file or not')
    parser.add_argument('--savepath',default='./results/',type = str,help='save paths')
    parser.add_argument('--name',default='default',type = str,help='experiment name for folder name')

    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--wireless_seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--scenario', default='PER', type = str, help='if consider packet error rate: PER, woPER')

    parser.add_argument('--selection', type = str,default='uni_random', #uni_random, solve_opti_loss_size,solve_opti_loss_size2,solve_opti_size, solve_opti_loss, 
                        #solve_opti_size, weighted_random, best_loss,  best_channel,best_channel_ratio, salehi, best_datasize_success_rate
                        # solve_opti_AoU
                        help='user selection strategy')
    parser.add_argument('--opti', default='LR', type = str,
                        help='optimization method to solve power allocation (and user selection): WR, P1, P2,P4, cstWObj')
    parser.add_argument('--sigma', default=1, type = float, help='fading variance')

    parser.add_argument('--aggregation',
                        default='oneKmomentum', #oneN, oneK, oneKmomentum
                        type = str,
                        help='optimization method')
    parser.add_argument('--formulation',
                        default= 'exact',#'orig', #exact, log-exact, salehi, log-salehi, product
                        type = str,
                        help='if optils2, problem formulation method')
    parser.add_argument('--vanish',
                        default= 1,
                        type = float,
                        help='if optils2, vanishing rounds')
    parser.add_argument('--process_function_form',
                        default= 'other_side1',#'default',  'curvy', 'other_side','other_side1','test'
                        type = str,
                        help='shape of the function representing function form.')
    parser.add_argument('--Pname',
                        default= 'heihei', # 'curvy', 'other_side'
                        type = str,
                        help='name of the run in Wandb.')
    parser.add_argument('--later_weights_coef',
                        default= 1, # 
                        type = float,
                        help='factor on later weights.')
    parser.add_argument('--cell_radius',
                        default= 1000, # 
                        type = float,
                        help='Wireless Cell radius.')
    parser.add_argument('--test_method',
                        default='normal', #uni_weights, exact_loss, sum_loss, datasize.
                        type = str,
                        help='test method default as normal')
    parser.add_argument('--curvy',
                        default= 3, 
                        type = float,
                        help='value of M in the expression of varphi and psi')
    parser.add_argument('--eta_init',
                        default= 4, 
                        type = float,
                        help='initial value of eta (ought to be positive large to encourage exploration)')
    parser.add_argument('--loss_type',
                        default='mean', #final
                        type = str,
                        help='type of the loss for eta')
    parser.add_argument('--weight_normalization',
                        default='norm', # mean, median, none, max, norm
                        type = str,
                        help='how to normalize the weights and laterweights')
    parser.add_argument('--no_later', action='store_true', help='if not consider later weights correction')
    parser.add_argument('--allocate_power', action='store_false', help='whether allocate power only available for non optil selection')
    parser.add_argument('--eval_trloss',
                        action= 'store_true', 
                        help='boolean for whether to evaluate training loss at each round (take computation time).')
    parser.add_argument('--num_workers',
                        type = int,
                        default = 0, 
                        help='number of workers for dataloader..')
    parser.add_argument('--no_FL', action='store_true', help='if centralized learning')
    parser.add_argument('--normalize_order',
                        type = str,
                        default = 'inf',  #1, 2, inf
                        help='normalizing order of the two balancing terms')
    parser.add_argument('--thrToNext',
                        type = float,
                        default = 1,  # 0.9
                        help='threshold for passing completely to random weighted random selection. (100 per cent in default but can be chosen less, not so much influence)')
    parser.add_argument('--data_distr_scenarios',
                        type = str,
                        default = 'none',  # "inverse_datasize"
                        help='f')
    args = parser.parse_args()
    return args

