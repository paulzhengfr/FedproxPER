#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.save_results import init_saving_doc, update_loss, update_acc, treat_docs_to_acc
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.sampling_func import DataPartitioner
from utils.options import args_parser
from models.Update import LocalUpdate, calc_exact_loss
from models.Nets import * #MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, FedAggregation
from models.evaluation import test_img
from selections.wireless import wireless_param, user_selection, update_wireless
from selections.opti import init_optil_weights, update_optil_weights, update_success_trained
# from models.vgg1 import *
from models.vgg import vgg11
# import wandb
import random


import os

from test_synthetic_dataset import read_data



if __name__ == '__main__':
    # parse args
    args = args_parser()
    print(args)
    if args.no_FL: # centralized learning option
        args.total_UE = 1
        args.active_UE = 1
        args.scenario = "woPER"
        args.local_ep = 1
        # wandb.init(project = 'noFL', name = args.dataset+'_'+args.Pname, config=args)
    # else:
        # wandb.init(project = 'FLper_{}'.format(args.dataset), name = args.Pname, config=args)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if args.name == 'default':
        args.name = args.Pname
    init_saving_doc(args)
    random.seed(1234+ args.seed)
    # dataset_rootdir = "/media/data/fl"
    dataset_rootdir = "../data"

    ## load dataset
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1751,), (0.3267,))])  #(0.1307,), (0.3081,)
        dataset_train = datasets.MNIST(root=dataset_rootdir+"/mnist/", train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(root=dataset_rootdir+"/mnist/", train=False, download=True, transform=trans_mnist)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(root=dataset_rootdir+"/cifar/", train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(root=dataset_rootdir+"/cifar/", train=False, download=True, transform=trans_cifar_test)
        

    elif args.dataset.find("synthetic") >= 0:
        train_data_dir = os.path.join(dataset_rootdir,args.dataset, 'train')
        test_data_dir = os.path.join(dataset_rootdir,args.dataset, 'test')
        clients, groups, dataset_train, dataset_test = read_data(train_data_dir, test_data_dir, args.dataset)
        dataset_train = [dataset_train[clients[i]] for i in range(args.total_UE)]
        datasize_p_UE = [len(dataset_train[i]['y']) for i in range(args.total_UE)]
        sum_datapoints = sum(datasize_p_UE)
        datasize_p_UE = np.array(datasize_p_UE)
        datasize_weight = np.array([l/sum_datapoints for l in datasize_p_UE])
        args.datasize_weight = datasize_weight


    else:
        exit('Error: unrecognized dataset')
    # if args.iid == 'iid':
    #     dict_users = mnist_iid(dataset_train, args.total_UE)
    # else:
    #     dict_users = mnist_noniid(dataset_train, args.total_UE)

    ## Split data to clients
    users_count_labels= {}
    if args.dataset.find("synthetic") == -1 and args.dataset.find("shakespeare") == -1:
        partition_obj = DataPartitioner(dataset_train, args.total_UE, NonIID=args.iid ,
                                alpha=args.alpha , seed = args.seed)
        dict_users, users_count_labels = partition_obj.use()
        args.users_count_labels = users_count_labels
        # print("dict_users", dict_users)
        # print("users count labels", users_count_labels)
        datasize_p_UE = np.array([len(dict_users[i]) for i in range(args.total_UE)])
        datasize_weight = np.array([l/sum(datasize_p_UE) for l in datasize_p_UE])
        args.datasize_weight = datasize_weight
    # datasize_p_UE = np.array([len(partition_obj.partitions[i]) for i in range(args.total_UE)])
    ## Test if dataset unbalanced
    # print('minimum datasize', np.min(datasize_p_UE), ' sum datapoint allocated ', sum(datasize_p_UE))
    # plt.figure()
    # plt.plot(np.arange(args.total_UE), datasize_p_UE[np.argsort(datasize_p_UE)])
    # plt.plot(np.arange(args.total_UE), datasize_p_UE)
    # plt.ylabel('datasize')
    # plt.savefig('./save/datasize5')

    if args.dataset.find("synthetic") == -1:

        img_size = dataset_train[0][0].shape
        print(img_size)
    else:
        img_size = len(dataset_train[0]['x'][0])

    ## build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model =='Mnist_oldMLP':
        net_glob = Mnist_oldMLP().to(args.device)
    elif args.model =='LeNet':
        net_glob = LeNet().to(args.device)
    elif args.model =="LeNet_cifar":
        net_glob = LeNet_cifar().to(args.device)
    elif args.model == 'synth_Net':
        net_glob = synth_Net().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'vgg':
        if args.dataset =='cifar':
            num_class_cifar = 10
        else:
            num_class_cifar = 100
        # net_glob = vgg11().to(args.device)
        net_glob = vgg11().to(args.device)
        #net_glob = VGG('VGG11').to(args.device)

    else:
        exit('Error: unrecognized model')
    print(net_glob)
    # wandb.watch(net_glob, log_freq=100)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training init
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    wireless_arg = {}
    if args.scenario == 'PER':
        wireless_arg = wireless_param(args, datasize_weight, np.sum(datasize_p_UE))
    # Initialize weights
    num_trained = 0
    list_trained = []
    bool_trained = []
    vanish_index = 1
    # loss_weights = np.ones(args.total_UE)*20
    loss_weights = np.ones(args.total_UE)*args.eta_init + args.datasize_weight/np.max(args.datasize_weight) * args.eta_init * 0.8
    weights, later_weights = init_optil_weights(args, wireless_arg)
    loss_avg = 10
    
    args.exact_loss= np.zeros(args.total_UE) # only when the exact loss option is on.
    
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.total_UE)]

    
    ## FL Training
    for iter in range(args.round):
        net_glob.train()
        exact_loss_Vn = np.zeros(args.total_UE)
        if args.test_method == "init_exact_loss" and iter == 0:
            args.exact_loss = calc_exact_loss(args, dataset_train, dict_users, net_glob)
            loss_weights = args.exact_loss
            update_acc(args, iter, 0, 0)
            # if args.scenario =='PER':
            #     wandb.log({"loss": 2, "accuracy": 2, "trained users":0, "successful transmitted users": 0,
            #         "max local loss": 2, "max index": 0,"coef increasing": 0, "coef decreasing": 0, "objective values": 0})
            # else:
            #     wandb.log({"loss": 2, "accuracy": 1})
            continue

        loss_locals = []
        if not args.all_clients:
            w_locals = []
        #m = max(int(args.frac * args.total_UE), 1)
        m = args.active_UE
        seed_round = 1234 + args.seed  + iter
        if args.scenario == 'woPER': # without packet error option.
            np.random.seed(seed_round)
            idxs_users = np.random.choice(range(args.total_UE), m, replace=False)
            wireless_arg = {}
        elif args.scenario == 'PER':
            wireless_arg = update_wireless(args, wireless_arg, seed_round)
            weights, later_weights, coef_dec, coef_inc = update_optil_weights(args, wireless_arg, loss_weights, num_trained)
            wireless_arg['incr'] = coef_inc
            wireless_arg['decr'] = coef_dec
            print("later weights zero? ", np.sum(later_weights)==0)
            idxs_users,proba_success_avg , fails, errorprob, obj_values = user_selection(args, wireless_arg, seed_round, datasize_weight,weights,later_weights)
            num_trained, list_trained,bool_trained,  vanish_index = update_success_trained(args, idxs_users, list_trained,bool_trained, vanish_index)
        else:
            print("args.scenario argument is wrong")
            np.random.seed(seed_round)
            idxs_users = np.random.choice(range(args.total_UE), m, replace=False)

        for idx in idxs_users:
            ## Local training
            if args.dataset.find("synthetic") == -1:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], user_id = idx)
            else:
                local = LocalUpdate(args=args, dataset=dataset_train[idx], idxs=idx, user_id = idx)
            w, loss, final_loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            if args.loss_type == 'mean':
                loss_weights[idx] = copy.deepcopy(loss)
            elif args.loss_type == 'final':
                loss_weights[idx] = copy.deepcopy(final_loss)

        # update global weights by aggregation.
        if args.scenario == 'woPER':
            w_glob = FedAvg(w_locals)
        else:
            w_glob = FedAggregation(w_locals, w_glob,datasize_weight,idxs_users, args, wireless_arg)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        
        exact_loss_Vn = np.zeros(args.total_UE)
        if args.test_method == "exact_loss" or args.selection == 'best_exact_loss':
            exact_loss_Vn = calc_exact_loss(args, dataset_train, dict_users, net_glob)
        args.exact_loss = exact_loss_Vn
            
        # print, log, and save loss and accuracy
        training_loss_avg = -1
        if len(idxs_users) != 0 and args.eval_trloss:
            loss_avg = sum(loss_locals) / len(loss_locals)
            if args.dataset.find("synthetic") == -1:
                exact_loss_Vn = calc_exact_loss(args, dataset_train,  net_glob, dict_users=dict_users)
            else:
                exact_loss_Vn = calc_exact_loss(args, dataset_train,  net_glob)
            training_loss_avg = np.sum(exact_loss_Vn * args.datasize_weight)
        print('Round {:3d}, Average loss {:.3f}, selected users {}'.format(iter, loss_avg,idxs_users))
        loss_train.append(loss_avg)

        net_glob.eval()
        acc_train = -1
        # if args.dataset.find("synthetic") == -1:
        #     acc_train, _, class_accuracy = test_img(net_glob, dataset_train, args)
        # else:
        #     acc_test, loss_test, class_accuracy = test_img(net_glob, dataset_test, args)
        acc_train, _, class_accuracy = test_img(net_glob, dataset_train, args)
        acc_test, loss_test, class_accuracy = test_img(net_glob, dataset_test, args)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        update_acc(args, iter, acc_test, acc_train, train_loss=training_loss_avg)
        if args.scenario =='PER':
            if len(idxs_users)!= 0:
                log_dict = {"loss": training_loss_avg, "accuracy": acc_test, "trained users":num_trained, "successful transmitted users": len(idxs_users),
                    "max local loss": max(loss_locals), "max index": idxs_users[np.argmax(np.array(loss_locals))],
                    "coef increasing": coef_inc, "coef decreasing": coef_dec, "objective values": obj_values}
                log_dict.update(class_accuracy)
                log_dict.update(args.trained_users_per_class)
                log_dict.update(args.trained_data_per_class)
                # wandb.log(log_dict)
            else:
                log_dict = {"loss": training_loss_avg, "accuracy":acc_test, "trained users":num_trained,"successful transmitted users":0,
                    "max local loss": 0, "max index": -10,"coef increasing":0, "coef decreasing": 1}
                log_dict.update(class_accuracy)
                log_dict.update(args.trained_users_per_class)
                log_dict.update(args.trained_data_per_class)
                # wandb.log({"loss": training_loss_avg, "accuracy":acc_test, "trained users":num_trained,"successful transmitted users":0,
                #     "max local loss": 0, "max index": -10,"coef increasing":0, "coef decreasing": 1})# loss: loss_avg
        else:
            log_dict = {"loss": training_loss_avg, "accuracy": acc_test}
            log_dict.update(class_accuracy)
            # wandb.log(log_dict)

    # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('train_loss')
    # plt.savefig('./save/fed_{}_{}_{}_K{}_iid{}.png'.format(args.dataset, args.model, args.round, args.active_UE, args.iid))
    treat_docs_to_acc(args)
