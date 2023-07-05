# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 11:56:24 2021

@author: Paul Zheng
"""

import numpy as np
import pandas as pd
from numpy import loadtxt
import copy
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
matplotlib.rc('xtick', labelsize=26)
matplotlib.rc('ytick', labelsize=26)


args = {'opt': "Prox", #"Avg" "Prox"
        'iid': "dirichlet", #"dirichlet" "niid"
        'dataset': 'mnist', #cifar, mnist, synthetic
        'method': "BL", #"BL" "UR", "opti", "WR", "Sal_WR"
        'ep':20,
        'seed_ind': 2
        }
multiple_seed = True
smooth_bool = False
def give_filename(args):
    # niidCapital = args['iid'].capitalize()

    # folder_name = "results/simu2_{}_{}{}".format(args['method'],args['opt'],args['iid'] )#'niid')
    folder_name = "./results/{}_{}_fc".format(args['dataset'],args['method'] )#'niid')
    tag = '{}/N{}_K{}_lr{:.3f}_bs{:d}_epo{:d}_mu{:.4f}_seed{}_Wseed{}_iid{}_alpha{:.3f}'
    seed_ind = 0
    if args['dataset'] == 'mnist':
        saveFileName = tag.format(folder_name, 500, 10 ,
                              0.1 , 64 , args['ep'] ,
                              1.0, 1 + args['seed_ind'] , 3 -args['seed_ind'] ,
                              args['iid'] , 0.5)
    elif args['dataset'] == 'cifar':
        saveFileName = tag.format(folder_name, 100, 10 ,
                              0.1 , 64 , args['ep'] ,
                              1.0,1+ args['seed_ind'], 3- args['seed_ind'] ,
                              'dirichlet' , 0.5)
    else: # synthetic cases
        saveFileName = tag.format(folder_name, 30, 10 ,
                              0.1 , 64 , args['ep'] ,
                              1.0,1 + args['seed_ind'], 3-args['seed_ind'] ,
                              'iid' , 0.5)
    acc = saveFileName+'_accuracy.csv'
    failed = saveFileName+'_failed.csv'
    loss = saveFileName+'_loss.csv'
    return saveFileName

def give_data(args):
    if multiple_seed:
        seedindVn = [0,1,2]
        acc_res = []
        for seed_ind in seedindVn:
            args['seed_ind'] = seed_ind
            pre_file_name = give_filename(args)
            print(pre_file_name)
            acc_file = pre_file_name+'_accuracy.csv'
            # failed_file = pre_file_name+'_failed_list.csv'
            # loss_file = pre_file_name+'_loss.csv'
            acc = loadtxt(acc_file, delimiter= ',')
            accMn = np.array([acc])
            if len(acc_res) == 0:
                acc_res = copy.deepcopy(accMn)
            else:
                acc_res = np.concatenate((acc_res, accMn), axis =0)
    else:
        pre_file_name = give_filename(args)
        print(pre_file_name)
        acc_file = pre_file_name+'_accuracy.csv'
        # failed_file = pre_file_name+'_failed_list.csv'
        # loss_file = pre_file_name+'_loss.csv'
        acc_res = loadtxt(acc_file, delimiter= ',')


    # fails = loadtxt(failed_file, delimiter= ',')
    # loss = loadtxt(loss_file, delimiter= ',')
    # return acc, fails, loss
    return acc_res
def give_full_method_name(abb):
    if abb == "optil":
        return "our method loss"
    if abb == "optils":
        return "our method loss and size"
    if abb == "opti":
        return "our method"
    if abb == "BL":
        return "best loss"
    if abb == "uni":
        return "uniform"
    if abb == "Sal_WR":
        return "weighted random"
    if abb == "WR":
        return "weighted random w/o aggregation correction"
    if abb == "BC":
        return "best channel"
    if abb == "Salehi":
        return "Salehi"

#%% Comparison client selection
# f = plt.figure(figsize=[20,20])
fig, ax = plt.subplots(figsize=[15,22])
# fig = plt.figure(figsize=[20,15])
# ax1 = plt.subplot(211)
# ax2= plt.subplot(212, sharex= ax1)
colors = ['r', 'b', 'g', 'c','m','y']
markers = ['o', 'v','^','x','d', '2']
id_color = 0
font_size = 30
marker_size = 20
if args['dataset'] == 'mnist':
    if args['iid'] == 'dirichlet':
        window_size = 21 # 51, 81
        nb_ite_div100= 1
        loweracc=7
    else:
        window_size = 41
        nb_ite_div100= 4
        loweracc=5
elif args['dataset'] == 'cifar':
    window_size = 51
    nb_ite_div100= 4
    loweracc=2
else:
    window_size = 1
    nb_ite_div100= 2
    loweracc=0
for id_meth, meth in enumerate(["opti","uni","BL", "BC", "WR","Sal_WR"]):#

    args['method'] = meth
    try:
        # acc, fails, loss = give_data(args)
        acc = give_data(args)
    except:
        print("haha", meth)
        continue
    # comm_rounds = np.arange(1,nb_ite_div100*100+1)
    index_rounds = np.arange(nb_ite_div100*100)
    if np.max(acc.shape)>nb_ite_div100*100:
        comm_rounds = np.arange(1, nb_ite_div100*100+1)
    else:
        comm_rounds = np.arange(1, np.max(acc.shape)+1)
    # acc = loss
    print(f"acc shape is {acc.shape}")
    if not multiple_seed:
        acc_pd= pd.Series(acc[0:nb_ite_div100*100])
    else:
        mean_acc_seed = np.mean(acc[:,index_rounds],0)
        acc_pd =  pd.Series(mean_acc_seed[0:nb_ite_div100*100])
    if window_size == 1 or not smooth_bool:

        
        if multiple_seed:
            print(f"acc shape is {np.mean(acc[:,index_rounds],0).shape}")
            print(f"comm rounds shape is {np.mean(comm_rounds,0).shape}")
            plt.plot(comm_rounds,np.mean(acc[:,index_rounds],0), linewidth = 4.0, 
              label=give_full_method_name(meth), color = colors[id_color],
              marker=markers[id_meth],markersize=marker_size, markevery=15, markeredgewidth = 6)
        else:
            plt.plot(comm_rounds,np.asarray(acc), linewidth = 4.0,
              label=give_full_method_name(meth), color = colors[id_color],
              marker=markers[id_meth],markersize=marker_size, markevery=15, markeredgewidth = 6)

    else:
        acc_smooth = acc_pd.rolling(window_size, center = True).mean()#.dropna()
        acc_std = acc_pd.rolling(window_size, center = True).std()#.dropna()


        # acc_smooth = savgol_filter(vec_avg_dict[key], 21, 3)
        # plt.plot(comm_rounds,acc,label=give_full_method_name(meth))
        plt.plot(comm_rounds,np.asarray(acc_smooth), linewidth = 4.0,
                  label=give_full_method_name(meth), color = colors[id_color],
                  marker=markers[id_meth],markersize=marker_size, markevery=15, markeredgewidth = 6) #[:len(comm_rounds)-window_size+1]
        indvec = np.arange(5*id_color,acc_smooth.size, step=nb_ite_div100*15)
        ax.errorbar(np.arange(1,nb_ite_div100*100)[indvec],
                    np.asarray(acc_smooth)[indvec],
                    yerr = np.asarray(acc_std)[indvec],
                    color = colors[id_color],
                    fmt='o', capsize=8,linewidth = 4)
    id_color += 1

plt.xlabel("Communication rounds",fontsize=font_size)
plt.ylabel("Test accuracy %",fontsize=font_size)
plt.legend(loc=0,prop={'size': font_size})
plt.xticks(np.arange(nb_ite_div100+1)*100, fontsize = font_size)
plt.yticks(np.arange(loweracc,11)*10, fontsize = font_size)
plt.ylim([loweracc*10,100])
# plt.title("Fed"+args['opt'] + " "+args['iid']+" alpha = 0.5", fontsize =font_size)
# plt.title("Fed"+args['opt'] + " "+args['iid'], fontsize =font_size)
# plt.title(args['dataset']+args['iid']+" alpha = 0.5", fontsize =font_size)
if args['dataset'] == 'mnist':
    plt.title(args['dataset']+' ' + args['iid'], fontsize =font_size)
elif args['dataset'] == 'cifar':
    plt.title(args['dataset']+' dirichlet', fontsize =font_size)
else:
    plt.title(args['dataset'], fontsize =font_size)
plt.grid()
ax.tick_params(color='#dddddd')
ax.spines['bottom'].set_color('#dddddd')
ax.spines['top'].set_color('#dddddd')
ax.spines['right'].set_color('#dddddd')
ax.spines['left'].set_color('#dddddd')
plt.tight_layout()
# figure_folder = "D:/Documents/Sciebo_groupfiles/PhD_Paul/figures2/"
# figure_folder = "D:/sciebo/files/PhD_Paul/figures/"
# plt.savefig(figure_folder+"comp_opti_"+args['opt']+args['iid']+".pdf")
plt.savefig("./figures2/comp_opti_"+args['opt']+args['iid']+".pdf")
plt.show()


# figure_folder = "D:/Documents/Sciebo_groupfiles/PhD_Paul/figures/"
# # figure_folder = "D:/sciebo/files/PhD_Paul/figures/"
# plt.savefig(figure_folder+"failed_transmission_{}.pdf".format(args['iid']))
# plt.show()

#%% Plot fails.
# fig, ax = plt.subplots(figsize=[15,12])
# colors = ['r', 'b', 'g', 'c','m','y']
# markers = ['o', '<','x','2','+','D']
# id_color = 0
# font_size = 40
# window_size = 11 # 51, 81
# # nb_ite_div100= 3
# for meth in ["optils2","uni","BL", "BC", "WR","Salehi"]:
#     args['method'] = meth
#     try:
#         acc, fails, loss = give_data(args)
#     except:
#         continue
#     if len(acc)>nb_ite_div100*100:
#         comm_rounds = np.arange(1, nb_ite_div100*100+1)
#     else:
#         comm_rounds = np.arange(1, len(acc)+1)
#     fails_s = pd.Series(fails[:nb_ite_div100*100,0])
#     fails_smooth = fails_s.rolling(window_size).mean()
#     plt.plot(comm_rounds,np.asarray(fails_smooth) , marker = markers[id_color],
#               label=give_full_method_name(meth), color = colors[id_color])
#     id_color += 1
# plt.xticks(np.arange(nb_ite_div100+1)*100, fontsize = font_size)
# plt.xlabel("Communication rounds",fontsize=font_size)
# plt.ylabel("Number of failed transmissions",fontsize=font_size)
# plt.yticks([0,2,4,6,8,10], fontsize = font_size)
# plt.legend(loc=4,prop={'size': font_size-10})
# plt.grid()
# # plt.title("Fed"+args['opt'] + " "+args['iid']+" alpha = 0.5", fontsize =font_size)
# #plt.title("Fed"+args['opt'] + " "+args['iid'], fontsize =font_size)

# ax.tick_params(color='#dddddd')
# ax.spines['bottom'].set_color('#dddddd')
# ax.spines['top'].set_color('#dddddd')
# ax.spines['right'].set_color('#dddddd')
# ax.spines['left'].set_color('#dddddd')
# plt.tight_layout()
# figure_folder = "D:/Documents/Sciebo_groupfiles/PhD_Paul/figures/"
# # figure_folder = "D:/sciebo/files/PhD_Paul/figures/"
# plt.savefig(figure_folder+"failed_transmission_{}.pdf".format(args['iid']))
# plt.show()
