import os
import csv, sys
import numpy as np
from numpy import savetxt


def init_saving_doc(args):
    save_path = args.savepath
    if os.path.isdir(save_path) == False and args.save:
        os.mkdir(save_path)
    folder_name = f'{args.dataset}_{args.name}'
    folder_path = save_path + folder_name
    args.folder_path = folder_path
    if os.path.isdir(folder_path)==False and args.save:
        os.mkdir(folder_path)
     # initiate log files
    tag = '{}/N{}_K{}_lr{:.3f}_bs{:d}_epo{:d}_mu{:.4f}_seed{}_Wseed{}_iid{}_alpha{:.3f}.csv'
    saveFileName = tag.format(folder_path, args.total_UE, args.active_UE ,
                              args.lr , args.local_bs , args.local_ep ,
                              args.mu , args.seed ,args.wireless_seed ,
                              args.iid , args.alpha )
    args.out_fname  = saveFileName
    with open(args.out_fname , 'w+') as f:
        print(
            'BEGIN-TRAINING\n'
            'World-Size,{ws}\n'
            'Batch-Size,{bs}\n'
            'Round, Epoch,batch_itr,'
            'User, Loss,avg:Loss,test_acc, train_acc, train_loss'.format(
                ws=args.total_UE ,
                bs=args.local_bs ),
            file=f)

def update_loss(args, round, ep_id,  batch_id, loss,loss_avg, user_id, train_loss=-1):
    with open(args.out_fname, '+a') as f:
        print('{rd},{ep},{itr},'
              '{userid},{loss:.4f},{loss_avg:.4f},'
              '-1,-1,-1'
              .format(rd=round, ep = ep_id, itr=batch_id,
                      loss=loss, loss_avg = loss_avg, userid = user_id), file=f)

def update_acc(args, round, test_acc, train_acc, train_loss=-1):
    with open(args.out_fname , '+a') as f:
        print('{rd},{filler},{filler},{filler},{filler},'
              '{filler},'
              '{val:.4f}, {val_tr:.4f}, {train_loss:.4f}'
              .format(rd=round, filler=-1, val=test_acc, val_tr = train_acc, train_loss=train_loss), file=f)

def treat_docs_to_acc(args):
    results = []
    count = 0
    with open(args.out_fname , newline='') as f:
        reader = csv.reader(f, delimiter=':')
        for row in reader:
            count += 1
            if count >= 5:
                results.append(row)

    results_tab = []
    for item in results:
        item_list = item[0].split(",")
        float_list = [0 for _ in range(len(item_list))]
        for _id, i in enumerate(item_list):
            float_list[_id] = float(i)
        results_tab.append(float_list)

    #%%
    results_tab = np.array(results_tab)
    acc_full = results_tab[:, -3]
    train_acc_full = results_tab[:, -2]
    train_loss_full = results_tab[:, -1]
    # print(len(acc_full))
    acc = [i for i in acc_full if i != -1]
    train_acc_full = [i for i in train_acc_full if i != -1]
    train_loss_full = [i for i in train_loss_full if i != -1]
    folder_name = args.folder_path
    tag = '{}/N{}_K{}_lr{:.3f}_bs{:d}_epo{:d}_mu{:.4f}_seed{}_Wseed{}_iid{}_alpha{:.3f}_accuracy.csv'
    saveFileName_bis = tag.format(folder_name, args.total_UE, args.active_UE ,
                              args.lr , args.local_bs , args.local_ep ,
                              args.mu , args.seed ,args.wireless_seed ,
                              args.iid , args.alpha )
    print(saveFileName_bis)
    savetxt(saveFileName_bis, acc, delimiter=',')
    tag = '{}/N{}_K{}_lr{:.3f}_bs{:d}_epo{:d}_mu{:.4f}_seed{}_Wseed{}_iid{}_alpha{:.3f}_tr_acc.csv'
    saveFileName_bis = tag.format(folder_name, args.total_UE, args.active_UE ,
                              args.lr , args.local_bs , args.local_ep ,
                              args.mu , args.seed ,args.wireless_seed ,
                              args.iid , args.alpha )
    print(saveFileName_bis)
    savetxt(saveFileName_bis, train_acc_full, delimiter=',')
    
    tag = '{}/N{}_K{}_lr{:.3f}_bs{:d}_epo{:d}_mu{:.4f}_seed{}_Wseed{}_iid{}_alpha{:.3f}_tr_loss.csv'
    saveFileName_bis = tag.format(folder_name, args.total_UE, args.active_UE ,
                              args.lr , args.local_bs , args.local_ep ,
                              args.mu , args.seed ,args.wireless_seed ,
                              args.iid , args.alpha )
    print(saveFileName_bis)
    savetxt(saveFileName_bis, train_loss_full, delimiter=',')
