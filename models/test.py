#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
def get_new_labelVn(labels, num_val):
    labels_fl = labels.flatten()
    one_Vn = np.arange(len(labels_fl)) * num_val
    labels_fl_lala = labels_fl + one_Vn
    labels_new = np.zeros((labels.shape[0],labels.shape[1],num_val))
    labels_new_fl = labels_new.flatten()
    labels_new_fl[labels_fl_lala]=1
    return labels_new_fl.reshape(labels_new.shape)

class Synthetic_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x,y

    def __len__(self):
        return len(self.target)
        
def test_img(net_g, datatest, args, vocab= None, char2index=None, n_sequence_length=80):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0

    if args.dataset.find("synthetic") > -1:
        data_x = torch.Tensor(datatest['x'])
        data_y = torch.Tensor(datatest['y'])
        data_y= data_y.type(torch.LongTensor)
        datatest = Synthetic_Dataset(data_x, data_y )
    elif args.dataset.find("shakespeare") > -1:
        data_x_array = np.array(datatest['x'])
        data_x = torch.as_tensor(data_x_array)
        data_y_array = np.array(datatest['y'])
        data_y = torch.as_tensor(data_y_array)
        data_y = data_y.type(torch.LongTensor)
        datatest = Synthetic_Dataset(data_x, data_y)

    # data_loader = DataLoader(datatest, batch_size=args.bs, num_workers =16)
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
     # initialize hidden states of lstm if using shakespeare dataset with lstm
    if args.dataset.find("shakespeare") > -1:
        hidden = net_g.init_hidden()

    # loss_Vn = np.zeros(args.total_UE)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            # if shakespeare, targets need to be processed first
            if args.dataset.find("shakespeare") == -1:
                target = target.cuda()

            data = data.cuda()
         # if using lstm, hidden states need to be considered as input and output of net
        if args.dataset.find("shakespeare") > -1:
            log_probs, hidden = net_g(data, hidden)
        else:
            log_probs = net_g(data)
        
        # processing targets
        if args.dataset == 'shakespeare':
            # target_new = np.zeros((len(target), n_sequence_length, len(vocab)))
            # for i in range(len(target)):
            #     for j in range(len(target[i])):
            #         target_new[i][j][target[i][j]] = 1
            target_new = get_new_labelVn(target, len(vocab))
            target = torch.as_tensor(target_new, dtype=torch.float)
            # print("target size ", target.size())
            if args.gpu != -1:
                target = target.cuda()

        #print("target", target)
        #print("log_probs", log_probs)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # if args.test_method=='exact_loss':
        #     loss_Vn[idx] = F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        if args.dataset.find("shakespeare") > -1:
            # get the index of the max log-probability
            # changed structure for log_probs (one additional dimension)
            y_pred = log_probs.data.max(dim=2, keepdim=True)[1]
            # get index of correct char
            target_index = target.data.max(dim=2, keepdim=True)[1]
            #print('pred', y_pred, 'test_loss', test_loss)
            correct += y_pred.eq(target_index.data.view_as(y_pred)).long().cpu().sum(dim=2).sum(dim=0)[n_sequence_length -1]
        #end erik
        else:
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            #print('pred', y_pred, 'test_loss', test_loss)
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()


    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss #, loss_Vn

