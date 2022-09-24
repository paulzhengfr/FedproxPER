#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy
from sklearn import metrics
from utils.save_results import update_loss

def get_new_labelVn(labels, num_val):
    labels_fl = labels.flatten()
    one_Vn = np.arange(len(labels_fl)) * num_val
    labels_fl_lala = labels_fl + one_Vn
    labels_new = np.zeros((labels.shape[0],labels.shape[1],num_val))
    labels_new_fl = labels_new.flatten()
    labels_new_fl[labels_fl_lala]=1
    return labels_new_fl.reshape(labels_new.shape)


class Synthetic_Dataset(Dataset):
        def __init__(self, data, target):
            self.data = data
            self.target = target

        def __getitem__(self, index):
            x = self.data[index]
            y = self.target[index]
            return x,y

        def __len__(self):
            return len(self.target)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    
    
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, user_id = None, vocab=None, char2index=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss().to(self.args.device)
        self.selected_clients = []
        if args.dataset.find("synthetic") == -1 and args.dataset.find("shakespeare") == -1:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        elif args.dataset.find("shakespeare") >= 0:
            data_x_array = np.array(dataset['x'])
            data_x = torch.as_tensor(data_x_array)
            data_y_array = np.array((dataset['y']))
            data_y = torch.as_tensor(data_y_array)
            
            data_y = data_y.type(torch.LongTensor)
            # print("data_y", data_y, "size", data_y.size())
            dataset_k = Synthetic_Dataset(data_x, data_y)
            self.ldr_train = DataLoader(dataset_k, batch_size=self.args.local_bs, shuffle=True)
            self.vocab = vocab

        else:
            data_x = torch.Tensor(dataset['x'])
            data_y = torch.Tensor(dataset['y'])
            data_y= data_y.type(torch.LongTensor)
            dataset_k = Synthetic_Dataset(data_x, data_y)
            self.ldr_train = DataLoader(dataset_k, batch_size=self.args.local_bs, shuffle=True, num_workers=16)
        self.user_id = user_id
        
    def train(self, net, n_sequence_length=80):
        net.train()
        # train and update
        #if self.args.dataset == "shakespeare": 
         #   optimizer = torch.optim.Adam(net.parameters(),lr=self.args.lr)
        #else:
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay = self.args.weight_decay)
        
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            # initialize hidden states for lstm
            if self.args.dataset.find("shakespeare") > -1:
                hidden = net.init_hidden()

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images = images.to(self.args.device)
                #start erik
                # do not move labels to device if shakespeare, need to be processed first
                if self.args.dataset.find('shakespeare') == -1:
                    labels = labels.to(self.args.device)

                # images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                # new variables because backprob, see udacity char rnn solution
                if self.args.dataset.find("shakespeare") > -1:
                    hidden = tuple([each.data for each in hidden])
                    # hidden = repackage_hidden( hidden)

                net.zero_grad()
                # added if-statement for shakespeare, hidden states need to be considered for lstm
                if self.args.dataset.find("shakespeare") > -1:
                    log_probs, hidden = net(images, hidden)
                else:
                    log_probs = net(images)

                # print("log_probs_training", log_probs)
                # processing labels for shakespeare dataset
                if self.args.dataset == 'shakespeare':
                    # labels_new = np.zeros((len(labels), n_sequence_length, len(self.vocab)))
                    # # print("labels train before processing ", labels, "size ", labels.size())
                    # for i in range(len(labels)):
                    #     for j in range(len(labels[i])):
                    #         labels_new[i][j][labels[i][j]] = 1 # Why 1.
                    labels_new = get_new_labelVn(labels, len(self.vocab))
                    # print("labels value example",labels[i][j])
                    labels = torch.as_tensor(labels_new, dtype=torch.float).to(self.args.device)
                    # print("labels", labels)
                    
                # labels = torch.max(labels, 1)[1].to(self.args.device)
                # print("log_probs", log_probs.size(), "labels", labels.size())
                loss = self.loss_func(log_probs, labels)
                # loss = self.loss_func(log_probs.t(), labels)

                loss.backward()
                if self.args.optimizer  == 'fedavg':
                    optimizer.step()
                elif self.args.optimizer  == 'fedprox':
                    for group in optimizer.param_groups:
                        for p in group['params'] :
                            if p.grad is None:
                                continue
                            d_p = p.grad.data
                            param_state = optimizer.state[p]
                            #param_state = copy.deepcopy(optimizer.state[p])
                            if 'old_init' not in param_state:
                                param_state['old_init']  = torch.clone(p.data).detach()
                            diff = p.data - param_state['old_init']
                            # d_p.add_(self.args.mu , diff)
                            # p.data.add_(-group['lr']  , d_p)
                            d_p.add_(diff, alpha = self.args.mu)
                            p.data.add_(d_p, alpha = -group['lr'])
                else:
                    print('Optimizer value wrong')
                    optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))

                batch_loss.append(loss.item())
                update_loss(self.args, -1, iter, batch_idx, loss.item(),-1, self.user_id)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            update_loss(self.args,-1, iter, batch_idx, -1,sum(batch_loss)/len(batch_loss), self.user_id)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), epoch_loss[-1]
    
    def forwardpass(self, net, n_sequence_length=80):
        net.eval()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay = self.args.weight_decay)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            # initialize hidden states for lstm
            if self.args.dataset.find("shakespeare") > -1:
                hidden = net.init_hidden()

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images = images.to(self.args.device)
                #start erik
                # do not move labels to device if shakespeare, need to be processed first
                if self.args.dataset.find('shakespeare') == -1:
                    labels = labels.to(self.args.device)

                # images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                # new variables because backprob, see udacity char rnn solution
                if self.args.dataset.find("shakespeare") > -1:
                    hidden = tuple([each.data for each in hidden])
                    # hidden = repackage_hidden( hidden)

                net.zero_grad()
                # added if-statement for shakespeare, hidden states need to be considered for lstm
                if self.args.dataset.find("shakespeare") > -1:
                    log_probs, hidden = net(images, hidden)
                else:
                    log_probs = net(images)

                # print("log_probs_training", log_probs)
                # processing labels for shakespeare dataset
                if self.args.dataset == 'shakespeare':
                    # labels_new = np.zeros((len(labels), n_sequence_length, len(self.vocab)))
                    # print("labels train before processing ", labels, "size ", labels.size())
                    # for i in range(len(labels)):
                    #     for j in range(len(labels[i])):
                    #         labels_new[i][j][labels[i][j]] = 1 # Why 1.
                    labels_new = get_new_labelVn(labels, len(self.vocab))
                    labels = torch.as_tensor(labels_new, dtype=torch.float).to(self.args.device)
                    # print("labels", labels)
                    
                # labels = torch.max(labels, 1)[1].to(self.args.device)
                # print("log_probs", log_probs.size(), "labels", labels.size())
                loss = self.loss_func(log_probs, labels)
                # if self.args.verbose and batch_idx % 10 == 0:
                #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         iter, batch_idx * len(images), len(self.ldr_train.dataset),
                #                100. * batch_idx / len(self.ldr_train), loss.item()))

                batch_loss.append(loss.item())
                # update_loss(self.args, -1, iter, batch_idx, loss.item(),-1, self.user_id)
                net.zero_grad()
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # update_loss(self.args,-1, iter, batch_idx, -1,sum(batch_loss)/len(batch_loss), self.user_id)
        return sum(epoch_loss) / len(epoch_loss)
        

def calc_exact_loss(args, dataset_train,  net_glob, dict_users = None, vocab = None, char2index= None):
    idxs_all_users = np.arange(args.total_UE)
    exact_loss = np.zeros(args.total_UE)
    for idx in idxs_all_users:
        if args.dataset.find("synthetic") == -1 and args.dataset.find("shakespeare") == -1:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], user_id = idx)
        elif args.dataset.find("shakespeare") > -1:
            local = LocalUpdate(args=args, dataset=dataset_train[idx], idxs=idx, user_id = idx, vocab = vocab, char2index = char2index)
        else:
            local = LocalUpdate(args=args, dataset=dataset_train[idx], idxs=idx, user_id = idx)
        loss = local.forwardpass(net=copy.deepcopy(net_glob).to(args.device))
        exact_loss[idx] = copy.deepcopy(loss)
    return exact_loss
