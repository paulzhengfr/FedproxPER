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
        
def test_img(net_g, datatest, args):
    """Calculate the test accuracy (Overall and per class)

    Args:
        net_g (_type_): model to be tested
        datatest (_type_): dataset to be tested on
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    
    net_g.eval()
    
    test_loss = 0
    correct = 0

    if args.dataset.find("synthetic") > -1:
        data_x = torch.Tensor(datatest['x'])
        data_y = torch.Tensor(datatest['y'])
        data_y= data_y.type(torch.LongTensor)
        datatest = Synthetic_Dataset(data_x, data_y )

    data_loader = DataLoader(datatest, batch_size=args.bs, num_workers=args.num_workers)
    l = len(data_loader)
    num_classes = len(data_loader.dataset.classes)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    total_samples = 0
    correct = 0
    class_accuracy = dict()

    for idx, (data, target) in enumerate(data_loader):
        target = target.cuda()
        data = data.cuda()
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct_batch = y_pred.eq(target.data.view_as(y_pred)).long().cpu()
        correct += correct_batch.sum().item()
        total_samples += len(target)

        for i in range(len(target)):
            label = target[i]
            class_correct[label] += correct_batch[i].item()
            class_total[label] += 1

    overall_accuracy = 100.0 * correct / total_samples
    accuracy = overall_accuracy
    # Print accuracy per class
    for i in range(num_classes):
        class_str = f"class {i} accuracy"
        class_accuracy[class_str] = 100.0 * class_correct[i] / class_total[i]
    #     print(f"Accuracy for class {i}: {class_accuracy}%")

    print(f"Overall Accuracy: {overall_accuracy}%")
    
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss, class_accuracy #, loss_Vn

