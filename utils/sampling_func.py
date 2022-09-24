# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:20:35 2021

@author: Paul Zheng
"""
import torch
from random import Random, shuffle
import numpy as np
from collections import OrderedDict
from tqdm import trange


#%% data Partitions
class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        to_return = self.data[data_idx]
        return to_return

class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """
    def __init__(self, data, num_clients, seed=1234, NonIID='iid', alpha=0.5, dataset=None):
        self.data = data
        self.dataset = dataset
        if NonIID == 'dirichlet':
            self.partitions, self.ratio, self.count = self.__getDirichletData__(data, num_clients, seed, alpha)

        elif NonIID == 'iid':
            self.partitions = []
            #self.ratio = sizes
            rng = Random()
            rng.seed(2020)
            data_len = len(data)
            indexes = [x for x in range(data_len)]
            rng.shuffle(indexes)
            frac = 1/num_clients
            for i in range(num_clients):
                part_len = int(frac * data_len)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]

        elif NonIID == 'niid':
            self.partitions, self.count = self.__getNonIIDdata__(data, num_clients, seed, alpha)
        else:
            print('NonIID value erroneous')

    #def use(self, partition):
    def use(self):
        #return Partition(self.data, self.partitions[partition])
        return self.partitions, self.count
    


    def __getNonIIDdata__(self, data, num_clients, seed, alpha):
        labelList = data.targets
        data_num = len(labelList)
        print('initial sum datapoint', data_num)
        rng = Random()
        rng.seed(12345) # + seed
        clients_num = num_clients
        # assign data to there labels
        a = [(label, idx) for idx, label in enumerate(labelList)]
        labelIdxDict = dict()
        for label, idx in a:
            label_int = int(label)
            labelIdxDict.setdefault(label_int,[])
            labelIdxDict[label_int].append(idx)
        labelNum = len(labelIdxDict)
        labelNameList = [key for key in labelIdxDict]
        num_data_perlabel = [len(labelIdxDict[i]) for i in range(labelNum)]
        print('number of data original to each label', num_data_perlabel)
        for i in range(labelNum):
            shuffle(labelIdxDict[i])
        labelIdxPointer = [0] * labelNum
        # sizes = number of nodes
        partitions = [list() for i in range(clients_num)]
        # partitions = {i: np.array([], dtype='int64') for i in range(clients_num)}
        eachPartitionLen= int(len(labelList)/clients_num)
        # majorLabelNumPerPartition = ceil(labelNum/len(partitions))
        majorLabelNumPerPartition = 2
        basicLabelRatio = alpha
        idx = np.zeros(labelNum, dtype = np.int64)
        idx_label_shuf = np.arange(labelNum)
        user_ind = np.zeros((clients_num,2))
        increment = int(data_num / clients_num / labelNum / majorLabelNumPerPartition * 0.8 )
        #increment = 21
        np.random.seed(2)
        for user in range(clients_num):
            #np.random.shuffle(idx_label_shuf)
            for j in range(majorLabelNumPerPartition):
                l = (user + j) % labelNum
                # partitions[user] = np.concatenate((partitions[user],labelIdxDict[l][idx[l]:idx[l] + increment]), axis =0)
                idl = idx_label_shuf[l]
                user_ind[user, j] = idl
                if idx[idl]+ increment < len(labelIdxDict[idl]):
                    partitions[user].append(labelIdxDict[idl][idx[idl]:idx[idl] + increment])
                    idx[idl]+=increment
                # if idl == 0:
                    # print('idx[idl]=', idx[idl])

        # Assign remaining sample
        user = 0
        props = np.random.lognormal(0, 2.0, (labelNum, clients_num // labelNum, 2))
        # props = np.ones((labelNum, clients_num // labelNum, 2))
        props = np.array([[[len(labelIdxDict[labelNameList[ind]]) - idxx]] for ind, idxx in enumerate(idx)]) * props/np.sum(props,(1,2),keepdims= True)
        for user in trange(clients_num):
            for j in range(majorLabelNumPerPartition):
                l = (user + j)%labelNum
                idl = int(user_ind[user,j])
                #print('idl',idl)
                num_samples = int(props[idl, user//labelNum, j])
                if idx[idl] + num_samples < len(labelIdxDict[idl]):
                    partitions[user].append(labelIdxDict[idl][idx[idl]:idx[idl]+num_samples])
                    # partitions[user] = np.concatenate((partitions[user],labelIdxDict[l][idx[l]:idx[l]+num_samples]), axis =0)
                    idx[idl]+=num_samples
                elif idx[idl] < len(labelIdxDict[idl]):
                    partitions[user].append(labelIdxDict[idl][idx[idl]:])
                    idx[idl] = len(labelIdxDict[idl])
                # if idl == 0:
                    # print('idx[idl]=', idx[idl], ' lala' ,len(labelIdxDict[idl]))
        #print(idx)
        print('number of data assigned to each label', idx)

        for user in range(clients_num):
            partitions[user] = [u for v in partitions[user] for u in v]
        # interval = 1
        # labelPointer = 0

        # #basic part
        # for partPointer in range(clients_num):
        #     requiredLabelList = list()
        #     for _ in range(majorLabelNumPerPartition):
        #         requiredLabelList.append(labelPointer)
        #         labelPointer += interval
        #         if labelPointer > labelNum - 1:
        #             labelPointer = interval
        #             interval += 1
        #         if interval >= labelNum /2 + 1:
        #             interval = 1
        #     for labelIdx in requiredLabelList:
        #         start = labelIdxPointer[labelIdx]
        #         idxIncrement = int(basicLabelRatio*len(labelIdxDict[labelNameList[labelIdx]]))
        #         partitions[partPointer].extend(labelIdxDict[labelNameList[labelIdx]][start:start+ idxIncrement])
        #         labelIdxPointer[labelIdx] += idxIncrement

        # #random part
        # remainLabels = list()
        # for labelIdx in range(labelNum):
        #     remainLabels.extend(labelIdxDict[labelNameList[labelIdx]][labelIdxPointer[labelIdx]:])
        # rng.shuffle(remainLabels)
        # for partPointer in range(clients_num):
        #     idxIncrement = eachPartitionLen - len(partitions[partPointer])
        #     partitions[partPointer].extend(remainLabels[:idxIncrement])
        #     rng.shuffle(partitions[partPointer])
        #     remainLabels = remainLabels[idxIncrement:]

        ###  Print the statistics
        net_dataidx_map = {}
        labelList = np.array(data.targets)
        for j in range(num_clients):
            np.random.shuffle(partitions[j])
            net_dataidx_map[j] = partitions[j]

        net_cls_counts = {}
        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp
        print('Data statistics: %s' % str(net_cls_counts))
        
        ### Create a list of users with corresponding 
        return partitions, net_cls_counts

    def __getDirichletData__(self, data, num_clients, seed, alpha):
        n_nets = num_clients
        K = 10 # number of class in the dataset
        labelList = np.array(data.targets)

        min_size = 0
        N = len(labelList)
        np.random.seed(2020) # + seed

        net_dataidx_map = {}
        while min_size < K:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(labelList == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp
        print('Data statistics: %s' % str(net_cls_counts))

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes/np.sum(local_sizes)
        print(weights)

        return idx_batch, weights, net_cls_counts


#%% Evaluate, metrics.
class Meter(object):
    """ Computes and stores the average, variance, and current value """

    def __init__(self, init_dict=None, ptag='Time', stateful=False,
                 csv_format=True):
        """
        :param init_dict: Dictionary to initialize meter values
        :param ptag: Print tag used in __str__() to identify meter
        :param stateful: Whether to store value history and compute MAD
        """
        self.reset()
        self.ptag = ptag
        self.value_history = None
        self.stateful = stateful
        if self.stateful:
            self.value_history = []
        self.csv_format = csv_format
        if init_dict is not None:
            for key in init_dict:
                try:
                    # TODO: add type checking to init_dict values
                    self.__dict__[key] = init_dict[key]
                except Exception:
                    print('(Warning) Invalid key {} in init_dict'.format(key))

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0
        self.sqsum = 0
        self.mad = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sqsum += (val ** 2) * n
        if self.count > 1:
            self.std = ((self.sqsum - (self.sum ** 2) / self.count)
                        / (self.count - 1)
                        ) ** 0.5
        if self.stateful:
            self.value_history.append(val)
            mad = 0
            for v in self.value_history:
                mad += abs(v - self.avg)
            self.mad = mad / len(self.value_history)

    def __str__(self):
        if self.csv_format:
            if self.stateful:
                return str('{dm.val:.3f},{dm.avg:.3f},{dm.mad:.3f}'
                           .format(dm=self))
            else:
                return str('{dm.val:.3f},{dm.avg:.3f},{dm.std:.3f}'
                           .format(dm=self))
        else:
            if self.stateful:
                return str(self.ptag) + \
                       str(': {dm.val:.3f} ({dm.avg:.3f} +- {dm.mad:.3f})'
                           .format(dm=self))
            else:
                return str(self.ptag) + \
                       str(': {dm.val:.3f} ({dm.avg:.3f} +- {dm.std:.3f})'
                           .format(dm=self))
def comp_accuracy(output_remote, target_remote, location = 'local', topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    if location == 'local':
        target = target_remote.copy().get()

        output = output_remote.copy().get()
    elif location == 'server':
        target = target_remote
        output = output_remote
    else:
        print('comp accuracy calculation location not specified.')
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def evaluate(model, test_loader, cuda_bool = False):
    model.eval()
    top1 = Meter(ptag='Acc')

    with torch.no_grad():
        for data, target in test_loader:
            if cuda_bool:
                data = data.cuda(non_blocking = True)
                target = target.cuda(non_blocking = True)
            outputs = model(data)

            acc1 = comp_accuracy(outputs, target, 'server')
            top1.update(acc1[0].item(), data.size(0))
    model.train()
    return top1.avg

def evaluate_loss(model, train_loader,criterion, cuda_bool = False):
    model.eval()
    top1 = Meter(ptag='loss')

    with torch.no_grad():
        for data, target in train_loader:
            if cuda_bool:
                data = data.cuda(non_blocking = True)
                target = target.cuda(non_blocking = True)
            outputs = model(data)
            loss = criterion(outputs, target)
            top1.update(loss.item(), data.size(0))
    model.train()
    return top1.avg


#%%
def if_toUpdate_salehi(weight, weight_pre, formu):
    if np.sum(np.abs(weight)) + np.sum(np.abs(weight_pre))== 0:
        if formu == 'salehi' or formu == 'log-salehi':
            return True
    return False

def calc_later_weights(nb_trained_users, args, loss_weights, datasize_weight,salehi_weight, salehi_weight_pre, calculate_coef_S, calculate_coef_rand):
    vanishing_rounds = args.vanish
    weights = (calculate_coef_S(nb_trained_users, vanishing_rounds*args.total_UE) * loss_weights)
    if args.formulation == 'orig':
        later_weights = calculate_coef_rand(nb_trained_users,2* args.total_UE) * (datasize_weight+0.01* np.random.rand(args.total_UE))
    elif args.formulation == 'exact':
        later_weights = calculate_coef_rand(nb_trained_users,vanishing_rounds* args.total_UE) * np.power(np.random.rand(args.total_UE),1/datasize_weight)
    elif args.formulation == 'log-exact':
        later_weights = - calculate_coef_rand(nb_trained_users,vanishing_rounds* args.total_UE) * np.log(np.random.rand(args.total_UE))/datasize_weight
    elif args.formulation == 'salehi':
        if if_toUpdate_salehi(salehi_weight,salehi_weight_pre, args.formulation):
            salehi_weight_pre = datasize_weight / wireless_arg['success prob']
            salehi_weight = CS_salehi(args.total_UE,args.active_UE, salehi_weight_pre, 1)
        later_weights = calculate_coef_rand(nb_trained_users,vanishing_rounds* args.total_UE) * np.power(np.random.rand(args.total_UE),1/salehi_weight)
    elif args.formulation == 'log-salehi':
        if if_toUpdate_salehi(salehi_weight,salehi_weight_pre, args.formulation):
            salehi_weight_pre = datasize_weight / wireless_arg['success prob']
            salehi_weight = CS_salehi(args.total_UE,args.active_UE, salehi_weight_pre, 1)
        later_weights = - calculate_coef_rand(nb_trained_users,vanishing_rounds* args.total_UE) *np.log(np.random.rand(args.total_UE))/salehi_weight
    else:
        later_weights = calculate_coef_rand(nb_trained_users,vanishing_rounds* args.total_UE) * np.power(np.random.rand(args.total_UE),1/datasize_weight)
    return later_weights, salehi_weight,salehi_weight_pre


def create_increasing_decr_functions(num_trained_users, args):
    N = args.total_UE
    if args.process_function_form == 'default':
        def calculate_coef_S(num_trained_users, N):
            return (np.exp((N - num_trained_users) / N) - 0.9) / (np.exp(1) - 0.9)
        def calculate_coef_rand(num_trained_users, N):
            return (np.exp(num_trained_users / N) - 0.9) / (np.exp(1) - 0.9) #* 0.002
    elif args.process_function_form == 'curvy':
        def calculate_coef_S(num_trained_users, N):
            return (np.exp((N - num_trained_users) / N * 3) - 0.9) / (np.exp(3) - 0.9)
        def calculate_coef_rand(num_trained_users, N):
            return (np.exp(num_trained_users / N * 3) - 0.9) / (np.exp(3) - 0.9) #* 0.002
    elif args.process_function_form == 'other_side':
        def calculate_coef_S(num_trained_users, N):
            return (1 - 0.99 * np.exp(-(N - num_trained_users) / N * 3)) / (1 - 0.99 * np.exp(-3))
        def calculate_coef_rand(num_trained_users, N):
            return (1 - 0.99 * np.exp(-num_trained_users / N * 3)) / (1 - 0.99 * np.exp(-3))
    elif args.process_function_form == 'other_side1':
        def calculate_coef_S(num_trained_users, N):
            return (1 - 0.99 * np.exp(-(N - num_trained_users) / N * 3)) / (1 - 0.99 * np.exp(-3))
        def calculate_coef_rand(num_trained_users, N):
            return (1 - calculate_coef_S(num_trained_users, N))
    elif args.process_function_form == 'test':
        def calculate_coef_S(num_trained_users, N):
            return 0
        def calculate_coef_rand(num_trained_users, N):
            return 1
    else:
        print("Process function form argument is erroneous")
        def calculate_coef_S(num_trained_users, N):
            return (1 - 0.99 * np.exp(-(N - num_trained_users) / N * 3)) / (1 - 0.99 * np.exp(-3))
        def calculate_coef_rand(num_trained_users, N):
            return (1 - 0.99 * np.exp(-num_trained_users / N * 3)) / (1 - 0.99 * np.exp(-3))
    return calculate_coef_S, calculate_coef_rand

#%% Wireless communication

#%% Later to add
# class FedProx(torch.optim.optimizer.Optimizer):
#     def __init__(self, params, ratio, gmf, lr=torch.optim.optimizer.required, momentum=0, dampening=0,
#                  weight_decay=0, nesterov=False, variance=0, mu=0):
#         self.gmf = gmf
#         self.ratio = ratio
#         self.itr = 0
#         self.a_sum = 0
#         self.mu = mu
#         if lr is not required and lr < 0.0:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
#                         weight_decay=weight_decay, nesterov=nesterov, variance=variance)
#         if nesterov and (momentum <= 0 or dampening != 0):
#             raise ValueError("Nesterov momentum requires a momentum and zero dampening")
#         super(FedProx, self).__init__(params, defaults)

#     def __setstate__(self, state):
#         super(FedProx, self).__setstate__(state)
#         for group in self.param_groups:
#             group.setdefault('nesterov', False)
#     def step(self, closure=None):
#         loss = None
#         for group in optimizer_c.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 d_p = p.grad.data
#                 param_state = optimizer_c.state[p]
#                 if 'old_init' not in param_state:
#                     param_state['old_init'] = torch.clone(p.data).detach()
#                 d_p.add_(mu, p.data - param_state['old_init'])
#                 p.data.add_(-group['lr'], d_p)
#         return loss
# algorithm = {'fedprox': FedProx}
# selected_opt = algorithm[args['optimizer']]
