# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 17:32:36 2021

@author: paulz

Have already a json file
already partitioned to each user.
We need to call data set when to use it.
Just call  dataloader as in https://github.com/CharlieDinh/DONE/blob/e5c62685c8e10e80d862f45c439c0bad8078013d/algorithms/edges/edgebase.py#L126

"""
import os
import json

#%% Read the json synthetic data
def format_train_data(train_data, test_data):
    new_train_data = dict()
    new_test_data = dict()
    for i in range(30):
        uname = 'f_{0:05d}'.format(i)
        newname = 'user{:d}'.format(i)
        new_train_data[newname] = train_data[uname]
        new_test_data[newname] = test_data[uname]
    return new_train_data, new_test_data
def aggregate_test_data(test_data):
    agg_test_data_x = []
    agg_test_data_y = []
    for i in range(len(test_data.keys())):
        k = 'user%d'%(i,)
        agg_test_data_x  = agg_test_data_x  + test_data[k]['x']
        agg_test_data_y = agg_test_data_y + test_data[k]['y']
    aggreg_test_data = {'x': agg_test_data_x,
                        'y': agg_test_data_y}
    return aggreg_test_data

# aggregation fct for shakespeare dataset;
# usually, we would append the whole dataset of user x in each iteration of the outer loop,
# but then test-dataset becomes to big, so we take every 100th sample
def aggregate_test_data_shakespeare(test_data, clients):
    agg_test_data_x = []
    agg_test_data_y = []
    for i in range(len(test_data.keys())):
        for j in range(len(test_data[clients[i]]['x'])):
            if j % 100 == 0:
                agg_test_data_x  = agg_test_data_x  + [(test_data[clients[i]]['x'])[j]]
                agg_test_data_y = agg_test_data_y + [(test_data[clients[i]]['y'])[j]]
    aggreg_test_data = {'x': agg_test_data_x,
                        'y': agg_test_data_y}
    return aggreg_test_data


def read_data(train_data_dir, test_data_dir, dataset):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])



    # added if statement for shakespeare
    if dataset.find("shakespeare") > -1:
        clients = list(train_data.keys())
        return clients, groups, train_data, aggregate_test_data_shakespeare(test_data, clients)
    else:
        new_train, new_test = format_train_data(train_data, test_data)
        clients = list(new_train.keys())
        return clients, groups, new_train, aggregate_test_data(new_test)


#dataset = 'synthetic'
#train_data_dir = os.path.join('data',dataset, 'train')
#test_data_dir = os.path.join('data',dataset, 'test')
#clients, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
