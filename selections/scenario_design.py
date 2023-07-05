import numpy as np
from random import shuffle

def get_best2labels(net_cls):
    # For Dirichelt distribution.
    best2_dict = {}
    for key, val in net_cls.items():
        # print(key)
        val_keys  = list(val.keys())
        val_vals = list(val.values())
        val_vals_sort = np.argsort(val_vals)


        if len(val) > 2:
            best_2labels = [val_keys[val_vals_sort[-1]], val_keys[val_vals_sort[-2]]]
            # print(best_2labels)
        # for label, nb in val.items()
        else:
            best_2labels = val_keys
        best2_dict[key] = best_2labels
    return best2_dict

## Choose assign user's distance to BS according to best2_dict.

# def assign_distance(dist_vec, user_labels):
#     dist_argsort = np.argsort(dist_vec)
#     new_dist = np.zeros(len(dist_vec))

#     ind_close = []
#     ind_far = []
#     # for user, labels in user_labels.items():
#     #     if 8 in labels or 9 in labels:
#     #         ind_close.append(user)
#     #     else:
#     #         ind_far.append(user)
#     for user, labels in user_labels.items():
#         if 9 in labels:
#             ind_close.append(user)
#         else:
#             ind_far.append(user)
#     dist_vec_np = np.array(dist_vec)
#     ind_close = np.array(ind_close)
#     ind_far = np.array(ind_far)
#     dist_close = dist_vec_np[dist_argsort[0:len(ind_close)]]
#     # shuffle(dist_close)
#     new_dist[ind_close-1] =dist_close

#     dist_far = dist_vec_np[dist_argsort[len(ind_close):]]
#     # shuffle(dist_far)
#     new_dist[ind_far-1] = dist_far
#     return new_dist

def assign_distance(args, wireless_arg):
    # print('length of user_labels',len(user_labels))
    # new_dist = np.zeros(len(user_labels)) + 10
    old_dist = wireless_arg['distance']
    new_dist = np.ones(len(old_dist))
    if args.data_distr_scenarios == "inverse_datasize":
        dist_sort = np.sort(old_dist)
        data_size_order = np.argsort(args.datasize_weight)
        new_dist = dist_sort[data_size_order]
    elif args.data_distr_scenarios == "label_separated-b":
        user_labels = args.users_count_labels
        for user, labels in user_labels.items():
            if 8 in labels and 9 in labels:
                new_dist[user] = wireless_arg['radius'] * 1.3 
            if 9 in labels and 0 in labels:
                new_dist[user] = wireless_arg['radius'] * 1.3
            if 8 in labels and 7 in labels:
                new_dist[user] = wireless_arg['radius'] * 1.3
            if 6 in labels and 7 in labels:
                new_dist[user] = wireless_arg['radius']
            if 6 in labels and 5 in labels:
                new_dist[user] = wireless_arg['radius']
            if 5 in labels and 4 in labels:
                new_dist[user] = wireless_arg['radius']
            if 4 in labels and 3 in labels:
                new_dist[user] = wireless_arg['radius']
            if 3 in labels and 2 in labels:
                new_dist[user] = wireless_arg['radius']
            if 2 in labels and 1 in labels:
                new_dist[user] = wireless_arg['radius']/10 
            if 1 in labels and 0 in labels:
                new_dist[user] = wireless_arg['radius']/10 
    elif args.data_distr_scenarios == "label_separated-a":
        user_labels = args.users_count_labels
        for user, labels in user_labels.items():
            if 8 in labels and 9 in labels:
                new_dist[user] = wireless_arg['radius'] 
            if 9 in labels and 0 in labels:
                new_dist[user] = wireless_arg['radius'] 
            if 8 in labels and 7 in labels:
                new_dist[user] = wireless_arg['radius']
            if 6 in labels and 7 in labels:
                new_dist[user] = wireless_arg['radius']
            if 6 in labels and 5 in labels:
                new_dist[user] = wireless_arg['radius']
            if 5 in labels and 4 in labels:
                new_dist[user] = wireless_arg['radius']
            if 4 in labels and 3 in labels:
                new_dist[user] = wireless_arg['radius']
            if 3 in labels and 2 in labels:
                new_dist[user] = wireless_arg['radius']
            if 2 in labels and 1 in labels:
                new_dist[user] = wireless_arg['radius']/10 
            if 1 in labels and 0 in labels:
                new_dist[user] = wireless_arg['radius']/10 

    return new_dist