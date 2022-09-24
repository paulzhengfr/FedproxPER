import numpy as np
import copy
def init_optil_weights(args, wireless_arg):
    weights = np.zeros(args.total_UE)
    later_weights = np.zeros(args.total_UE)
    if args.selection == 'solve_opti_loss_size':
        weights = copy.deepcopy(args.datasize_weight)
    elif args.selection == 'solve_opti_loss_size2' or args.selection == 'solve_opti_loss_size4':
        # weights = args.datasize_weight * 60000
        # weights = args.datasize_weight / np.max(args.datasize_weight) * 10
        weights =  np.ones(args.total_UE)*args.eta_init + args.datasize_weight/np.mean(args.datasize_weight)
        later_weights = np.zeros(args.total_UE)
    elif args.selection == 'solve_opti_loss':
        weights =  np.ones(args.total_UE)
    return weights, later_weights

def update_optil_weights(args, wireless_arg, loss_weights, num_trained):
    weights = np.zeros(args.total_UE)
    later_weights = np.zeros(args.total_UE)
    coef_decr = 0
    coef_incr = 0
    if args.selection == 'solve_opti_loss_size':
        weights = args.datasize_weight + loss_weights
    elif args.selection == 'solve_opti_loss_size2' or args.selection == 'solve_opti_loss_size3':
        vanishing_rounds = args.vanish
        calculate_coef_S, calculate_coef_rand = create_increasing_decr_functions(num_trained, args, vanishing_rounds*args.total_UE)
        
        salehi_weight = wireless_arg['salehi_weight_sampling']
        if not args.no_later:
            later_weights, coef_incr = calc_later_weights(num_trained, args, salehi_weight, calculate_coef_rand)
        
        coef_decr = calculate_coef_S(num_trained, vanishing_rounds * args.total_UE)
        if args.no_later:
            weight_normalization = 1
            weights = loss_weights
        else:
            weight_normalization = np.mean(later_weights / coef_incr)/np.max(loss_weights)
            if args.weight_normalization == 'mean':
                weight_normalization = np.mean(later_weights / coef_incr)/np.mean(loss_weights)
            elif args.weight_normalization == 'median':
                weight_normalization = np.mean(later_weights / coef_incr)/np.median(loss_weights)
            weights = coef_decr * loss_weights * weight_normalization
        # normalize the loss weights to one so it's comparable to the random weights (between 0,1).
    elif args.selection == 'solve_opti_loss_size4':
        vanishing_rounds = args.vanish
        calculate_coef_S, calculate_coef_rand = create_increasing_decr_functions(num_trained, args, vanishing_rounds*args.total_UE)
        coef_decr = calculate_coef_S(num_trained, vanishing_rounds * args.total_UE)
        coef_normalization = 1/ np.max(args.datasize_weight) / np.max(loss_weights)
        weights = coef_decr * loss_weights * args.datasize_weight * coef_normalization
        salehi_weight = wireless_arg['salehi_weight_sampling']
        later_weights, coef_incr = calc_later_weights(num_trained, args, salehi_weight, calculate_coef_rand)
    elif args.selection == 'solve_opti_loss':
        weights =  loss_weights
    return weights, later_weights, coef_decr, coef_incr






#%%

def calc_later_weights(nb_trained_users, args, salehi_weight,calculate_coef_rand):
    vanishing_rounds = args.vanish
    datasize_weight = copy.deepcopy(args.datasize_weight)
    if args.formulation == 'orig':
        later_weights = calculate_coef_rand(nb_trained_users,vanishing_rounds* args.total_UE) * (datasize_weight+0.01* np.random.rand(args.total_UE))
    elif args.formulation == 'exact':
        later_weights = calculate_coef_rand(nb_trained_users,vanishing_rounds* args.total_UE) * np.power(np.random.rand(args.total_UE),1/datasize_weight)
    elif args.formulation == 'log-exact':
        later_weights = - calculate_coef_rand(nb_trained_users,vanishing_rounds* args.total_UE) * np.log(np.random.rand(args.total_UE))/datasize_weight
    elif args.formulation == 'salehi':
        #salehi_weight = wireless_arg['salehi_weight_sampling']
        later_weights = calculate_coef_rand(nb_trained_users,vanishing_rounds* args.total_UE) * np.power(np.random.rand(args.total_UE),1/salehi_weight)
    elif args.formulation == 'log-salehi':
        #salehi_weight = wireless_arg['salehi_weight_sampling']
        later_weights = - calculate_coef_rand(nb_trained_users,vanishing_rounds* args.total_UE) *np.log(np.random.rand(args.total_UE))/salehi_weight
    elif args.formulation == 'product':
        #salehi_weight = wireless_arg['salehi_weight_sampling']
        later_weights = 1+ np.power(np.random.rand(args.total_UE),1/datasize_weight)
    else:
        later_weights = calculate_coef_rand(nb_trained_users,vanishing_rounds* args.total_UE) * np.power(np.random.rand(args.total_UE),1/datasize_weight)
    return later_weights * args.later_weights_coef, calculate_coef_rand(nb_trained_users, vanishing_rounds*args.total_UE)


def create_increasing_decr_functions(num_trained_users, args, N):
    if num_trained_users >= N:
        def calculate_coef_S(num_trained_users,N):
            return 0
        def calculate_coef_rand(num_trained_users,N):
            return 1
        return calculate_coef_S, calculate_coef_rand
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
    elif args.process_function_form == 'other_side_curv':
        def calculate_coef_S(num_trained_users, N):
            return (1 - 0.999 * np.exp(-(N - num_trained_users) / N * 10)) / (1 - 0.999 * np.exp(-10))
        def calculate_coef_rand(num_trained_users, N):
            return (1 - calculate_coef_S(num_trained_users, N))
    elif args.process_function_form == 'other_side1':
        def calculate_coef_S(num_trained_users, N):
            return (1 - np.exp(-(N - num_trained_users) / N * args.curvy)) / (1 -  np.exp(-args.curvy))
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



def update_success_trained(args, active_clients, list_trained, bool_trained, vanish_index):
    if len(list_trained) == 0:
        print("len list_trained is 0")
        list_trained =  np.zeros(args.total_UE)
        bool_trained =  np.zeros(args.total_UE)
    list_trained[active_clients] += 1
    bool_trained[list_trained >= vanish_index] = vanish_index
    nb_trained_users = np.sum(bool_trained)
    if nb_trained_users / args.total_UE / vanish_index > 0.9:
        vanish_index = vanish_index + 1
    return nb_trained_users, list_trained, bool_trained, vanish_index
