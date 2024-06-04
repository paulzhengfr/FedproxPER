import numpy as np
from selections.function_user_selection import user_selection_opti, naif_power_allocation
from selections.function_user_selection_allCombinations import choose_comb, solve_opti_all, solve_opti_wrt_P
from selections.function_LR import opti_LR, opti_LR_for_naif
from selections.main_otherClientselection import CS_salehi
from selections.scenario_design import get_best2labels, assign_distance
from selections.function_AoU import solve_cstObj_pb
import torch
import copy
import pickle 
# import wandb


def dB2power(x):
    return np.exp(x/10*np.log(10))

def wireless_param(args, data_weight, nb_data_assigned):
    wireless_arg = {
        'radius':args.cell_radius,
        'ampli': 15,
        'N0':dB2power(-150),
        'B': 1e6,
        'm': dB2power(0.023),
        'M' : 16,
        'Mprime': 15,
        'E_max':60, #mJ
        'Tslot': 1.3,
        'sigma':args.sigma, #1
        'freq' : 2400, #Mhz
        'P_max': 10, #mW
        'alpha':0.1,
        'beta': 0.001,
        'kappa':10**(-28),
        'freq_comp': 2*10**9,
        'C': 2*10**4
        }
    np.random.seed(args.wireless_seed )
    wireless_arg['distance']  = np.sqrt(np.random.uniform(1, wireless_arg['radius']  ** 2, args.total_UE))
    
    if args.data_distr_scenarios != "none":
        wireless_arg['distance']  = assign_distance(args, wireless_arg)
    # print("Final Distance vector:",wireless_arg['distance'])
    
    FSPL = 20 * np.log10(wireless_arg['distance']) + 20 * np.log10(wireless_arg['freq']) - 27.55
    wireless_arg['FSPL'] = dB2power(FSPL)

    wireless_arg['P_sum'] = wireless_arg['E_max'] / wireless_arg['Tslot']
    if wireless_arg['sigma'] == 0:
        o_avg = 2
    else:
        o_avg = wireless_arg['sigma']   * np.sqrt(np.pi /2) # Rayleigh distribution mean
    wireless_arg['h_avg'] = o_avg  / wireless_arg['FSPL']

    snr = wireless_arg['P_max']  * wireless_arg['h_avg'] / wireless_arg['N0'] /wireless_arg['B']
    wireless_arg['success prob'] = np.exp(-wireless_arg['m'] / snr)
    print("Average success probability", np.mean(wireless_arg['success prob']))
    wireless_arg['theta'] = wireless_arg['kappa']*wireless_arg['freq_comp']**2 *wireless_arg['C']*args.local_ep * nb_data_assigned

    all_comb = None
    if args.opti == 'P1' or args.opti == 'P2' or args.opti =='P1_3' or args.opti == 'P1_4':
        wireless_arg['list2choose'] = list(range(args.total_UE))
        wireless_arg['all_comb'] = choose_comb(wireless_arg['list2choose'], args.active_UE)
    #if args.selection == 'salehi' or args.formulation == 'salehi' or args.formulation == 'log-salehi':
    salehi_weight = data_weight / wireless_arg['success prob']
    weight_sampling = CS_salehi(args.total_UE,args.active_UE, salehi_weight, 1)
    wireless_arg['salehi_weight'] = copy.deepcopy(salehi_weight)
    wireless_arg['salehi_weight_sampling'] = copy.deepcopy(weight_sampling)
    return wireless_arg

def update_wireless(args, wireless_arg, seed):
    # Update fading coefficient
    np.random.seed(seed)
    if wireless_arg['sigma'] == 0:
        h_i = 2  / wireless_arg['FSPL']
        o_i = np.ones(args.total_UE)
    else:
        o_i = wireless_arg['sigma']   * np.sqrt(np.square(np.random.randn(args.total_UE)) + np.square(np.random.randn(args.total_UE))) # Rayleigh distribution
        h_i = o_i  / wireless_arg['FSPL']
    
    #h_avg = (h_avg * training_round + h_i) / (training_round + 1)
    wireless_arg['o_i']   = copy.deepcopy(o_i)
    wireless_arg['h_i']   = copy.deepcopy(h_i)
    return wireless_arg

def objective_funtion(x, alpha, h_i, S_i, data_size, const, later_weights,P_max, incr,decr):
    vec = S_i * np.exp(-alpha / h_i / x) + later_weights
    return np.sum(vec[x>0])


def user_selection(args, wireless_arg, seed,  data_size,weights,later_weights):
    if args.opti != "WF" and args.opti !="LR" and args.opti !="cstWObj":
        all_comb = copy.deepcopy(wireless_arg['all_comb'])
        list2choose = copy.deepcopy(wireless_arg['list2choose'])
    user_indices = [k for k in range(args.total_UE )]
    h_i = copy.deepcopy(wireless_arg['h_i'])
    h_avg =  copy.deepcopy(wireless_arg['h_avg'])
    N0 = wireless_arg['N0']
    B = wireless_arg['B']
    m = wireless_arg['m']
    const_alpha = N0 * B / m
    P_max = wireless_arg['P_max']
    P_sum = wireless_arg['P_sum']
    theta = wireless_arg['theta']
    Tslot = wireless_arg['Tslot']
    
    #print("decreasing value", wireless_arg['decr'] )
    K = args.active_UE
    if args.selection  == 'uni_random' or args.selection  == 'best_channel' or args.selection  == 'best_channel_ratio' or args.selection  == 'best_loss' or args.selection  == 'weighted_random' \
        or args.selection == 'salehi' or (wireless_arg['decr'] == 0 and (not args.no_later or not args.selection  == 'solve_opti_laterW')):
        if args.selection == 'uni_random':
            np.random.seed(seed)
            active_clients = np.random.choice(user_indices, args.active_UE , replace = False)
        elif args.selection  == 'best_channel':
            active_clients =  np.argsort(-h_avg)[0:args.active_UE]
        elif args.selection  == 'best_channel_ratio':
            active_clients = np.argsort(-h_i / wireless_arg['h_avg'])[0:args.active_UE]
        elif args.selection  == 'best_datasize_success_rate':
            importance_vector = wireless_arg['success prob'] * data_size
            active_clients =  np.argsort(-importance_vector)[0:args.active_UE]
        elif args.selection  == 'best_loss':
            active_clients = np.argsort(-weights)[0:args.active_UE]
        elif args.selection == 'best_exact_loss':
            active_clients = np.argsort(-args.exact_loss)[0:args.active_UE]
        elif args.selection  == 'weighted_random':
            torch.manual_seed(seed )
            active_clients = list(torch.utils.data.WeightedRandomSampler(data_size, args.active_UE, replacement = False))
        elif wireless_arg['decr'] == 0:
            torch.manual_seed(seed + 12345)
            active_clients = list(torch.utils.data.WeightedRandomSampler(data_size, args.active_UE, replacement = False))
        elif args.selection == 'salehi':
            salehi_weight = data_size / wireless_arg['success prob']
            weight_sampling = CS_salehi(args.total_UE,args.active_UE, salehi_weight, 1)
            active_clients = list(torch.utils.data.WeightedRandomSampler(weight_sampling , args.active_UE, replacement = False))
            # if checkIfDuplicates_1(active_clients):
            #     print("-----------------Duplicates of users--------")

        h_avg_p = copy.deepcopy(h_avg[active_clients])
        data_size_p = copy.deepcopy(data_size[active_clients])
        later_weights_p = copy.deepcopy(later_weights[active_clients])
        weights_uni = np.ones(args.active_UE)
        power_allocated = np.zeros(len(user_indices))
        if args.allocate_power:
            if args.opti == 'LR':
                _, power_allocated[active_clients], _, _ = opti_LR_for_naif( const_alpha, h_avg_p, weights_uni, P_max, P_sum,data_size_p, theta, later_weights_p)
            else:
                try:
                    _, power_allocated[active_clients] = solve_opti_wrt_P(weights_uni, h_avg_p, N0,B, m, args.active_UE, wireless_arg['alpha']  , wireless_arg['beta'] , P_max,P_sum, 'P1',data_size_p, theta, Tslot, later_weights_p)
                except:
                    print("-------------------------------------")
                    print("solve_opti_wrt_P error occurs")
                    try:
                        _, power_allocated[active_clients] = solve_opti_wrt_P(np.ones(args.active_UE), h_avg_p, N0,B, m, args.active_UE, wireless_arg['alpha']  , wireless_arg['beta'] , P_max,P_sum, 'P1',data_size_p,theta, Tslot, later_weights_p)
                    except:
                        print("Still errors")
                        power_allocated[active_clients] = np.ones(args.active_UE) * min(P_max, P_sum / args.active_UE)
        else:
            la = 1
            power_allocated[active_clients] = naif_power_allocation(P_max, la, P_sum, theta, Tslot, data_size_p, m, B, N0, h_avg_p)
        #power_allocated[active_clients] = min(P_sum / args.active_UE , P_max)
        
        run_time = 0
        message = ''
        obj_value = objective_funtion(power_allocated, const_alpha, h_avg, np.ones(args.total_UE), data_size, theta, later_weights,P_max, 1,1)
    elif (args.selection  == 'solve_opti_loss_size' or args.selection  == 'solve_opti_loss_size2' or args.selection  == 'solve_opti_loss_size3' or 
          args.selection  == 'solve_opti_loss_size4' or args.selection  =='solve_opti_size' or args.selection  == 'solve_opti_loss' or \
              args.selection == 'solve_opti_AoU' or args.selection == 'solve_opti_laterW'):
        if args.opti == 'P4':
            user_selected,power_allocated, message,_,run_time=user_selection_opti(args.active_UE , wireless_arg['M']  ,wireless_arg['Mprime'],
                                                                weights, h_avg, N0*B, m,
                                                                wireless_arg['alpha']  , wireless_arg['beta'] ,
                                                                P_max, P_sum)
            active_clients = np.nonzero(user_selected)[0]
        elif args.opti == 'P1' or args.opti == 'P2' or args.opti == 'P1_3' or args.opti == 'P1_4':
            message = 'yoyo'
            active_clients,power_active, run_time= solve_opti_all(all_comb, list2choose,  weights, h_avg, N0,B, m, K,
                                                              wireless_arg['alpha']  , wireless_arg['beta'],
                                                              P_max,P_sum,args.opti,data_size,theta, Tslot, later_weights)

            power_allocated = np.zeros(len(user_indices))
            power_allocated[active_clients] = power_active
        elif args.opti == 'LR' :
            message = 'yoyo'
            const_alpha = N0 * B * m
            const = theta / Tslot
            weights_to_use = copy.deepcopy(weights)
            
            if args.test_method == 'uni_weights':
                weights_to_use = np.ones(args.total_UE) * wireless_arg['decr']
            elif args.test_method == 'exact_loss': # sum loss
                weights_to_use = args.exact_loss* wireless_arg['decr']
                
            
            if args.test_method == 'sum_loss':
                _, power_allocated, _ ,_= opti_LR( K, const_alpha, h_avg, np.ones(args.total_UE)* wireless_arg['decr'], P_max, P_sum,data_size,const,
                                                 later_weights+weights_to_use ) # remember to add 
            else:
                if args.no_later:
                    _, power_allocated, _ ,_= opti_LR( K, const_alpha, h_avg, weights_to_use, P_max, P_sum,data_size,const,
                                                     np.zeros(args.total_UE)) # remember to add data_size_weight
                else:
                    _, power_allocated, _ ,_= opti_LR( K, const_alpha, h_avg, weights_to_use *wireless_arg['decr'] , P_max, P_sum,data_size,const,
                                                        later_weights) # remember to add data_size_weight
            
            active_clients = np.nonzero(power_allocated)[0]
            print('active clients number is ', len(active_clients) )
            # if len(active_clients) != K:
            #     print('active clients number is ', len(active_clients) )
                # logging.info("active clients number is {}".format( len(active_clients)))
            run_time = 0
        elif args.opti == 'cstWObj':
            const_alpha = N0 * B * m
            const = theta / Tslot
            weights_to_use = copy.deepcopy(weights)
            # print("age of updates vector", args.age_Updates)
            if args.selection == "solve_opti_laterW":
                power_allocated = solve_cstObj_pb(K, const_alpha, h_avg, np.zeros(args.total_UE), P_max, P_sum,data_size,const, later_weights)
            else:
                power_allocated = solve_cstObj_pb(K, const_alpha, h_avg, weights_to_use*wireless_arg['decr'], P_max, P_sum,data_size,const, later_weights)
            active_clients = np.nonzero(power_allocated)[0]
            print('active clients number is ', len(active_clients) )
        obj_value = objective_funtion(power_allocated, const_alpha, h_avg, weights, data_size, theta, later_weights,P_max, 1,1)
        
    else:
        print("args selection value is erroneous, uniform random selection is considered instead")
        active_clients = np.random.choice(user_indices, args.active_UE , replace = False)
        power_allocated = np.zeros(len(user_indices))
        power_allocated[active_clients] = min(P_sum / args.active_UE , P_max)
        run_time = 0
        message = ''
    # Communication error
    print("Intended transmission users:", active_clients)
    snr = power_allocated * h_i / N0 / B
    transmission_rate = np.log2(1 + snr)
    proba_success = np.zeros(args.total_UE)
    proba_success[snr>0] =  np.exp(-m / snr[snr>0])
    snr_avg = power_allocated * h_avg / N0 / B
    proba_success_avg = np.zeros(args.total_UE)
    proba_success_avg[snr_avg>0] =  np.exp(-m / snr_avg[snr_avg>0])

    active_success_clients = []
    fails = 0
    np.random.seed(seed+123)
    for index, c in enumerate(active_clients):
        success = np.random.binomial(1, proba_success[c])
        if success:
            active_success_clients.append(c)
        else:
            fails += 1
    print("number of failed clients," ,fails)
    return list(set(active_success_clients)), proba_success_avg, fails, np.mean(proba_success[active_clients]), obj_value
