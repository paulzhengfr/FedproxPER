import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


dataf = pd.read_csv('results/data_failed_transmission_mnist.csv')
# dataf = pd.read_csv('results/data_failed_transmission_cifar.csv')


args = {'opt': "Prox", #"Avg" "Prox"
        'iid': "dirichlet", #"dirichlet" "niid"
        'dataset': 'mnist', #cifar, mnist, synthetic
        'method': "BL", #"BL" "UR", "opti", "WR", "Sal_WR"
        'ep':20,
        'seed_ind': 0
        }


colnames = list(dataf.columns)

useful_col = []
for col in colnames:
    if '_MIN' in col or '_MAX' in col:
        continue
    useful_col.append(col)
    
diri_col = []
niid_col = []
for col in useful_col:
    if 'niid' in col:
        niid_col.append(col)
    else:
        diri_col.append(col)
if args['dataset'] == 'mnist':
    df_diri = dataf[diri_col][0:400]
else:
    df_diri = dataf[diri_col]
df_niid = dataf[niid_col]



meth_Vn = {'BC', 'UR', 'BL', 'WR', 'Sal_WR', 'opti'}
df_diri_avg = pd.DataFrame()
for meth in meth_Vn:
    meth_col = []
    for col in df_diri.columns:
        if meth in col:
            meth_col.append(col)
    df_diri_avg[meth] = df_diri[meth_col].mean(axis = 1)

df_niid_avg = pd.DataFrame()
for meth in meth_Vn:
    meth_col = []
    for col in df_niid.columns:
        if meth in col:
            meth_col.append(col)
    df_niid_avg[meth] = df_niid[meth_col].mean(axis = 1)
    
def give_full_method_name(abb):
    if abb == "optil":
        return "our method loss"
    if abb == "optils":
        return "our method loss and size"
    if abb == "opti":
        return "our method"
    if abb == "BL":
        return "best loss"
    if abb == "UR":
        return "uniform"
    if abb == "Sal_WR":
        return "weighted random"
    if abb == "WR":
        return "weighted random w/o aggregation correction"
    if abb == "BC":
        return "best channel"
    if abb == "Salehi":
        return "Salehi"
if args['dataset'] == 'mnist':
    if args['iid'] == 'dirichlet':
        window_size = 21 # 51, 81
        nb_ite_div100= 4
        loweracc=2
    else:
        window_size = 41
        nb_ite_div100= 8
        loweracc=2
elif args['dataset'] == 'cifar':
    window_size = 41
    nb_ite_div100= 8
    loweracc=3
else:
    window_size = 1
    nb_ite_div100= 2
    loweracc=0

#%% Plot fails.
fig, ax = plt.subplots(figsize=[30,24])
colors = ['r', 'b', 'g', 'c','m','y']
markers = ['o', '<','x','2','+','D']
id_color = 0
font_size = 40
window_size = 21 # 51, 81
# nb_ite_div100= 3
for meth in ["opti","UR","BL", "BC", "WR","Sal_WR"]:
    transpar = 1
    if meth == 'opti' or meth == 'Sal_WR':
        transpar = 0.5
    if args['iid'] == "dirichlet":
        fails = df_diri_avg[meth]
    else:
        fails = df_niid_avg[meth]
    if len(fails)>nb_ite_div100*100:
        comm_rounds = np.arange(1, nb_ite_div100*100+1)
    else:
        comm_rounds = np.arange(1, len(fails)+1)
    # fails_s = pd.Series(fails[:nb_ite_div100*100,0])
    fails_smooth = fails.rolling(window_size).mean()
    plt.plot(comm_rounds,np.asarray(fails_smooth) , marker = markers[id_color],  
              label=give_full_method_name(meth), color = colors[id_color], alpha = transpar)
    id_color += 1

plt.xticks(np.arange(nb_ite_div100+1)*100, fontsize = font_size)
plt.xlabel("Communication rounds",fontsize=font_size)
plt.ylabel("Number of failed transmissions",fontsize=font_size)
plt.yticks([0,2,4,6,8,10], fontsize = font_size)
plt.legend(loc=4,prop={'size': font_size-10})
plt.grid()

# plt.title("Fed"+args['opt'] + " "+args['iid']+" alpha = 0.5", fontsize =font_size)
#plt.title("Fed"+args['opt'] + " "+args['iid'], fontsize =font_size)

ax.tick_params(color='#dddddd')
ax.spines['bottom'].set_color('#dddddd')
ax.spines['top'].set_color('#dddddd') 
ax.spines['right'].set_color('#dddddd')
ax.spines['left'].set_color('#dddddd')
plt.tight_layout()

# figure_folder = "D:/Documents/Sciebo_groupfiles/PhD_Paul/figures/"
# figure_folder = "D:/sciebo/files/PhD_Paul/figures/"
# plt.savefig(figure_folder+"failed_transmission_{}.pdf".format(args['iid']))
# plt.show()