# Fedprox PER (packet error rate)

This code consists in federated learning under transmission packet error rate for the paper "Federated Learning in Heterogeneous Networks with Unreliable Communication".



The main program is *main_fed.py*.

Please first install the requirements using pip:  
```pip install -r requirement.txt``` file. 


You can run the program by:

```
python main_fed.py --dataset mnist --model Mnist_oldMLP --round 200 --gpu 0 --iid niid --optimizer fedprox --total_UE 500 --active_UE 10 --selection "weighted_random"  --name 'mnist_WR' --local_ep 20 --lr 0.1 --scenario woPER

```

Many options of the training can be found at the *argparse* part in the main program.
* `--optimizer`: FedAvg and FedProx are implemented and can be changed: fedavg, fedprox.
* `--iid`: type of non-iid: `IID` for iid, `NIID` for 2 digits per client, and `Dirichlet` for dirichlet non-iid distribution tuned by `--alpha` (represent the similarity of data).
* `--batch_size`: batch size.
* `--lr`: learning rate.
* `--mu`: FedProx parameter.
* `--round`: total communication rounds to run
* `--epochs`: local computation
* `--seed` and `--wireless_seed`:random seeds
* `--total_UE`: total participating users
* `--active_UE`: number of active users at each round.
* `--selection`: user selection strategy `weighted_random`, and our method is denoted as `solve_opti_loss_size2`.
* `--name`: experiment name for folder name where all results are stored.

More options can be found in `./utils/options.py`.

All simulations in the paper can be reproced by their corresponding bash scripts in the folder `bash_scripts/`.
All figures can be reproduced by `Fig_X_*.ipynb/py` file in this main folder.

Results csv files to reproduce the figures will all be updated after paper official publication.


The paper reference is as belows:
---
```
@ARTICLE{10253642,
  author={Zheng, Paul and Zhu, Yao and Hu, Yulin and Zhang, Zhengming and Schmeink, Anke},
  journal={IEEE Transactions on Wireless Communications}, 
  title={Federated Learning in Heterogeneous Networks With Unreliable Communication}, 
  year={2024},
  volume={23},
  number={4},
  pages={3823-3838},
  keywords={Training;Convergence;Computational modeling;Servers;Resource management;Data models;Wireless networks;Distributed learning;federated learning;wireless networks;packet error rate;convergence analysis},
  doi={10.1109/TWC.2023.3311824}}
```
