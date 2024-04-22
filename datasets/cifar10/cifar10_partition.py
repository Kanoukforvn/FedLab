import sys
import torch
import torchvision
import os
import ssl
import matplotlib.pyplot as plt
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from fedlab.utils.dataset.functional import noniid_slicing, random_slicing, partition_report
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.dataset import functional as F
#from fedlab.utils.functional import save_dict

sys.path.append("../../")

trainset = torchvision.datasets.CIFAR10(root="./", train=True, download=True) 

num_clients = 100
num_classes = 10


# perform partition
hetero_dir_part = CIFAR10Partitioner(trainset.targets, 
                                num_clients,
                                balance=None, 
                                partition="dirichlet",
                                dir_alpha=0.3,)

torch.save(hetero_dir_part.client_dict, "cifar10_hetero_dir.pkl")
print(len(hetero_dir_part))


# generate partition report
csv_file = "./partition-reports/cifar10_hetero_dir_0.3_100clients.csv"
partition_report(trainset.targets, hetero_dir_part.client_dict, 
                 class_num=num_classes, 
                 verbose=False, file=csv_file)


hetero_dir_part_df = pd.read_csv(csv_file,header=0)
#print(hetero_dir_part_df.columns)
hetero_dir_part_df = hetero_dir_part_df.set_index('cid')
col_names = [f"class-{i}" for i in range(num_classes)]
for col in col_names:
    hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['TotalAmount']).astype(int)


#select first 10 clients for bar plot
hetero_dir_part_df[col_names].iloc[:10].plot.barh(stacked=True)  
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('sample num')
plt.savefig(f"./imgs/cifar10_hetero_dir_0.3_100clients.png", dpi=400, bbox_inches = 'tight')


