import sys
import torch
import torchvision
import os
import ssl

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from fedlab.utils.dataset.functional import noniid_slicing, random_slicing
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.dataset import functional as F
#from fedlab.utils.functional import save_dict

sys.path.append("../../")

trainset = torchvision.datasets.CIFAR10(root="./", train=True, download=True)

num_clients = 10
num_classes = 10

seed = 2021

# perform partition
hetero_dir_part = CIFAR10Partitioner(trainset.targets, 
                                num_clients,
                                balance=None, 
                                partition="dirichlet",
                                dir_alpha=0.3,
                                seed=seed)

torch.save(hetero_dir_part.client_dict, "cifar10_hetero_dir.pkl")
print(len(hetero_dir_part))


