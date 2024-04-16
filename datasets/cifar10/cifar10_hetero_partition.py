import sys
import torch
import torchvision

from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.dataset import functional as F
from fedlab.utils.functional import partition_report


trainset = torchvision.datasets.MNIST(root="./", train=True, download=False)

num_clients = 100
num_classes = 10

seed = 2021


hetero_dir_part = CIFAR10Partitioner(trainset.targets, 
                                num_clients,
                                balance=None, 
                                partition="dirichlet",
                                dir_alpha=0.3,
                                seed=seed)

torch.save(hetero_dir_part.client_dict, "cifar10_hetero_dir.pkl")
print(len(hetero_dir_part))