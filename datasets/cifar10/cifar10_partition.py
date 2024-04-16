import sys
import torch
import torchvision


from fedlab.utils.dataset.slicing import noniid_slicing, random_slicing
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.dataset import functional as F
from fedlab.utils.functional import partition_report, save_dict


trainset = torchvision.datasets.MNIST(root="./", train=True, download=False)

num_clients = 100
num_classes = 10

seed = 2021

data_indices = noniid_slicing(trainset, num_clients=100, num_shards=200)
save_dict(data_indices, "cifar10_noniid.pkl")

data_indices = random_slicing(trainset, num_clients=100)
save_dict(data_indices, "cifar10_iid.pkl")


