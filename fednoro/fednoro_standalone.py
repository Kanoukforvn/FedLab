import sys
import os
import torch

sys.path.append("../")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# configuration
from munch import Munch
from fedlab.models.mlp import MLP
from fedlab.models.build_model import build_model


args = Munch

args.total_client = 5
args.alpha = 0.5
args.seed = 42
args.preprocess = True
args.cuda = True
args.dataname = "cifar10"
args.model = "Resnet18"
args.pretrained = 1

if args.dataname == "cifar10":
    args.n_classes = 10

args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
model = build_model(args)

# We provide a example usage of patitioned CIFAR10 dataset
# Download raw CIFAR10 dataset and partition them according to given configuration

from torchvision import transforms
from fedlab.contrib.dataset.partitioned_cifar10 import PartitionedCIFAR10

fed_cifar10 = PartitionedCIFAR10(root="../datasets/cifar10/",
                         path="../datasets/cifar10/fedcifar10/",
                         dataname = args.dataname,
                         num_clients=args.total_client,
                         partition="dirichlet",
                         dir_alpha=args.alpha,
                         seed=args.seed,
                         preprocess=args.preprocess,
                         download=True,
                         verbose=True,
                         transform=transforms.ToTensor())

dataset = fed_cifar10.get_dataset(0) # get the 0-th client's dataset
dataloader = fed_cifar10.get_dataloader(0, batch_size=128) # get the 0-th client's dataset loader with batch size 128

# client
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer, SGDClientTrainer

# local train configuration
args.epochs = 5
args.batch_size = 128
args.lr = 0.1

trainer = SGDSerialClientTrainer(model, args.total_client, cuda=args.cuda) # serial trainer
# trainer = SGDClientTrainer(model, cuda=True) # single trainer

trainer.setup_dataset(fed_cifar10)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

# server
from fedlab.contrib.algorithm.basic_server import SyncServerHandler

# global configuration
args.com_round = 10
args.sample_ratio = 0.1

handler = SyncServerHandler(model=model, global_round=args.com_round, sample_ratio=args.sample_ratio, cuda=args.cuda)

import matplotlib.pyplot as plt
import numpy as np

from fedlab.utils.functional import evaluate
from fedlab.core.standalone import StandalonePipeline

from torch import nn
from torch.utils.data import DataLoader
import torchvision

class EvalPipeline(StandalonePipeline):
    def __init__(self, handler, trainer, test_loader):
        super().__init__(handler, trainer)
        self.test_loader = test_loader 
        self.loss = []
        self.acc = []
        
    def main(self):
        t=0
        while self.handler.if_stop is False:
            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package
            
            # client side
            self.trainer.local_process(broadcast, sampled_clients)
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load(pack)

            loss, acc = evaluate(self.handler.model, nn.CrossEntropyLoss(), self.test_loader)
            print("Round {}, Loss {:.4f}, Test Accuracy {:.4f}".format(t, loss, acc))
            t+=1
            self.loss.append(loss)
            self.acc.append(acc)
    
    def show(self):
        plt.figure(figsize=(8, 4.5))
        ax = plt.subplot(1, 2, 1)
        ax.plot(np.arange(len(self.loss)), self.loss, color='blue', linestyle='-')
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Loss")
    
        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(np.arange(len(self.acc)), self.acc, color='red', linestyle='-')
        ax2.set_xlabel("Communication Round")
        ax2.set_ylabel("Accuracy")
        
        plt.show()
        
    
        
test_data = torchvision.datasets.CIFAR10(root="../datasets/cifar10/",
                                       train=False,
                                       transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=1024)

standalone_eval = EvalPipeline(handler=handler, trainer=trainer, test_loader=test_loader)
standalone_eval.main()

standalone_eval.show()
plt.show()