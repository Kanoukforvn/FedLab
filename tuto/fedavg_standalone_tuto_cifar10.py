# configuration
from munch import Munch
import matplotlib.pyplot as plt
import torchvision

args = Munch

args.total_client = 20
args.alpha = 0.1
args.seed = 0
args.preprocess = True
args.cuda = True
args.dataname = "cifar10"
args.num_classes = 10

import logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO)

import sys
sys.path.append("../")


from fedlab.models.mlp import MLP
from fedlab.utils.dataset.functional import partition_report
from fedlab.models.build_model import build_model

model = build_model()

# We provide a example usage of patitioned CIFAR10 dataset
# Download raw CIFAR10 dataset and partition them according to given configuration

from torchvision import transforms
from fedlab.contrib.dataset.partitioned_cifar10 import PartitionedCIFAR10
import pandas as pd

trainset = torchvision.datasets.CIFAR10(root="../../../../data/CIFAR10/", train=True, download=True)

fed_cifar10 = PartitionedCIFAR10(root="../datasets/cifar10/",
                                  path="../datasets/cifar10/fedcifar10_2/",
                                  dataname=args.dataname,
                                  num_clients=args.total_client,
                                  num_classes=args.num_classes,
                                  balance=False,
                                  partition="dirichlet",
                                  seed=args.seed,
                                  dir_alpha=args.alpha,
                                  preprocess=args.preprocess,
                                  download=True,
                                  verbose=True,
                                  transform=transforms.ToTensor())

dataset = fed_cifar10.get_dataset(0) # get the 0-th client's dataset
dataloader = fed_cifar10.get_dataloader(0, batch_size=16) # get the 0-th client's dataset loader with batch size 128

# generate partition report
csv_file = f"./partition-reports/{args.dataname}_hetero_dir_{args.alpha}_{args.total_client}clients.csv"
partition_report(fed_cifar10.targets_train, fed_cifar10.data_indices_train, 
                 class_num=args.num_classes, 
                 verbose=False, file=csv_file)

hetero_dir_part_df = pd.read_csv(csv_file,header=0)
hetero_dir_part_df = hetero_dir_part_df.set_index('cid')
col_names = [f"class-{i}" for i in range(args.num_classes)]
for col in col_names:
    hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['TotalAmount']).astype(int)

#select first 10 clients for bar plot
hetero_dir_part_df[col_names].iloc[:5].plot.barh(stacked=True)  
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('sample num')
plt.savefig(f"./imgs/{args.dataname}_dir_alpha_{args.alpha}_{args.total_client}clients.png", dpi=400, bbox_inches = 'tight')
plt.show()


# client
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer, SGDClientTrainer
from fedlab.contrib.algorithm.fedmdcs import FedMDCSSerialClientTrainer

# local train configuration
args.epochs = 5
args.batch_size = 16
args.lr = 0.0003

trainer = SGDSerialClientTrainer(model, args.total_client, cuda=args.cuda) # serial trainer
#trainer = SGDClientTrainer(model, cuda=True) # single trainer

trainer.setup_dataset(fed_cifar10)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

# server
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.fedmdcs import FedMDCSServerHandler

# global configuration
args.com_round = 100
args.sample_ratio = 0.5
args.top_n_clients = args.sample_ratio*args.total_client 

handler = SyncServerHandler(model=model, global_round=args.com_round, sample_ratio=args.sample_ratio, cuda=args.cuda, num_clients=args.total_client, 
                            )#top_n_clients=args.top_n_clients)

import matplotlib.pyplot as plt
import numpy as np

from fedlab.utils.functional import evaluate
from fedlab.core.standalone import StandalonePipeline

from torch import nn
from torch.utils.data import DataLoader

class EvalPipeline(StandalonePipeline):
    def __init__(self, handler, trainer, test_loader):
        super().__init__(handler, trainer)
        self.test_loader = test_loader 
        self.loss = []
        self.acc = []
        self.bacc = []
        self.best_performance = 0
        self.best_balanced_accuracy = 0
        
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

            loss, acc, bacc = evaluate(self.handler.model, nn.CrossEntropyLoss(), self.test_loader)
            print("Round {}, Loss {:.4f}, Test Accuracy {:.4f}, Balanced Accuracy {:.4f}".format(t, loss, acc, bacc))
            
            if bacc > self.best_balanced_accuracy:
                self.best_balanced_accuracy = bacc
                logging.info(f'Best balanced accuracy: {self.best_balanced_accuracy:.4f}')

            if acc > self.best_performance:
                self.best_performance = acc
                logging.info(f'Best accuracy: {self.best_performance:.4f}')

            t+=1
            self.loss.append(loss)
            self.acc.append(acc)
            self.bacc.append(bacc)
    
        logging.info(f'Final Best Balanced Accuracy: {self.best_balanced_accuracy:.4f}')

    def show(self):
        plt.figure(figsize=(8,4.5))
        ax = plt.subplot(1,2,1)
        ax.plot(np.arange(len(self.loss)), self.loss)
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Loss")
        
        ax2 = plt.subplot(1,2,2)
        ax2.plot(np.arange(len(self.acc)), self.acc)
        ax2.set_xlabel("Communication Round")
        ax2.set_ylabel("Accuarcy")
        
        plt.savefig(f"./imgs/{args.dataname}_dir_alpha_{args.alpha}_loss_accuracy.png", dpi=400, bbox_inches = 'tight')
   
        
test_data = torchvision.datasets.CIFAR10(root="../datasets/cifar10/",
                                       train=False,
                                       transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=1024)

standalone_eval = EvalPipeline(handler=handler, trainer=trainer, test_loader=test_loader)
standalone_eval.main()
standalone_eval.show()