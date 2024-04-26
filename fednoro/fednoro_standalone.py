import sys
import os
import copy
import logging
import torch
import pandas as pd

import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import torch.nn as nn

from tensorboardX import SummaryWriter

from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

sys.path.append("../")

cwd = os.getcwd()
project_root = os.path.abspath(os.path.join(cwd, "../.."))
sys.path.append(project_root)

# configuration
from munch import Munch
from fedlab.models.mlp import MLP
from fedlab.models.build_model import build_model
from fedlab.utils.dataset.functional import partition_report
from fedlab.utils.fednoro_utils import add_noise, set_seed, set_output_files
from fedlab.contrib.algorithm.fednoro import FedNoRoSerialClientTrainerS1, FedAvgServerHandler
from fedlab.contrib.algorithm.basic_server import SyncServerHandler

logging.basicConfig(level = logging.INFO)
logging.getLogger().setLevel(logging.INFO)

args = Munch

args.total_client = 10
args.alpha = 0.5
args.seed = 42
args.preprocess = True
args.dataname = "cifar10"
args.model = "Resnet18"
args.pretrained = 1
args.num_users = args.total_client
#args.device = "cuda" if torch.cuda.is_available() else "cpu"
args.device = "cuda"


if args.dataname == "cifar10":
    args.n_classes = 10


# We provide a example usage of patitioned CIFAR10 dataset
# Download raw CIFAR10 dataset and partition them according to given configuration

from torchvision import transforms
from fedlab.contrib.dataset.partitioned_cifar10 import PartitionedCIFAR10

############################################
#           Set up the dataset             #
############################################


fed_cifar10 = PartitionedCIFAR10(root="../datasets/cifar10/",
                                  path="../datasets/cifar10/fedcifar10/",
                                  dataname=args.dataname,
                                  num_clients=args.total_client,
                                  num_classes=args.n_classes,
                                  partition="dirichlet",
                                  dir_alpha=args.alpha,
                                  seed=args.seed,
                                  preprocess=args.preprocess,
                                  download=True,
                                  verbose=True,
                                  transform=transforms.ToTensor())

# Get the dataset for the 0-th client
dataset_train = fed_cifar10.get_dataset(0, type="train")
dataset_test = fed_cifar10.get_dataset(0, type="test")

# Get the dataloaders
dataloader_train = fed_cifar10.get_dataloader(0, batch_size=128, type="train")
dataloader_test = fed_cifar10.get_dataloader(0, batch_size=128, type="test")

logging.info(
    f"train: {Counter(fed_cifar10.targets_train)}, total: {len(fed_cifar10.targets_train)}")
logging.info(
    f"test: {Counter(fed_cifar10.targets_test)}, total: {len(fed_cifar10.targets_test)}")


############################################
#           Dataset visualization          #
############################################

# generate partition report
csv_file = "./partition-reports/cifar10_hetero_dir_0.3_10clients.csv"
partition_report(fed_cifar10.targets_train, fed_cifar10.data_indices_train, 
                 class_num=args.n_classes, 
                 verbose=False, file=csv_file)


hetero_dir_part_df = pd.read_csv(csv_file,header=0)
hetero_dir_part_df = hetero_dir_part_df.set_index('cid')
col_names = [f"class-{i}" for i in range(args.n_classes)]
for col in col_names:
    hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['TotalAmount']).astype(int)

#select first 10 clients for bar plot
hetero_dir_part_df[col_names].iloc[:10].plot.barh(stacked=True)  
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('sample num')
plt.savefig(f"./imgs/cifar10_hetero_dir_0.3_10clients.png", dpi=400, bbox_inches = 'tight')
plt.show()

#noise
args.level_n_lowerb = 0.3
args.level_n_upperb = 0.5
args.level_n_system = 0.4
args.n_type = "random"

y_train = np.array(fed_cifar10.targets_train)
y_train_noisy, gamma_s, real_noise_level = add_noise(args, y_train, fed_cifar10.data_indices_train)
fed_cifar10.targets_train = y_train_noisy

# generate partition report
csv_file = "./partition-reports/cifar10_dir_aft_noise_0.3_10clients.csv"
partition_report(fed_cifar10.targets_train, fed_cifar10.data_indices_train, 
                 class_num=args.n_classes, 
                 verbose=False, file=csv_file)


hetero_dir_part_df = pd.read_csv(csv_file,header=0)
hetero_dir_part_df = hetero_dir_part_df.set_index('cid')
col_names = [f"class-{i}" for i in range(args.n_classes)]
for col in col_names:
    hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['TotalAmount']).astype(int)

#select first 10 clients for bar plot
hetero_dir_part_df[col_names].iloc[:10].plot.barh(stacked=True)  
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('sample num')
plt.savefig(f"./imgs/cifar10_dir_aft_noise_0.3_10clients.png", dpi=400, bbox_inches = 'tight')
plt.show()


# local train configuration
args.epochs = 5
args.batch_size = 128
args.lr = 0.1

model = build_model(args)

args.base_lr = 3e-4
args.warm = 1
args.s1 = 15
        
set_seed(args.seed)

############################################
#           Stage 1 - Warm Up              #
############################################

print("\n ---------------------begin training---------------------")

# client
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer, SGDClientTrainer

# Create client trainer and server handler
args = lambda: None
args.total_client = 10
args.epochs = 5
args.batch_size = 128
args.lr = 0.1
args.com_round = 10
args.sample_ratio = 0.1
args.cuda = True
args.device = "cuda"

trainer = FedNoRoSerialClientTrainerS1(model, args.total_client, cuda=args.cuda)
trainer.setup_dataset(fed_cifar10)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

from fedlab.utils.functional import evaluate, globaltest
from fedlab.core.standalone import StandalonePipeline

handler = FedAvgServerHandler(model=model, global_round=args.com_round, sample_ratio=args.sample_ratio, cuda=args.cuda)

from torch import nn
from torch.utils.data import DataLoader
import torchvision

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

############################################
#      Stage 1 - Evaluation Pipeline       #
############################################

label_counts_per_client = trainer.get_num_of_each_class(fed_cifar10)
for client_index, label_counts in enumerate(label_counts_per_client):
    print(f"Client {client_index} label counts: {label_counts}")

class EvalPipeline(StandalonePipeline):
    def __init__(self, handler, trainer, test_loader):
        super().__init__(handler, trainer)
        self.test_loader = test_loader
        self.loss = []
        self.acc = []
        self.best_performance = 0
        
    def main(self):
        t = 0
        while self.handler.if_stop is False:
            # Server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package
            
            # Client side
            self.trainer.local_process(broadcast, sampled_clients)
            uploads = self.trainer.uplink_package

            # Server side
            for pack in uploads:
                self.handler.load(pack)

            loss, acc = evaluate(self.handler.model, nn.CrossEntropyLoss(), self.test_loader)
            print("Round {}, Loss {:.4f}, Test Accuracy {:.4f}".format(t, loss, acc))
            self.loss.append(loss)
            self.acc.append(acc)

            pred = globaltest(copy.deepcopy(model).to(
                'cuda'), self.test_loader, 'cuda')
            acc = accuracy_score(fed_cifar10.targets_test, pred)
            bacc = balanced_accuracy_score(fed_cifar10.targets_test, pred)
            print("bacc : ", bacc)
            # Save model if best performance
            if bacc > self.best_performance:
                self.best_performance = bacc
                logging.info(f'Best balanced accuracy: {self.best_performance:.4f}')

                # Save model state_dict
                model_path = f'fednoro/stage1_model_{t}.pth'
                torch.save(self.handler.model.state_dict(), model_path)
                logging.info(f'Saved model state_dict to: {model_path}')

            
            self.loss.append(loss)
            self.acc.append(acc)
            t += 1

    def show(self):
        plt.figure(figsize=(8, 4.5))
        ax = plt.subplot(1, 2, 1)
        ax.plot(np.arange(len(self.loss)), self.loss)
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Loss")
        
        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(np.arange(len(self.acc)), self.acc)
        ax2.set_xlabel("Communication Round")
        ax2.set_ylabel("Accuracy")

        plt.savefig(f"./imgs/cifar10_dir_loss_accuracy_s1.png", dpi=400, bbox_inches = 'tight')
        
        

test_data = torchvision.datasets.CIFAR10(root="../datasets/cifar10/",
                                       train=False,
                                       transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=1024)

# Run evaluation
eval_pipeline = EvalPipeline(handler=handler, trainer=trainer, test_loader=test_loader)
eval_pipeline.main()
eval_pipeline.show()

model_path = os.path.join("model", "s1_model_params.pth")
torch.save(model.state_dict(), model_path)

"""
# trainer = SGDClientTrainer(model, cuda=True) # single trainer
trainer = FedNoRoSerialClientTrainerS1(model, args.total_client, cuda=args.cuda) # serial trainer

trainer.setup_dataset(fed_cifar10)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)


# server
from fedlab.contrib.algorithm.basic_server import SyncServerHandler

# global configuration
args.com_round = 10
args.sample_ratio = 0.1

handler = SyncServerHandler(model=model, global_round=args.com_round, sample_ratio=args.sample_ratio, cuda=args.cuda)


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
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot loss
        axs[0].plot(np.arange(len(self.loss)), self.loss, color='blue')
        axs[0].set_title('Loss')
        axs[0].set_xlabel('Communication Round')
        axs[0].set_ylabel('Loss')

        # Plot accuracy
        axs[1].plot(np.arange(len(self.acc)), self.acc, color='red')
        axs[1].set_title('Accuracy')
        axs[1].set_xlabel('Communication Round')
        axs[1].set_ylabel('Accuracy')

        plt.savefig(f"./imgs/cifar10_hetero_dir_loss_accuracy.png", dpi=400, bbox_inches = 'tight')
        plt.tight_layout()
        plt.show()
        
    
        
test_data = torchvision.datasets.CIFAR10(root="../datasets/cifar10/",
                                       train=False,
                                       transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=1024)

standalone_eval = EvalPipeline(handler=handler, trainer=trainer, test_loader=test_loader)
standalone_eval.main()

standalone_eval.show()
"""