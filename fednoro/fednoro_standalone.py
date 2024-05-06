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
from fedlab.utils.fednoro_utils import add_noise, set_seed, set_output_files, get_output
from fedlab.contrib.algorithm.fednoro import FedNoRoSerialClientTrainerS1, FedAvgServerHandler
from fedlab.contrib.algorithm.basic_server import SyncServerHandler

args = Munch

args.total_client = 5
args.alpha = 2
args.seed = 1
args.preprocess = True
args.dataname = "cifar10"
args.model = "Resnet18"
args.pretrained = 1
args.num_users = args.total_client
#args.device = "cuda" if torch.cuda.is_available() else "cpu"
args.device = "cuda"
args.cuda = True

if args.dataname == "cifar10":
    args.n_classes = 10


logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', 
                        datefmt='%H:%M:%S',
                        stream=sys.stdout)

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


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
                                  balance=True,
                                  partition="dirichlet",
                                  seed=args.seed,
                                  dir_alpha=args.alpha,
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
#                  Dataset                 #
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
hetero_dir_part_df[col_names].iloc[:5].plot.barh(stacked=True)  
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('sample num')
plt.savefig(f"./imgs/cifar10_hetero_dir_0.3_10clients.png", dpi=400, bbox_inches = 'tight')
plt.show()


############################################
#            Noise Generation              #
############################################

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
hetero_dir_part_df[col_names].iloc[:5].plot.barh(stacked=True)  
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('sample num')
plt.savefig(f"./imgs/cifar10_dir_aft_noise_0.3_10clients.png", dpi=400, bbox_inches = 'tight')
plt.show()


# local train configuration
args.epochs = 5
args.batch_size = 128
args.lr = 0.0003

model = build_model(args)
        
set_seed(args.seed)

############################################
#           Stage 1 - Warm Up              #
############################################

logging.info("\n ---------------------begin training---------------------")

# client
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer, SGDClientTrainer

# Create client trainer and server handler
args.com_round = 15
args.sample_ratio = 0.1

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
#      Stage 1-1 - Evaluation Pipeline       #
############################################

label_counts_per_client = trainer.get_num_of_each_class_global(fed_cifar10)
for client_index, label_counts in enumerate(label_counts_per_client):
    logging.info(f"Client {client_index} label counts: {label_counts}")

class EvalPipelineS1(StandalonePipeline):
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
            logging.info("Round {}, Loss {:.4f}, Test Accuracy {:.4f}".format(t, loss, acc))

            if acc > self.best_performance:
                self.best_performance = acc
                logging.info(f'Best accuracy: {self.best_performance:.4f}')

                # Save model state_dict
                model_path = f'model/stage1_model_{t}.pth'
                self.best_round_number = t
                torch.save(self.handler.model.state_dict(), model_path)
                # logging.info(f'Saved model state_dict to: {model_path}')
            
            self.loss.append(loss)
            self.acc.append(acc)
            t += 1
        
        logging.info('Final best accuracy: {:.4f}, Best model number : {} '.format(self.best_performance, self.best_round_number))

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
eval_pipeline_s1 = EvalPipelineS1(handler=handler, trainer=trainer, test_loader=test_loader)
eval_pipeline_s1.main()
eval_pipeline_s1.show()

############################################
#      Stage 1-2 - Client Selection        #
############################################

model_path = f"./model/stage1_model_{eval_pipeline_s1.best_round_number}.pth"
logging.info(
    f"********************** load model from: {model_path} **********************")

model.load_state_dict(torch.load(model_path))

from sklearn.mixture import GaussianMixture

criterion = nn.CrossEntropyLoss(reduction='none')
local_output, loss = get_output(dataloader_train, model.to(args.device), args, False, criterion)

metrics = np.zeros((args.total_client, args.n_classes)).astype("float")
num = np.zeros((args.total_client, args.n_classes)).astype("float")
user_id = list(range(args.total_client))

for id in range(args.total_client):
    idxs = fed_cifar10.data_indices_train[id]
    for idx in idxs:
        c = fed_cifar10.targets_train[idx]
        logging.info("c : ")
        logging.info(c)

        logging.info("id : ")
        logging.info(id)

        num[id, c] += 1
        metrics[id, c] += loss[idx]
metrics = metrics / num
for i in range(metrics.shape[0]):
    for j in range(metrics.shape[1]): 
        if np.isnan(metrics[i, j]):
            metrics[i, j] = np.nanmin(metrics[:, j])
for j in range(metrics.shape[1]):
    metrics[:, j] = (metrics[:, j] - metrics[:, j].min()) / (metrics[:, j].max() - metrics[:, j].min())
logging.info("metrics:")
logging.info(metrics)

vote = []
for i in range(9):
    gmm = GaussianMixture(n_components=2, random_state=i).fit(metrics)
    gmm_pred = gmm.predict(metrics)
    noisy_clients = np.where(gmm_pred == np.argmax(gmm.means_.sum(1)))[0]
    noisy_clients = set(list(noisy_clients))
    vote.append(noisy_clients)
cnt = []
for i in vote:
    cnt.append(vote.count(i))
noisy_clients = list(vote[cnt.index(max(cnt))])

logging.info(f"selected noisy clients: {noisy_clients}, real noisy clients: {np.where(gamma_s>0.)[0]}")
clean_clients = list(set(user_id) - set(noisy_clients))
logging.info(f"selected clean clients: {clean_clients}")