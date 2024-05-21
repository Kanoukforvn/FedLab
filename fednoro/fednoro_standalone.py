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

from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

from munch import Munch

args = Munch

args.total_client = 20
args.alpha = 2
args.seed = 0
args.preprocess = True
args.dataname = "cifar10"
args.model = "Resnet18"
args.pretrained = 1
args.num_users = args.total_client
#args.device = "cuda" if torch.cuda.is_available() else "cpu"
args.device = "cuda"
args.cuda = True
args.level_n_lowerb = 0.5
args.level_n_upperb = 0.7
args.level_n_system = 0.4
args.n_type = "random"
args.epochs = 5
args.batch_size = 16
args.lr = 0.0003
args.warm_up_round = 15
args.sample_ratio = 1
args.begin = 10
args.end = 49
args.a = 0.8 
args.exp = "Fed"       
args.com_round = 100-args.warm_up_round

if args.dataname == "cifar10":
    args.n_classes = 10

#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(f'log_dataset_{args.dataname}_noise_lvl_{args.level_n_system}_num_client_{args.total_client}')])

logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', 
                        datefmt='%H:%M:%S',
                        stream=sys.stdout)

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) #useful for kaggle

sys.path.append("../")

cwd = os.getcwd()
project_root = os.path.abspath(os.path.join(cwd, "../.."))
sys.path.append(project_root)

# configuration
from fedlab.models.build_model import build_model
from fedlab.utils.dataset.functional import partition_report
from fedlab.utils import Logger, SerializationTool, Aggregators, LogitAdjust, LA_KD, DaAggregator
from fedlab.utils.fednoro_utils import add_noise, set_seed, get_output, get_current_consistency_weight
from fedlab.contrib.algorithm.fednoro import FedNoRoSerialClientTrainer, FedNoRoServerHandler, FedAvgServerHandler
from fedlab.contrib.algorithm.basic_server import SyncServerHandler


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
dataloader_train = fed_cifar10.get_dataloader(0, args.batch_size, type="train")
dataloader_test = fed_cifar10.get_dataloader(0, args.batch_size, type="test")

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

from matplotlib import pyplot as plt
from fedlab.utils.dataset.functional import feddata_scatterplot

title = 'Data Distribution over Clients for Each Class'
fig = feddata_scatterplot(fed_cifar10.targets_train,
                          fed_cifar10.data_indices_train,
                          args.total_client,
                          args.n_classes,
                          figsize=(6, 4),
                          max_size=200,
                          title=title)
fig.savefig(f'./imgs/feddata-scatterplot-vis.png') 


############################################
#            Noise Generation              #
############################################


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

model = build_model(args)
        
set_seed(args.seed)

############################################
#           Stage 1 - Warm Up              #
############################################

logging.info("\n ---------------------begin training---------------------")

# client
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer, SGDClientTrainer

# Create client trainer and server handler

trainer = FedNoRoSerialClientTrainer(model, args.total_client, cuda=args.cuda)
trainer.setup_dataset(fed_cifar10)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

from fedlab.utils.functional import evaluate
from fedlab.core.standalone import StandalonePipeline

handler = FedAvgServerHandler(model=model, global_round=args.warm_up_round, sample_ratio=1, cuda=args.cuda, num_clients=args.total_client)

from fedlab.core.standalone import StandalonePipeline

from torch import nn
from torch.utils.data import DataLoader
import torchvision

############################################
#      Stage 1-1 - Evaluation Pipeline       #
############################################

label_counts_per_client = trainer.get_num_of_each_class_global(fed_cifar10)
for client_index, label_counts in enumerate(label_counts_per_client):
    logging.info(f"Client {client_index} label counts: {label_counts} total")

class EvalPipelineS1(StandalonePipeline):
    def __init__(self, handler, trainer, test_loader):
        super().__init__(handler, trainer)
        self.test_loader = test_loader
        self.loss = []
        self.acc = []
        self.bacc = []
        self.best_performance = 0
        self.best_balanced_accuracy = 0

    def main(self):
        t = 0
        while self.handler.if_stop is False:
            logging.info("Round {}".format(t+1))

            # Server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package
            
            # Client side
            self.trainer.local_process_s1(broadcast, sampled_clients)
            uploads = self.trainer.uplink_package


            # Server side
            for pack in uploads:
                self.handler.load(pack)

            loss, acc, bacc = evaluate(self.handler.model, nn.CrossEntropyLoss(), self.test_loader)
            logging.info("Loss {:.4f}, Test Accuracy {:.4f}, Balanced Accuracy {:.4f}".format(loss, acc, bacc))

            if bacc > self.best_balanced_accuracy:
                self.best_balanced_accuracy = bacc
                logging.info(f'Best balanced accuracy: {self.best_balanced_accuracy:.4f}')

                # Save model state_dict
                model_path = f'model/stage1_model_{t}.pth'
                self.best_round_number = t
                torch.save(self.handler.model.state_dict(), model_path)
                # logging.info(f'Saved model state_dict to: {model_path}')


            if acc > self.best_performance:
                self.best_performance = acc
                logging.info(f'Best accuracy: {self.best_performance:.4f}')

            self.loss.append(loss)
            self.acc.append(acc)
            self.bacc.append(bacc)
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
test_loader = DataLoader(test_data, batch_size=32)

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


train_data = torchvision.datasets.CIFAR10(root="../datasets/cifar10/",
                                       train=True,
                                       transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=32)

criterion = nn.CrossEntropyLoss(reduction='none')
local_output, loss = get_output(train_loader, model.to(args.device), args, False, criterion)

metrics = np.zeros((args.total_client, args.n_classes)).astype("float")
num = np.zeros((args.total_client, args.n_classes)).astype("float")
user_id = list(range(args.total_client))

np.set_printoptions(precision=2)

for id in range(args.total_client):
    idxs = fed_cifar10.data_indices_train[id]
    for idx in idxs:
        c = fed_cifar10.targets_train[idx]
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
    gmm = GaussianMixture(n_components=2, random_state=i).fit(metrics) #n_component classement niveau de bruit
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

############################################
#    Stage 2 - Noise-Robust Training       #
############################################


trainer = FedNoRoSerialClientTrainer(model, args.total_client, cuda=args.cuda, lr=args.lr)
trainer.setup_dataset(fed_cifar10)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

handler = FedNoRoServerHandler(model=model, global_round=args.com_round, sample_ratio=args.sample_ratio, cuda=args.cuda, num_clients=args.total_client)

class EvalPipelineS2(StandalonePipeline):
    def __init__(self, args, handler, trainer, test_loader, clean_clients, noisy_clients):
        super().__init__(handler, trainer)
        self.test_loader = test_loader
        self.loss = []
        self.acc = []
        self.bacc = []
        self.best_performance = 0
        self.best_balanced_accuracy = 0
        self.begin = args.begin
        self.end = args.end
        self.a = args.a
        self.noisy_clients = noisy_clients
        self.clean_clients = clean_clients

    def main(self):
        t = 0
        while self.handler.if_stop is False:
            logging.info("Round {}".format(t+1))

            # Server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package

            logging.info("Training on K={} clients".format(len(sampled_clients)))

            # Client side
            self.trainer.local_process_s2(
                broadcast, 
                sampled_clients, 
                t, 
                begin=self.begin, 
                end=self.end, 
                a=self.a, 
                noisy_clients=self.noisy_clients, 
                clean_clients=self.clean_clients)
            
            uploads = self.trainer.uplink_package

            # Server side
            for pack in uploads:
                self.handler.load(pack, self.clean_clients, self.noisy_clients)

            loss, acc, bacc = evaluate(self.handler.model, nn.CrossEntropyLoss(), self.test_loader)
            logging.info("Loss {:.4f}, Test Accuracy {:.4f}, Balanced Accuracy {:.4f}".format(loss, acc, bacc))
            
            if acc > self.best_performance:
                self.best_performance = acc
                logging.info(f'Best accuracy: {self.best_performance:.4f}')


            # Update best balanced accuracy
            if bacc > self.best_balanced_accuracy:
                self.best_balanced_accuracy = bacc
                logging.info(f'Best balanced accuracy: {self.best_balanced_accuracy:.4f}')
                # Save model state_dict
                model_path = f'model/stage2_model_{t}.pth'
                self.best_round_number = t
                torch.save(self.handler.model.state_dict(), model_path)
                # logging.info(f'Saved model state_dict to: {model_path}')


            self.loss.append(loss)
            self.acc.append(acc)
            self.bacc.append(bacc)
            t += 1
        
        logging.info('Final best accuracy: {:.4f}, Best balanced accuracy {:.4f}, Best model number : {} '.format(self.best_performance, self.best_balanced_accuracy, self.best_round_number))


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

        plt.savefig(f"./imgs/cifar10_dir_loss_accuracy_s2.png", dpi=400, bbox_inches = 'tight')
        
# Run evaluation
eval_pipeline_s2 = EvalPipelineS2(handler=handler, trainer=trainer, noisy_clients=noisy_clients, clean_clients=clean_clients, test_loader=test_loader, args=args)
eval_pipeline_s2.main()
eval_pipeline_s2.show()

torch.cuda.empty_cache()
