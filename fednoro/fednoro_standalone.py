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

from config import args

logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', 
                        datefmt='%H:%M:%S',
                        stream=sys.stdout)

logging.info(f"{args.noisy_selection}")

#logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) #useful for kaggle

sys.path.append("../")

cwd = os.getcwd()
project_root = os.path.abspath(os.path.join(cwd, "../.."))
sys.path.append(project_root)

# configuration
from fedlab.models.build_model import build_model
from fedlab.utils.dataset.functional import partition_report
from fedlab.utils import Logger, SerializationTool, Aggregators, LogitAdjust, LA_KD, DaAggregator
from fedlab.utils.fednoro_utils import add_noise, set_seed, get_output, get_current_consistency_weight, set_output_files
from fedlab.contrib.algorithm.fednoro import FedNoRoSerialClientTrainer, FedNoRoServerHandler, FedAvgServerHandler, FedAvgServerHandlerS2
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

from torch import nn
from torch.utils.data import DataLoader
import torchvision

# generate partition report
csv_file = f"./partition-reports/{args.dataname}_{args.level_n_system}_{args.total_client}_clients.csv"
partition_report(fed_cifar10.targets_train, fed_cifar10.data_indices_train, 
                 class_num=args.n_classes, 
                 verbose=False, file=csv_file)


hetero_dir_part_df = pd.read_csv(csv_file,header=0)
hetero_dir_part_df = hetero_dir_part_df.set_index('cid')

col_names = [f"class-{i}" for i in range(args.n_classes)]
for col in col_names:
    hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['TotalAmount']).astype(int)

train_data = torchvision.datasets.CIFAR10(root="../datasets/cifar10/",
                                          train=True,
                                          transform=transforms.ToTensor())

############################################
#            Noise Generation              #
############################################


y_train = np.array(fed_cifar10.targets_train)
y_train_noisy, gamma_s, real_noise_level = add_noise(args, y_train, fed_cifar10.data_indices_train)
fed_cifar10.targets_train = y_train_noisy

# Noise Generation Bis
y_train = np.array(train_data.targets)
y_train_noisy, gamma_s, real_noise_level = add_noise(args, y_train, fed_cifar10.data_indices_train)
train_data.targets = y_train_noisy

# generate partition report
csv_file = f"./partition-reports/{args.dataname}_nlvl_{args.level_n_system}_{args.total_client}_clients.csv"
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

trainer = FedNoRoSerialClientTrainer(model, args.total_client, base_lr=args.lr, cuda=args.cuda)
trainer.setup_dataset(fed_cifar10)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

from fedlab.utils.functional import evaluate, globaltest
from fedlab.core.standalone import StandalonePipeline

handler = FedAvgServerHandler(model=model, global_round=args.warm_up_round, sample_ratio=1, cuda=args.cuda, num_clients=args.total_client)

from fedlab.core.standalone import StandalonePipeline


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
        while not self.handler.if_stop and t < args.warm_up_round:
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

            loss = evaluate(self.handler.model, nn.CrossEntropyLoss(), self.test_loader)

            pred = globaltest(copy.deepcopy(self.handler.model).to(
                args.device), test_data, args)
            acc = accuracy_score(fed_cifar10.targets_test, pred)
            bacc = balanced_accuracy_score(fed_cifar10.targets_test, pred)
            cm = confusion_matrix(fed_cifar10.targets_test, pred)

            logging.info("Loss {:.4f}, Balanced Accuracy {:.4f}".format(loss, bacc))
            logging.info(cm)

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
            
            self.loss.append(loss)
            self.acc.append(acc)
            self.bacc.append(bacc)
            t += 1
        
        logging.info('Final best balanced accuracy: {:.4f}, Best model number : {} '.format(self.best_balanced_accuracy, self.best_round_number))

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

        plt.savefig(f"./imgs/s1_fednoro_{args.dataname}_nlvl_{args.level_n_system}_loss_balanced_accuracy_{self.best_balanced_accuracy}.png", dpi=400, bbox_inches = 'tight')
        

test_data = torchvision.datasets.CIFAR10(root="../datasets/cifar10/",
                                       train=False,
                                       transform=transforms.ToTensor())        


test_loader = DataLoader(test_data, batch_size=32)

if args.warm:    
    # Run evaluation
    eval_pipeline_s1 = EvalPipelineS1(handler=handler, trainer=trainer, test_loader=test_loader)
    eval_pipeline_s1.main()
    eval_pipeline_s1.show()
    model_path = f"./model/stage1_model_{eval_pipeline_s1.best_round_number}.pth" #eval_pipeline_s1.best_round_number
else:
    best_round_number=14
    model_path = f"./model/stage1_model_{best_round_number}.pth"


############################################
#      Stage 1-2 - Client Selection        #
############################################


logging.info(
    f"********************** load model from: {model_path} **********************")

model.load_state_dict(torch.load(model_path))

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler

train_loader = DataLoader(train_data, batch_size=32, shuffle=False, num_workers=4)

criterion = nn.CrossEntropyLoss(reduction='none')
local_output, loss = get_output(train_loader, model.to(args.device), args, False, criterion)

metrics = np.zeros((args.total_client, args.n_classes)).astype("float")
num = np.zeros((args.total_client, args.n_classes)).astype("float")
user_id = list(range(args.total_client))

np.set_printoptions(precision=2)

for id in range(args.total_client):
    idxs = fed_cifar10.data_indices_train[id]
    for idx in idxs:
        c = train_data.targets[idx]
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

# Voting mechanism to identify the most consistent noisy clients
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

plt.figure(figsize=(10, 8))

is_noisy = np.zeros(metrics.shape[0], dtype=bool)
is_noisy[list(noisy_clients)] = True

for i in range(metrics.shape[0]):
    plt.text(metrics[i, 0], metrics[i, 1], str(i), fontsize=8, ha='right')

pca = PCA(n_components=2)
reduced_metrics = pca.fit_transform(metrics)

plt.figure(figsize=(10, 8))

plt.scatter(reduced_metrics[is_noisy, 0], reduced_metrics[is_noisy, 1], color='red', label='Noisy Clients', alpha=0.6)
plt.scatter(reduced_metrics[~is_noisy, 0], reduced_metrics[~is_noisy, 1], color='blue', label='Clean Clients', alpha=0.6)

for i in range(reduced_metrics.shape[0]):
    plt.text(reduced_metrics[i, 0], reduced_metrics[i, 1], str(i), fontsize=8, ha='right')

plt.title('Visualization of Noisy and Clean Clusters in Reduced 2D Space')
plt.legend()
plt.savefig(f'./imgs/{args.dataname}_nlvl_{args.level_n_system}_{args.level_n_lowerb}_{args.level_n_upperb}_noisy_clean_clusters_pca.png')
plt.show()

# Calculate the centroid of the clean clients' cluster
clean_metrics = metrics[~is_noisy]
clean_centroid = np.mean(clean_metrics, axis=0)

# Calculate EMD and Euclidian distances of noisy clients from the clean centroid
euclidean_distances = {}
emd_distances = {}
for client in noisy_clients:
    euclidean_distance = np.linalg.norm(metrics[client] - clean_centroid)
    emd_distance = wasserstein_distance(metrics[client], clean_centroid)
    euclidean_distances[client] = euclidean_distance
    emd_distances[client] = emd_distance

# Function to scale values to the range of target values
def scale_to_range(values, target_min, target_max):
    min_value = np.min(values)
    max_value = np.max(values)
    scaled_values = (values - min_value) / (max_value - min_value) * (target_max - target_min) + target_min
    return scaled_values

# Define the range for scaling based on the real noise levels
min_noise_level = np.min(real_noise_level[noisy_clients])
max_noise_level = np.max(real_noise_level[noisy_clients])

# Scale the distances with respect to the real noise level range
scaled_euclidean_distances = scale_to_range(np.array(list(euclidean_distances.values())), min_noise_level, max_noise_level)
scaled_emd_distances = scale_to_range(np.array(list(emd_distances.values())), min_noise_level, max_noise_level)
scaled_real_noise = real_noise_level[noisy_clients]  # No need to scale this as it is already in the desired range

# Create a dictionary for logging purposes
scaled_distances = {
    client: (scaled_euclidean_distances[idx], scaled_emd_distances[idx], scaled_real_noise[idx])
    for idx, client in enumerate(noisy_clients)
}

# Plot the histogram with superposed distances
plt.figure(figsize=(10, 8))
bar_width = 0.25
indices = np.arange(len(noisy_clients))

plt.bar(indices, scaled_euclidean_distances, bar_width, label='Euclidean Distance', color='b')
plt.bar(indices + bar_width, scaled_emd_distances, bar_width, label='EMD', color='g')
plt.bar(indices + 2 * bar_width, scaled_real_noise, bar_width, label='Real Noise Level', color='r')

plt.xlabel('Client Index')
plt.ylabel('Scaled Value')
plt.title('Comparison of Distances and Real Noise Level for Noisy Clients')
plt.xticks(indices + bar_width, noisy_clients)
plt.legend()
plt.tight_layout()

plt.savefig(f'./imgs/nlvl_{args.level_n_system}_distances_and_noise_level_scaled.png')
plt.show()

# Sort the noisy clients by normalized distances and noise ratio
sorted_clients_by_euclidean = sorted(noisy_clients, key=lambda x: scaled_distances[x][0], reverse=True)
sorted_clients_by_emd = sorted(noisy_clients, key=lambda x: scaled_distances[x][1], reverse=True)
sorted_clients_by_noise_ratio = sorted(noisy_clients, key=lambda x: scaled_distances[x][2], reverse=True)

# Change this value for testing different distances
noisy_distances = emd_distances

# Calculate the median distance
distances = np.array(list(noisy_distances.values()))
median_distance = np.median(distances)


if args.noisy_selection == True:
    # Select noisy clients whose distances are above the median
    logging.info("noisy selection")
    selected_noisy_clients = [client for client, distance in noisy_distances.items() if distance < median_distance]
if args.noisy_selection == False:
    # Select all noisy client
    selected_noisy_clients = list(noisy_distances.keys())

# Sort the noisy clients by distance
sorted_noisy_clients = sorted(noisy_distances, key=noisy_distances.get, reverse=True)

# Print the ranking of noisy clients by distance and noise ratio
logging.info("Ranking of noisy clients by normalized Euclidean distance from the clean cluster centroid:")
for rank, client in enumerate(sorted_clients_by_euclidean, 1):
    logging.info(f"Rank {rank}: Client {client}, Normalized Euclidean Distance: {scaled_distances[client][0]:.4f}")

logging.info("Ranking of noisy clients by normalized EMD from the clean cluster centroid:")
for rank, client in enumerate(sorted_clients_by_emd, 1):
    logging.info(f"Rank {rank}: Client {client}, Normalized EMD: {scaled_distances[client][1]:.4f}")

logging.info("Ranking of noisy clients by normalized real noise ratio:")
for rank, client in enumerate(sorted_clients_by_noise_ratio, 1):
    logging.info(f"Rank {rank}: Client {client}, Normalized Real Noise Ratio: {scaled_distances[client][2]:.4f}")

# Plotting the ranking of noisy clients by distance
plt.figure(figsize=(10, 8))
plt.bar(range(len(sorted_noisy_clients)), [noisy_distances[client] for client in sorted_noisy_clients], color='red')
plt.xticks(range(len(sorted_noisy_clients)), sorted_noisy_clients)
plt.xlabel('Client ID')
plt.ylabel('Distance from Clean Cluster Centroid')
plt.title('Ranking of Noisy Clients by Distance from Clean Cluster Centroid')
plt.savefig(f'./imgs/nlvl_{args.level_n_system}_noisy_clients_ranking.png')
plt.show()

logging.info(f"predicted noisy clients: {noisy_clients}, real noisy clients: {np.where(gamma_s > 0)[0]}")
logging.info(f"selected noisy clients : {selected_noisy_clients}")
clean_clients = list(set(user_id) - set(noisy_clients))
logging.info(f"selected clean clients: {clean_clients}")

############################################
#           Stage 2 - Training             #
############################################


class EvalPipelineS2Alt(StandalonePipeline):
    def __init__(self, handler, trainer, test_loader, clean_clients, selected_noisy_clients):
        super().__init__(handler, trainer)
        self.test_loader = test_loader
        self.loss = []
        self.acc = []
        self.bacc = []
        self.best_performance = 0
        self.best_balanced_accuracy = 0
        self.selected_noisy_clients = selected_noisy_clients
        self.clean_clients = clean_clients


    def main(self):
        t = 0
        while not self.handler.if_stop and t < args.com_round:
            
            logging.info("Round {}".format(t+1))

            # Server side
            sampled_clients = self.handler.sample_clients(self.selected_noisy_clients, self.clean_clients,len(self.clean_clients+self.selected_noisy_clients))
            broadcast = self.handler.downlink_package
            
            # Client side
            self.trainer.local_process_s2alt(broadcast, sampled_clients)
            uploads = self.trainer.uplink_package

            # Server side
            for pack in uploads:
                self.handler.load(pack)

            loss = evaluate(self.handler.model, nn.CrossEntropyLoss(), self.test_loader)

            pred = globaltest(copy.deepcopy(self.handler.model).to(
                args.device), test_data, args)
            acc = accuracy_score(fed_cifar10.targets_test, pred)
            bacc = balanced_accuracy_score(fed_cifar10.targets_test, pred)
            cm = confusion_matrix(fed_cifar10.targets_test, pred)

            logging.info("Loss {:.4f}, Balanced Accuracy {:.4f}".format(loss, bacc))
            logging.info(cm)
            
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
            
            self.loss.append(loss)
            self.acc.append(acc)
            self.bacc.append(bacc)
            t += 1
        
        logging.info('Final best balanced accuracy: {:.4f}, Best model number : {} '.format(self.best_balanced_accuracy, self.best_round_number))
        logging.info(self.bacc)

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
        if args.noisy_selection == True:
            plt.savefig(f"./imgs/s2_noisy_selection_fedavg_{args.dataname}_nlvl_{args.level_n_system}_loss_accuracy_{self.best_performance}.png", dpi=400, bbox_inches = 'tight')
        if args.noisy_selection == False:  
            plt.savefig(f"./imgs/s2_fedavg_{args.dataname}_nlvl_{args.level_n_system}_loss_accuracy_{self.best_performance}.png", dpi=400, bbox_inches = 'tight')
        
    def show_b(self):
        plt.figure(figsize=(8,4.5))
        ax = plt.subplot(1,2,1)
        ax.plot(np.arange(len(self.loss)), self.loss)
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Loss")
        
        ax2 = plt.subplot(1,2,2)
        ax2.plot(np.arange(len(self.bacc)), self.bacc)
        ax2.set_xlabel("Communication Round")
        ax2.set_ylabel("Balanced Accuarcy")
        if args.noisy_selection == True:
            plt.savefig(f"./imgs/s2_noisy_selection_fedavg_{args.dataname}_nlvl_{args.level_n_system}_loss_balanced_accuracy_{self.best_balanced_accuracy}.png", dpi=400, bbox_inches = 'tight')
        if args.noisy_selection == False:
            plt.savefig(f"./imgs/s2_noisy_selection_fedavg_{args.dataname}_nlvl_{args.level_n_system}_loss_balanced_accuracy_{self.best_balanced_accuracy}.png", dpi=400, bbox_inches = 'tight')

               
class EvalPipelineS2(StandalonePipeline):
    def __init__(self, args, handler, trainer, test_loader, clean_clients, selected_noisy_clients):
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
        self.selected_noisy_clients = selected_noisy_clients
        self.clean_clients = clean_clients

    def main(self):
        t = 0
        while not self.handler.if_stop and t < args.com_round:
            logging.info("Round {}".format(t+1))
            logging.info(len(self.clean_clients+self.selected_noisy_clients))
            # Server side
            sampled_clients = self.handler.sample_clients(self.selected_noisy_clients, self.clean_clients,len(self.clean_clients+self.selected_noisy_clients))
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
                noisy_clients=self.selected_noisy_clients, 
                clean_clients=self.clean_clients)
            
            uploads = self.trainer.uplink_package

            # Server side
            for pack in uploads:
                self.handler.load(pack, self.clean_clients, self.selected_noisy_clients)

            loss = evaluate(self.handler.model, nn.CrossEntropyLoss(), self.test_loader)

            pred = globaltest(copy.deepcopy(self.handler.model).to(
                args.device), test_data, args)
            acc = accuracy_score(fed_cifar10.targets_test, pred)
            bacc = balanced_accuracy_score(fed_cifar10.targets_test, pred)
            cm = confusion_matrix(fed_cifar10.targets_test, pred)

            logging.info("Loss {:.4f}, Balanced Accuracy {:.4f}".format(loss, bacc))
            logging.info(cm)
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
            
            self.loss.append(loss)
            self.acc.append(acc)
            self.bacc.append(bacc)
            t += 1
        
        logging.info('Final best balanced accuracy: {:.4f}, Best model number : {} '.format(self.best_balanced_accuracy, self.best_round_number))
        logging.info(f"Balanced Accuaracy of each rounds : {self.bacc}")

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
        if args.noisy_selection == True:
            plt.savefig(f"./imgs/s2_noisy_selection_fednoro_{args.dataname}_nlvl_{args.level_n_system}_loss_accuracy_{self.best_performance}.png", dpi=400, bbox_inches = 'tight')
        if args.noisy_selection == False:
            plt.savefig(f"./imgs/s2_fednoro_{args.dataname}_nlvl_{args.level_n_system}_loss_balanced_accuracy_{self.best_balanced_accuracy}.png", dpi=400, bbox_inches = 'tight')

    def show_b(self):
        plt.figure(figsize=(8,4.5))
        ax = plt.subplot(1,2,1)
        ax.plot(np.arange(len(self.loss)), self.loss)
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Loss")
        
        ax2 = plt.subplot(1,2,2)
        ax2.plot(np.arange(len(self.bacc)), self.bacc)
        ax2.set_xlabel("Communication Round")
        ax2.set_ylabel("Balanced Accuarcy")
        if args.noisy_selection == True:
            plt.savefig(f"./imgs/s2_noisy_selection_fednoro_{args.dataname}_nlvl_{args.level_n_system}_loss_balanced_accuracy_{self.best_balanced_accuracy}.png", dpi=400, bbox_inches = 'tight')
        if args.noisy_selection == False:
            plt.savefig(f"./imgs/s2_fednoro_{args.dataname}_nlvl_{args.level_n_system}_loss_balanced_accuracy_{self.best_balanced_accuracy}.png", dpi=400, bbox_inches = 'tight')
        
### Training with distance aware aggregator and logit adjustment ###
        
if args.aggregator == 'fedavg':

    logging.info("fedavg")

    trainer = FedNoRoSerialClientTrainer(model, args.total_client, base_lr=args.lr, cuda=args.cuda)
    trainer.setup_dataset(fed_cifar10)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

    handler = FedAvgServerHandlerS2(model=model, global_round=args.com_round, sample_ratio=1, cuda=args.cuda, num_clients=args.total_client)
    
    # Run evaluation
    eval_pipeline_s2alt = EvalPipelineS2Alt(handler=handler, trainer=trainer, test_loader=test_loader, clean_clients=clean_clients, selected_noisy_clients=selected_noisy_clients)
    eval_pipeline_s2alt.main()
    #eval_pipeline_s2alt.show()
    eval_pipeline_s2alt.show_b()    
   

### Training with fedavg aggregator and logit adjustment ###

if args.aggregator == 'fednoro':
    
    logging.info("fednoro")
    
    trainer = FedNoRoSerialClientTrainer(model, args.total_client, base_lr=args.lr, cuda=args.cuda)
    trainer.setup_dataset(fed_cifar10)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

    handler = FedNoRoServerHandler(model=model, global_round=args.com_round, sample_ratio=args.sample_ratio, cuda=args.cuda, num_clients=args.total_client)

    # Run evaluation
    eval_pipeline_s2 = EvalPipelineS2(handler=handler, trainer=trainer, selected_noisy_clients=selected_noisy_clients, clean_clients=clean_clients, test_loader=test_loader, args=args)
    eval_pipeline_s2.main()
    #eval_pipeline_s2.show()
    eval_pipeline_s2.show_b()
    
