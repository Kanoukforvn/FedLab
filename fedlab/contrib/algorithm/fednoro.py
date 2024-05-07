import sys
sys.path.append("../")

from copy import deepcopy
import torch
from fedlab.core.client.trainer import ClientTrainer, SerialClientTrainer
from fedlab.contrib.algorithm import SyncServerHandler, SGDSerialClientTrainer
from fedlab.utils import Logger, SerializationTool, Aggregators, LogitAdjust, LA_KD
from fedlab.utils.fednoro_utils import get_current_consistency_weight

import logging
import copy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import Counter

from typing import List
import numpy as np

##################
#
#      Server
#
##################


class FedAvgServerHandler(SyncServerHandler):
    """FedNoRo server handler."""
    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        weights = [ele[1] for ele in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, weights)
        SerializationTool.deserialize_model(self._model, serialized_parameters)

class FedNoRoServerHandler(SyncServerHandler):
    """FedNoRo server handler."""
    def __init__(self, model, global_round, sample_ratio, cuda, noisy_clients, clean_clients):
        super().__init__(model, global_round, sample_ratio, cuda)
        self.noisy_clients = noisy_clients
        self.clean_clients = clean_clients

    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        weights = [ele[1] for ele in buffer]
        logging.info("weights: {}".format(weights))
        serialized_parameters = DaAggregator.DaAgg(parameters_list, weights, clean_clients=self.clean_clients, noisy_clients=self.noisy_clients)
        SerializationTool.deserialize_model(self._model, serialized_parameters)


class DaAggregator(object):
    def __init__(self, device=torch.device('cuda')):
        self.device = device

    @staticmethod
    def model_dist(w_1, w_2):
        assert w_1.keys() == w_2.keys(), "Error: cannot compute distance between dicts with different keys"
        dist_total = torch.zeros(1).float()
        for key in w_1.keys():
            if "int" in str(w_1[key].dtype):
                continue
            dist = torch.norm(w_1[key] - w_2[key])
            dist_total += dist.cpu()

        return dist_total.cpu().item()

    @staticmethod
    def DaAgg(serialized_params_list, weights, clean_clients, noisy_clients):
        """Data-aware aggregation

        Args:
            serialized_params_list (list[torch.Tensor]): List of serialized model parameters from each client.
            clean_clients (list[int]): List of indices of clean clients.
            noisy_clients (list[int]): List of indices of noisy clients.

        Returns:
            torch.Tensor: Aggregated serialized parameters.
        """

        if weights is None:
            weights = torch.ones(len(serialized_params_list))

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)

        # Move weights to the same device as the first tensor in serialized_params_list
        device = serialized_params_list[0].device
        weights = weights.to(device)

        # Initialize client weights
        num_params = [len(params) for params in serialized_params_list]
        client_weight = torch.tensor(num_params, dtype=torch.float)
        client_weight /= torch.sum(client_weight)

        device = serialized_params_list[0].device

        # Calculate distance from noisy clients
        distance = torch.zeros(len(num_params))
        for n_idx in noisy_clients:
            dis = []
            for c_idx in clean_clients:
                dis.append(DaAggregator.model_dist(weights[n_idx], weights[c_idx]))
            distance[n_idx] = min(dis)
        distance /= torch.max(distance)

        # Update client weights based on distance
        client_weight *= torch.exp(-distance)
        client_weight /= torch.sum(client_weight)

        # Perform aggregation
        serialized_params_list = [params.to(device) for params in serialized_params_list]
        serialized_parameters = torch.sum(
            torch.stack(serialized_params_list, dim=-1) * client_weight, dim=-1)

        return serialized_parameters


##################
#
#      Client
#
##################
        
class FedNoRoSerialClientTrainer(SGDSerialClientTrainer):

    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False,
                 warmup_rounds=5, lr_warmup=0.0003, epochs_warmup=5, lr=0.01, epochs=5, num_class = 10) -> None:
        super().__init__(model, num_clients, cuda, device, personal)
        self._LOGGER = logger if logger is not None else Logger()
        self.warmup_rounds = warmup_rounds #FIXME replace  com round with warm up round
        self.lr_warmup = lr_warmup
        self.epochs_warmup = epochs_warmup
        self.lr = lr
        self.epochs = epochs
        self.iteration = 0
        self.num_class = num_class

    def setup_dataset(self, dataset):
        self.dataset = dataset
    
    def setup_optim(self, epochs, batch_size, lr):

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        cls_num_list = self.get_num_of_each_class_global(self.dataset)
        for client_index, label_counts in enumerate(cls_num_list):
            self.ce_criterion = LogitAdjust(label_counts)
            self.cb_criterion = LA_KD(label_counts)

    def get_num_of_each_class_global(self, fed_dataset):

        label_counts_per_client = []
        
        for client_index in range(self.num_clients):
            dataset_train_client = fed_dataset.get_dataset(client_index, type="train")
            label_counts = Counter()
            for _, label in dataset_train_client:
                label_counts[label] += 1
            label_counts_per_client.append([label_counts[class_label] for class_label in sorted(label_counts.keys())])
        
        return label_counts_per_client

    def local_process_s1(self, payload, id_list):
        model_parameters = payload[0]
        w_local, loss_local = [], []

        for id in (progress_bar := tqdm(id_list)):
            progress_bar.set_description(f"Training on client {id}", refresh=True)
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            w_local, loss_local = self.train_LA(model_parameters.cuda(self.device), data_loader)
            pack = [w_local, loss_local]
            self.cache.append(pack)

    def local_process_s2(self, payload, id_list, t, begin, end, a, clean_clients, noisy_clients):
        model_parameters = payload[0]
        w_local, loss_local = [], []

        weight_kd = get_current_consistency_weight(
            t, begin, end) * a

        for id in (progress_bar := tqdm(id_list)):
            progress_bar.set_description(f"Training on client {id}", refresh=True)
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            
            if id in clean_clients:
                w_local, loss_local = self.train_LA(model_parameters.cuda(self.device), data_loader)
                pack = [w_local, loss_local]
                self.cache.append(pack)

            elif id in noisy_clients:
                w_local, loss_local = self.train_fednoro(model_parameters.cuda(self.device), data_loader, weight_kd=weight_kd)
                pack = [w_local, loss_local]
                self.cache.append(pack)


    def train_LA(self, model_parameters, train_loader):
    
        self.set_model(model_parameters.cuda(self.device))
        self._model.train()

        optimizer = torch.optim.SGD(self._model.parameters(), lr=self.lr, momentum=0.5)
        
        for epoch in range(self.epochs):
            
            epoch_loss = []                        
            
            for data, target in train_loader:

                data, target = data.cuda(self.device), target.cuda(self.device)
                optimizer.zero_grad()
                logits = self._model(data)
                loss = self.ce_criterion(logits, target)
                loss.backward()

                optimizer.step()
                epoch_loss.append(loss.item())

            avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)

        
        return [SerializationTool.serialize_model(self._model), avg_epoch_loss]

    def train_fednoro(self, model_parameters, train_loader, weight_kd):

        self.set_model(model_parameters.cuda(self.device))

        self.student_net = self._model
        self.teacher_net = self._model

        self.student_net.train()
        self.teacher_net.eval()

        self.optimizer = torch.optim.SGD(self._model.parameters(), lr=self.lr, momentum=0.5)

        for epoch in range(self.epochs):
        
            epoch_loss = []
        
            batch_loss = []

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self._model(images)

                with torch.no_grad():
                    teacher_output = self.teacher_net(images)
                    soft_label = torch.softmax(teacher_output / 0.8, dim=1)

                loss = self.cb_criterion(logits, labels, soft_label, weight_kd)
                loss = loss.float()
                loss.backward()

                self.optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(np.array(batch_loss).mean())

        return SerializationTool.serialize_model(self._model), np.array(epoch_loss).mean()

