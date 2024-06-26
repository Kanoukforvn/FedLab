import sys
sys.path.append("../")

from copy import deepcopy
import torch
from fedlab.core.client.trainer import ClientTrainer, SerialClientTrainer
from fedlab.contrib.algorithm import SyncServerHandler, SGDSerialClientTrainer
from fedlab.utils import Logger, SerializationTool, Aggregators, LogitAdjust, LA_KD, DaAggregator
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
import torch.nn.functional as F
from ..client_sampler.uniform_sampler import RandomSampler

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

class FedAvgServerHandlerS2(SyncServerHandler):
    """FedNoRo server handler."""
    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        weights = [ele[1] for ele in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, weights)
        SerializationTool.deserialize_model(self._model, serialized_parameters)

    def sample_clients(self, selected_noisy_clients, clean_clients, num_to_sample=None):
   
        # Combine the selected noisy clients and clean clients
        total_clients = selected_noisy_clients + clean_clients
        total_clients_count = len(total_clients)
        
        # Number of clients to sample per round
        num_to_sample = self.num_clients_per_round if num_to_sample is None else num_to_sample

        num_to_sample = min(num_to_sample, total_clients_count)

        if self.sampler is None:
            self.sampler = RandomSampler(len(total_clients))
        
        # Sample from the combined list of clients
        sampled_indices = self.sampler.sample(num_to_sample)
        sampled_clients = [total_clients[i] for i in sampled_indices]
        
        #assert self.num_clients_per_round == len(sampled_clients)
        return sorted(sampled_clients)


class FedNoRoServerHandler(SyncServerHandler):
    """FedNoRo server handler."""

    def global_update_daagg(self, buffer, clean_clients, noisy_clients):
        parameters_list = [ele[0] for ele in buffer]
        weights = [ele[1] for ele in buffer]        
        serialized_parameters = DaAggregator.DaAgg(parameters_list, clean_clients, noisy_clients)
        SerializationTool.deserialize_model(self._model, serialized_parameters)
    
    def sample_clients(self, selected_noisy_clients, clean_clients, num_to_sample=None):
   
        # Combine the selected noisy clients and clean clients
        total_clients = selected_noisy_clients + clean_clients
        total_clients_count = len(total_clients)
        
        # Number of clients to sample per round
        num_to_sample = self.num_clients_per_round if num_to_sample is None else num_to_sample

        num_to_sample = min(num_to_sample, total_clients_count)

        if self.sampler is None:
            self.sampler = RandomSampler(len(total_clients))
        
        # Sample from the combined list of clients
        sampled_indices = self.sampler.sample(num_to_sample)
        sampled_clients = [total_clients[i] for i in sampled_indices]
        
        #assert self.num_clients_per_round == len(sampled_clients)
        return sorted(sampled_clients)
        
    def load(self, payload: List[torch.Tensor], clean_clients, noisy_clients) -> bool:
        """Update global model with collected parameters from clients.

        Note:
            Server handler will call this method when its ``client_buffer_cache`` is full. User can
            overwrite the strategy of aggregation to apply on :attr:`model_parameters_list`, and
            use :meth:`SerializationTool.deserialize_model` to load serialized parameters after
            aggregation into :attr:`self._model`.

        Args:
            payload (list[torch.Tensor]): A list of tensors passed by manager layer.
        """
        assert len(payload) > 0
        self.client_buffer_cache.append(deepcopy(payload))

        assert len(self.client_buffer_cache) <= self.num_clients_per_round

        if len(self.client_buffer_cache) == self.num_clients_per_round:
            self.global_update_daagg(self.client_buffer_cache, clean_clients, noisy_clients)
            self.round += 1

            # reset cache
            self.client_buffer_cache = []

            return True  # return True to end this round.
        else:
            return False



##################
#
#      Client
#
##################
        
class FedNoRoSerialClientTrainer(SGDSerialClientTrainer):

    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False,
                base_lr=0.0003, epochs=5, num_class = 10) -> None:
        super().__init__(model, num_clients, cuda, device, personal)
        self.lr = base_lr
        self.epochs = epochs
        self.iteration = 0
        self.num_class = num_class

    def setup_dataset(self, dataset):
        self.dataset = dataset
    
    def setup_optim(self, epochs, batch_size, lr):

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr=self.lr)
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
            w_local, loss_local = self.train_LA_s1(model_parameters.cuda(self.device), data_loader)
            pack = [w_local, loss_local]
            self.cache.append(pack)

    def local_process_s2alt(self, payload, id_list):
        model_parameters = payload[0]
        w_local, loss_local = [], []

        for id in (progress_bar := tqdm(id_list)):
            progress_bar.set_description(f"Training on client {id}", refresh=True)
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            w_local, loss_local = self.train_LA_s2(model_parameters.cuda(self.device), data_loader)
            pack = [w_local, loss_local]
            self.cache.append(pack)

    def local_process_s2(self, payload, id_list, t, begin, end, a, clean_clients, noisy_clients):
        model_parameters = payload[0]

        w_locals, loss_locals = [], []

        weight_kd = get_current_consistency_weight(
            t, begin, end) * a

        for id in (progress_bar := tqdm(id_list)):
            
            
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            
            if id in clean_clients:
                progress_bar.set_description(f"Training on clean client {id}", refresh=True)
                w_local, loss_local = self.train_LA_s2(model_parameters.cuda(self.device), data_loader)
                pack = [w_local, loss_local]
                self.cache.append(pack)
            
            elif id in noisy_clients:
                progress_bar.set_description(f"Training on noisy client {id}", refresh=True)
                w_local, loss_local = self.train_fednoro(model_parameters.cuda(self.device), data_loader, weight_kd=weight_kd)
                pack = [w_local, loss_local]
                self.cache.append(pack)
            
    def train_LA_s1(self, model_parameters, train_loader):
    
        self.set_model(model_parameters.cuda(self.device))
        self._model.train()

        # set the optimizer
        #optimizer = torch.optim.Adam(
        #     self._model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=5e-4)
        
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

    def train_LA_s2(self, model_parameters, train_loader):
        
            self.set_model(model_parameters.cuda(self.device))
            self._model.train()
    
            # set the optimizer
            optimizer = torch.optim.Adam(
                 self._model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=5e-4)
            
            #optimizer = torch.optim.SGD(self._model.parameters(), lr=self.lr, momentum=0.5)
            
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
        def gaussian_ramp_up(epoch, total_epochs, lambda_max=0.8):
            return lambda_max * np.exp(-5 * (1 - (epoch / total_epochs))**2)

        self.set_model(model_parameters.cuda(self.device))

        self.student_net = self._model
        self.teacher_net = self._model

        self.student_net.train()
        self.teacher_net.eval()

        #optimizer = torch.optim.SGD(self._model.parameters(), lr=self.lr, momentum=0.5)
        optimizer = torch.optim.Adam(
             self._model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=5e-4)
        
        for epoch in range(self.epochs):
            epoch_loss = []
            lambda_kd = gaussian_ramp_up(epoch, self.epochs)

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                logits = self._model(images)

                with torch.no_grad():
                    teacher_logits = self.teacher_net(images)
                    soft_labels = F.softmax(teacher_logits / 0.8, dim=1)

                loss_kl = F.kl_div(F.log_softmax(logits / 0.8, dim=1), soft_labels, reduction='batchmean')
                loss_ce = F.cross_entropy(logits, labels)
                loss = lambda_kd * loss_kl + (1 - lambda_kd) * loss_ce
                loss = loss.float()
                loss.backward()

                optimizer.step()
                epoch_loss.append(loss.item())

            epoch_mean_loss = np.mean(epoch_loss)

        return SerializationTool.serialize_model(self._model), epoch_mean_loss    
