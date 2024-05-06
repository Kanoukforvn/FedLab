import sys
sys.path.append("../")

from copy import deepcopy
import torch
from fedlab.core.client.trainer import ClientTrainer, SerialClientTrainer
from fedlab.contrib.algorithm import SyncServerHandler, SGDSerialClientTrainer
from fedlab.utils import Logger, SerializationTool, Aggregators, LogitAdjust

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
    """FedAvg server handler."""
    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        weights = [ele[1] for ele in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, weights)
        SerializationTool.deserialize_model(self._model, serialized_parameters)


##################
#
#      Client
#
##################
        
class FedNoRoSerialClientTrainerS1(SGDSerialClientTrainer):

    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False,
                 warmup_rounds=15, lr_warmup=0.0003, epochs_warmup=5, lr=0.01, epochs=5, num_class = 10) -> None:
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

    def get_num_of_each_class_global(self, fed_dataset):

        label_counts_per_client = []
        
        for client_index in range(self.num_clients):
            dataset_train_client = fed_dataset.get_dataset(client_index, type="train")
            label_counts = Counter()
            for _, label in dataset_train_client:
                label_counts[label] += 1
            label_counts_per_client.append([label_counts[class_label] for class_label in sorted(label_counts.keys())])
        
        return label_counts_per_client

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        w_local, loss_local = [], []

        for id in (progress_bar := tqdm(id_list)):
            progress_bar.set_description(f"Training on client {id}", refresh=True)
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            w_local, loss_local = self.train_LA(model_parameters.cuda(self.device), data_loader)
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

        # Increment iteration
        self.iteration += 1

        return [SerializationTool.serialize_model(self._model), avg_epoch_loss]

    def train_warmup(self, model_parameters, train_loader):
        """Warm-up phase training using FedAvg.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data during warm-up phase.

        Returns:
            list: A list containing serialized model parameters and average loss.
        """
        # Deserialize model parameters and move them to the GPU device
        self.set_model(model_parameters.cuda(self.device))
        self._model.train()

        # Set up optimizer with warm-up learning rate
        optimizer = torch.optim.SGD(self._model.parameters(), self.lr_warmup)

        # Warm-up phase training loop
        for epoch in range(self.epochs):
            epoch_loss = []

            # Training loop
            for data, target in train_loader:
                data, target = data.cuda(self.device), target.cuda(self.device)
                optimizer.zero_grad()
                output = self._model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())

            # Calculate average loss for the epoch
            avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)

        # Increment iteration
        self.iteration += 1

        # Return serialized model parameters and average loss
        return [SerializationTool.serialize_model(self._model), avg_epoch_loss]
