import sys
sys.path.append("../")

from copy import deepcopy
import torch
from fedlab.core.client.trainer import ClientTrainer, SerialClientTrainer
from fedlab.contrib.algorithm import SyncServerHandler, SGDSerialClientTrainer
from fedlab.utils import Logger, SerializationTool, Aggregators, LogitAdjust

import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from typing import List
import numpy as np

def globaltest(net, test_dataset, args):
    net.eval()
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    pred = np.array([])
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            pred = np.concatenate([pred, predicted.detach().cpu().numpy()], axis=0)
    return pred


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
    """
    Train multiple clients in a single process using FedNoRo algorithm with warm-up phase.

    Args:
        model (torch.nn.Module): Model used in this federation.
        num_clients (int): Number of clients in current trainer.
        cuda (bool): Use GPUs or not. Default: ``False``.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None.
        logger (Logger, optional): Object of :class:`Logger`.
        personal (bool, optional): If True is passed, SerialModelMaintainer will generate the copy of local parameters list and maintain them respectively. These parameters are indexed by [0, num-1]. Defaults to False.
        warmup_rounds (int): Number of warm-up rounds for FedAvg.
        lr_warmup (float): Learning rate for warm-up phase.
        epochs_warmup (int): Number of epochs for warm-up phase.
        lr (float): Learning rate for FedNoRo algorithm.
        epochs (int): Number of epochs for FedNoRo algorithm.
    """
    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False,
                 warmup_rounds=1, lr_warmup=0.01, epochs_warmup=1, lr=0.01, epochs=10) -> None:
        super().__init__(model, num_clients, cuda, device, personal)
        self._LOGGER = logger if logger is not None else Logger()
        self.warmup_rounds = warmup_rounds
        self.lr_warmup = lr_warmup
        self.epochs_warmup = epochs_warmup
        self.lr = lr
        self.epochs = epochs
        self.iteration = 0
        self.device = device

    def setup_dataset(self, dataset):
        self.dataset = dataset

    def setup_optim(self, epochs, batch_size, lr):
        """Set up local optimization configuration.

        Args:
            epochs (int): Local epochs.
            batch_size (int): Local batch size. 
            lr (float): Learning rate.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for id in (progress_bar := tqdm(id_list)):
            progress_bar.set_description(f"Training on client {id}", refresh=True)
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            if self.iteration < self.warmup_rounds:
                pack = self.train_warmup(model_parameters, data_loader)
            else:
                pack = self.train_fednoro(model_parameters, data_loader)
            self.cache.append(pack)

    def train_warmup(self, model_parameters, train_loader):
        """Warm-up phase training using FedAvg.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data during warm-up phase.

        Returns:
            list: A list containing serialized model parameters and average loss.
        """
        self.set_model(model_parameters)
        self._model.train()

        # Set up optimizer with warm-up learning rate
        optimizer = torch.optim.SGD(self._model.parameters(), self.lr_warmup)

        # Warm-up phase training loop
        for _ in range(self.epochs_warmup):
            epoch_loss = []

            # Training loop
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
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

    def train_fednoro(self, model_parameters, train_loader):
        """FedNoRo algorithm training.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data during FedNoRo algorithm phase.

        Returns:
            list: A list containing serialized model parameters and average loss.
        """
        self.set_model(model_parameters)
        self._model.train()

        # Set up optimizer with FedNoRo learning rate
        optimizer = torch.optim.SGD(self._model.parameters(), self.lr)

        # FedNoRo algorithm training loop
        for _ in range(self.epochs):
            epoch_loss = []

            # Training loop
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
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