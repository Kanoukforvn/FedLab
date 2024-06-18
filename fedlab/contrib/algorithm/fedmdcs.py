import torch
from typing import List
from tqdm import tqdm

from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
from ...utils.aggregator import Aggregators
from ...utils.serialization import SerializationTool

from fedlab.contrib.client_sampler.base_sampler import FedSampler
from fedlab.utils.logger import Logger

##################
#
#      Server
#
##################


import torch
from typing import List
from copy import deepcopy
from .basic_server import SyncServerHandler
from ...utils.serialization import SerializationTool
from ...utils.aggregator import Aggregators

class FedMDCSServerHandler(SyncServerHandler):
    def __init__(self, model, global_round, num_clients=0, sample_ratio=1, cuda=False, device=None, sampler=None, logger=None, top_n_clients=10):
        super(FedMDCSServerHandler, self).__init__(model, global_round, num_clients, sample_ratio, cuda, device, sampler, logger)
        self.top_n_clients = top_n_clients
        self.past_performances = [0] * num_clients  # Initialize past performances

    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        data_sizes = [ele[1] for ele in buffer]

        aggregated_parameters_tensor = Aggregators.FedMDCSAgg(parameters_list, self.model_parameters, self.past_performances)
        SerializationTool.deserialize_model(self._model, aggregated_parameters_tensor)

        # Update past performances (Example: you can update based on some metric)
        # For simplicity, let's say we measure performance as the inverse of divergence
        for idx, params in enumerate(parameters_list):
            divergence = torch.mean(torch.abs((torch.tensor(params) - torch.tensor(self.model_parameters)) / (torch.tensor(self.model_parameters) + 1e-10)))
            self.past_performances[idx] = 1 / (divergence + 1e-10)  # Avoid division by zero

        # Normalize past performances to keep them in a reasonable range
        max_perf = max(self.past_performances)
        self.past_performances = [perf / max_perf for perf in self.past_performances]

##################
#
#      Client
#
##################


class FedMDCSSerialClientTrainer(SGDSerialClientTrainer):
    """Federated client with local SGD solver."""
    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self._model.train()

        data_size = 0
        for _ in range(self.epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                data_size += len(target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return [self.model_parameters, data_size]
