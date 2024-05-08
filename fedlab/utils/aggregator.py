# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import copy
import logging

class Aggregators(object):
    """Define the algorithm of parameters aggregation"""

    @staticmethod
    def fedavg_aggregate(serialized_params_list, weights=None):
        """FedAvg aggregator

        Paper: http://proceedings.mlr.press/v54/mcmahan17a.html

        Args:
            serialized_params_list (list[torch.Tensor])): Merge all tensors following FedAvg.
            weights (list, numpy.array or torch.Tensor, optional): Weights for each params, the length of weights need to be same as length of ``serialized_params_list``

        Returns:
            torch.Tensor
        """
        if weights is None:
            weights = torch.ones(len(serialized_params_list))

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)

        # Move weights to the same device as the first tensor in serialized_params_list
        device = serialized_params_list[0].device
        weights = weights.to(device)

        logging.info("in function :  {}".format(weights))

        weights = weights / torch.sum(weights)
        assert torch.all(weights > 0), "weights should be non-negative values"

        # Move all tensors in serialized_params_list to the same device as weights
        serialized_params_list = [params.to(device) for params in serialized_params_list]

        # Perform the aggregation operation
        serialized_parameters = torch.sum(
            torch.stack(serialized_params_list, dim=-1) * weights, dim=-1)

        return serialized_parameters


    @staticmethod
    def fedasync_aggregate(server_param, new_param, alpha):
        """FedAsync aggregator
        
        Paper: https://arxiv.org/abs/1903.03934
        """
        serialized_parameters = torch.mul(1 - alpha, server_param) + \
                                torch.mul(alpha, new_param)
        return serialized_parameters
    
class DaAggregator(object):
    def __init__(self, device=torch.device('cuda')):
        self.device = device

    @staticmethod
    def DaAgg(serialized_params_list, w, clean_clients, noisy_clients):
        dict_len = [len(params) for params in serialized_params_list]
        client_weight = np.array(dict_len)
        client_weight = client_weight / client_weight.sum()
        distance = np.zeros(len(dict_len))
        for n_idx in noisy_clients:
            dis = []
            for c_idx in clean_clients:
                dis.append(DaAggregator.model_dist(w[n_idx], w[c_idx]))
            distance[n_idx] = min(dis)
        distance = distance / distance.max()
        client_weight = client_weight * np.exp(-distance)
        client_weight = client_weight / client_weight.sum()
        # print(client_weight)

        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * client_weight[0] 
            for i in range(1, len(w)):
                w_avg[k] += w[i][k] * client_weight[i]
        
        
        # Move all tensors in serialized_params_list to the same device as weights
        serialized_params_list = [params.to(DaAggregator.device) for params in serialized_params_list]

        # Perform the aggregation operation
        serialized_parameters = torch.sum(
            torch.stack(serialized_params_list, dim=-1) * w_avg, dim=-1)

        return w_avg

    @staticmethod
    def model_dist(w_1, w_2):
        assert w_1.keys() == w_2.keys(), "Error: cannot compute distance between dict with different keys"
        dist_total = torch.zeros(1).float()
        for key in w_1.keys():
            if "int" in str(w_1[key].dtype):
                continue
            dist = torch.norm(w_1[key] - w_2[key])
            dist_total += dist.cpu()

        return dist_total.cpu().item()


    """

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
    
       
        
        Data-aware aggregation

        Args:
            serialized_params_list (list[torch.Tensor]): List of serialized model parameters from each client.
            clean_clients (list[int]): List of indices of clean clients.
            noisy_clients (list[int]): List of indices of noisy clients.

        Returns:
            torch.Tensor: Aggregated serialized parameters.
        
        
        

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

        """