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

        #logging.info("in function :  {}".format(weights))

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


import torch
import numpy as np

class DaAggregator:
    @staticmethod
    def DaAgg(parameters_list, weights, clean_clients, noisy_clients):
        # Define the client-wise distance metric
        def compute_distance(wi, wj):
            return torch.norm(wi - wj, p=2)

        distances = []
        for i, wi in enumerate(parameters_list):
            min_distance = min(compute_distance(wi.cpu(), wj.cpu()) for j, wj in enumerate(parameters_list) if j in clean_clients)
            distances.append(min_distance)

        max_distance = max(distances) if distances else 1  # Avoid division by zero
        normalized_distances = [d / max_distance for d in distances]

        aggregation_weights = []
        for i, d in enumerate(normalized_distances):
            if i in clean_clients:
                aggregation_weights.append(1.0)
            elif i in noisy_clients:
                aggregation_weights.append(np.exp(-d.cpu().item()))
            else:
                aggregation_weights.append(0.0)

        # Normalize aggregation weights
        total_weights = sum(aggregation_weights)
        normalized_weights = [w / total_weights for w in aggregation_weights]

        # Aggregate local models
        global_model = sum(parameters_list[i].cpu() * weight for i, weight in enumerate(normalized_weights))

        return global_model