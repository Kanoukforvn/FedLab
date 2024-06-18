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
from torch.multiprocessing import Pool
import torch.multiprocessing as mp
from multiprocessing import get_context
import numpy as np
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
    
    @staticmethod
    def calculate_divergence(client_model, global_model):
        total_divergence = 0.0

        # Ensure models are on the same device
        device = client_model[0].device
        for w_client, w_global in zip(client_model, global_model):
            # Ensure both client and global weights are on the GPU
            w_client = w_client.to(device)
            w_global = w_global.to(device)
            
            # Flatten and calculate divergence on the GPU
            w_client_flat = w_client.flatten()
            w_global_flat = w_global.flatten()
            total_divergence += torch.sum(torch.abs((w_client_flat - w_global_flat) / w_global_flat))

        # Calculate divergence and move result back to CPU
        divergence = total_divergence.item() / len(global_model)
        return divergence

    @staticmethod
    def FedMDCSAgg(parameters_list, global_model_parameters, past_performances):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Convert parameters_list and global_model_parameters to tensors and move to GPU
        parameters_tensor = torch.stack([torch.tensor(params).to(device) for params in parameters_list])
        global_params_tensor = torch.tensor(global_model_parameters).to(device)
        
        # Calculate normalized model divergence for each client
        epsilon = 1e-10
        divergences = torch.mean(torch.abs((parameters_tensor - global_params_tensor) / (global_params_tensor + epsilon)), dim=1)

        # Normalize divergences and past_performances
        norm_divergences = divergences / divergences.max()
        norm_past_performances = torch.tensor(past_performances).to(device) / max(past_performances)

        # Calculate composite score
        # Weights can be adjusted based on importance
        alpha, beta = 0.4, 0.3
        scores = alpha * norm_divergences + beta * norm_past_performances

        # Determine the median score
        median_score = scores.median().item()

        #For using median (better)
        # Select clients with scores above the median
        #selected_clients_indices = (scores > median_score).nonzero(as_tuple=True)[0].tolist()

        # Sort scores and select top 50%
        num_clients_to_select = len(scores) // 2
        _, selected_clients_indices = torch.topk(scores, num_clients_to_select)

        # If no clients are selected, use all clients
        if len(selected_clients_indices) == 0:
            selected_clients_indices = list(range(len(parameters_list)))

        # Log selected client indices
        logging.info(f"Selected client indices:{selected_clients_indices}")

        # Aggregate the selected clients' parameters
        selected_parameters_tensor = parameters_tensor[selected_clients_indices]
        aggregated_parameters_tensor = torch.mean(selected_parameters_tensor, dim=0)

        return aggregated_parameters_tensor
class DaAggregator:
    @staticmethod
    def DaAgg(parameters_list, clean_clients, noisy_clients):
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
    
    @staticmethod
    def DaFedMDCSAgg(client_models, global_model, top_n_clients, clean_clients, noisy_clients):
        """
        Ranks clients based on the normalized model divergence and aggregates using the top N clients,
        then uses FedAvg to perform the final aggregation.
        
        Args:
            client_models (list): List of client model weights.
            global_model (list): Weights of the global model.
            top_n_clients (int): Number of top clients to use for aggregation.

        Returns:
            aggregated_model: Aggregated model parameters.
        """
        device = global_model[0].device  # Ensure the global model is on the appropriate device

        # Calculate divergences on the same device as the global model
        divergences = [Aggregators.calculate_divergence(
            [tensor.to(device) for tensor in client_model], global_model) for client_model in client_models]

        # Rank clients based on normalized divergence in descending order
        ranked_clients = sorted(enumerate(divergences), key=lambda x: x[1], reverse=True)

        # Select the top N clients
        selected_clients_indices = [client[0] for client in ranked_clients[:top_n_clients]]
        
        # Aggregate the selected clients' models using FedAvg
        selected_models = [client_models[idx] for idx in selected_clients_indices]
        selected_weights = [1.0 / top_n_clients] * top_n_clients  # Uniform weights for simplicity

        aggregated_model = DaAggregator.DaAgg(selected_models, clean_clients, noisy_clients)

        return aggregated_model
        