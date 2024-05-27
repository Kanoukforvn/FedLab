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

import os

import torch
from torch.utils.data import DataLoader
import torchvision

from .basic_dataset import FedDataset, BaseDataset
from ...utils.dataset.partition import CIFAR10Partitioner


class PartitionedCIFAR10(FedDataset):
    """Class for partitioning the CIFAR10 dataset for federated learning."""

    def __init__(
        self,
        root,
        path,
        dataname,
        num_clients,
        num_classes,
        download=True,
        preprocess=False,
        balance=True,
        partition="iid",
        unbalance_sgm=0,
        num_shards=None,
        dir_alpha=None,
        verbose=True,
        seed=None,
        transform=None,
        target_transform=None,
    ) -> None:
        """Initialize PartitionedCIFAR10."""
        self.dataname = dataname
        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.transform = transform
        self.target_transform = target_transform
        self.targets_train = None
        self.targets_test = None

        if preprocess:
            self.preprocess(
                balance=balance,
                partition=partition,
                unbalance_sgm=unbalance_sgm,
                num_shards=num_shards,
                dir_alpha=dir_alpha,
                verbose=verbose,
                seed=seed,
                download=download,
            )

    def preprocess(
        self,
        balance=True,
        partition="iid",
        unbalance_sgm=0,
        num_shards=None,
        dir_alpha=None,
        verbose=True,
        seed=None,
        download=True,
    ):
        """Perform FL partition on the dataset."""
        self.download = download

        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "test"))

        # train and test dataset partitioning
        trainset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=True,
            transform=self.transform,
            download=self.download,
        )
        testset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=False,
            transform=self.transform,
            download=self.download,
        )

        self.targets_train = trainset.targets
        self.targets_test = testset.targets

        partitioner_train = CIFAR10Partitioner(
            trainset.targets,
            self.num_clients,
            balance=balance,
            partition=partition,
            unbalance_sgm=unbalance_sgm,
            num_shards=num_shards,
            dir_alpha=dir_alpha,
            verbose=verbose,
            seed=seed,
        )
        partitioner_test = CIFAR10Partitioner(
            testset.targets,
            self.num_clients,
            balance=balance,
            partition=partition,
            unbalance_sgm=unbalance_sgm,
            num_shards=num_shards,
            dir_alpha=dir_alpha,
            verbose=verbose,
            seed=seed,
        )

        self.data_indices_train = partitioner_train.client_dict
        self.data_indices_test = partitioner_test.client_dict

        samples_train, labels_train = [], []
        samples_test, labels_test = [], []

        for x, y in trainset:
            samples_train.append(x)
            labels_train.append(y)

        for x, y in testset:
            samples_test.append(x)
            labels_test.append(y)

        for cid_train, indices_train in self.data_indices_train.items():
            data_train, label_train = [], []
            for idx in indices_train:
                x_train, y_train = samples_train[idx], labels_train[idx]
                data_train.append(x_train)
                label_train.append(y_train)
            dataset_train = BaseDataset(data_train, label_train)
            torch.save(
                dataset_train,
                os.path.join(self.path, "train", f"data{cid_train}.pkl"),
            )

        for cid_test, indices_test in self.data_indices_test.items():
            data_test, label_test = [], []
            for idx in indices_test:
                x_test, y_test = samples_test[idx], labels_test[idx]
                data_test.append(x_test)
                label_test.append(y_test)
            dataset_test = BaseDataset(data_test, label_test)
            torch.save(
                dataset_test,
                os.path.join(self.path, "test", f"data{cid_test}.pkl"),
            )

    def get_dataset(self, cid, type="train"):
        """Load subdataset for client with client ID ``cid`` from local file."""
        dataset = torch.load(
            os.path.join(self.path, type, f"data{cid}.pkl")
        )
        return dataset

    def get_dataloader(self, cid, batch_size=None, type="train"):
        """Return dataload for client with client ID ``cid``."""
        dataset = self.get_dataset(cid, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return data_loader

