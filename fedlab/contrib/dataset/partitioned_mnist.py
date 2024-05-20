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
from torchvision import transforms


from .basic_dataset import FedDataset, BaseDataset
from ...utils.dataset.partition import CIFAR10Partitioner, CIFAR100Partitioner, MNISTPartitioner


class PartitionedMNIST(FedDataset):
    """:class:`FedDataset` with partitioning preprocess. For detailed partitioning, please
    check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.

    
    Args:
        root (str): Path to download raw dataset.
        path (str): Path to save partitioned subdataset.
        num_clients (int): Number of clients.
        download (bool): Whether to download the raw dataset.
        preprocess (bool): Whether to preprocess the dataset.
        partition (str, optional): Partition name. Only supports ``"noniid-#label"``, ``"noniid-labeldir"``, ``"unbalance"`` and ``"iid"`` partition schemes.
        dir_alpha (float, optional): Dirichlet distribution parameter for non-iid partition. Only works if ``partition="dirichlet"``. Default as ``None``.
        verbose (bool, optional): Whether to print partition process. Default as ``True``.
        seed (int, optional): Random seed. Default as ``None``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """
    def __init__(self,
                 root,
                 path,
                 num_clients,
                 download=True,
                 preprocess=False,
                 partition="iid",
                 dir_alpha=None,
                 verbose=True,
                 seed=None,
                 transform=None,
                 target_transform=None) -> None:

        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients
        self.transform = transform
        self.targt_transform = target_transform

        if preprocess:
            self.preprocess(partition=partition,
                            dir_alpha=dir_alpha,
                            verbose=verbose,
                            seed=seed,
                            download=download,
                            transform=transform,
                            target_transform=target_transform)

    def preprocess(self,
                   partition="iid",
                   dir_alpha=None,
                   verbose=True,
                   seed=None,
                   download=True,
                   transform=None,
                   target_transform=None):
        """Perform FL partition on the dataset, and save each subset for each client into ``data{cid}.pkl`` file.

        For details of partition schemes, please check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.
        """
        self.download = download

        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "var"))
            os.mkdir(os.path.join(self.path, "test"))

        trainset = torchvision.datasets.MNIST(root=self.root,
                                                train=True,
                                                download=download)

        testset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=False,
            download=self.download,
        )


        self.targets_train = trainset.targets
        self.targets_test = testset.targets


        partitioner_train = MNISTPartitioner(trainset.targets,
                                        self.num_clients,
                                        partition=partition,
                                        dir_alpha=dir_alpha,
                                        verbose=verbose,
                                        seed=seed)

        partitioner_test = MNISTPartitioner(testset.targets,
                                        self.num_clients,
                                        partition=partition,
                                        dir_alpha=dir_alpha,
                                        verbose=verbose,
                                        seed=seed)

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
        """Load subdataset for client with client ID ``cid`` from local file.

        Args:
             cid (int): client id
             type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.

        Returns:
            Dataset
        """
        dataset = torch.load(
            os.path.join(self.path, type, "data{}.pkl".format(cid)))
        return dataset

    def get_dataloader(self, cid, batch_size=None, type="train"):
        """Return dataload for client with client ID ``cid``.

        Args:
            cid (int): client id
            batch_size (int, optional): batch size in DataLoader.
            type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.
        """
        dataset = self.get_dataset(cid, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader
