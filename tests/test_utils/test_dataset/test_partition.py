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

import unittest
import os
import numpy as np
from fedlab.utils.dataset.partition import CIFAR10Partitioner


#
# class DataPartitionerTestCase(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls) -> None:
#         cls.num_samples = 10000
#         cls.num_classes = 10
#         cls.num_clients = 100
#
#     def setUp(self) -> None:
#         np.random.seed(2021)
#
#     def test_len(self):
#         pass
#
#     def test_item(self):
#         pass


class CIFAR10PartitionerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.num_classes = 10
        cls.num_clients = 50
        cls.seed = 2021
        targets = []
        for k in range(cls.num_classes):
            targets.append([k for _ in range(5000)])
        cls.num_samples = len(targets)
        targets = np.array(targets)
        np.random.seed(cls.seed)
        cls.targets = targets[np.random.permutation(cls.num_samples)]  # shuffle

    def test_len(self):
        partitioner = CIFAR10Partitioner(self.targets,
                                         self.num_clients,
                                         balance=True,
                                         partition="iid",
                                         seed=self.seed)
        self.assertEqual(len(partitioner), self.num_clients)

    def test_item(self):
        partitioner = CIFAR10Partitioner(self.targets,
                                         self.num_clients,
                                         balance=True,
                                         partition="iid",
                                         seed=self.seed)
        res = [all(partitioner[cid] == partitioner.client_dict[cid]) for cid in
               range(self.num_clients)]
        self.assertTrue(all(res))

    # def test_hetero_dir(self):
    #     # perform partition
    #     partition = CIFAR10Partitioner(self.targets,
    #                                    self.num_clients,
    #                                    balance=None,
    #                                    partition="dirichlet",
    #                                    dir_alpha=0.3,
    #                                    seed=self.seed)
    #     self.assertEqual(len(partition), self.num_clients)
    #     client_sample_nums = np.array([len(partition[cid]) for cid in range(self.num_clients)])
    #     # sample number of each client should no less than number of classes
    #     self.assertTrue(all(client_sample_nums >= self.num_classes))
    #     # sample number of each client should not be the same
    #     self.assertTrue(len(set(client_sample_nums)) >= 2)

    def test_shards(self):
        num_shards = 200
        partition = CIFAR10Partitioner(self.targets,
                                       self.num_clients,
                                       balance=None,
                                       partition="shards",
                                       num_shards=num_shards,
                                       seed=self.seed)
        self.assertEqual(len(partition), self.num_clients)
        self.assertTrue(all([len(partition) == (
                int(num_shards / self.num_clients) * int(self.num_samples / num_shards))]))

    def test_balance_iid(self):
        partition = CIFAR10Partitioner(self.targets,
                                       self.num_clients,
                                       balance=True,
                                       partition="iid",
                                       seed=self.seed)
        self.assertEqual(len(partition), self.num_clients)
        # check balance
        client_sample_nums = np.array([len(partition[cid]) for cid in range(self.num_clients)])
        self.assertTrue(all(client_sample_nums == int(self.num_samples / self.num_clients)))

    def test_unbalance_iid(self):
        partition = CIFAR10Partitioner(self.targets,
                                       self.num_clients,
                                       balance=False,
                                       partition="iid",
                                       unbalance_sgm=0.3,
                                       seed=self.seed)
        self.assertEqual(len(partition), self.num_clients)
        # check unbalance
        client_sample_nums = [len(partition[cid]) for cid in range(self.num_clients)]
        self.assertTrue(len(set(client_sample_nums)) >= 2)

    def test_balance_dir(self):
        pass

    def test_unbalance_dir(self):
        pass
