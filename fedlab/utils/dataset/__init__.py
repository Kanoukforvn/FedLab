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

from .partition import DataPartitioner, BasicPartitioner, VisionPartitioner
from .partition import CIFAR10Partitioner, CIFAR100Partitioner, FMNISTPartitioner, MNISTPartitioner, \
    SVHNPartitioner
from .partition import FCUBEPartitioner
from .partition import AdultPartitioner, RCV1Partitioner, CovtypePartitioner
from .partition import ISICPartitioner
