import os
import argparse
import random
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
from torch import nn
import sys
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

sys.path.append("../../")

# Import necessary modules from your codebase
from fednoro.utils.options import args_parser
from fednoro.utils.local_training import LocalUpdate, globaltest
from fednoro.utils.FedAvg import FedAvg, DaAgg
from fednoro.utils.utils import add_noise, set_seed, set_output_files, get_output, get_current_consistency_weight
from fednoro.model.build_model import build_model

# Import necessary modules from FedLab
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.core.standalone import StandalonePipeline
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST

# Parse arguments
parser = argparse.ArgumentParser(description="Standalone training example")

# Add arguments for the script
parser.add_argument("--total_clients", type=int, default=100)
parser.add_argument("--com_round", type=int, default=10)
parser.add_argument("--sample_ratio", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.01)
# system setting
parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=0, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')

# basic setting
parser.add_argument('--exp', type=str,
                    default='Fed', help='experiment name')
parser.add_argument('--dataset', type=str,
                    default='ICH', help='dataset name')
parser.add_argument('--model', type=str,
                    default='Resnet18', help='model name')
parser.add_argument('--base_lr', type=float,  default=3e-4,
                        help='base learning rate')
parser.add_argument('--pretrained', type=int,  default=0)

    # for FL
parser.add_argument('--n_clients', type=int,  default=20,
                        help='number of users') 
parser.add_argument('--iid', type=int, default=0, help="i.i.d. or non-i.i.d.")
parser.add_argument('--non_iid_prob_class', type=float,
                        default=0.9, help='parameter for non-iid')
parser.add_argument('--alpha_dirichlet', type=float,
                        default=2.0, help='parameter for non-iid')
parser.add_argument('--local_ep', type=int, default=5, help='local epoch')
parser.add_argument('--rounds', type=int,  default=100, help='rounds')

parser.add_argument('--s1', type=int,  default=10, help='stage 1 rounds')
parser.add_argument('--begin', type=int,  default=10, help='ramp up begin')
parser.add_argument('--end', type=int,  default=49, help='ramp up end')
parser.add_argument('--a', type=float,  default=0.8, help='a')
parser.add_argument('--warm', type=int,  default=1)

    # noise
parser.add_argument('--level_n_system', type=float, default=0.4, help="fraction of noisy clients")
parser.add_argument('--level_n_lowerb', type=float, default=0.5, help="lower bound of noise level")
parser.add_argument('--level_n_upperb', type=float, default=0.7, help="upper bound of noise level")
parser.add_argument('--n_type', type=str, default="instance", help="type of noise")
parser.add_argument("--n_classes", type=int, default=10)  
# Parse the arguments

args = parser.parse_args()
args.num_users = args.n_clients
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize torch seed
torch.manual_seed(0)

# Initialize FedLab environment
serialization_tool = SerializationTool()

# Define the model
model = build_model(args)  # Your custom model creation function

# Server setup
handler = SyncServerHandler(
    model, args.com_round, args.total_clients, args.sample_ratio
)

# Client setup
trainer = SGDSerialClientTrainer(model, args.total_clients, cuda=torch.cuda.is_available())  # Ensure CUDA availability
dataset = PathologicalMNIST(
    root="../../datasets/mnist/",
    path="../../datasets/mnist/",
    num_clients=args.total_clients,
)
dataset_train, _, _ = dataset  # Assuming you're using a similar dataset structure as before

# Preprocess the dataset (if needed)
dataset.preprocess()

trainer.setup_dataset(dataset_train)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

handler.setup_dataset(dataset_train)

# Main pipeline
pipeline = StandalonePipeline(handler, trainer)
pipeline.main()