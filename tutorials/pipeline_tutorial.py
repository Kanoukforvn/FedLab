import sys
import os
sys.path.append("../")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

sys.path.append("../../../")


# configuration
from munch import Munch
from fedlab.models.mlp import MLP

model = MLP(784, 10)
args = Munch

args.total_client = 100
args.alpha = 0.5
args.seed = 42
args.preprocess = True
args.cuda = False

# We provide a example usage of patitioned MNIST dataset
# Download raw MNIST dataset and partition them according to given configuration

from torchvision import transforms
from fedlab.contrib.dataset.partitioned_mnist import PartitionedMNIST

fed_mnist = PartitionedMNIST(root="../datasets/mnist/",
                         path="../datasets/mnist/fedmnist/",
                         num_clients=args.total_client,
                         partition="noniid-labeldir",
                         dir_alpha=args.alpha,
                         seed=args.seed,
                         preprocess=args.preprocess,
                         download=True,
                         verbose=True,
                         transform=transforms.Compose(
                             [transforms.ToPILImage(), transforms.ToTensor()]))

dataset = fed_mnist.get_dataset(0) # get the 0-th client's dataset
dataloader = fed_mnist.get_dataloader(0, batch_size=128) # get the 0-th client's dataset loader with batch size 128