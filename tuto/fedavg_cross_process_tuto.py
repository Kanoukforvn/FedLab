import sys
sys.path.append("../")

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
args.dataname = "mnist"
args.num_classes = 10

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


dataset = fed_mnist.get_dataset(0) # get the 0-th client's dataset
dataloader = fed_mnist.get_dataloader(0, batch_size=128) # get the 0-th client's dataset loader with batch size 128

# client
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer, SGDClientTrainer

# local train configuration
args.epochs = 5
args.batch_size = 128
args.lr = 0.1

trainer = SGDSerialClientTrainer(model, args.total_client, cuda=args.cuda) # serial trainer
# trainer = SGDClientTrainer(model, cuda=True) # single trainer

trainer.setup_dataset(fed_mnist)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

# server
from fedlab.contrib.algorithm.basic_server import SyncServerHandler

# global configuration
args.com_round = 10
args.sample_ratio = 0.1

handler = SyncServerHandler(model=model, global_round=args.com_round, sample_ratio=args.sample_ratio, cuda=args.cuda)
