import sys
sys.path.append("../")

import logging
logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', 
                        datefmt='%H:%M:%S',
                        stream=sys.stdout)

import torchvision
import copy
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

# configuration
from munch import Munch
import matplotlib.pyplot as plt
from fedlab.models.mlp import MLP, MLR
from fedlab.models.cnn import CNNMnistPns, CNNCifarPns
from fedlab.utils.dataset.functional import partition_report

#model = MLP(784, 10)
model = MLR(784, 10)
#model = CNNMnistPns()
args = Munch

args.total_client = 10
args.alpha = 0.1
args.seed = 0
args.preprocess = True
args.cuda = True
args.dataname = "mnist"
args.num_classes = 10
args.device = "cuda"
args.noniid_percentage = 0.5

# We provide a example usage of patitioned MNIST dataset
# Download raw MNIST dataset and partition them according to given configuration

from torchvision import transforms
from fedlab.contrib.dataset.partitioned_mnist import PartitionedMNIST
import pandas as pd
from collections import Counter

trainset = torchvision.datasets.MNIST(root="../../../../data/MNIST/", train=True, download=True)

fed_mnist = PartitionedMNIST(
    root="../datasets/mnist/",
    path="../datasets/mnist/fedmnist/",
    num_clients=args.total_client,
    partition="noniid-labeldir",
    dir_alpha=args.alpha,
    seed=args.seed,
    preprocess=args.preprocess,
    download=True,
    verbose=True,
    #noniid_percentage=args.noniid_percentage, 
    transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
)
dataset = fed_mnist.get_dataset(0) # get the 0-th client's dataset
dataloader = fed_mnist.get_dataloader(0, batch_size=16) # get the 0-th client's dataset loader with batch size 128

logging.info(
    f"train: {Counter(fed_mnist.targets_train)}, total: {len(fed_mnist.targets_train)}")
logging.info(
    f"test: {Counter(fed_mnist.targets_test)}, total: {len(fed_mnist.targets_test)}")

# generate partition report
csv_file = f"./partition-reports/{args.dataname}_hetero_dir_{args.alpha}_{args.total_client}clients.csv"
partition_report(fed_mnist.targets, fed_mnist.client_dict, 
                 class_num=args.num_classes, 
                 verbose=False, file=csv_file)

hetero_dir_part_df = pd.read_csv(csv_file,header=0)
hetero_dir_part_df = hetero_dir_part_df.set_index('cid')
col_names = [f"class-{i}" for i in range(args.num_classes)]
for col in col_names:
    hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['TotalAmount']).astype(int)

#select first 10 clients for bar plot
hetero_dir_part_df[col_names].iloc[:10].plot.barh(stacked=True)  
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('sample num')
plt.savefig(f"./imgs/{args.dataname}_dir_alpha_{args.alpha}_{args.total_client}_clients.png", dpi=400, bbox_inches = 'tight')
plt.show()



# client
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer, SGDClientTrainer
from fedlab.contrib.algorithm.fedmdcs import FedMDCSSerialClientTrainer

# local train configuration
args.epochs = 5
args.batch_size = 16
args.lr = 0.01

trainer = FedMDCSSerialClientTrainer(model, args.total_client, cuda=args.cuda) # serial trainer
#trainer = SGDSerialClientTrainer(model, args.total_client, cuda=args.cuda) # serial trainer
#trainer = SGDClientTrainer(model, cuda=True) # single trainer

trainer.setup_dataset(fed_mnist)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

# server
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.fedmdcs import FedMDCSServerHandler

# global configuration
args.com_round = 200
args.sample_ratio = 0.5
args.top_n_clients = int((args.sample_ratio/2)*args.total_client) 

handler = FedMDCSServerHandler(model=model, global_round=args.com_round, sample_ratio=1, cuda=args.cuda, num_clients=args.total_client, 
                            top_n_clients=args.top_n_clients)
#handler = SyncServerHandler(model=model, global_round=args.com_round, sample_ratio=args.sample_ratio, cuda=args.cuda, num_clients=args.total_client)

import matplotlib.pyplot as plt
import numpy as np

from fedlab.utils.functional import evaluate, globaltest
from fedlab.core.standalone import StandalonePipeline

from torch import nn
from torch.utils.data import DataLoader

class EvalPipeline(StandalonePipeline):
    def __init__(self, handler, trainer, test_loader):
        super().__init__(handler, trainer)
        self.test_loader = test_loader 
        self.loss = []
        self.acc = []
        self.bacc = []
        self.best_performance = 0
        self.best_balanced_accuracy = 0
        
    def main(self):
        t=0
        while self.handler.if_stop is False:
            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package
            
            # client side
            self.trainer.local_process(broadcast, sampled_clients)
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load(pack)

            loss = evaluate(self.handler.model, nn.CrossEntropyLoss(), self.test_loader)

            pred = globaltest(copy.deepcopy(self.handler.model).to(
                args.device), test_data, args)
            acc = accuracy_score(test_data.targets, pred)
            bacc = balanced_accuracy_score(test_data.targets, pred)

            logging.info("Round : {}, Loss {:.4f}, Balanced Accuracy {:.4f}".format(t,loss, bacc))

            if bacc > self.best_balanced_accuracy:
                self.best_round_number = t
                self.best_balanced_accuracy = bacc
                logging.info(f'Best balanced accuracy: {self.best_balanced_accuracy:.4f}')

            if acc > self.best_performance:
                self.best_performance = acc
            
            self.loss.append(loss)
            self.acc.append(acc)
            self.bacc.append(bacc)
            t += 1
        
        logging.info('Final best balanced accuracy: {:.4f}, Best model number : {} '.format(self.best_balanced_accuracy, self.best_round_number))
        logging.info(self.bacc)
            
    def show(self):
        plt.figure(figsize=(8,4.5))
        ax = plt.subplot(1,2,1)
        ax.plot(np.arange(len(self.loss)), self.loss)
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Loss")
        
        ax2 = plt.subplot(1,2,2)
        ax2.plot(np.arange(len(self.acc)), self.acc)
        ax2.set_xlabel("Communication Round")
        ax2.set_ylabel("Accuarcy")
        
        plt.savefig(f"./imgs/mdcs{args.dataname}_dir_alpha_{args.alpha}_loss_accuracy.png", dpi=400, bbox_inches = 'tight')

    def show_b(self):
        plt.figure(figsize=(8,4.5))
        ax = plt.subplot(1,2,1)
        ax.plot(np.arange(len(self.loss)), self.loss)
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Loss")
        
        ax2 = plt.subplot(1,2,2)
        ax2.plot(np.arange(len(self.bacc)), self.bacc)
        ax2.set_xlabel("Communication Round")
        ax2.set_ylabel("Balanced Accuarcy")
        
        plt.savefig(f"./imgs/mdcs_{args.dataname}_dir_alpha_{args.alpha}_loss_balanced_accuracy.png", dpi=400, bbox_inches = 'tight')
   
        
test_data = torchvision.datasets.MNIST(root="../datasets/mnist/",
                                       train=False,
                                       transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=32)

standalone_eval = EvalPipeline(handler=handler, trainer=trainer, test_loader=test_loader)
standalone_eval.main()
standalone_eval.show()
standalone_eval.show_b()
