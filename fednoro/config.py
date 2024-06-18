import argparse

# Define the command-line arguments
parser = argparse.ArgumentParser(description='Your description here')

# Add arguments
parser.add_argument('--total_client', type=int, default=20, help='Total number of clients')
parser.add_argument('--alpha', type=int, default=2, help='Alpha value')
parser.add_argument('--seed', type=int, default=100, help='Seed value')
parser.add_argument('--preprocess', type=bool, default=True, help='Preprocess flag')
parser.add_argument('--dataname', type=str, default='cifar10', help='Dataset name')
parser.add_argument('--model', type=str, default='Resnet18', help='Model architecture')
parser.add_argument('--pretrained', type=int, default=1, help='Pretrained flag')
parser.add_argument('--num_users', type=int, help='Number of users (defaults to total_client)')
parser.add_argument('--device', type=str, default='cuda', help='Device type')
parser.add_argument('--cuda', type=bool, default=True, help='CUDA flag')
parser.add_argument('--level_n_lowerb', type=float, default=0.5, help='Lower bound of level n')
parser.add_argument('--level_n_upperb', type=float, default=0.7, help='Upper bound of level n')
parser.add_argument('--level_n_system', type=float, default=0.4, help='System level n')
parser.add_argument('--n_type', type=str, default='random', help='Type of n')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
parser.add_argument('--warm_up_round', type=int, default=15, help='Warm-up rounds')
parser.add_argument('--sample_ratio', type=int, default=1, help='Sample ratio')
parser.add_argument('--begin', type=int, default=10, help='Begin value')
parser.add_argument('--end', type=int, default=49, help='End value')
parser.add_argument('--a', type=float, default=0.8, help='A value')
parser.add_argument('--exp', type=str, default='Fed', help='Experiment name')
parser.add_argument('--com_round', type=int, default=100, help='communication rounds (defaults to 100)')
parser.add_argument('--warm', type=int, default=1, help='Warm flag')
parser.add_argument('--n_classes', type=int, default=10, help='Number of classes (defaults to 10)')
parser.add_argument('--noisy_selection', type=bool, default=False, help='True : Activate noisy client selection False : keep classic fednoro strategy')
parser.add_argument('--aggregator', type=str, default="fednoro", help='Number of classes (defaults to 10)')


# Parse the arguments
args = parser.parse_args()

# If num_users is not provided, set it to total_client
if not args.num_users:
    args.num_users = args.total_client

# Convert boolean arguments to bool
args.preprocess = bool(args.preprocess)
args.cuda = bool(args.cuda)
args.warm = bool(args.warm)
