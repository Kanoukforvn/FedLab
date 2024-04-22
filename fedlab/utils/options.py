import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # system setting
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

    # basic setting
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')
    parser.add_argument('--model', type=str, default='Resnet18', help='model name')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float, default=0.1, help='base learning rate')
    parser.add_argument('--pretrained', type=int, default=1)

    # for FL
    parser.add_argument('--total_client', type=int, default=20, help='number of users') 
    parser.add_argument('--alpha', type=float, default=0.5, help='parameter for non-iid')
    parser.add_argument('--preprocess', type=int, default=1, help='preprocess data')
    parser.add_argument('--cuda', type=int, default=1, help='use CUDA')
    parser.add_argument('--dataname', type=str, default='cifar10_Resnet18', help='dataset name + model')

    # training settings
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')

    # server settings
    parser.add_argument('--com_round', type=int, default=10, help='number of communication rounds')
    parser.add_argument('--sample_ratio', type=float, default=0.1, help='ratio of clients sampled per round')

    args = parser.parse_args()
    return args
