from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='MNIST NN Training in PyTorch')

    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='input batch size for testing (default: 1000)')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--num_hiddens', type=int, default=256, help='number of hidden units (default: 256)')

    return parser.parse_args()