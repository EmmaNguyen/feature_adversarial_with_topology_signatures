"""
Author: Emma Nguyen

Inspired by a source code: https://github.com/eifuentes/swae-pytorch

Usage:

$python main.py

"""

import argparse

from experiments.vaegan import simple_demo

def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch Implementation")
    parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                        help='input batch size for training (default: 500)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--alpha', type=float, default=0.9, metavar='A',
                        help='RMSprop alpha/rho (default: 0.9)')
    parser.add_argument('--distribution', type=str, default='circle', metavar='DIST',
                        help='Latent Distribution (default: circle)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                        help='number of dataloader workers if device is CPU (default: 8)')
    parser.add_argument('--seed', type=int, default=7, metavar='S',
                        help='random seed (default: 7)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='number of batches to log training status (default: 10)')

    return parser.parse_args()

if __name__=="__main__":
    opt = parse_arguments()
    simple_demo(opt)
