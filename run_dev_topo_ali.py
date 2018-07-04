import os
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from architecture.neural_topo_nets import PHConvNet, ConvNetDecoder, SimpleConvNetEncoder, train_PHConvNet, AdversarialTopologicalLearningNets
from architecture.neural_topo_nets import get_loss_discriminator, get_loss_generator
from utils.topo_utils import Provider, train_test_from_dataset, LearningRateScheduler
from utils.topo_utils import ConsoleBatchProgress, PredictionMonitor, load_dgm_data
from utils.utils import export_result
from utils.utils import get_MNIST, weights_init_normal

os.makedirs('images', exist_ok=True)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dgm_data_path', type=str, default=None, help='an absolute path to dgm (h5).')
    parser.add_argument('--raw_data_path', type=str, default=None, help='an absolute path to raw data, e.g. images.')
    parser.add_argument('--output_file', type=str, default="test_deployment.txt", help='filename of results from all experiments.')
    parser.add_argument('--writing_mode', type=bool, default=False, help="False if write a new file unless True")
    parser.add_argument('--num_epochs', type=int, default=3, help='a number of epochs, e.g. number of times running entire of data.')
    parser.add_argument('--momentum', type=float, default=0.7, help='a number indicating momentum.')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--lr_start', type=float, default=0.1, help='a starting value of learning rate (lr) to be tried.')
    parser.add_argument('--lr_step', type=float, default=20, help='a number of incremental steps in learning rate (lr).')
    parser.add_argument('--lr_adaption', type=float, default=0.5, help='a coefficient to jump between values of learning rate (lr).')
    parser.add_argument('--test_ratio', type=float, default=0.5, help='a ratio between test set and training set.')
    parser.add_argument('--batch_size', type=int, default=128, help='a number indicating a batch size.')
    parser.add_argument('--latent_dim', type=int, default=5, help='dimensionality of the latent space')
    parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--num_experiment', type=int, default=5, help='a number of experiments repeated.')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    return parser.parse_args()

opt = parse_arguments()

def main():
    print("[ Start experimenting ]")
    data_loader = get_MNIST(opt)
    data_train, data_test, subscripted_views = load_dgm_data(opt)

    generator = ConvNetDecoder(opt.latent_dim, opt.img_size, opt.channels)
    generator.apply(weights_init_normal)

    discriminator = PHConvNet(subscripted_views)
    discriminator.apply(weights_init_normal)

    encoder = SimpleConvNetEncoder(opt.img_size, opt.latent_dim)
    encoder.apply(weights_init_normal)

    adversarial_loss = nn.BCELoss()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        encoder.cuda()
        adversarial_loss.cuda()

    for i in range(1, opt.num_experiment + 1):
        print('[ Run experiment {} ]'.format(i))
        adversarial_nets = AdversarialTopologicalLearningNets(generator, discriminator, encoder, adversarial_loss, opt)
        adversarial_nets.train(data_loader, opt.num_epochs, opt.lr, opt.b1, opt.b2)  #Adam optimizer
        average_ten_accuracy = adversarial_nets.top_accuracies(k=10, scoring='accuracy')/ 10.0
        export_result(opt.output_file, average_ten_accuracy, opt.writing_mode)

    print("[ End experimenting ]")

if __name__=='__main__':
    # run_experiment(opt)
    main()
