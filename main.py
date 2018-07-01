import argparse
import os
import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from generative_model.emma_ali import run_generative_adversarial_network
from architecture.neural_nets import Generator, Discriminator, Decoder
from architecture.neural_nets import train_discriminator, get_loss_generator, get_loss_discriminator
from architecture.neural_nets import get_loss_wasserstein_discriminator, get_loss_wasserstein_generator
from utils.utils import print_progress, get_sample_data, view_generated_data, get_MNIST, weights_init_normal

os.makedirs('images', exist_ok=True)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
    parser.add_argument('--latent_dim', type=int, default=5, help='dimensionality of the latent space')
    parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
    return parser.parse_args()

def run_topological_signature(opt):
        data_loader = get_MNIST(opt)

        discriminator = Discriminator(opt.img_size, opt.latent_dim)
        discriminator.apply(weights_init_normal)
        discriminator_solver = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        if cuda:
            discriminator.cuda()
            adversarial_loss.cuda()

        adversarial_loss = nn.BCELoss()

        for epoch in range(opt.n_epochs):
            for i, (imgs, _) in enumerate(data_loader):

                if cuda: imgs = imgs.type(torch.cuda.FloatTensor)

                discriminator_solver.zero_grad()
                discriminator_loss = adversarial_loss(discriminator(imgs), imgs)
                discriminator_loss.backward()
                discriminator_solver.step()

                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f]" % (epoch, opt.n_epochs, i, len(data_loader),
                                                                discriminator_loss.data.item()))

                batches_done = epoch * len(data_loader) + i

                if batches_done % opt.sample_interval == 0:
                    save_image(fake_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)

if __name__=="__main__":
    opt = parse_arguments()
    run_topological_signature(opt)
