import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

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
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
    parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
    parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
    return parser.parse_args()

opt = parse_arguments()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, opt.img_size**2),
            nn.Tanh()
        )

    def forward(self, noise):
        # import pdb;
        img = self.model(noise)
        # pdb.set_trace(im)dd
        img = img.view(img.size()[0], opt.channels, opt.img_size, opt.img_size)

        return img

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.img_size**2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size()[0], -1)
        validity = self.model(img_flat)
        return validity

def get_MNIST(opt):
    os.makedirs('../../data/mnist', exist_ok=True)
    return torch.utils.data.DataLoader(
        datasets.MNIST('../../data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=opt.batch_size, shuffle=True)

def get_sample_data(generator, imgs, latent_dim):
    return generator(Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))), Variable(imgs)

def train_discriminator(discriminator, fake_data, real_data=None):
    if real_data is not None:
        return discriminator(fake_data), discriminator(real_data)
    else:
        return discriminator(fake_data)

def get_loss_discriminator(discriminator, fake_imgs, real_imgs):
    adversarial_loss = nn.BCELoss()
    # minibatch_size = discriminator_real.size()[0]
    minibatch_size = real_imgs.size()[0]
    valid = Variable(Tensor(minibatch_size, 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(minibatch_size, 1).fill_(0.0), requires_grad=False)
    real_loss = adversarial_loss(discriminator(real_imgs), valid)
    fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake)
    return (real_loss + fake_loss) / 2
def get_loss_generator(discriminator_fake):
    objection = nn.BCELoss()
    minibatch_size = discriminator_fake.size()[0]
    # minibatch_size = opt.batch_size
    valid = Variable(Tensor(minibatch_size, 1).fill_(1.0), requires_grad=False)
    return objection(discriminator_fake, valid)

def print_progress(discriminator_loss, generator_loss, iteration, periodic_iteration=10000):
    if iteration % periodic_iteration == 0:
            print("Iteration: {0}; discriminator_loss: {1}; generator_loss: {2}".\
                format(iteration, \
                discriminator_loss.data.numpy(),
                generator_loss.data.numpy()))

def view_generated_data(generator, iteration, num_picture=16, periodic_iteration=10000):
    if iteration % 1000 == 0:
        sample_data = generator.data.numpy()[:num_picture]
        fig = plt.figure(figsize=(4, 4))
        grid = gridspec.GridSpec(4, 4)
        grid.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(sample_data):
            ax = plt.subplot(grid[i])
            plt.axis('off')

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')

            plt.imshow(sample.reshape(28,28), cmap='Greys_r')

        if not os.path.exists('out_emma/'):
            os.makedirs('out_emma/')

        plt.savefig("out_emma/{}.png".format(str(iteration).zfill(3)), bbox_inches='tight')
        plt.close(fig)

def run_generative_adversarial_network(opt):
    dataloader = get_MNIST(opt)

    generator = Generator(opt)
    generator.apply(weights_init_normal)
    generator_solver = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    discriminator = Discriminator(opt)
    discriminator.apply(weights_init_normal)
    discriminator_solver = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    adversarial_loss = nn.BCELoss()

    batches_done = 0
    for epoch in range(opt.n_epochs):

        # Batch iterator
        data_iter = iter(dataloader)

        for i in range(len(data_iter) // opt.n_critic):
            # Train discriminator for n_critic times
            for _ in range(opt.n_critic):
                (imgs, _) = data_iter.next()

                # Adversarial ground truths
                valid = Variable(Tensor(imgs.shape[0], 1).fill_(-1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)

                if cuda:
                    imgs = imgs.type(torch.cuda.FloatTensor)

                real_imgs = Variable(imgs)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                discriminator_solver.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

                # Generate a batch of images
                fake_imgs = generator(z)


                # Train on real images
                real_validity = discriminator(real_imgs)
                real_validity.backward(valid)
                # Train on fake images
                fake_validity = discriminator(fake_imgs)
                fake_validity.backward(fake)

                d_loss = real_validity - fake_validity

                discriminator_solver.step()

                # Clip weights of discriminator
                for p in discriminator.parameters():
                    p.data.clamp_(-opt.clip_value, opt.clip_value)

            # -----------------
            #  Train Generator
            # -----------------

            generator_solver.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            gen_validity = discriminator(fake_imgs)
            gen_validity.backward(valid)

            generator_solver.step()

            # import pdb; pdb.set_trace()
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs,
                                                            batches_done % len(dataloader), len(dataloader),
                                                            d_loss.data[0][0], gen_validity.data[0][0]))

            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)
            batches_done += 1

if __name__ == "__main__":
    run_generative_adversarial_network(opt)
