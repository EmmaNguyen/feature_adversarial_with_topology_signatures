import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Generator(nn.Module):
    def __init__(self, latent_dim, img_size, channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
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
            nn.Linear(1024, self.img_size**2),
            nn.Tanh()
        )

    def forward(self, noise):
        # import pdb;
        img = self.model(noise)
        # pdb.set_trace(im)
        img = img.view(img.size()[0], self.channels, self.img_size, self.img_size)

        return img

class Generator(nn.Module):
    def __init__(self, latent_dim, img_size, channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
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
            nn.Linear(1024, self.img_size**2),
            nn.Tanh()
        )

    def forward(self, noise):
        # import pdb;
        img = self.model(noise)
        # pdb.set_trace(im)
        img = img.view(img.size()[0], self.channels, self.img_size, self.img_size)

        return img

class Discriminator(nn.Module):
    def __init__(self, img_size, latent_dim):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(self.img_size**2 + self.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, latent_vector):
        img_flat = img.view(img.size()[0], -1)
        validity = self.model(torch.cat([img_flat, latent_vector],1))
        return validity

class Decoder(nn.Module):
    def __init__(self, img_size, latent_dim):
        super(Decoder, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(self.img_size**2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self.latent_dim),
            nn.Sigmoid()
        )
    def forward(self, img):
        # import pdb; pdb.set_trace()
        img_flat = img.view(img.size()[0], -1)
        validity = self.model(img_flat) #64x784
        return validity

def train_discriminator(discriminator, imgs, latent_vector):
    # imgs = imgs.view(imgs.size()[0], -1)
    # vector = torch.cat([imgs, latent_vector], 1)
    # return discriminator(vector)
    return discriminator(imgs, latent_vector)

def get_loss_discriminator(discriminator, fake_imgs, z, real_imgs, fake_z):
    adversarial_loss = nn.BCELoss()
    # minibatch_size = discriminator_real.size()[0]
    minibatch_size = real_imgs.size()[0]
    valid = Variable(Tensor(minibatch_size, 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(minibatch_size, 1).fill_(0.0), requires_grad=False)
    real_loss = adversarial_loss(train_discriminator(discriminator, real_imgs, fake_z), valid)
    fake_loss = adversarial_loss(train_discriminator(discriminator, fake_imgs.detach(), z), fake)
    return (real_loss + fake_loss) / 2

def get_loss_generator(discriminator, fake_imgs, z, real_imgs, fake_z):
    objection = nn.BCELoss()
    minibatch_size = fake_imgs.size()[0]
    # minibatch_size = self.batch_size
    valid = Variable(Tensor(minibatch_size, 1).fill_(1.0), requires_grad=False)
    valid_prediction = train_discriminator(discriminator, fake_imgs, z)
    # import pdb; pdb.set_trace()
    return objection(valid_prediction, valid)

def get_loss_wasserstein_discriminator(discriminator, fake_imgs, z, real_imgs, fake_z):
    real_validity = discriminator(real_imgs, fake_z)
    fake_validity = discriminator(fake_imgs, z)
    return real_validity - fake_validity

def get_loss_wasserstein_generator(discriminator, fake_imgs, z, real_imgs, fake_z):
    return torch.mean(discriminator(fake_imgs, z))
