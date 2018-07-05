import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets

from utils.topo_utils import UpperDiagonalThresholdedLogTransform, pers_dgm_center_init
from utils.persistent_homology_transform import SLayer, SLayerPHT, reduce_essential_dgm

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class PHConvNet(torch.nn.Module):
    def __init__(self, subscripted_views):
        super(PHConvNet, self).__init__()
        self.subscripted_views = subscripted_views
        n_elements = 75
        n_filters = 32
        stage_2_out = 15
        n_neighbor_directions = 1
        output_size = 10
        self.transform = UpperDiagonalThresholdedLogTransform(0.1)

        # Stacking
        self.pht_sl = SLayerPHT(len(subscripted_views),n_elements,2,n_neighbor_directions=n_neighbor_directions,
                                center_init=self.transform(pers_dgm_center_init(n_elements)),
                                sharpness_init=torch.ones(n_elements, 2) * 4)
        self.stage_1 = []
        for i in range(len(subscripted_views)):
            seq = nn.Sequential()
            seq.add_module('conv_1', nn.Conv1d(1 + 2 * n_neighbor_directions, n_filters, 1, bias=False))
            seq.add_module('conv_2', nn.Conv1d(n_filters, 8, 1, bias=False))
            self.stage_1.append(seq)
            self.add_module('stage_1_{}'.format(i), seq)

        self.stage_2 = []
        for i in range(len(subscripted_views)):
            seq = nn.Sequential()
            seq.add_module('linear_1', nn.Linear(n_elements, stage_2_out))
            seq.add_module('batch_norm', nn.BatchNorm1d(stage_2_out))
            seq.add_module('linear_2', nn.Linear(stage_2_out, stage_2_out))
            seq.add_module('relu', nn.ReLU())
            seq.add_module('Dropout', nn.Dropout(0.4))
            self.stage_2.append(seq)
            self.add_module('stage_2_{}'.format(i), seq)

        linear_1 = nn.Sequential()
        linear_1.add_module('linear', nn.Linear(len(subscripted_views) * stage_2_out, 50))
        linear_1.add_module('batchnorm', torch.nn.BatchNorm1d(50))
        linear_1.add_module('drop_out', torch.nn.Dropout(0.3))
        self.linear_1 = linear_1
        linear_2 = nn.Sequential()
        linear_2.add_module('linear', nn.Linear(50, output_size))
        self.linear_2 = linear_2

    def forward(self, batch):
        x = [batch[n] for n in self.subscripted_views]
        x = [[self.transform(dgm) for dgm in view_batch] for view_batch in x]
        x = self.pht_sl(x)
        x = [l(xx) for l, xx in zip(self.stage_1, x)]
        x = [torch.squeeze(torch.max(xx, 1)[0]) for xx in x]
        x = [l(xx) for l, xx in zip(self.stage_2, x)]
        x = torch.cat(x, 1)
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x

class ConvNetDecoder(nn.Module):
    def __init__(self, latent_dim, img_size, channels):
        super(ConvNetDecoder, self).__init__()
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

class SimpleConvNetEncoder(nn.Module):
    def __init__(self, img_size, latent_dim):
        super(SimpleConvNetEncoder, self).__init__()
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


class AdversarialTopologicalLearningNets(object):
    def __init__(self, generator, discriminator, encoder, latent_dim, adversarial_loss):
        self.discriminator = discriminator
        self.generator = generator
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.adversarial_loss = adversarial_loss

    def set_momentum_solver(self, lr, b1, b2):
        self.generator_solver = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        self.discriminator_solver = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        self.encoder_solver = torch.optim.Adam(self.encoder.parameters(), lr=lr, betas=(b1, b2))

    def train(self, data_loader, num_epochs):
        for epoch in range(num_epochs):
            for i, (imgs, _) in enumerate(data_loader):
                if cuda: imgs = imgs.type(torch.cuda.FloatTensor)

                valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

                real_imgs = Variable(imgs)

                self.generator_solver.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))

                # Generate a batch of images
                fake_imgs = self.generator(z)
                fake_z = self.encoder(real_imgs)

                generator_loss = get_loss_generator(fake_imgs, z, real_imgs, fake_z)
                generator_loss.backward()
                self.generator_solver.step()

                self.discriminator_solver.zero_grad()
                discriminator_loss = get_loss_discriminator(fake_imgs, z, real_imgs, fake_z)
                self.discriminator_loss.backward()
                self.discriminator_solver.step()

                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, num_epochs, i, len(data_loader),
                                                                discriminator_loss.data.item(), generator_loss.data.item()))

                batches_done = epoch * len(data_loader) + i

                if batches_done % opt.sample_interval == 0:
                    save_image(fake_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)

        def train_PHConvNet(self, img, z):
            return Trainer(model=self.discriminator,
                              optimizer=self.discriminator_solver,
                              loss=self.adversarial_loss,
                              train_data=torch.cat((img, z), 1),
                              n_epochs=opt.num_epochs,
                              cuda=True,
                              variable_created_by_model=True)

        def get_loss_discriminator(self, fake_imgs, z, real_imgs, fake_z):
            minibatch_size = real_imgs.size()[0]
            valid = Variable(Tensor(minibatch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(minibatch_size, 1).fill_(0.0), requires_grad=False)
            real_loss = self.adversarial_loss(train_PHConvNet(real_imgs, fake_z), valid)
            fake_loss = self.adversarial_loss(train_PHConvNet(fake_imgs.detach(), z), fake)
            return (real_loss + fake_loss) / 2

        def get_loss_generator(self, fake_imgs, z, real_imgs, fake_z):
            minibatch_size = fake_imgs.size()[0]
            valid = Variable(Tensor(minibatch_size, 1).fill_(1.0), requires_grad=False)
            valid_prediction = train_PHConvNet(fake_imgs, z)
            return self.adversarial_loss(valid_prediction, valid)
