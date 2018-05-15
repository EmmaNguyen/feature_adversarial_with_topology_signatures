import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Topo_Generator(nn.Module):
    def __init__(self, latent_dim, img_size, channels, subscripted_views):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels
        self.subscripted_views = subscripted_views
        self.transform = UpperDiagonalThresholdedLogTransform(0.1)

        def get_init(num_elements):
            transform = UpperDiagonalThresholdedLogTransform(0.1)
            return transform(pers_dgm_center_init(num_elements))

        self.dim_0 = SLayer(150, 2, get_init(150), torch.ones(150, 2) * 3)
        self.dim_0_ess = SLayer(50, 1)
        self.dim_1_ess = SLayer(50, 1)
        self.slayers = [self.dim_0,
                        self.dim_0_ess,
                        self.dim_1_ess
                        ]

        self.stage_1 = []
        stage_1_outs = [75, 25, 25]

        for i, (n_in, n_out) in enumerate(zip([150, 50, 50], stage_1_outs)):
            seq = nn.Sequential()
            seq.add_module('linear_1', nn.Linear(n_in, n_out))
            seq.add_module('batch_norm', nn.BatchNorm1d(n_out))
            seq.add_module('drop_out_1', nn.Dropout(0.1))
            seq.add_module('linear_2', nn.Linear(n_out, n_out))
            seq.add_module('relu', nn.ReLU())
            seq.add_module('drop_out_2', nn.Dropout(0.1))

            self.stage_1.append(seq)
            self.add_module('stage_1_{}'.format(i), seq)

        linear_1 = nn.Sequential()
        linear_1.add_module('linear_1', nn.Linear(sum(stage_1_outs), 200))
        linear_1.add_module('batchnorm_1', torch.nn.BatchNorm1d(200))
        linear_1.add_module('relu_1', nn.ReLU())
        linear_1.add_module('linear_2', nn.Linear(200, 100))
        linear_1.add_module('batchnorm_2', torch.nn.BatchNorm1d(100))
        linear_1.add_module('drop_out_2', torch.nn.Dropout(0.1))
        linear_1.add_module('relu_2', nn.ReLU())
        linear_1.add_module('linear_3', nn.Linear(100, 50))
        linear_1.add_module('batchnorm_3', nn.BatchNorm1d(50))
        linear_1.add_module('relu_3', nn.ReLU())
        linear_1.add_module('linear_4', nn.Linear(50, 5))
        linear_1.add_module('batchnorm_4', nn.BatchNorm1d(5))
        self.linear_1 = linear_1

    def forward(self, noise):
        x = [batch[n] for n in self.subscripted_views]

        x = [
             [self.transform(dgm) for dgm in x[0]],
             [reduce_essential_dgm(dgm) for dgm in x[1]],
             [reduce_essential_dgm(dgm) for dgm in x[2]]
            ]

        x_sl = [l(xx) for l, xx in zip(self.slayers, x)]

        x = [l(xx) for l, xx in zip(self.stage_1, x_sl)]

        x = torch.cat(x, 1)

        x = self.linear_1(x)

        return x

#class Discriminator(nn.Module):
    #def __init__(self, img_size, latent_dim):
        #super(Discriminator, self).__init__()
        #self.img_size = img_size
        #self.latent_dim = latent_dim
        #self.model = nn.Sequential(
            #nn.Linear(self.img_size**2 + self.latent_dim, 512),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Linear(512, 256),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Linear(256, 1),
            #nn.Sigmoid()
        #)
#
    #def forward(self, img, latent_vector):
        #img_flat = img.view(img.size()[0], -1)
        #validity = self.model(torch.cat([img_flat, latent_vector],1))
        #return validity

class Topo_Decoder(nn.Module):
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
