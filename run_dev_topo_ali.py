import os
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from architecture.neural_topo_nets import PHConvNet, ConvNet, SimpleConvNet
from architecture.neural_topo_nets import get_loss_discriminator, get_loss_generator
from utils.topo_utils import Provider, train_test_from_dataset, Trainer, LearningRateScheduler, train_PHConvNet
from utils.topo_utils import ConsoleBatchProgress, PredictionMonitor
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

def load_data(opt):
    view_name_template = 'dim_0_dir_{}'
    subscripted_views = sorted([view_name_template.format(i) for i in range(32)])
    assert (str(len(subscripted_views)) in opt.dgm_data_path)
    print("[ Load provider data ]")
    dataset = Provider()
    dataset.read_from_h5(opt.dgm_data_path)
    assert all(view_name in dataset.view_names for view_name in subscripted_views)
    print("[ Create data loader]")
    data_train, data_test = train_test_from_dataset(dataset,
                                                    test_size=opt.test_ratio,
                                                    batch_size=opt.batch_size)
    return data_train, data_test, subscripted_views

def run_experiment(opt):
    print("[ Start experimenting ]")
    data_train, data_test, subscripted_views = load_data(opt)

    for i in range(1, opt.num_experiment + 1):
        print('[ Run experiment {} ]'.format(i))
        topo_discriminator = PHConvNet(subscripted_views)   #subscripted_views is a number of directions to reconstruct a image
        train_topo_discriminator = train_PHConvNet(topo_discriminator, opt, data_train, data_test)
        train_topo_discriminator.run()
        last_ten_accuracies = list(train_topo_discriminator.prediction_monitor.accuracies.values())[-10:]
        average_accuracy = sum(last_ten_accuracies)/ 10.0
        export_result(opt.output_file, average_accuracy, opt.writing_mode)
    print("[ End experimenting ]")

def run_adversarial_learning_topo_features(opt):
        data_loader = get_MNIST(opt)
        data_train, data_test, subscripted_views = load_data(opt)

        generator = ConvNet(opt.latent_dim, opt.img_size, opt.channels)
        generator.apply(weights_init_normal)
        generator_solver = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        discriminator = PHConvNet(subscripted_views)
        discriminator.apply(weights_init_normal)
        discriminator_solver = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        decoder = SimpleConvNet(opt.img_size, opt.latent_dim)
        decoder.apply(weights_init_normal)
        decoder_solver = torch.optim.Adam(decoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


        if cuda:
            generator.cuda()
            discriminator.cuda()
            adversarial_loss.cuda()

        adversarial_loss = nn.BCELoss()

        for epoch in range(opt.num_epochs):
            for i, (imgs, _) in enumerate(data_loader):

                if cuda: imgs = imgs.type(torch.cuda.FloatTensor)

                valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

                real_imgs = Variable(imgs)

                generator_solver.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

                # Generate a batch of images
                fake_imgs = generator(z)
                fake_z = decoder(real_imgs)

                generator_loss = get_loss_generator(discriminator, fake_imgs, z, real_imgs, fake_z)
                generator_loss.backward()
                generator_solver.step()

                discriminator_solver.zero_grad()
                discriminator_loss = get_loss_discriminator(discriminator, fake_imgs, z, real_imgs, fake_z)
                discriminator_loss.backward()
                discriminator_solver.step()

                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(data_loader),
                                                                discriminator_loss.data.item(), generator_loss.data.item()))

                batches_done = epoch * len(data_loader) + i

                if batches_done % opt.sample_interval == 0:
                    save_image(fake_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)

if __name__=='__main__':
    opt = parse_arguments()
    # run_experiment(opt)
    run_adversarial_learning_topo_features(opt)
