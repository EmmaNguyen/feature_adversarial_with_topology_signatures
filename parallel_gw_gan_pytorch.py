import torch
import torch.nn
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import scipy as sp
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data
from parallel_geometry_score.gs import geom_score, rlt, rlts
from ot import gromov_wasserstein2, unif

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 32
z_dim = 128
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
cnt = 0
lr = 1e-4
lam1 = 1e-2
lam2 = 1e-2

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

old_err_state = np.seterr(divide='raise')
ignored_states = np.seterr(**old_err_state)

def log(x):
    return torch.log(x + 1e-8)


E = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, z_dim)
)

G = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, X_dim),
    torch.nn.Sigmoid()
)

D = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, 1),
    torch.nn.Sigmoid()
)


def reset_grad():
    G.zero_grad()
    D.zero_grad()
    E.zero_grad()


def sample_X(size, include_y=False):
    X, y = mnist.train.next_batch(size)
    X = Variable(torch.from_numpy(X))

    if include_y:
        y = np.argmax(y, axis=1).astype(np.int)
        y = Variable(torch.from_numpy(y))
        return X, y

    return X


E_solver = optim.Adam(E.parameters(), lr=lr)
G_solver = optim.Adam(G.parameters(), lr=lr)
D_solver = optim.Adam(D.parameters(), lr=lr)

def l2_distance(X, Y):
    return torch.sum((X - Y)**2, 1)

def geometry_score(X, Y):
    if torch.cuda.is_available():
        rlts1 = rlts(X.data.cpu().numpy(), n=mb_size)
        rlts2 = rlts(Y.data.cpu().numpy(), n=mb_size)
    else:
        rlts1 = rlts(X.data.numpy(), n=mb_size)
        rlts2 = rlts(Y.data.numpy(), n=mb_size)
    return Variable(Tensor(geom_score(rlts1, rlts2)),
                    requires_grad=False)

def gromov_wasserstein_distance(X, Y):
    import concurrent.futures
    gw_dist = np.zeros(mb_size)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in executor.map(range(mb_size)):
            C1 = sp.spatial.distance.cdist(X[i,:].reshape(28,28).data.cpu().numpy(), X[i,:].reshape(28,28).data.cpu().numpy()) #Convert data back to an image from one hot encoding with size 28x28
            C2 = sp.spatial.distance.cdist(Y[i,:].reshape(28,28).data.cpu().numpy(), Y[i,:].reshape(28,28).data.cpu().numpy())
            C1 /= C1.max()
            C2 /= C2.max()
            p = unif(28)
            q = unif(28)
            gw_dist[i] = gromov_wasserstein2(C1, C2, p, q, loss_fun='square_loss', epsilon=5e-4) 
    return Variable(Tensor(gw_dist))  

# metric_regularized = l2_distance
# metric_regularized = geometry_score
metric_regularized = gromov_wasserstein_distance

for it in range(1000000):
    """ Discriminator """
    # Sample data
    X = sample_X(mb_size)
    z = Variable(torch.randn(mb_size, z_dim))

    # Dicriminator_1 forward-loss-backward-update
    G_sample = G(z)
    D_real = D(X)
    D_fake = D(G_sample)

    D_loss = -torch.mean(log(D_real) + log(1 - D_fake))

    D_loss.backward()
    D_solver.step()

    # Housekeeping - reset gradient
    reset_grad()

    """ Generator """
    # Sample data
    X = sample_X(mb_size)
    z = Variable(torch.randn(mb_size, z_dim))

    # Generator forward-loss-backward-update
    G_sample = G(z)
    G_sample_reg = G(E(X))
    D_fake = D(G_sample)
    D_reg = D(G_sample_reg)

    # geometry_score_G = metric_regularized(X, G_sample_reg)
    # import pdb; pdb.set_trace()
    
    reg = torch.mean(metric_regularized(X, G_sample_reg))
    G_loss = reg

    G_loss.backward()
    G_solver.step()

    # Housekeeping - reset gradient
    reset_grad()

    """ Encoder """
    # Sample data
    X = sample_X(mb_size)
    z = Variable(torch.randn(mb_size, z_dim))

    G_sample_reg = G(E(X))
    D_reg = D(G_sample_reg)

    E_loss = reg

    E_loss.backward()
    E_solver.step()

    # Housekeeping - reset gradient
    reset_grad()

    # Print and plot every now and then
    if it % 10 == 0:
        print('Iter-{}; D_loss: {}; E_loss: {}; G_loss: {}'
              .format(it, D_loss.data.numpy(), E_loss.data.numpy(), G_loss.data.numpy()))

        samples = G(z).data.numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'
                    .format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)
