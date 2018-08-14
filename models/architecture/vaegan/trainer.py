import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .distributions import rand_circle2d
from ot import gromov_wasserstein2, unif


def rand_projections(embedding_dim, num_samples=50):
    """This fn generates `L` random samples from the latent space's unit sphere.

        Args:
            embedding_dim (int): embedding dimension size
            num_samples (int): number of random projection samples

        Return:
            torch.Tensor
    """
    theta = [w / np.sqrt((w**2).sum()) for w in np.random.normal(size=(num_samples, embedding_dim))]
    theta = np.asarray(theta)
    return torch.from_numpy(theta).type(torch.FloatTensor)


def _sliced_wasserstein_distance(encoded_samples, distribution_samples, num_projections=50, p=2):
    """Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): embedded training tensor samples
            distribution_samples (torch.Tensor): distribution training tensor samples
            num_projections (int): number of projectsion to approximate sliced wasserstein distance
            p (int): power of distance metric

        Return:
            torch.Tensor
    """
    # derive latent space dimension size from random samples drawn from a distribution in it
    embedding_dim = distribution_samples.size(1)
    # generate random projections in latent space
    projections = rand_projections(embedding_dim, num_projections)
    # calculate projection of the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
    # calculate projection of the random distribution samples
    distribution_projections = distribution_samples.matmul(projections.transpose(0, 1))
    # calculate the sliced wasserstein distance by
    # sorting the samples per projection and
    # calculating the difference between the
    # encoded samples and drawn samples per projection
    wasserstein_distance = torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]
    # distance between them (L2 by default for Wasserstein-2)
    wasserstein_distance_p = torch.pow(wasserstein_distance, p)
    # approximate wasserstein_distance for each projection
    return wasserstein_distance_p.mean()


def sliced_wasserstein_distance(encoded_samples, distribution_fn=rand_circle2d, num_projections=50, p=2):
    """Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): embedded training tensor samples
            distribution_fn (callable): callable to draw random samples
            num_projections (int): number of projectsion to approximate sliced wasserstein distance
            p (int): power of distance metric

        Return:
            torch.Tensor
    """
    # derive batch size from encoded samples
    batch_size = encoded_samples.size(0)
    # draw samples from latent space prior distribution
    z = distribution_fn(batch_size)
    # approximate wasserstein_distance between encoded and prior distributions
    # for average over each projection
    swd = _sliced_wasserstein_distance(encoded_samples, z, num_projections, p)
    return swd

def _topology_persistence(encoded_samples, distribution_samples, num_projections=50, p=2):
    prior_subcripted_views = distribution_samples
    posterior_subscripted_views = encoded_samples

    adversarial_learner = AdversariallearnerBatchTrainer()
    adversarial_learner.train_on_batch(prior_subcripted_views)

    posterior_pred = adversarial_learner.eval_on_batch(posterior_subscripted_views)
    bce = F.binary_cross_entropy(posterior_pred)

    # derive latent space dimension size from random samples drawn from a distribution in it
    embedding_dim = distribution_samples.size(1)
    # generate random projections in latent space
    projections = rand_projections(embedding_dim, num_projections)
    # calculate projection of the encoded samples
    #import pdb; pdb.set_trace()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1).cuda())
    # calculate projection of the random distribution samples
    distribution_projections = distribution_samples.matmul(projections.transpose(0, 1))
    # calculate the sliced wasserstein distance by
    # sorting the samples per projection and
    # calculating the difference between the
    # encoded samples and drawn samples per projection
    wasserstein_distance = torch.sort(encoded_projections.transpose(0, 1).cuda(), dim=1)[0] - torch.sort(distribution_projections.transpose(0, 1).cuda(), dim=1)[0]
    # distance between them (L2 by default for Wasserstein-2)
    wasserstein_distance_p = torch.pow(wasserstein_distance, p)
    # approximate wasserstein_distance for each projection
    return wasserstein_distance_p.mean()

def topology_persistence(encoded_samples, distribution_fn=rand_cirlce2d, num_projections=50, p=2):
    batch_size = encoded_samples.size(0)
    z = distribution_fn(batch_size)
    return _topology_persistence(encoded_samples, self._distribution_fn, self.num_projections_, self.p_)


def gromov_wasserstein_distance(X, Y, device):
    import concurrent.futures
    # import pdb; pdb.set_trace()
    mb_size = X.size(0)
    gw_dist = np.zeros(mb_size)
    Tensor = torch.FloatTensor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in executor.map(range(mb_size)):
            C1 = sp.spatial.distance.cdist(X[i,:].reshape(28,28).data.cpu().numpy(), X[i,:].reshape(28,28).data.cpu().numpy()) #Convert data back to an image from one hot encoding with size 28x28
            C2 = sp.spatial.distance.cdist(Y[i,:].reshape(28,28).data.cpu().numpy(), Y[i,:].reshape(28,28).data.cpu().numpy())
            C1 /= C1.max()
            C2 /= C2.max()
            p = unif(28)
            q = unif(28)
            gw_dist[i] = gromov_wasserstein2(C1, C2, p, q, loss_fun='square_loss', epsilon=5e-4)
    print("*"*100)
    return Variable(Tensor(gw_dist), requires_grad=True).sum()

class SWAEBatchTrainer:
    """Sliced Wasserstein Autoencoder Batch Trainer.

        Args:
            autoencoder (torch.nn.Module): module which implements autoencoder framework
            optimizer (torch.optim.Optimizer): torch optimizer
            distribution_fn (callable): callable to draw random samples
            num_projections (int): number of projectsion to approximate sliced wasserstein distance
            p (int): power of distance metric
            weight_swd (float): weight of divergence metric compared to reconstruction in loss
            device (torch.Device): torch device
    """
    def __init__(self, autoencoder, optimizer, distribution_fn,
                 num_projections=50, p=2, weight_swd=10.0, device=None):
        self.model_ = autoencoder
        self.optimizer = optimizer
        self._distribution_fn = distribution_fn
        self.embedding_dim_ = self.model_ .encoder.embedding_dim_
        self.num_projections_ = num_projections
        self.p_ = p
        self.weight_swd = weight_swd
        self._device = device if device else torch.device('cpu')

    def __call__(self, x):
        return self.eval_on_batch(x)

    def train_on_batch(self, x):
        # reset gradients
        self.optimizer.zero_grad()
        # autoencoder forward pass and loss
        evals = self.eval_on_batch(x)
        # backpropagate loss
        evals['loss'].backward()
        # update encoder and decoder parameters
        self.optimizer.step()
        return evals

    def test_on_batch(self, x):
        # reset gradients
        self.optimizer.zero_grad()
        # autoencoder forward pass and loss
        evals = self.eval_on_batch(x)
        return evals

    def eval_on_batch(self, x):
        x = x.to(self._device)
        recon_x, z = self.model_(x)
        # Equation 4 - this works for 1D
        bce = F.binary_cross_entropy(recon_x, x)
        gw = gromov_wasserstein_distance(recon_x, x, self._device)
        # Equation 15, this is only works for 2D
        w2 = float(self.weight_swd) * sliced_wasserstein_distance(z, self._distribution_fn, self.num_projections_, self.p_)
        # Equation 16: but why there is a bce. Following the original implementation with Keras
        # it is said that (bce and l1) is the first term for equation 16, and w2 for the second term.
        loss = bce + gw + w2
        return {'loss': loss, 'bce': bce, 'gw': gw, 'w2': w2, 'encode': z, 'decode': recon_x}
