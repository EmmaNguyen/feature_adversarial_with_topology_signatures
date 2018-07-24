from __future__ import absolute_import
from __future__ import print_function
from torch_geometry_score.gs.utils import relative, witness
import torch

def rlt(X, L_0=64, gamma=None, i_max=100):
    """
      This function implements Algorithm 1 for one sample of landmarks.

    Args:
      X: np.array representing the dataset.
      L_0: number of landmarks to sample.
      gamma: float, parameter determining the maximum persistence value.
      i_max: int, upper bound on the value of beta_1 to compute.

    Returns
      An array of size (i_max, ) containing RLT(i, 1, X, L)
      for randomly sampled landmarks.
    """
    N = X.shape[0]
    if gamma is None:
        gamma = 1.0 / 128 * N / 5000
    I_1, alpha_max = witness(X, L_0=L_0, gamma=gamma)
    res = relative(I_1, alpha_max, i_max=i_max)
    return res


def rlts(X, L_0=64, gamma=None, i_max=100, n=1000):
    """
      This function implements Algorithm 1.

    Args:
      X: np.array representing the dataset.n
      L_0: number of landmarks to sample.
      gamma: float, parameter determining the maximum persistence value.
      i_max: int, upper bound on the value of beta_1 to compute.
      n: int, number of samples
    Returns
      An array of size (n, i_max) containing RLT(i, 1, X, L)
      for n collections of randomly sampled landmarks.
    """
    rlts = torch.zeros((n, i_max))
    for i in range(n):
        rlts[i, :] = rlt(X, L_0, gamma, i_max)
    # import pdb; pdb.set_trace()
    return rlts


def geom_score(rlts1, rlts2):
    """
      This function implements Algorithm 2.

    Args:
       rlts1 and rlts2: arrays as returned by the function "rlts".
    Returns
       Float, a number representing topological similarity of two datasets.

    """
    #mrlt1 = torch.mean(rlts1, dim=0)
    #mrlt2 = torch.mean(rlts2, dim=0)
    #import pdb; pdb.set_trace()
    #return torch.sum((mrlt1 - mrlt2) ** 2)
    return torch.sum((rlts1 - rlts2) ** 2, dim =0)
