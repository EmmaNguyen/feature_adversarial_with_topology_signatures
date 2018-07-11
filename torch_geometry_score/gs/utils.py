import numpy as torch
import torch
from scipy.spatial.distance import cdist, pdist, squareform
import gudhi
import matplotlib.pyplot as plt


def relative(I_1, alpha_max, i_max=100):
    """
      For a collection of intervals I_1 this functions computes
      RLT by formulas (2) and (3). This function will be typically called
      on the output of the gudhi persistence_intervals_in_dimension function.

    Args:
      I_1: list of intervals e.g. [[0, 1], [0, 2], [0, torch.inf]].
      alpha_max: float, the maximal persistence value
      i_max: int, upper bound on the value of beta_1 to compute.

    Returns
      An array of size (i_max, ) containing desired RLT.
    """

    persistence_intervals = []
    # If for some interval we have that it persisted up to torch.inf
    # we replace this point with alpha_max.
    # import pdb; pdb.set_trace()
    def is_inf(interval):
        return True if interval is 'inf' else False

    for interval in I_1:
        if not is_inf(interval[1]):
            persistence_intervals.append(list(interval))
        else:
            persistence_intervals.append([interval[0], alpha_max])

    # If there are no intervals in H1 then we always observed 0 holes.
    if len(persistence_intervals) == 0:
        rlt = torch.zeros(i_max)
        rlt[0] = 1.0
        return rlt

    persistence_intervals_ext = persistence_intervals + [[0, alpha_max]]
    persistence_intervals_ext = torch.tensor(persistence_intervals_ext)
    persistence_intervals = torch.tensor(persistence_intervals)

    # Change in the value of beta_1 may happen only at the boundary points
    # of the intervals
    def flatten(torch_tensor):
        """
        Ref. https://discuss.pytorch.org/t/any-alternatives-to-flat-for-tensor/3106
        """
        return torch_tensor.view(torch_tensor.numel())
    switch_points, _ = torch.sort(torch.unique(flatten(persistence_intervals_ext)))
    rlt = torch.zeros(i_max, dtype=torch.float64)
    for i in range(0, switch_points.shape[0] - 1):
        midpoint = ((switch_points[i] + switch_points[i + 1]) / 2).float()
        count_midpoint_interval = 0
        for interval in persistence_intervals:
            # Count how many intervals contain midpoint
            if midpoint >= interval[0] and midpoint < interval[1]:
                count_midpoint_interval += 1
        if (count_midpoint_interval < i_max):
            # import pdb; pdb.set_trace()
            rlt[count_midpoint_interval] += (switch_points[i + 1] - switch_points[i]).double()
    return rlt / alpha_max


def lmrk_table(W, L):
    """
      Helper function to construct an itorchut for the gudhi.WitnessComplex
      function.

    Args:
      W: 2d array of size w x d, containing witnesses
      L: 2d array of size l x d containing landmarks

    Returns
      Return a 3d array D of size w x l x 2 and the maximal distance
      between W and L.

      D satisfies the property that D[i, :, :] is [idx_i, dists_i],
      where dists_i are the sorted distances from the i-th witness to each
      point in L and idx_i are the indices of the corresponding points
      in L, e.g.,
      D[i, :, :] = [[0, 0.1], [1, 0.2], [3, 0.3], [2, 0.4]]
    """
    # import pdb; pdb.set_trace()
    a = torch.from_numpy(cdist(W, L))
    max_val = torch.max(a)
    _, idx = torch.sort(a)
    # import pdb; pdb.set_trace()
    range_id = torch.range(0, a.shape[0]-1, dtype=torch.int64).unsqueeze(-1)
    b = a[range_id, idx].long()
    idx = idx.long()
    # print(b)
    return torch.stack([idx, b], dim=-1), max_val


def random_landmarks(X, L_0=32):
    """
      Randomly sample L_0 points from X.
    """
    sz = X.shape[0]
    idx = torch.LongTensor(L_0).random_(sz)
    L = X[idx]
    return L


def witness(X, gamma=1.0 / 128, L_0=64):
    """
      This function computes the persistence intervals for the dataset
      X using the witness complex.

    Args:
      X: 2d array representing the dataset.
      gamma: parameter determining the maximal persistence value.
      L_0: int, number of landmarks to use.

    Returns
      A list of persistence intervals and the maximal persistence value.
    """
    L = random_landmarks(X, L_0)
    W = X
    lmrk_tab, max_dist = lmrk_table(W, L)
    wc = gudhi.WitnessComplex(lmrk_tab)
    alpha_max = max_dist * gamma
    # import pdb; pdb.set_trace()
    st = wc.create_simplex_tree(max_alpha_square=alpha_max, limit_dimension=2)
    big_diag = st.persistence(homology_coeff_field=2)
    diag = st.persistence_intervals_in_dimension(1)
    return diag, alpha_max


def fancy_plot(y, color='C0', label='', alpha=0.3):
    """
      A function for a nice visualization of MRLT.
    """
    n = y.shape[0]
    x = torch.arange(n)
    xleft = x - 0.5
    xright = x + 0.5
    X = torch.array([xleft, xright]).T.flatten()
    Xn = torch.zeros(X.shape[0] + 2)
    Xn[1:-1] = X
    Xn[0] = -0.5
    Xn[-1] = n - 0.5
    Y = torch.array([y, y]).T.flatten()
    Yn = torch.zeros(Y.shape[0] + 2)
    Yn[1:-1] = Y
    plt.bar(x, y, width=1, alpha=alpha, color=color, edgecolor=color)
    plt.plot(Xn, Yn, c=color, label=label, lw=3)
