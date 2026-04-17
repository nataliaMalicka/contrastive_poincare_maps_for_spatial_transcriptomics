"""
Some methods have been adapted from Poincare Maps: https://github.com/facebookresearch/PoincareMaps
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from torch.autograd import Function
from .poincare_manifold import PoincareBall

eps = 1e-5
boundary = 1 - eps

def poincare_translation(v, x):
    """
    Computes the translation of x  when we move v to the center.
    Hence, the translation of u with -u should be the origin.
    """
    xsq = (x ** 2).sum(axis=1)
    vsq = (v ** 2).sum()
    xv = (x * v).sum(axis=1)
    a = np.matmul((xsq + 2 * xv + 1).reshape(-1, 1),
                  v.reshape(1, -1)) + (1 - vsq) * x
    b = xsq * vsq + 2 * xv + 1
    return np.dot(np.diag(1. / b), a)


def poincare_root(root_name, labels, features):
    if root_name is not None:
        head_idx = np.where(labels == root_name)[0]

        if len(head_idx) > 1:
            # medoids in Euclidean space
            D = pairwise_distances(features[head_idx, :], metric='euclidean')
            return head_idx[np.argmin(D.mean(axis=0))]
        elif len(head_idx) == 1:
            return head_idx[0]
        else:
            return -1

    return -1


def grad(x, v, sqnormx, sqnormv, sqdist):
    alpha = (1 - sqnormx)
    beta = (1 - sqnormv)
    z = 1 + 2 * sqdist / (alpha * beta)
    a = ((sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) /
            torch.pow(alpha, 2)).unsqueeze(-1).expand_as(x)
    a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
    z = torch.sqrt(torch.pow(z, 2) - 1)
    z = torch.clamp(z * beta, min=eps).unsqueeze(-1)
    return 4 * a / z.expand_as(x)


class PoincareDistance(Function):
    @staticmethod
    def forward(self, u, v):
        self.save_for_backward(u, v)
        self.squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, boundary)
        self.sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, boundary)
        self.sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
        x = self.sqdist / ((1 - self.squnorm) * (1 - self.sqvnorm)) * 2 + 1
        # arcosh
        z = torch.sqrt(torch.pow(x, 2) - 1)
        return torch.log(x + z)

    @staticmethod
    def backward(self, g):
        u, v = self.saved_tensors
        g = g.unsqueeze(-1)
        gu = grad(u, v, self.squnorm, self.sqvnorm, self.sqdist)
        gv = grad(v, u, self.sqvnorm, self.squnorm, self.sqdist)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv

def PoincareSim(x, y, manifold):
    '''Poincare similarity is the inverse of Poincare distance
    '''
    dist = PoincareDistance().apply(x, y)
    return 1. / (1 + dist)

class Poincare_loss(nn.Module):
    def __init__(self, temperature=1.0):
        """NT-Xent loss for contrastive learning using cosine distance as similarity metric as used in [SimCLR](https://arxiv.org/abs/2002.05709).
        Implementation adapted from https://theaisummer.com/simclr/#simclr-loss-implementation

        Args:
            temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
        """
        super().__init__()
        self.temperature = temperature
        self.manifold = PoincareBall()

    def forward(self, z_i, z_j):
        """Compute NT-Xent loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch

        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples

        Returns:
            float: loss
        """
        batch_size = z_i.size(0)
        emb_size = z_i.size(1)

        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        z_a = z.unsqueeze(1)
        z_b = z.unsqueeze(0)
        z_a = z_a.expand(2*batch_size, 2*batch_size, emb_size)
        z_b = z_b.expand_as(z_a)
        similarity = PoincareSim(z_a, z_b, self.manifold).squeeze(-1)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=positives.device)).float()
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss

class WeightedKNNLoss(nn.Module):
    def __init__(self, W, lambda_rep=1.0):
        super().__init__()
        self.W = W                      # precomputed kNN
        self.lambda_rep = lambda_rep    # repulsion factor

    def forward(self, z, indices):
        batch_size, emb_dim = z.size()

        # sub sampled graph from the dataloader
        W_batch = self.W[indices][:, indices].to(z.device)

        # pairwise hyperbolic distances
        z_a = z.unsqueeze(1).expand(batch_size, batch_size, emb_dim)
        z_b = z.unsqueeze(0).expand(batch_size, batch_size, emb_dim)
        D_low = PoincareDistance.apply(z_a, z_b).squeeze(-1)

        # diagonal where point compared with itself
        eye = torch.eye(batch_size, dtype=torch.bool, device=z.device)

        # remove self-connections so they dont contribute to the loss
        W_batch = W_batch.masked_fill(eye, 0.0)

        # identify non-neighbor pairs (used for repulsion term)
        # 1 for non-neighbors, 0 for neighbors and self-pairs
        non_neighbors = (W_batch == 0).float().masked_fill(eye, 0.0)

        # weighted attraction, strongly connected neighbours have a higher weight
        # non-neigh pushed farther apart
        loss_attract = (W_batch * D_low).sum() / (W_batch.sum() + 1e-8)

        # repulsion
        loss_repel = (non_neighbors * torch.exp(-D_low)).sum() / (non_neighbors.sum() + 1e-8)

        return loss_attract + self.lambda_rep * loss_repel