import torch
import numpy as np
from scipy.spatial import KDTree


def subsample_points_by_radius(x: torch.Tensor, 
                               radius: float) -> torch.LongTensor:
    """
    Subsample points by radius. This is used in simulating mean curvature flow.

    Args:
        x (torch.Tensor): Points to subsample
        radius (float): Radius for subsampling

    Returns:
        torch.LongTensor: Indices of subsampled points
    """    
    
    tree = KDTree(x.cpu().numpy())
    inds = tree.query_ball_point(x.cpu().numpy(), r=radius)
    lengths = np.array([len(i) for i in inds])
    if (lengths == 1).any():
        keepers = np.concatenate(inds[lengths == 1])
    else:
        keepers = np.empty((0))
    if keepers.shape[0] == x.shape[0]:
        return torch.LongTensor(keepers)

    other_inds = np.unique(np.concatenate(inds[lengths > 1]))
    other_x = x[other_inds]
    otree = KDTree(other_x.cpu().numpy())
    pd = otree.sparse_distance_matrix(otree, max_distance=radius).tocoo()
    pd.data = pd.data.astype(bool)
    pdmask = np.logical_not(pd.toarray())
    mask = np.ones(pdmask.shape[0], dtype=bool)
    arange = np.arange(mask.shape[0])
    inds = []
    while mask.any():
        new_idx = np.random.choice(arange[mask])
        inds.append(new_idx)
        mask[new_idx] = False
        mask = np.logical_and(mask, pdmask[new_idx])
        
    keepers = np.concatenate((keepers, other_inds[inds]))
    return torch.LongTensor(keepers)

def smooth_values_by_gaussian(x: torch.Tensor, 
                              values: torch.Tensor, 
                              radius: float) -> torch.Tensor:
    """
    Smooth values by a truncated Gaussian kernel over a given radius. This is used
    in simulating mean curvature flow.

    Args:
        x (torch.Tensor): Input data points
        values (torch.Tensor): Values to be smoothed
        radius (float): Radius for truncating the Gaussian kernel. The standard deviation
            of the Gaussian is set to one third of the radius.

    Returns:
        torch.Tensor: Smoothed values
    """
    
    device = x.device
    tree = KDTree(x.cpu().numpy())
    k = max(
        [len(i) for i in tree.query_ball_point(x.cpu().numpy(), 
                                                r=radius)]
        )
    dist, ind = tree.query(x.cpu().numpy(), k=k)
    dist = torch.from_numpy(dist).float().to(device)
    ind = torch.from_numpy(ind).long().to(device)
    weights = torch.exp(-dist.pow(2) / (2 * (radius/3)**2))
    weights[dist > radius] = 0
    weights = weights / weights.sum(dim=1, keepdim=True)
    values = (weights * values[ind]).sum(dim=1, keepdim=True)
    return values
    
    