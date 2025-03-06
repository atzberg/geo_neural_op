import torch
import torch.nn as nn
from typing import List
from torch_geometric.nn import global_mean_pool

from .layers import BlockConv

class BlockGNP(nn.Module):
    """
    Block Geometric Neural Operator

    Parameters
    ----------
    node_dim : int
        Dimension of the input node features.
    edge_dim : int
        Dimension of the input edge features.
    out_dim : int
        Dimension of the output node features.
    layers : list of int
        List of integers representing the dimensions of the hidden layers.
    num_channels : int
        Number of blocks in the kernel factorization.
    neurons : int
        Number of neurons in the kernel MLP.
    nonlinearity : str
        Nonlinearity to use in the hidden layers.
    device : torch.device
        Device to run the model on.
    """
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 out_dim: int,
                 layers: List[int],
                 num_channels: int,
                 neurons: int,
                 nonlinearity: str,
                 device: torch.device):
        super().__init__()
        self.device = device
        self.layers = layers
        self.depth = len(layers) - 1
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.channels = num_channels
        self.neurons = neurons
        self.nonlinearity = nonlinearity
        self.lift = nn.Linear(node_dim, layers[0])
        self.proj = nn.Linear(layers[-1], out_dim)
        self.W = nn.ModuleList([nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
                                for d_in, d_out in zip(layers[:-1], layers[1:])])
        self.mixes = nn.ModuleList([nn.Linear(d_out, d_out) for d_out in layers[1:]])
        self.convs = nn.ModuleList([BlockConv(edge_dim,
                                              d_in,
                                              d_out,
                                              self.channels,
                                              neurons,
                                              nonlinearity) for d_in, d_out in zip(layers[:-1], layers[1:])])
        self.activ = getattr(nn, nonlinearity)()
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.lift(x)
        for i in range(self.depth):
            z = self.convs[i](x, edge_index, edge_attr)
            x = self.W[i](x) + self.mixes[i](z)
            if i < self.depth - 1:
                x = self.activ(x)
        x = self.proj(x)
        return x
    
class PatchGNP(nn.Module):
    """
    Patch Geometric Neural Operator

    Parameters
    ----------
    model : torch.nn.Module
        Model to use for processing the input data.
    out_dim : int
        Dimension of the output basis.
    device : torch.device
        Device to run the model on.
    """
    def __init__(self,
                 model: nn.Module,
                 out_dim: int,
                 device: torch.device):
        super().__init__()
        self.model = model
        self.device = device
        self.out_dim = out_dim
        self.v_dim = model.out_dim
        self.nonlinearity = model.nonlinearity
        self.activ = getattr(nn, self.nonlinearity)()
        self.mlp = nn.Sequential(nn.Linear(self.v_dim, 2 * self.v_dim),
                                 self.activ,
                                 nn.Linear(2 * self.v_dim, self.out_dim))

        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, data):
        x = self.model(data)
        x = global_mean_pool(x[data.mask], batch=data.batch[data.mask])
        x = self.mlp(x)
        return x