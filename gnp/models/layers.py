import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing

class BlockKernel(nn.Module):
    """
    MLP to compute the block factorized kernel

    Parameters
    ----------
    edge_dim : int
        Dimension of the input edge features.
    d_in : int
        Dimension of the input node features.
    d_out : int
        Dimension of the output node features.
    num_channels : int
        Number of blocks in the kernel factorization.
    neurons : int
        Number of neurons in the kernel MLP.
    nonlinearity : str, optional
        Nonlinearity to use in the hidden layers (default is 'ReLU').
    """
    def __init__(self, edge_dim, d_in, d_out, num_channels, neurons, 
                 nonlinearity='ReLU'):
        super().__init__()
        self.d_x = edge_dim
        self.channels = num_channels
        self.d_in = d_in
        self.d_out = d_out
        self.head_in = d_in // num_channels
        self.head_out = d_out // num_channels
        self.neurons = neurons
        self.d_out = self.channels * self.head_in * self.head_out

        self.layers = nn.ModuleList([nn.Linear(self.d_x, neurons),
                                     nn.Linear(neurons, self.d_out)])
        if nonlinearity is not None:
            self.activ = getattr(nn, nonlinearity)()

    def forward(self, x):
        z = self.activ(self.layers[0](x))
        z = self.layers[1](z)
        return z.view(-1, self.channels, self.head_in, self.head_out)
    
class BlockConv(MessagePassing):
    """
    Edge convolution layer with the block factorized kernel.

    Parameters
    ----------
    edge_dim : int
        Dimension of the input edge features.
    d_in : int
        Dimension of the input node features.
    d_out : int
        Dimension of the output node features.
    num_channels : int
        Number of blocks in the kernel factorization.
    neurons : int
        Number of neurons in the kernel MLP.
    nonlinearity : str
        Nonlinearity to use in the hidden layers.
    """
    def __init__(self, edge_dim, d_in, d_out, num_channels, neurons, nonlinearity='ReLU'):
        super().__init__(aggr='mean')
        self.edge_dim = edge_dim
        self.d_in = d_in
        self.d_out = d_out
        self.head_in = d_in // num_channels
        self.channels = num_channels
        self.nn = BlockKernel(edge_dim, d_in, d_out, num_channels, neurons, nonlinearity)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        weights = self.nn(edge_attr)
        return torch.matmul(x_j.view(-1, self.channels, 1, self.head_in), weights).view(-1, self.d_out)

    def update(self, aggr_out):
        return aggr_out
    
