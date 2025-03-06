import numpy as np
import torch
from typing import Optional
import torch_geometric as tg
from scipy.spatial import KDTree
from torch_geometric.data import Batch, Data
from torch_geometric.nn import radius_graph

from .pca import PCABatch

def graph_from_dict(sample: dict, 
                    radius: float, 
                    max_num_neighbors: int=32):
    """
    Construct a graph from a dictionary of node features and other data.

    Parameters
    ----------
    sample : dict
        Dictionary containing the node features and other data.
    radius : float
        Radius for the radius graph.
    max_num_neighbors : int, optional
        Maximum number of neighbors per node. Defaults to 32.

    Returns
    -------
    torch_geometric.data.Data
        A graph data object containing node features, edge indices, and edge attributes.
    """    
    x = sample.get('x')
    x_device = x.device
    edge_data = sample.get('edge_data')
    if edge_data is None:
        edge_data = x

    tree = KDTree(x.cpu().detach().numpy())
    cols = tree.query_ball_point(x.cpu().detach().numpy(), r=radius)
    cols = [
        np.random.choice(c, size=min(max_num_neighbors, len(c)), replace=False) for c in cols
        ]
    cols = [torch.tensor(c).long() for c in cols]
    rows = [torch.full_like(c, i) for i, c in enumerate(cols)]
    edge_ind = torch.stack((torch.cat(rows), torch.cat(cols)), dim=0)

    edge_attr = edge_data[edge_ind.t()].reshape(-1, 2 * edge_data.shape[1])

    data = Data(x=x,
                edge_index=edge_ind,
                edge_attr=edge_attr)
    for k, v in sample.items():
        if k not in ['x', 'edge_data']:
            setattr(data, k, v)
    return data.to(x_device)


class PatchGenerator:
    """
    PatchLoader class that generates patches from a point cloud.
    
    Parameters
    ----------
        data : dict
            Point cloud data. Must include keys 'x' for point cloud and 
            'normals' which must have the correct orientation of the pointcloud.
            The normals do not need to be accurate other than the orientation.
        graph_radius : float
            Radius for graph construction.
        batch_size : int, optional
            Batch size. Defaults to 1.
        center : str, optional
            Method for choosing patch centers. If 'tree', a KDTree is used to ensure patches 
            completely cover the point cloud. If 'gmls', there is one patch for every point 
            in the data. Defaults to 'tree'.
        shuffle_patches : bool, optional
            Whether to shuffle patches. Defaults to True.
        knn : int, optional
            Number of neighbors to choose for patch construction. Defaults to 50.
        min_radius : float, optional
            Minimum patch radius. Defaults to 0.01.
        pca : bool, optional
            If True, returns patch in local PCA coordinates. Defaults to True.
        degree : int, optional
            Maximum degree of 1D Legendre basis if pca is True. Defaults to 3.
        orientation : Optional[torch.Tensor], optional
            Orientation used if pca is True. Defaults to None.
        scale : bool, optional
            If True, scaling is used if pca is also True. Defaults to True.
        include_scale : bool, optional
            If True, includes scale in the output. Defaults to True.
        min_z_scale : float, optional
            Minimum scale used in pca rescaling. Defaults to 1e-3.
        max_num_neighbors : int, optional
            Maximum neighbors used in graph construction. Defaults to 32.
        device : str, optional
            Device to use. Defaults to 'cpu'.
    """    
    def __init__(self,
                 data: dict,
                 graph_radius: float,
                 batch_size: int=1,
                 center: str='tree',
                 shuffle_patches: bool=True,
                 knn: int=50,
                 min_radius: float=0.01,
                 pca: bool=True,
                 degree: int=3,
                 orientation: Optional[torch.Tensor]=None,
                 min_z_scale: float=1e-3,
                 max_num_neighbors: int=32,
                 device: str ='cpu'):     
        
        self.current_idx = 0
        self.data = data.copy()
        self.x = self.data.get('x').squeeze(0)
        self.graph_radius = graph_radius
        self.batch_size = batch_size
        self.shuffle = shuffle_patches
        self.device = device
        self.tree = KDTree(self.x.cpu().numpy())
        self.center = center
        self.knn = knn
        self.min_radius = torch.tensor(min_radius, device=self.device)
        self.centers = self.get_centers()
        self.max_num_neighbors = max_num_neighbors
        
        if self.data.get('path') is not None:
            self.data.pop('path')
        
        if self.shuffle:
            self.order = torch.randperm(self.centers.shape[0])
        else:
            self.order = torch.arange(self.centers.shape[0])

        self.center_data = self.x[self.centers]
        self.num_patches = self.centers.shape[0]

        knn_dist, self.knn_ind = self.tree.query(
            self.center_data.cpu().numpy(), k=self.knn, eps=0.05
            )
        self.knn_dist = torch.from_numpy(knn_dist[:, -1]).to(self.device)

        if not hasattr(self, 'clusters'):
            self.clusters = tg.nn.knn(self.center_data, self.x, k=1)[1].squeeze()
            
        self.patch_idx, self.buffer_idx = self.get_patch_indices()
        self.pca = pca
        self.degree = degree
        self.min_z_scale = min_z_scale
        
    def get_centers(self) -> torch.LongTensor:

        """        
        Choose the centers of the patches.
        If `self.center` is 'tree', the centers are chosen so that there is sufficient
        coverage of the data. If `self.center` is 'gmls', there is one patch center
        Returns
        -------
        torch.LongTensor
            Indices of the patch centers.
        """        
        
        if self.center == 'tree':
            _, ind = self.tree.query(self.x.cpu().numpy(), k=int(self.knn//3))
            centers = []
            mask = np.ones(self.x.shape[0], dtype=bool)
            arange = np.arange(self.x.shape[0])
            while mask.any():
                idx = np.random.choice(arange[mask])
                centers.append(idx)
                mask[ind[idx]] = False
            centers = torch.LongTensor(centers)
            
            knn_dist, self.knn_ind = self.tree.query(self.x.cpu()[centers].numpy(), 
                                                          k=self.knn)
            self.knn_dist = torch.from_numpy(knn_dist[:, -1]).to(self.device)
            
            clusters = torch.zeros(self.x.shape[0], dtype=torch.long)
            arange = torch.arange(self.knn_ind.shape[0])
            for j in range(self.knn_ind.shape[1]-1, -1, -1):
                clusters[self.knn_ind[:, j]] = arange
            self.clusters = clusters.to(self.device)
            
        elif self.center == 'gmls':
            centers = torch.arange(self.x.shape[0])
            knn_dist, self.knn_ind = self.tree.query(self.x.cpu().numpy(), 
                                                          k=self.knn)
            self.knn_dist = torch.from_numpy(knn_dist[:, -1]).to(self.device)
            self.clusters = centers.clone().to(self.device)
            
        return centers
    
    def get_patch_indices(self) -> tuple[list[int], list[int]]:
        
        """
        Returns indices of the patches and buffer regions.
        Returns
        -------
        tuple of list of int
            patch_idx : list of int
                List of indices of the patches.
            buffer_idx : list of int
                List of indices of the buffer regions.
        """     
           
        radii = torch.maximum(1.1 * self.knn_dist, self.min_radius).cpu().numpy()
        patch_idx = self.tree.query_ball_point(self.center_data.cpu().numpy(),
                                               r=radii, p=2, eps=0.01)
        buffer_idx = self.tree.query_ball_point(self.center_data.cpu().numpy(),
                                                r=1.25*radii, p=2, eps=0.01)
        return patch_idx, buffer_idx
        
        

    def get_patch(self, idx: int) -> Data:
        """
        Returns a single patch in the form of a torch_geometric Data object.

        Parameters
        ----------
        idx : int
            Index of the patch to return.

        Returns
        -------
        Data
            torch_geometric Data object containing the patch and all the associated data.
        """        
        
        eval_patch_idx = torch.tensor(self.patch_idx[idx], device=self.device)
        buffer_patch_idx = torch.tensor(self.buffer_idx[idx], device=self.device)
        
        extra_patch_idx = torch.tensor([x for x in buffer_patch_idx 
                                        if x not in eval_patch_idx], 
                                       device=self.device)

        patch_idx = torch.cat((eval_patch_idx, extra_patch_idx)).long()
        
        cluster_mask = (
            self.clusters[patch_idx] == self.clusters[self.centers[idx]]
            ).view(-1, 1)
        mask = torch.cat((torch.ones_like(eval_patch_idx),
                          torch.zeros_like(extra_patch_idx)), 
                         dim=0).bool().view(-1, 1)
        
        patch_data = {}
        for k, v in self.data.items():
            patch_data[k] = v.squeeze(0)[patch_idx]
        
        data = graph_from_dict(patch_data, 
                               radius=self.graph_radius,
                               max_num_neighbors=self.max_num_neighbors)

        data.mask = mask
        data.cluster_mask = cluster_mask
        data.ind = patch_idx

        return data.to(self.device)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx >= self.order.shape[0]:
            raise StopIteration
        else:
            idx = self.current_idx
            diff = min(self.order.shape[0] - self.current_idx, self.batch_size)
            self.current_idx += diff
        batch = Batch.from_data_list([self.get_patch(self.order[idx + i]) for i in range(diff)])
        
        if self.pca:
            pca_mapper = PCABatch(num_graphs=batch.num_graphs,
                                  degree=self.degree,
                                  min_z_scale=self.min_z_scale)
            return pca_mapper.to_pca(batch)
        else:
            return batch


class PatchLoader:
    """
    PatchLoader class that outputs a PatchGenerator instance when given PCD data.
    
    Parameters
        ----------
        graph_radius : float
            Radius for graph construction.
        batch_size : int, optional
            Batch size. Defaults to 1.
        center : str, optional
            Method for choosing patch centers. If 'tree', a KDTree is used to ensure patches 
            completely cover the point cloud. If 'gmls', there is one patch for every point 
            in the data. Defaults to 'tree'.
        shuffle_patches : bool, optional
            Whether to shuffle patches. Defaults to True.
        knn : int, optional
            Number of neighbors to choose for patch construction. Defaults to 50.
        min_radius : float, optional
            Minimum patch radius. Defaults to 0.01.
        pca : bool, optional
            If True, returns patch in local PCA coordinates. Defaults to True.
        degree : int, optional
            Maximum degree of 1D Legendre basis if pca is True. Defaults to 3.
        min_z_scale : float, optional
            Minimum scale used in pca rescaling. Defaults to 1e-3.
        max_num_neighbors : int, optional
            Maximum neighbors used in graph construction. Defaults to 32.
        device : str, optional
            Device to use. Defaults to 'cpu'.
    """    
    def __init__(self,
                 graph_radius: float,
                 batch_size: int=1,
                 center: str='tree',
                 shuffle_patches: bool=True,
                 knn: int=50,
                 min_radius: float=0.01,
                 pca: bool=True,
                 degree: int=3,
                 min_z_scale: float=1e-3,
                 max_num_neighbors: int=32,
                 device: str ='cpu'):
       
        self.graph_radius = graph_radius
        self.batch_size = batch_size
        self.shuffle = shuffle_patches
        self.device = device
        self.center = center
        self.knn = knn
        self.min_radius = min_radius
        self.pca = pca
        self.degree = degree
        self.min_z_scale = min_z_scale
        self.max_num_neighbors = max_num_neighbors
        

    def __call__(self, data: dict) -> PatchGenerator:
        """
        Returns a PatchGenerator object.

        Parameters
        ----------
        data : dict
            Point cloud data. Must include keys 'x' for point cloud and 
            'normals' which must have the correct orientation of the pointcloud.
            The normals do not need to be accurate other than the orientation.

        Returns
        -------
        PatchGenerator
            PatchGenerator object.
        """        
        return PatchGenerator(data=data,
                              graph_radius=self.graph_radius,
                              batch_size=self.batch_size,
                              shuffle_patches=self.shuffle,
                              center=self.center,
                              knn=self.knn,
                              min_radius=self.min_radius,
                              pca=self.pca,
                              degree=self.degree,
                              min_z_scale=self.min_z_scale,
                              max_num_neighbors=self.max_num_neighbors,
                              device=self.device)

