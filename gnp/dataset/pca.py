import torch
from torch_geometric.data import Data, Batch
from typing import Optional

class PCAPatch:
    def __init__(self,
                 data: Optional[dict]=None,
                 degree: int=3,
                 min_z_scale: float=1e-3):
        
        self.data = data
        self.mask = None
        self.degree = degree
        self.basis = 'legendre'
        self.center = None
        self.pca_vectors = None
        self.xy_scale = None
        self.z_scale = None
        self.min_z_scale = min_z_scale

        if self.data is not None:
            self.initialize(data)

    def initialize(self, data: dict):
        """
        Initialize the PCA basis for the input patch

        Parameters
        ----------
        data : dict
            Dictionary containing the input patch data
        """             
        
        self.data = data
        self.mask = data.mask.view(-1)
        patch_data = self.data.x[self.mask]
        self.center = patch_data.mean(dim=0, keepdim=True)
        _, S, Vh = torch.linalg.svd(patch_data - self.center, full_matrices=False)
        pca_vectors = Vh.clone()


        z_scale = S[2] * torch.ones(1, device=S.device)
        sign = [-1, 1][torch.randperm(2)[0]]
        z_scale *= sign
        
        if hasattr(data, 'normals'):
            self.orientation = data.normals.mean(dim=0, keepdim=True)
            self.orientation /= self.orientation.norm(dim=-1, keepdim=True)
            if self.orientation.isnan().any():
                self.orientation = self.center
        else:
            self.orientation = self.center
            
        cross = torch.linalg.cross(Vh[0], Vh[1])
        if (cross * self.orientation).sum() < 0:
            pca_vectors[0] = Vh[1].clone()
            pca_vectors[1] = Vh[0].clone()
        pca_vectors[2] = torch.linalg.cross(pca_vectors[0], pca_vectors[1])

        self.pca_vectors = pca_vectors
        self.xy_scale = None
        self.z_scale = 2 * (z_scale / (patch_data.shape[0] - 1) ** 0.5)
        if self.z_scale.abs() < self.min_z_scale:
            self.z_scale = sign * self.min_z_scale * torch.ones(1, device=S.device)

    def to_pca(self) -> Data:
        """
        Perform change of coordinates to local PCA basis for input patch.

        Returns
        -------
        Data
            Data object of patch in PCA basis.
        """
        
        projection = self.pca_vectors.t()
        pca_coords = (self.data.x - self.center) @ projection
        self.xy_scale = pca_coords[self.mask, :2].norm(dim=-1).max()
        scaling = torch.tensor([self.xy_scale,  self.xy_scale, self.z_scale], 
                               device=self.xy_scale.device)
        
        pca_coords = pca_coords / scaling.view(-1, 3)
        original_x = self.data.get('original_x', None)
        
        if original_x is not None:
            orig_pca_coords = ((original_x - self.center) 
                                @ projection) / scaling.view(-1, 3)
        else:
            orig_pca_coords = pca_coords
        
        pca_edges = ((self.data.edge_attr - self.center.repeat(1, 2))
                     @ torch.block_diag(projection, projection))
        pca_edges /= scaling.view(-1, 3).repeat(1, 2)
        normals = self.data.normals @ self.pca_vectors.t()

        out_data = Data(
            x=pca_coords,
            original_x=orig_pca_coords,
            normals=normals,
            edge_attr=pca_edges,
            mask=self.mask,
            degree=self.degree,
            xy_scale=self.xy_scale,
            z_scale=self.z_scale,
            pca_vectors=self.pca_vectors,
            center=self.center,
            basis=self.basis
        )
        for k, v in self.data.items():
            if k not in ['x', 'original_x', 'normals', 'mask', 'edge_attr']:
                out_data[k] = v
                
        return out_data

class PCABatch:
    """
    Class for handling multiple PCA patches.
    """    
    def __init__(self, num_graphs: int, **kwargs):
        """
        Initialize the PCABatch object

        Parameters
        ----------
        num_graphs : int
            Number of PCA patches in the batch.
        """        
        
        self.num_graphs = num_graphs
        self.pcas = [PCAPatch(**kwargs) for _ in range(num_graphs)]
        self.normal_scale = 1

    def to_pca(self, data: Batch) -> Batch:
        """
        Convert the input batch to PCA basis.

        Parameters
        ----------
        data : Batch
            Input batch of data.

        Returns
        -------
        Batch
            Batch of data in PCA basis.
        """         
        
        for j in range(self.num_graphs):
            self.pcas[j].initialize(data[j])
        batch = Batch.from_data_list([pca.to_pca() for pca in self.pcas])
        return batch
