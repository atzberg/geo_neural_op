import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
from torch_geometric.data import Batch
from torch_geometric.nn import fps
from tqdm import tqdm
import scipy.sparse as sp

from .config import load_config, load_model, load_patchloader
from .geometry.surface import SurfacePatch
from .utils import smooth_values_by_gaussian, subsample_points_by_radius 

MODEL_PATH = Path.joinpath(Path(__file__).parent, 'model_weights')

class GeometryEstimator:
    """
    Geometry estimator for a point cloud.

    Parameters
    ----------
    pcd : torch.Tensor
        Point cloud data.
    orientation : torch.Tensor
        Orientation data.
    function_values : Optional[torch.Tensor]
        Function values for the point cloud.
    model : str
        Model name.
    device : torch.device
        Device to run the model on.
    data : dict
        Dictionary containing the input data.
    model_path : Path
        Path to the model state dictionary.
    model : nn.Module
        Loaded model.
    config_path : Path
        Path to the configuration file.
    cfg : dict
        Configuration dictionary.
    patch_loader : PatchLoader
        Patch loader object.
    """
    
    
    def __init__(self, 
                 pcd: torch.Tensor,
                 orientation: torch.Tensor,
                 function_values: Optional[torch.Tensor]=None,
                 model: str='clean_30k',
                 device: torch.device='cpu',
                 **patch_kwargs: Optional[dict]):
        
        assert model in ['clean_30k', 'clean_50k', 'noise_70k', 'outlier_50k']
        self.pcd = pcd.to(device)
        self.orientation = orientation.to(device)
        self.device = device
        
        self.data = {'x': self.pcd, 
                     'normals': self.orientation}
        
        self.model_path = Path.joinpath(MODEL_PATH, model, 'state_dict.pth')
        self.model = load_model(self.model_path)
        self.model.to(device)
        
        self.config_path = Path.joinpath(MODEL_PATH, model, 'config.yaml')
        self.cfg = load_config(self.config_path)
        self.cfg['device'] = device
        self.patch_loader = load_patchloader(self.cfg, **patch_kwargs)
        
        if function_values is not None:
            self.function_values = function_values.to(device)
            self.data['function_values'] = self.function_values
    
    def surface_patch(self, patch_data: Batch) -> SurfacePatch:
        """
        Create a SurfacePatch from the input patch data.
        
        Parameters
        ----------
        patch_data : Batch
            Batch containing the input patch data.
        
        Returns
        -------
        SurfacePatch
            SurfacePatch object.
        """
        with torch.no_grad():
            surface_coefficients = self.model(patch_data)
        return SurfacePatch(patch_data, surface_coefficients)
    
    def estimate_quantities(self, scalar_names: list[str]) -> dict:
        """
        Estimate geometric quantities on the point cloud. This function returns 
        a dictionary containing the estimated scalar and/or vector values.

        Args:
            scalar_names (list[str]): List of scalar names to estimate. This can
                be any of the following: 'xyz_coordinates', 'normals', 'tangents', 
                'mean_curvature', 'gaussian_curvature', 'pca_coordinates', 'normals_pca',
                'tangents_pca', 'metric', 'shape', 'weingarten', 'inverse_metric',
                'inverse_metric_derivatives', 'det_metric', 'laplace_beltrami_from_coefficients'

        Returns:
            dict: Dictionary containing the estimated scalar values.
        """        
        data_size = self.pcd.size(0)
        patch_dataloader = self.patch_loader(self.data)
        output = {}
        
        for batch in patch_dataloader:
            mask = batch.cluster_mask.flatten()
            indices = batch.ind[mask]
            
            patch = self.surface_patch(batch)
            
            for name in scalar_names:
                quantity = getattr(patch, name, None)
                if quantity is None:
                    raise ValueError(f'Unknown scalar name: {name}')
                if name not in output:
                    output[name] = torch.zeros((data_size, *quantity.shape[1:]), 
                                               device=self.device)
                output[name][indices] = quantity[mask]
        
        return output
    
    
    def flow_step(self, 
                  delta_t: float, 
                  subsample_radius: float, 
                  smooth_radius: float, 
                  smooth_x: bool) -> dict:
        """
        Perform a single step of mean curvature flow on the point cloud.

        Args:
            delta_t (float): Time step for the flow.
            subsample_radius (float): Radius used for subsampling points.
            smooth_radius (float): Radius used for smoothing mean curvature.
            smooth_x (bool): Whether to smooth the point cloud before flow.

        Returns:
            dict: Dictionary containing the update point cloud data, normals, 
            and mean curvature.
        """        
        
        if smooth_x:
            estimate = self.estimate_quantities(['xyz_coordinates'])
            x = estimate['xyz_coordinates']
            self.pcd = x
            self.data['x'] = x
        
        estimate = self.estimate_quantities(['normals', 'mean_curvature'])
        x = self.pcd
        normals = estimate['normals']
        mean_curvature = smooth_values_by_gaussian(x=x,
                                                   values=estimate['mean_curvature'],
                                                   radius=smooth_radius)
        new_x = x + delta_t * mean_curvature.view(-1, 1) * normals
        subsampled_indices = subsample_points_by_radius(new_x, subsample_radius)
        new_x = new_x[subsampled_indices]
        new_normals = normals[subsampled_indices]
        mean_curvature = mean_curvature[subsampled_indices]

        new_data = {'x': new_x, 
                    'normals': new_normals, 
                    'mean_curvature': mean_curvature}
        
        return new_data

        
    def mean_flow(self,
                  num_steps: int,
                  save_data_per_step: int,
                  delta_t: float,
                  subsample_radius: float,
                  smooth_radius: float,
                  smooth_x: bool) -> list[dict]:
        """
        Perform mean curvature flow on the point cloud.

        Args:
            num_steps (int): Number of steps to perform.
            save_data_per_step (int): Save data every n steps.
            delta_t (float): Time step for the flow.
            subsample_radius (float): Radius used for subsampling points. 
            smooth_radius (float): Radius used for smoothing mean curvature.
            smooth_x (bool): Whether to smooth the point cloud before flow.

        Returns:
            list[dict]: List of dictionaries containing the updated point cloud data
            and normals at each saved time step.
        """        
        
        save_data = []
        for i in tqdm(range(num_steps)):
            new_data = self.flow_step(delta_t=delta_t,
                                      subsample_radius=subsample_radius,
                                      smooth_radius=smooth_radius,
                                      smooth_x=smooth_x)
            self.data = new_data.copy()
            self.pcd = new_data['x']
            self.orientation = new_data['normals']
            if i % save_data_per_step == 0:
                save_data.append(new_data.copy())
        
        return save_data
    
    def gmls_weights(self, batch: Batch, 
                     radius: float=1., 
                     p: int=4) -> list[torch.Tensor]:
        """
        Compute the weights for the generalized moving least squares (GMLS) method.

        Args:
            batch (Batch): Batch containing the input data.
            radius (float, optional): Radius to truncate weight function. Defaults to 1..
            p (int, optional): Degree p of the weight function. Defaults to 4.

        Returns:
            list[torch.Tensor]: List of weight matrices for each batch.
        """     
        xs = [
            batch[i].x[batch[i].mask.flatten(), :2] for i in range(batch.num_graphs)
            ]
        centers = batch.x[batch.cluster_mask.flatten(), :2]
        cxs = [x - centers[i].view(1, 2) for i, x in enumerate(xs)]
        Ws = [
            torch.diag(
                F.relu(1 - cxs[j].norm(dim=1) / radius).pow(p))
            for j in range(batch.num_graphs)
            ]
        return Ws
    
    def laplace_beltrami_legendre_blocks(self, 
                                         surface: SurfacePatch
                                         ) -> list[torch.Tensor]:
        """Laplace-Beltrami operator of Legendre basis functions at each center point.

        Args:
            surface (SurfacePatch): Surface patch object.

        Returns:
            list[torch.Tensor]: List of outputs at each center point.
        """
        
        lb_values = surface.laplace_beltrami_legendre_basis[surface.cluster_mask]
        return torch.split(lb_values, split_size_or_sections=1, dim=0)
    
    def legendre_blocks(self, surface: SurfacePatch) -> list[torch.Tensor]:
        """Legendre basis functions blocked by batch.

        Args:
            surface (SurfacePatch): Surface patch object.

        Returns:
            list[torch.Tensor]: List of outputs per batch.
        """
        mask = surface.mask
        batch = surface.batch[mask]
        
        legendre_values = surface.basis.evaluate(surface.x[mask, :2])
        legendre_blocks = [
            legendre_values[batch == i] for i in range(batch.max() + 1)
            ]
        return legendre_blocks
    
    def stiffness_on_batch(self, 
                           batch: Batch, 
                           radius: float=1., 
                           p: int=4) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the stiffness matrix on a batch of data.

        Args:
            batch (Batch): Batch containing the input data.
            radius (float, optional): Radius to truncate weight function. Defaults to 1..
            p (int, optional): Degree p of the weight function. Defaults to 4.

        Returns:
            torch.Tensor: Indices and values of the stiffness matrix.
        """        
        Ws = self.gmls_weights(batch, radius=radius, p=p)
        surface = self.surface_patch(batch)
        legendre_blocks = self.legendre_blocks(surface)
        lb_blocks = self.laplace_beltrami_legendre_blocks(surface)
        
        outputs = [
            torch.linalg.lstsq((leg.T @ W @ leg).cpu(), (leg.T @ W).cpu(), driver='gelsd')
            for leg, W in zip(legendre_blocks, Ws)
        ]
        stiffness_values = torch.cat([
            -lb @ output.solution.to(self.device) 
            for lb, output in zip(lb_blocks, outputs)
            ], dim=1).flatten()
        
        row_inds = batch.ind[surface.cluster_mask]
        column_inds = [
            batch[i].ind[batch[i].mask].view(1, -1) for i in range(batch.num_graphs)
            ]
        stiffness_indices = torch.cat([
            torch.cat(
                (row_inds[i] * torch.ones((1, column_inds[i].shape[1]), device=self.device), 
                 column_inds[i]), dim=0) 
            for i in range(batch.num_graphs)], dim=1
            )
        return stiffness_indices, stiffness_values
    
    
    def stiffness_matrix_gmls(self, 
                              drop_ratio: float=0.1,
                              radius: float=1., 
                              p: int=4,
                              remove_outliers: bool=False) -> sp.coo_matrix:
        """
        Compute the stiffness matrix using the generalized moving least squares (GMLS) method.

        Args:
            drop_ratio (float, optional): Ratio of points to drop. Defaults to 0.1.
            radius (float, optional): Radius to truncate weight function. Defaults to 1..
            p (int, optional): Degree p of the weight function. Defaults to 4.
            remove_outliers (bool, optional): Whether to remove outliers. Defaults to False.

        Returns:
            sp.coo_matrix: Stiffness matrix.
        """
        
        if remove_outliers:
            outputs = self.estimate_quantities(['x', 'pca_coordinates'])
            outlier_mask = (outputs['x' - outputs['pca_coordinates']].norm(dim=1) < 0.1)
            for k, v in self.data.items():
                self.data[k] = v[outlier_mask]
            self.pcd = self.pcd[outlier_mask]
            self.orientation = self.orientation[outlier_mask]
        
        drop_inds = fps(self.pcd, ratio=drop_ratio)
        self.data['stiffness_mask'] = torch.ones(self.pcd.size(0), 
                                                 dtype=torch.bool,
                                                 device=self.device)
        self.data['stiffness_mask'][drop_inds] = False
        
        self.patch_loader.center = 'gmls'
        patch_dataloader = self.patch_loader(self.data)
        stiffness_indices = []
        stiffness_values = []
        
        for batch in tqdm(patch_dataloader):
            batch.mask = batch.mask & batch.stiffness_mask
            indices, values = self.stiffness_on_batch(batch, radius=radius, p=p)
            stiffness_indices.append(indices)
            stiffness_values.append(values)
        
        stiffness_indices = torch.cat(stiffness_indices, dim=1)
        stiffness_values = torch.cat(stiffness_values)
        
        notmask = torch.logical_not(self.data['stiffness_mask'])
        skipped = torch.arange(self.data['stiffness_mask'].shape[0], 
                               device=self.device)[notmask]
        if len(skipped) > 0:
            subtract_index_value = torch.zeros_like(stiffness_indices[1], 
                                                    device=self.device)
            num_iters = skipped.shape[0] / 10
            num_iters = int(num_iters) if num_iters % 1 == 0 else int(num_iters) + 1
            for i in range(num_iters):
                start = i * 10
                end = min((i+1) * 10, skipped.shape[0])
                subtract_index_value += (
                    stiffness_indices[1].view(-1, 1) > skipped[start:end].view(1, -1)
                    ).sum(dim=1)
            stiffness_indices[1] = stiffness_indices[1] - subtract_index_value.long()
        
        stiffness = sp.coo_matrix((
            stiffness_values.cpu().numpy(), 
            stiffness_indices.cpu().numpy().astype(np.int32)
        ))
        return stiffness
