import torch
import yaml
from pathlib import Path
from .models.gnp import PatchGNP, BlockGNP
from .dataset.patch import PatchLoader

def load_config(path: Path) -> dict:
    """
    Load a configuration file from a yaml file.

    Parameters
    ----------
    path : str
        Path to the yaml file.

    Returns
    -------
    dict
        Dictionary containing the configuration parameters.
    """
    assert path.exists()
    with open(path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg

def load_model(model_path: Path) -> PatchGNP:
    """
    Load a model from a directory.

    Parameters
    ----------
    model_dir : Path
        Path to the model directory.

    Returns
    -------
    torch.nn.Module
        The loaded model.
    """    
    assert model_path.exists()
    
    backbone = BlockGNP(node_dim=3,
                        edge_dim=6,
                        out_dim=64,
                        layers=6*[64],
                        num_channels=4,
                        nonlinearity='ReLU',
                        neurons=256,
                        device='cpu')
    model = PatchGNP(model=backbone,
                     out_dim=16,
                     device='cpu')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    return model

def load_patchloader(cfg: dict, **kwargs) -> PatchLoader:
    """
    Load a PatchLoader using a yaml file in data_dir.

    Parameters
    ----------
    cfg: dict
        Dictionary containing the configuration parameters.
    """    
    for k, v in kwargs.items():
        cfg[k] = v
    return PatchLoader(**cfg)