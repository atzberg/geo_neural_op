### Examples
----
GNP models can be instantiated using the ``GNP`` model class:
```python
from gnp.models import GNP

full_model = GNP(
    node_dim=3,
    edge_dim=6,
    out_dim=1,
    layers=[64] * 10,
    conv_name="GraphConvolution",
    conv_args={"neurons": 128},
    nonlinearity="ReLU",
    skip_connection=True,
    device="cuda",
)

```

See how to load more variants of the ``GNP`` in our [Example Code](./models_01/models.ipynb)

Pretrained models are loaded when instantiatiating the ```GeometryEstimator``` class. This can be used to gather geometric quantities and more by instantiating:
```python
import numpy as np
import torch
from gnp import GeometryEstimator
pcd = torch.from_numpy(np.load('example_data/spot/xyz.npy'))
orientation = torch.from_numpy(np.load('example_data/spot/normals.npy'))
estimator = GeometryEstimator(pcd=pcd,
                              orientation=orientation,
                              model_name='clean_30k')
```


#### GNPs for Estimating Curvatures 
Geometric quantities can be easily generated using
```python
outputs = estimator.estimate_quantities(['mean_curvature', 'gaussian_curvature'])
```

<p align="center">
<br>
<img src="./curvatures_01/images/spot_curvatures.png" width="70%"> 
<img src="./curvatures_01/images/toroidal_curvatures.png" width="70%"> 
</p>

See our [example code](./curvatures_01/curvatures.ipynb).

----
#### GNPs for Solving PDEs on Manifolds 
A stiffness matrix for the Laplace-Beltrami equation $-\Delta_{\text{LB}} u = f$
using Generalized Moving Least Squares (GMLS) can be constructed
```python
stiffness_matrix, collocation_mask, outlier_mask = estimator.stiffness_matrix_gmls(
    drop_ratio=0.1, radius=1, p=4, remove_outliers=False
)
```
<p align="center">
<br>
<img src="./collocation_01/images/spot_collocation.png" width="100%"> 
</p>

This can be paired with your favorite linear solvers and/or preconditioners. 
We use Scipy's LGMRES and PyAMG for preconditioning. See our collocation 
[example code](./collocation_01/collocation.ipynb)

----
#### GNPs for Curvature Driven Flows

Mean curvature flows (MCF) can be simulated using 
```python
flow_data = estimator.mean_flow(
    num_steps=250,
    save_data_per_step=25,
    delta_t=0.0002,
    subsample_radius=0.005,
    smooth_radius=0.06,
    smooth_x=False,
)
```

<p align="center">
<br>
<img src="./mean_flows_01/images/spot_flow.png" width="100%"> 
</p>

See our [example code](./mean_flows_01/mean_flow.ipynb).

----
#### GNP Curvature Estimation: Training Example

We show a standalone example of how to train a **Geometric Neural
Operator (PatchGNP)** from scratch to estimate surface curvatures on 3D point clouds.

#### Overview

The script `train_curvature_estimator.py`:

1. **Generates synthetic surfaces** -- analytically extracted mean and Gaussian curvatures as ground truth (GT).

   **Example surfaces:** -- (i) unit sphere, (ii) torus, (iii) paraboloid.
2. **Trains a `PatchGNP`** model end-to-end using MSE loss on both mean and gaussian curvature types. 

    **Remark:** The model learns Legendre polynomial coefficients for local surface patches; 
    curvatures are computed differentiably from those coefficients via the standard first and 
    second fundamental forms.

3. **Evaluates the trained model** on each test surface to obtain the mean absolute errors of the curvature estimates.

#### Training 
```bash
python train_curvature_estimator.py --epochs 1000 --num_points 15000 \
  --num_patches 64 --device cuda
```

See our training
[example code](./train_curvature_01).

----

