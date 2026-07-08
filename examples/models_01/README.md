# Transferable Pre-trained GNP Models 

GNP models can be instantiated in general using the ``GNP`` model class:
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

Pretrained model weights are loaded when instantiatiating the ```GeometryEstimator``` class. This can be used to gather geometric quantities and more by instantiating:
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

For more details see the notebook [[models.ipynb]](./models.ipynb).
