<p align="left">
<img src="https://github.com/atzberg/geo_neural_op/blob/main/images/docs/geo_neural_op_software.png" width="90%"> 
</p>

[Documentation](https://web.math.ucsb.edu/~atzberg/geo_neural_op_docs/html/index.html) |
[Examples](./examples) |
[Paper 1](https://doi.org/10.1088/2632-2153/ad8980) |
[Paper 2](https://doi.org/10.1088/2632-2153/ae1bf8) |
[Paper 3](https://arxiv.org/abs/2603.03621)
                                                                                                
### Geometric Neural Operators (GNPs) 

GNPs enable deep learning of features from point clouds and geometric datasets. 
Provides data-driven tools for learning and evaluating differential operators and solving PDEs on manifolds.

#### 🟣 Core Functionality

- **Geometric Feature Learning:** Extract intrinsic properties directly from point-cloud representations.
- **Differential Operators:** Evaluate local curvatures and Laplace-Beltrami operators using learned protocols.
- **Advanced PDE Solvers:** Robust solvers for Partial Differential Equations (PDEs) on manifolds.
- **Transferable Pretrained Models:** Available models for immediate use with pretrained weights, see examples section.

#### 🔵 Additional Functionality
- **Shape Evolution:** Support for mean-curvature shape flows and other dynamic geometric tasks.
- **Efficient Architectures:** Implementations using sparsity and factorizations for efficiency and trade-offs.
- **Modular Architecture:** Easily integrate GNP components into larger data-processing pipelines.

#### Robust Estimation Approach 

Our pre-trained GNP models and training methods also allow for 
coping with noise and other artifacts that arise when processing point-clouds.
This allows for robust estimates of the curvature and other geometric properties even when 
point-clouds have artifacts, such as outliers as shown below. 
<p align="left">
<img src="https://github.com/atzberg/geo_neural_op/blob/main/images/docs/point_cloud_outliers.png" width="60%"> 
</p>

#### ✴️ Examples

<p align="left">
<br>
<img src="./examples/mean_flows_01/images/mean_curv_flow_01.gif" width="60%"> <br>
</p>

A few applications of GNPs include
- [[mean curvature flows]](https://github.com/atzberg/geo_neural_op/tree/main/examples/mean_flows_01)
- [[pde solvers]](https://github.com/atzberg/geo_neural_op/tree/main/examples/pde_solver_01)
- [[curvature estimation]](https://github.com/atzberg/geo_neural_op/tree/main/examples/curvatures_01)
- [[transferable models]](https://github.com/atzberg/geo_neural_op/tree/main/examples/models_01)

[[examples folder]](https://github.com/atzberg/geo_neural_op/tree/main/examples)

We provide practical demonstrations for how GNPs can be used in
practice.  This includes examples (i) to estimate geometric properties, such as
the metric and curvatures of surfaces, (ii) to approximate solutions of
geometric partial differential equations (PDEs) on manifolds, and (iii) to
perform curvature-driven flows of shapes. These results show a few ways GNPs
can be used for incorporating the roles of geometry into machine learning
processing pipelines and solvers. See the [examples folder](https://github.com/atzberg/geo_neural_op/tree/main/examples).  

#### 📦 __Installation__


```bash
git clone git@github.com:atzberg/geo_neural_op.git
conda create -n gnp python=3.12
conda activate gnp
```
Install PyTorch prior to installing the repo to avoid installation errors 
related to torch-cluster and torch-scatter. To install PyTorch with cpu, use:
```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```
To install with CUDA, use one of the following, replacing X with the correct version:
```bash
# CUDA 11.X
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu11X
# CUDA 12.X
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu12X 
```
For available CUDA versions, see 
[PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/).
Once PyTorch is installed, you can install this repository using pip:
```bash
cd geo_neural_op
pip install .
```
If you want to run the example notebooks, you can install the additional dependencies using:
```bash
pip install .[dev]
```
If there is an error installing because of torch-cluster or torch-scatter, adding
the flag `--no-build-isolation` should fix this. Building wheels for torch-cluster and torch-scatter can be quite time consuming.

Alternatively, you can 
install torch-cluster and torch-scatter separately using the pre-built wheels 
corresponding to your PyTorch installation. After they are installed, you can proceed with 
installing this repository. Installing using the pre-built wheels can be done using the appropriate command below:
```bash
# CPU Build
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
# CUDA 11.X
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.6.0+cu121X.html
# CUDA 12.X 
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.6.0+cu12X.html 
```

#### 🤖 __Testing__

You can run tests for the package using

```bash
python -m unittest discover tests
```
For use of the package see the [examples folder](https://github.com/atzberg/geo_neural_op/tree/main/examples).  
More information on the structure of the package also can be found on the
[documentation pages](https://web.math.ucsb.edu/~atzberg/geo_neural_op_docs/html/index.html).


#### 💡 __Usage__

For information on how to use the package, see

- [Examples](./examples) 

- [Documentation](https://web.math.ucsb.edu/~atzberg/geo_neural_op_docs/html/index.html) 


__Version 2.0 Efficiency Gains__

``geo_neural_op`` v2.0.0 sees significant efficiency gains by leveraging the
optimized data processing of ``PatchTensor`` and inference of the ``PatchGNP``.
This uses the new separable, block-factorized kernels (see papers). Below, we display a
comparison of the average running times for versions 1.0.0 and 2.0.0 on both CPU
and CUDA devices. Each task is run 10 times on each of the example data sets found
in ``geo_neural_op/example_data``. We display the average times to perform each
task on one data sample from ``example_data``. In all cases, v2.0 sees about >18x
speed up on CPU and about a >7x speed up on CUDA devices. 

| Task | v1.0.0 CPU | v1.0.0 CUDA | v2.0.0 CPU | v2.0.0 CUDA|
|---------------------|------------|-------------|------------|------------|
| Geometric Quantities| 139.39s | 22.27s | 7.55s | 2.86s |
| Stiffness Matrix Construction | 2,591.73s | 524.32s | 117.85s | 7.85s |
| Mean Flow (10 steps) | 1,484.20s | 242.42s | 81.61s | 28.10s |




#### 📚 __Additional Information__

For the package, please cite: 



*Geometric Neural Operators (GNPs) for Data-Driven Deep Learning in Non-Euclidean Settings,*
B. Quackenbush and P. J. Atzberger, Machine Learning: Science and Technology, 5.4, 045033, (2024), 
[paper](https://doi.org/10.1088/2632-2153/ad8980), [arXiv](https://arxiv.org/abs/2404.10843).
```
@article{quackenbush_atzberger_gnps_2024,
  title={Geometric neural operators (gnps) for data-driven deep learning in non-euclidean settings},
  author={Quackenbush, Blaine and Atzberger, PJ},
  journal={Machine Learning: Science and Technology},
  volume={5},
  number={4},
  pages={045033},
  url={https://doi.org/10.1088/2632-2153/ad8980},
  publisher={IOP Publishing},
  year={2024}
}
```

*Transferable Foundation Models for Geometric Tasks on Point Cloud Representations: Geometric Neural Operators,*
B. Quackenbush and P. J. Atzberger, Machine Learning: Science and Technology, 6.4, 045045, (2025), 
[paper](https://doi.org/10.1088/2632-2153/ae1bf8), [arxiv](https://arxiv.org/abs/2503.04649).
```
@article{quackenbush_atzberger_gnp_transfer_2025,
  title={Transferable Foundation Models for Geometric Tasks on Point Cloud Representations: Geometric Neural Operators},  
  author={Quackenbush, Blaine and Atzberger, Paul},  
  journal={Machine Learning: Science and Technology},
  month = {11},  
  volume = {6},
  number = {4},
  pages = {045045},
  url={https://doi.org/10.1088/2632-2153/ae1bf8}
  publisher = {IOP Publishing},
  year={2025},
}
```


__Acknowledgements__
This work was supported by NSF Grant DMS-1616353 and NSF-DMS-2306345.

__Additional Information__ <br>
https://web.atzberger.org

----

[Documentation](https://web.math.ucsb.edu/~atzberg/geo_neural_op_docs/html/index.html) |
[Examples](./examples) |
[Paper 1](https://doi.org/10.1088/2632-2153/ad8980) |
[Paper 2](https://doi.org/10.1088/2632-2153/ae1bf8) | 
[Paper 3](https://arxiv.org/abs/2603.03621)
