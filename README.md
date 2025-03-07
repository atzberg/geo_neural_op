<p align="left">
<img src="https://github.com/atzberg/geo_neural_op/blob/main/images/docs/geo_neural_op_software.png" width="90%"> 
</p>

[Documentation](https://web.math.ucsb.edu/~atzberg/geo_neural_op_docs/html/index.html) |
[Examples](./examples) |
[Paper](https://doi.org/10.1088/2632-2153/ad8980) |
[ArXiv](https://arxiv.org/abs/2404.10843)
                                                                                                
### Geometric Neural Operators (GNPs) 

Geometric Neural Operators (GNPs) allow for data-driven deep learning
of features from point-cloud representations and other 
datasets for tasks involving geometry.   This includes training 
protocols and learned operators for estimating local curvatures,
evaluating geometric differential operators, solvers for 
PDEs on manifolds, mean-curvature shape flows, and other tasks.
The package provides practical neural network architectures and factorizations 
for training to accounting for geometric contributions and features.  The package also 
has a modular design allowing for use of GNPs within other data-processing pipelines.
Pretrained models are also provided for estimating curvatures, Laplace-Beltrami operators,
components for PDE solvers, and other geometric tasks.

Examples are included that demonstrate how GNPs can be used.  This includes (i) to estimate geometric properties, such as the metric and curvatures of surfaces, (ii) to approximate solutions of geometric partial differential equations (PDEs) on manifolds, and (iii) to perform curvature-driven flows of shapes. These results show a few ways GNPs can be used for incorporating the roles of geometry into machine learning processing pipelines and solvers.

__Quick Start__


```bash
git clone git@github.com:atzberg/geo_neural_op.git
conda create -n gnp
conda activate gnp
pip install -r requirements.txt
```
You may also need to first install `pip`,
```bash
conda insall pip 
```

For use of the package see the [examples folder](https://github.com/atzberg/geo_neural_op__staging/tree/main/examples).  More
information on the structure of the package also can be found on the
[documentation pages](https://web.math.ucsb.edu/~atzberg/geo_neural_op_docs/html/index.html).

__Packages__ 

The pip install should automatically handle most of the dependencies.  If there are
issues, please be sure to install [pytorch](https://pytorch.org/) package version >= 2.0.0.
The full set of dependencies can be found in the [requirements.txt](./requirements.txt).
You may want to first install pytorch package manually to configure it for your specific
GPU system and platform.

__Usage__

For information on how to use the package, see

- [Examples](./examples) |

- [Documentation](https://web.math.ucsb.edu/~atzberg/geo_neural_op_docs/html/index.html) |

__Additional Information__

When using this package, please cite: 

*Geometric Neural Operators (GNPs) for Data-Driven Deep Learning in Non-Euclidean Settings,*
B. Quackenbush and P. J. Atzberger, Machine Learning: Science and Technology, 5.4, 045033, (2024), 
[Paper](https://doi.org/10.1088/2632-2153/ad8980) [ArXiv](https://arxiv.org/abs/2404.10843).
```
@article{quackenbush2024gnp,
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
B. Quackenbush and P. J. Atzberger, arXiv, (2025), 
[ArXiv](https://arxiv.org/pdf/2503.04649).
```
@article{quackenbush2025geotasks,
  title={Transferable Foundation Models for Geometric Tasks on Point Cloud Representations: Geometric Neural Operators},
  author={Quackenbush, Blaine and Atzberger, PJ},
  journal={arXiv:2503.04649},
  url={https://arxiv.org/pdf/2503.04649},
  year={2025}
}
```

__Acknowledgements__
This work was supported by grants from NSF Grant DMS-1616353 and NSF Grant DMS-1616353.

----

[Documentation](https://web.math.ucsb.edu/~atzberg/geo_neural_op/docs/index.html) |
[Examples](./examples) |
[Paper](https://doi.org/10.1088/2632-2153/ad8980) |
[ArXiv](https://arxiv.org/abs/2404.10843)



