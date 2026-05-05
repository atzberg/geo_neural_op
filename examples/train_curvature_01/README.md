# GNP Curvature Estimation: Training Example

We show a standalone example of how to train a **Geometric Neural
Operator (PatchGNP)** from scratch to estimate surface curvatures on 3D point clouds.

## Overview

The script `train_curvature_estimator.py`:

1. **Generates synthetic surfaces** -- analytically extracted mean and Gaussian curvatures as ground truth (GT).

   **Example surfaces:** -- (i) unit sphere, (ii) torus, (iii) paraboloid.
2. **Trains a `PatchGNP`** model end-to-end using MSE loss on both mean and gaussian curvature types. 

    **Remark:** The model learns Legendre polynomial coefficients for local surface patches; 
    curvatures are computed differentiably from those coefficients via the standard first and 
    second fundamental forms.

3. **Evaluates the trained model** on each test surface to obtain the mean absolute errors of the curvature estimates.

4. **Saves matplotlib plots** to `./plots/`:
   - `loss_curve.png` -- training loss (log scale) vs epoch
   - `mean_curvature_scatter.png` -- predicted vs ground-truth mean curvature
     scatter (R², MAE)
   - `gaussian_curvature_scatter.png` -- same for Gaussian curvature
   - `3d_{surface}.png` -- 3D point cloud colored by ground truth (GT) and predicted
     curvatures (2×2 panel)
   - `dist_{surface}.png` -- histogram comparing predicted and GT curvature
     distributions

### Surfaces and analytical curvatures

| Surface | Mean curvature H | Gaussian curvature K |
|---------|-----------------|---------------------|
| Sphere (radius R=1) | H = −1/R (constant) | K = 1/R² (constant) |
| Torus (R=1, r=0.4) | varies with tube angle | varies with tube angle |
| Paraboloid z = x²+y² | varies with position | varies with position |

The sign convention for H follows the **outward-normal** convention (positive
curvature = surface bends inward relative to the outward normal).

---

## Running

**From the repository root:**
```bash
python examples/train_curvature_01/train_curvature_estimator.py
```

**From inside `examples/train_curvature_01/`:**
```bash
cd examples/train_curvature_01
python train_curvature_estimator.py
```

---

## Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 30 | Number of training epochs |
| `--lr` | 2e-5 | Adam learning rate |
| `--num_patches` | 256 | Training patches sampled per surface per step |
| `--k` | 50 | k-NN neighbours per patch |
| `--num_points` | 15000 | Points per synthetic surface |
| `--device` | auto | Force `cuda` or `cpu` |
| `--seed` | 42 | Random seed |
| `--loss_weight_mean` | 1.0 | MSE weight for mean curvature |
| `--loss_weight_gauss` | 0.5 | MSE weight for Gaussian curvature |

Check the codes for the latest values for the defaults. 

### Quick test run (~30 seconds on CPU):
```bash
python examples/train_curvature_01/train_curvature_estimator.py \
    --epochs 3 --num_points 500 --num_patches 64
```

### Full training run (~30-50 minutes on CPU):
```bash
python examples/train_curvature_01/train_curvature_estimator.py --epochs 10000
```

### GPU run:
```bash
python examples/train_curvature_01/train_curvature_estimator.py --epochs 10000 --device cuda
```

---

## Additional notes

- **Training mode vs test mode**: `PatchTensor(mode='train')` randomly samples
  patch centers each call (data augmentation), while `mode='test'` uses a
greedy covering strategy for complete, non-overlapping coverage. The `Surface`
class is only used during evaluation (test mode), where cluster assignments are
meaningful.
- **3D plots**: Color patterns in the predicted curvature should visually
  match the ground-truth panels, especially the high/low curvature regions of
the torus (inner vs. outer tube).
- **Curvature computation during training**: Uses a direct Legendre-basis
  computation on patch-indexed points, bypassing the `Surface` class (which
requires proper cluster assignments unavailable in train mode).
- **Random rotation augmentation**: Each training step applies a random 3D
  rotation to the surface, demonstrates the model is orientation-invariant.

