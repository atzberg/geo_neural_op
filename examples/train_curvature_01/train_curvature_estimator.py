"""
Train a Geometric Neural Operator (PatchGNP) from scratch to estimate 
from a point cloud surface the mean curvatures and Gaussian curvatures.

Training surfaces with analytically known curvatures: (i) unit sphere, (ii) 
torus, and (iii) paraboloid. The model learns to predict Legendre 
polynomial representations for local surface patches, from which 
curvatures are computed differentiably.

Usage (from repo root):
    python examples/train_curvature_01/train_curvature_estimator.py

Usage (from examples/train_curvature_01/):
    python train_curvature_estimator.py

Quick test run:
    python train_curvature_estimator.py --epochs 3 --num_points 500 --num_patches 64
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # allows for plots without a display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as torch_F
from mpl_toolkits.mplot3d import Axes3D  # set up for 3D projection

# Developed with assistance of AI Anthropic Claude Code.

# -- Path setup
# Works whether the script is run from the repo root or from examples/train_curvature_01/.
base_dir = Path(__file__).resolve().parent # always examples/train_curvature_01/
repo_dir = base_dir.parent.parent  # always the repo root
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

from gnp.dataset.patch import PatchTensor
from gnp.geometry.legendre import Legendre2D
from gnp.geometry.surface import Surface
from gnp.models.gnp import PatchGNP 

# -- Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PatchGNP to estimate surface curvatures from scratch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=10000,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Adam learning rate")
    parser.add_argument("--save_skip", type=int, default=500,
                        help="How often to save results (-1 indicates only final)")
    parser.add_argument("--num_patches", type=int, default=256,
                        help="Training patches sampled per surface per step")
    parser.add_argument("--k", type=int, default=50,
                        help="k-NN neighbours for PatchTensor")
    parser.add_argument("--num_points", type=int, default=15000,
                        help="Points per synthetic training/test surface")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: 'cuda' or 'cpu'. Default: auto-detect.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--loss_weight_mean", type=float, default=1.0,
                        help="MSE weight for mean curvature loss")
    parser.add_argument("--loss_weight_gauss", type=float, default=0.5,
                        help="MSE weight for Gaussian curvature loss")
    return parser.parse_args()

# -- Synthetic surface generation
def generate_sphere(num_points: int, radius: float = 1.0, device: str = "cpu",
                    rng=None) -> dict:
    """
    Unit sphere sampled uniformly via the normal-vector method.

    Analytical curvatures (outward-normal convention):
        H = -1/R   (surface bends away from the outward normal)
        K =  1/R²
    """
    if rng is None:
        rng = np.random.default_rng()
    raw = rng.standard_normal((num_points, 3)).astype(np.float32)
    xyz = raw / np.linalg.norm(raw, axis=1, keepdims=True) * radius
    normals = xyz / radius  # outward unit normal

    mean_curv = np.full(num_points, -1.0 / radius, dtype=np.float32)
    gauss_curv = np.full(num_points, 1.0 / radius ** 2, dtype=np.float32)

    results = {
        "xyz": torch.from_numpy(xyz).to(device),
        "normals": torch.from_numpy(normals).to(device),
        "mean_curvature": torch.from_numpy(mean_curv).to(device),
        "gaussian_curvature": torch.from_numpy(gauss_curv).to(device),
        "name": f"sphere_R{radius}",
    }

    return results


def generate_torus(num_points: int, R: float = 1.0, r: float = 0.4,
                   device: str = "cpu", rng=None) -> dict:
    """
    Torus with major radius R and tube radius r (r < R).
    Points sampled uniformly on the surface via rejection sampling.

    Analytical curvatures (outward-normal convention):
        H = -(R + 2r·cos v) / (2r·(R + r·cos v))
        K =   cos v         / (r·(R + r·cos v))
    """
    if rng is None:
        rng = np.random.default_rng()
    assert r < R, "Tube radius r must be less than major radius R."

    accepted_u, accepted_v = [], []
    max_w = R + r
    while len(accepted_u) < num_points:
        n = max(num_points * 6, 20_000)
        u_c = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        v_c = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        accept = rng.uniform(0, 1, n).astype(np.float32) < (R + r * np.cos(v_c)) / max_w
        accepted_u.extend(u_c[accept])
        accepted_v.extend(v_c[accept])

    u = np.asarray(accepted_u[:num_points], dtype=np.float32)
    v = np.asarray(accepted_v[:num_points], dtype=np.float32)

    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    xyz = np.stack([x, y, z], axis=1)

    nx = np.cos(u) * np.cos(v)
    ny = np.sin(u) * np.cos(v)
    nz = np.sin(v)
    normals = np.stack([nx, ny, nz], axis=1)

    denom = r * (R + r * np.cos(v))
    mean_curv = -(R + 2 * r * np.cos(v)) / (2 * denom)
    gauss_curv = np.cos(v) / denom

    results = {
        "xyz": torch.from_numpy(xyz).to(device),
        "normals": torch.from_numpy(normals).to(device),
        "mean_curvature": torch.from_numpy(mean_curv).to(device),
        "gaussian_curvature": torch.from_numpy(gauss_curv).to(device),
        "name": f"torus_R{R}_r{r}",
    }

    return results


def generate_paraboloid(num_points: int, a: float = 1.0, b: float = 1.0,
                         xy_range: float = 0.8, device: str = "cpu",
                         rng=None) -> dict:
    """
    Paraboloid  z = a·x² + b·y²  sampled uniformly in (x,y) ∈ [-xy_range, xy_range]².
    Normal points upward (into the bowl, positive-z component).

    Analytical curvatures:
        H = (a(1+4b²y²) + b(1+4a²x²)) / (1+4a²x²+4b²y²)^(3/2)
        K = 4ab / (1+4a²x²+4b²y²)²
    """
    if rng is None:
        rng = np.random.default_rng()
    xy = rng.uniform(-xy_range, xy_range, (num_points, 2)).astype(np.float32)
    x, y = xy[:, 0], xy[:, 1]
    z = a * x ** 2 + b * y ** 2
    xyz = np.stack([x, y, z], axis=1)

    raw_n = np.stack([-2 * a * x, -2 * b * y, np.ones_like(x)], axis=1)
    normals = raw_n / np.linalg.norm(raw_n, axis=1, keepdims=True)

    denom_sq = 1 + 4 * a ** 2 * x ** 2 + 4 * b ** 2 * y ** 2
    mean_curv = (a * (1 + 4 * b ** 2 * y ** 2) + b * (1 + 4 * a ** 2 * x ** 2)) / denom_sq ** 1.5
    gauss_curv = 4 * a * b / denom_sq ** 2

    results = {
        "xyz": torch.from_numpy(xyz).to(device),
        "normals": torch.from_numpy(normals).to(device),
        "mean_curvature": torch.from_numpy(mean_curv).to(device),
        "gaussian_curvature": torch.from_numpy(gauss_curv).to(device),
        "name": f"paraboloid_a{a}_b{b}",
    }

    return results


# -- Data augmentation
def random_rotation_matrix(rng) -> np.ndarray:
    """Uniformly random 3×3 rotation matrix via QR decomposition."""
    M = rng.standard_normal((3, 3)).astype(np.float32)
    Q, _ = np.linalg.qr(M)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def apply_rotation(surface_data: dict, R: np.ndarray) -> dict:
    """Rotate xyz and normals; curvature scalars are rotation-invariant."""
    dev = surface_data["xyz"].device
    Rt = torch.from_numpy(R).to(dev)
    rotated = dict(surface_data)
    rotated["xyz"] = surface_data["xyz"] @ Rt.T
    rotated["normals"] = surface_data["normals"] @ Rt.T
    return rotated



# -- Model construction
def build_model(device: str) -> PatchGNP:
    """
    Smaller PatchGNP for efficient training from scratch

    Architecture:
        layers=[32,32,32,32], neurons=128, num_channels=4, out_dim=16
        (out_dim=16 corresponds to (basis_degree+1)² = 4² Legendre coefficients)
    """
    model = PatchGNP(
        node_dim=3,
        out_dim=16,
        layers=[32, 32, 32, 32],
        num_channels=4,
        neurons=128,
        nonlinearity="ReLU",
        device=device,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  PatchGNP -- trainable parameters: {n_params:,}")
    return model



# -- Curvature computation from patch data (training mode)
def compute_patch_curvatures(
    patch_data, coefficients: torch.Tensor, basis: Legendre2D
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and Gaussian curvatures for all points inside the training patches.

    This mirrors Surface._compute_geometry but uses patch-indexed data
    (patch_indices / patch_number) rather than the full-cloud cluster assignment,
    which is always zero in train mode and therefore unsuitable for the Surface class.

    Parameters
    ----------
    patch_data : PatchData
        Output of PatchTensor.as_patch_data() in 'train' mode.
    coefficients : torch.Tensor
        Model output, shape (num_patches, 16).
    basis : Legendre2D
        Legendre basis with degree=3.

    Returns
    -------
    (mean_curvature, gaussian_curvature) each of shape (M,),
    where M = len(patch_data.patch_indices).
    """
    batch = patch_data.patch_number           # (M,)
    uv = patch_data.local_coordinates[:, :2]  # (M, 2)
    xy_scale = patch_data.xy_scale[batch]     # (M, 1)
    z_scale = patch_data.z_scale              # (num_patches, 1)

    # Scale coefficients by z-axis scale (matches Surface._compute_geometry)
    scaled_coeffs = z_scale * coefficients    # (num_patches, 16)

    # Derivatives in normalised local coordinates → shape (M, 5)
    # Order: [h_u, h_v, h_uv, h_uu, h_vv]
    raw_deriv = basis.derivatives_from_coeffs(uv, scaled_coeffs[batch])

    # Convert to physical (unscaled) derivatives
    # First-order: divide by xy_scale; second-order: divide by xy_scale²
    deriv_scale = torch.cat(
        (xy_scale.repeat(1, 2), xy_scale.pow(2).repeat(1, 3)), dim=1
    ).reciprocal()
    h_u, h_v, h_uv, h_uu, h_vv = torch.split(raw_deriv * deriv_scale, 1, dim=1)

    # First fundamental form  g = [[E, F], [F, G]]
    E = 1.0 + h_u.pow(2)
    F_val = h_u * h_v
    G = 1.0 + h_v.pow(2)
    det_g = E * G - F_val.pow(2)

    # Second fundamental form  shape = [[L, M], [M, N]] / sqrt(1+|grad h|²)
    denom = (1.0 + h_u.pow(2) + h_v.pow(2)).sqrt()
    L = h_uu / denom
    M_val = h_uv / denom
    N_val = h_vv / denom

    # Weingarten map  W = g⁻¹ @ shape  (computed element-wise)
    W11 = (G * L - F_val * M_val) / det_g
    W22 = (E * N_val - F_val * M_val) / det_g
    W12 = (G * M_val - F_val * N_val) / det_g
    W21 = (E * M_val - F_val * L) / det_g

    mean_curv = 0.5 * (W11 + W22)          # (M, 1)
    gauss_curv = W11 * W22 - W12 * W21     # (M, 1)

    return mean_curv.squeeze(1), gauss_curv.squeeze(1)



# -- Training loop
def train(args, model: PatchGNP, training_surfaces: list, device: str,
          basis: Legendre2D, flag_init: bool = True, t_context: dict = None, 
          num_epochs: int = -1) -> tuple[list[float], dict]:

    if flag_init:

        if t_context is None:
          t_context = {}

        rng = tuple(map(t_context.get,['rng']))
        if rng is None: 
            rng = np.random.default_rng(args.seed + 1)
            t_context.update({'rng':rng})

        """Train model and return per-epoch average loss."""
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5
        )

        t_context.update({
           'epoch':0,
           'optimizer':optimizer,
           'scheduler':scheduler,
        })

    epoch_losses = []
    epoch,rng,optimizer,scheduler = tuple(map(t_context.get,['epoch','rng','optimizer','scheduler'])) 

    if num_epochs == -1:
        num_epochs = args.epochs

    disp_skip = 10 

    model.train()
    for ii in range(1, num_epochs + 1):
        epoch_loss = 0.0
        n_steps = 0

        # Shuffle surface order for each epoch
        order = list(range(len(training_surfaces)))
        rng.shuffle(order)

        for idx in order:
            surf = training_surfaces[idx]
            rotated = apply_rotation(surf, random_rotation_matrix(rng))

            # Build fresh patches (mode='train' resamples centres each call)
            patch_tensor = PatchTensor(
                data={"x": rotated["xyz"], "normals": rotated["normals"]},
                k=args.k,
                mode="train",
                num_training_patches=args.num_patches,
                basis_degree=3,
                device=device,
            )
            patch_data = patch_tensor.as_patch_data()

            optimizer.zero_grad()

            # Forward pass -- no torch.no_grad(), gradients required
            coefficients = model(
                patch_data.local_coordinates,
                patch_data.patch_number,
            )

            # Curvatures at patch points (differentiable)
            pred_mean, pred_gauss = compute_patch_curvatures(
                patch_data, coefficients, basis
            )

            # Ground truth indexed by patch_indices (global point indices in patches)
            gt_mean = rotated["mean_curvature"][patch_data.patch_indices]
            gt_gauss = rotated["gaussian_curvature"][patch_data.patch_indices]

            loss = (
                args.loss_weight_mean * torch_F.mse_loss(pred_mean, gt_mean)
                + args.loss_weight_gauss * torch_F.mse_loss(pred_gauss, gt_gauss)
            )

            if not torch.isfinite(loss):
                print(f"  [epoch {epoch}] Non-finite loss on '{surf['name']}', skipping step.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1

        scheduler.step()
        avg = epoch_loss / max(n_steps, 1)
        epoch_losses.append(avg)
        epoch += 1; 

        if epoch % disp_skip == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg:.6f}  lr={lr_now:.2e}")

    t_context.update({
        'epoch':epoch,
        'optimizer':optimizer,
        'scheduler':scheduler,
    })
    return epoch_losses, t_context  


# -- Evaluation
def evaluate(model: PatchGNP, test_surfaces: list, args, device: str) -> list[dict]:
    """
    Evaluate in test mode (greedy cover patches, deterministic assignment).

    Uses the Surface class -- valid because mode='test' produces correct
    cluster assignments for all points.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for surf in test_surfaces:
            patch_tensor = PatchTensor(
                data={"x": surf["xyz"], "normals": surf["normals"]},
                k=args.k,
                mode="test",
                basis_degree=3,
                device=device,
            )
            patch_data = patch_tensor.as_patch_data()

            # Batch-iterate to avoid OOM on large clouds
            coeffs_list = []
            for pd_batch in patch_data.batch_iterator(512):
                c = model(pd_batch.local_coordinates, pd_batch.patch_number)
                coeffs_list.append(c)
            coefficients = torch.cat(coeffs_list, dim=0)

            # Surface class works correctly in test mode (proper cluster assignments)
            surface = Surface(patch_data, coefficients)

            xyz_np = surf["xyz"].cpu().numpy()
            results.append({
                "name": surf["name"],
                "xyz": xyz_np,
                "pred_mean": surface.mean_curvature.cpu().numpy(),
                "pred_gauss": surface.gaussian_curvature.cpu().numpy(),
                "gt_mean": surf["mean_curvature"].cpu().numpy(),
                "gt_gauss": surf["gaussian_curvature"].cpu().numpy(),
            })

    return results


# -- Plotting
def plot_loss_curve(epoch_losses: list, base_plot_dir: Path, cur_epoch: int = 0):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = list(range(1, len(epoch_losses) + 1))
    ax.plot(epochs, epoch_losses, linewidth=2, color="steelblue",
            marker="o", markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("MSE Loss (log scale)", fontsize=13)
    ax.set_title("Training Loss -- PatchGNP Curvature Estimator", fontsize=14)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = base_plot_dir / f"loss_curve_{cur_epoch:07}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def _scatter_panel(ax, gt, pred, title, xlabel, ylabel):
    """Draw a single predicted-vs-ground-truth scatter panel."""
    # Clip to 5th–95th percentile for display only
    lo, hi = np.nanpercentile(gt, 5), np.nanpercentile(gt, 95)
    mask = np.isfinite(pred) & np.isfinite(gt) & (gt >= lo) & (gt <= hi)
    gt_p, pr_p = gt[mask], pred[mask]

    ax.scatter(gt_p, pr_p, s=3, alpha=0.35, color="steelblue", rasterized=True)

    #lims = [min(gt_p.min(), pr_p.min()), max(gt_p.max(), pr_p.max())]
    #ax.plot(lims, lims, "r--", linewidth=1.5, label="y = x")
    lims = [-2.0, 2.0]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="y = x")

    ss_res = np.sum((pr_p - gt_p) ** 2)
    ss_tot = np.sum((gt_p - gt_p.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    mae = np.mean(np.abs(pr_p - gt_p))

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{title}\n$R^2$={r2:.3f}  MAE={mae:.4f}", fontsize=10)
    ax.legend(fontsize=8)


def plot_curvature_scatter(results: list, base_plot_dir: Path, cur_epoch: int = 0):
    """One figure each for mean curvature and Gaussian curvature."""
    specs = [
        ("Mean Curvature", "gt_mean", "pred_mean",
        f"mean_curvature_scatter_{cur_epoch:07}.png"), 
        ("Gaussian Curvature", "gt_gauss", "pred_gauss",
        f"gaussian_curvature_scatter_{cur_epoch:07}.png"),
    ]
    for label, gt_key, pred_key, fname in specs:
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, res in zip(axes, results):
            ax.set_xlim(-2.0, 2.0);
            ax.set_ylim(-2.0, 2.0);
            _scatter_panel(
                ax,
                gt=res[gt_key],
                pred=res[pred_key],
                title=res["name"],
                xlabel=f"Ground truth {label}",
                ylabel=f"Predicted {label}",
            )
            
        fig.suptitle(f"{label}: Predicted vs Ground Truth", fontsize=13)
        fig.tight_layout()
        out = base_plot_dir / fname
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out.name}")


def plot_3d_surfaces(results: list, base_plot_dir: Path, cur_epoch: int = 0):
    """
    For each test surface, produce a 2×2 figure:
        [GT mean curv]  [Predicted mean curv]
        [GT Gauss curv] [Predicted Gauss curv]
    """
    for res in results:
        xyz = res["xyz"]
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        specs = [
            ("GT Mean Curv",   res["gt_mean"],   "coolwarm", 1),
            ("Pred Mean Curv", res["pred_mean"],  "coolwarm", 2),
            ("GT Gauss Curv",  res["gt_gauss"],  "RdYlBu",  3),
            ("Pred Gauss Curv",res["pred_gauss"], "RdYlBu",  4),
        ]

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(res["name"], fontsize=14, fontweight="bold")
  
        ii = 0 
        for sub_title, values, cmap, pos in specs:
            ax = fig.add_subplot(2, 2, pos, projection="3d")
            if ii % 2 == 0:
                flag_GT = True
            else:
                flag_GT = False
            if flag_GT: # keep range the same for GT and Pred
              vmin = np.nanpercentile(values, 5)
              vmax = np.nanpercentile(values, 95)
            clipped = np.clip(values, vmin, vmax)
            sc = ax.scatter(x, y, z, c=clipped, cmap=cmap, s=3,
                            vmin=vmin, vmax=vmax, rasterized=True)
            fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.1)
            ax.set_title(sub_title, fontsize=10)
            ax.set_xlabel("X", fontsize=8)
            ax.set_ylabel("Y", fontsize=8)
            ax.set_zlabel("Z", fontsize=8)
            ax.tick_params(labelsize=7)
            ii += 1

        fig.tight_layout()
        safe = res["name"].replace(".", "p").replace("/", "_")
        out = base_plot_dir / f"3d_{safe}_{cur_epoch:07}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  Saved: {out.name}")


def plot_curvature_distributions(results: list, base_plot_dir: Path, cur_epoch: int = 0):
    """
    Histogram comparing predicted and ground-truth curvature distributions
    for each surface. Useful for checking scale and sign alignment.
    """
    for res in results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{res['name']} -- Curvature Distributions", fontsize=13)

        for ax, gt_key, pred_key, label in [
            (axes[0], "gt_mean", "pred_mean", "Mean Curvature"),
            (axes[1], "gt_gauss", "pred_gauss", "Gaussian Curvature"),
        ]:
            gt = res[gt_key]
            pred = res[pred_key]
            lo = np.nanpercentile(gt, 2)
            hi = np.nanpercentile(gt, 98)
            if np.abs(hi - lo) < 1e-3: # if too close together
              lo = -2.0; hi = 2.0; 
            bins = np.linspace(lo, hi, 60)
            ax.hist(gt[np.isfinite(gt)], bins=bins, alpha=0.55,
                    label="Ground truth", color="steelblue", density=True)
            ax.hist(pred[np.isfinite(pred)], bins=bins, alpha=0.55,
                    label="Predicted", color="darkorange", density=True)
            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.legend(fontsize=9)

        fig.tight_layout()
        safe = res["name"].replace(".", "p").replace("/", "_")
        out = base_plot_dir / f"dist_{safe}_{cur_epoch:07}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out.name}")


# -- Main
def main():
    args = parse_args()

    # Save seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("GNP Surface Curvature Training Example")
    print(f"{'='*60}")
    print(f"Device : {device}")
    print(f"Epochs : {args.epochs}  |  LR : {args.lr}")
    print(f"Points per surface : {args.num_points}")
    print(f"Training patches   : {args.num_patches}")
    print(f"Save Skip          : {args.save_skip}")

    base_plot_dir = base_dir / "plots"
    base_plot_dir.mkdir(exist_ok=True)

    # ── Generate surfaces ──────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    print(f"\nGenerating synthetic surfaces ({args.num_points} points each) …")

    sphere = generate_sphere(args.num_points, radius=1.0, device=device, rng=rng)
    torus = generate_torus(args.num_points, R=1.0, r=0.4, device=device, rng=rng)
    paraboloid = generate_paraboloid(args.num_points, a=1.0, b=1.0, device=device, rng=rng)
    training_surfaces = [sphere, torus, paraboloid]

    # Quick sanity print on analytical curvatures
    for surf in training_surfaces:
        h_vals = surf["mean_curvature"].cpu().numpy()
        k_vals = surf["gaussian_curvature"].cpu().numpy()
        print(f"  {surf['name']:28s}  "
              f"H ∈ [{h_vals.min():.3f}, {h_vals.max():.3f}]  "
              f"K ∈ [{k_vals.min():.3f}, {k_vals.max():.3f}]")

    # -- Build model
    print("\nBuilding PatchGNP model …")
    model = build_model(device)
    basis = Legendre2D(degree=3)

    # -- Train by iterating over training and evaluating model 
    cur_epoch = 0 
    if args.save_skip == -1: # default
        args.save_skip = args.epochs 

    num_iter = np.ceil(args.epochs/args.save_skip).astype(int)

    epoch_losses = []
    for iter in range(0,num_iter):

        if iter == 0:
          flag_init = True
          t_context = {'rng':rng};
        else:
          flag_init = False 

        num_epochs = min(args.save_skip,args.epochs - cur_epoch)

        # -- Train
        print(f"\nTraining for another {num_epochs} epochs …")
        epoch_losses_iter, t_context = train(args, model, training_surfaces, 
                                        device, basis, flag_init, t_context,
                                        num_epochs)
        cur_epoch = t_context['epoch']
        epoch_losses += epoch_losses_iter

        # -- Evaluate
        print("\nEvaluating on test surfaces (mode='test', greedy cover) …")
        results = evaluate(model, training_surfaces, args, device)

        print("\nTest metrics:")
        for res in results:
            mask = np.isfinite(res["pred_mean"]) & np.isfinite(res["gt_mean"])
            mae_m = np.mean(np.abs(res["pred_mean"][mask] - res["gt_mean"][mask]))
            mask_g = np.isfinite(res["pred_gauss"]) & np.isfinite(res["gt_gauss"])
            mae_g = np.mean(np.abs(res["pred_gauss"][mask_g] - res["gt_gauss"][mask_g]))
            print(f"  {res['name']:30s}  mean_curv MAE={mae_m:.4f}  gauss_curv MAE={mae_g:.4f}")

        # -- Plots 
        print(f"\nSaving plots to {base_plot_dir} …")
        plot_loss_curve(epoch_losses, base_plot_dir, cur_epoch)
        plot_curvature_scatter(results, base_plot_dir, cur_epoch)
        plot_3d_surfaces(results, base_plot_dir, cur_epoch)
        plot_curvature_distributions(results, base_plot_dir, cur_epoch)

    print(f"\n{'='*60}")
    print("Done. All plots saved to examples/train_curvature_01/plots")
    print(f"{'='*60}\n")


# -- Script entry function
if __name__ == "__main__":
    main()
