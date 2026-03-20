import torch
import numpy as np
import cv2
from functools import lru_cache
import matplotlib.pyplot as plt
import os

def _nonzero_sign(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def _safe_norm_dim0(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(x.pow(2).sum(dim=0, keepdim=True) + eps)


def _compute_tensor_dd_duv(tensor: torch.Tensor) -> torch.Tensor:
    """Compute finite-difference gradients du and dv of a [1, H, W] tensor."""
    assert tensor.ndim == 3, tensor.shape
    dd_du = torch.zeros_like(tensor)
    dd_dv = torch.zeros_like(tensor)

    dd_du[:, :, 1:-1] = tensor[:, :, 2:] - tensor[:, :, :-2]
    dd_dv[:, 1:-1, :] = tensor[:, 2:, :] - tensor[:, :-2, :]

    return torch.cat([dd_du, dd_dv], dim=0) / 2


def _cosine_dudv_loss(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    t1 = tensor1 / _safe_norm_dim0(tensor1, eps=1e-8)
    t2 = tensor2 / _safe_norm_dim0(tensor2, eps=1e-8)
    cos = (t1 * t2).sum(dim=0)
    cos = torch.nan_to_num(cos, nan=0.0, posinf=0.0, neginf=0.0)
    return (1 - cos).mean()


@lru_cache(maxsize=None)
def _create_grid(width, height, device, dtype=torch.float):
    yy, xx = torch.meshgrid(
        torch.arange(0, height, device=device, dtype=dtype),
        torch.arange(0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=0) + 0.5


def _compute_dd_duv(normal, depth, focal_x, focal_y, cx, cy):
    """Compute depth gradients du/dv from normals and camera intrinsics."""
    grid = _create_grid(
        width=depth.shape[2],
        height=depth.shape[1],
        device=normal.device,
        dtype=normal.dtype,
    )

    nz = normal[2:3]
    nz_safe = torch.abs(nz).clamp(min=1e-4) * _nonzero_sign(nz)
    Z = (
        1
        + (normal[0:1] * (grid[0:1] - cx) / nz_safe / focal_x)
        + (normal[1:2] * (grid[1:2] - cy) / nz_safe / focal_y)
    ) * nz_safe

    Z_safe = torch.abs(Z).clamp(min=1e-6) * _nonzero_sign(Z)
    dU_du = (-normal[0:1] * depth) / focal_x / Z_safe
    dU_dv = (-normal[1:2] * depth) / focal_y / Z_safe

    return torch.cat([dU_du, dU_dv], dim=0)


def disparity_normal_loss(
    render_depth: torch.Tensor,
    gt_disparity: torch.Tensor,
    weight: float = 0.1,
    l1_weight: float = 0.4,
    render_normals: torch.Tensor = None,
    surface_normals: torch.Tensor = None,
    focal_x: float = None,
    focal_y: float = None,
    cx: float = None,
    cy: float = None,
    guided_blur_radius: int = 800,
    guided_blur_eps: float = 1.0,
) -> torch.Tensor:

    render_depth = torch.nan_to_num(render_depth, nan=0.0, posinf=0.0, neginf=0.0).float()
    gt_disparity = torch.nan_to_num(gt_disparity, nan=0.0, posinf=0.0, neginf=0.0).float()

    render_disp = 1.0 / render_depth

    gt_disparity = gt_disparity * (render_disp.max() - render_disp.min()) + render_disp.min()

    d_gt_dudv = _compute_tensor_dd_duv(gt_disparity)
    d_render_dudv = _compute_tensor_dd_duv(render_depth)

    d_gt_norm = torch.sqrt(d_gt_dudv.pow(2).sum(dim=0) + 1e-8)
    d_render_norm = torch.sqrt(d_render_dudv.pow(2).sum(dim=0) + 1e-8)

    gt_safe = gt_disparity

    coefficients = d_gt_norm / (gt_safe.squeeze(0) ** 2 * d_render_norm + 1e-5)

    # Clamp 5% of outliers (bottom 2.5% and top 2.5%) 
    lower = torch.quantile(coefficients.float(), 0.025)
    upper = torch.quantile(coefficients.float(), 0.975)
    coefficients = coefficients.clamp(min=lower, max=upper)

    beta = 1.0 / gt_safe.squeeze(0) - coefficients * render_depth.squeeze(0)

    gt_guide = gt_safe[0].detach().cpu().numpy().astype(np.float32)

    coefficients = torch.from_numpy(
        cv2.ximgproc.guidedFilter(gt_guide, coefficients.detach().cpu().numpy().astype(np.float32), radius=guided_blur_radius, eps=guided_blur_eps, dDepth=-1)
    ).to(gt_safe.device)

    beta = torch.from_numpy(
        cv2.ximgproc.guidedFilter(gt_guide, beta.detach().cpu().numpy().astype(np.float32), radius=guided_blur_radius, eps=guided_blur_eps, dDepth=-1)
    ).to(gt_safe.device)

    d_gt_dudv_pred = -coefficients * gt_safe.squeeze(0) ** 2 * d_render_dudv

    d_gt_dudv_pred_norm = torch.sqrt(d_gt_dudv_pred.pow(2).sum(dim=0) + 1e-8)
    d_gt_dudv_norm = torch.sqrt(d_gt_dudv.pow(2).sum(dim=0) + 1e-8)

    # Clamp 5% of outliers (bottom 2.5% and top 2.5%) 
    lower = torch.quantile(d_gt_dudv_pred_norm.float(), 0.025)
    upper = torch.quantile(d_gt_dudv_pred_norm.float(), 0.975)
    d_gt_dudv_pred_norm = d_gt_dudv_pred_norm.clamp(min=lower, max=upper)

    lower = torch.quantile(d_gt_dudv_norm.float(), 0.025)
    upper = torch.quantile(d_gt_dudv_norm.float(), 0.975)
    d_gt_dudv_norm = d_gt_dudv_norm.clamp(min=lower, max=upper)

    render_depth_reconstructed = (1 - beta * gt_safe.squeeze(0)) / (coefficients * gt_safe.squeeze(0))

    # Align reconstructed depth to render depth using least squares
    with torch.no_grad():
        _fx = render_depth_reconstructed.flatten()
        _fy = render_depth.squeeze(0).flatten()
        _fA = torch.stack([_fx, torch.ones_like(_fx)], dim=1)
        _sol = torch.linalg.solve(_fA.T @ _fA, _fA.T @ _fy)
        _fa, _fb = _sol[0], _sol[1]
    render_depth_reconstructed = _fa * render_depth_reconstructed + _fb

    # L1 loss (in depth space)
    L1_depth_loss = (render_depth_reconstructed - render_depth.squeeze(0)).abs().mean()

    loss = (
        (d_gt_dudv_pred_norm - d_gt_dudv_norm).pow(2).mean()
        + _cosine_dudv_loss(
            tensor1=d_gt_dudv_pred,
            tensor2=d_gt_dudv,
        )
    )

    # Render normals branch
    if render_normals is not None:
        d_render_dudv_from_normal = _compute_dd_duv(
            normal=render_normals,
            depth=render_depth,
            focal_x=focal_x,
            focal_y=focal_y,
            cx=cx,
            cy=cy,
        )
        d_render_dudv_from_normal = torch.nan_to_num(
            d_render_dudv_from_normal, nan=1e-6
        )

        d_gt_dudv_pred2 = -coefficients * gt_safe.squeeze(0) ** 2 * d_render_dudv_from_normal

        d_gt_dudv_pred2_norm = torch.sqrt(
            d_gt_dudv_pred2.pow(2).sum(dim=0) + 1e-8
        )

        loss = (
            loss
            + ((d_gt_dudv_pred2_norm - d_gt_dudv_norm).pow(2).mean())
            + _cosine_dudv_loss(
                tensor1=d_gt_dudv_pred2,
                tensor2=d_gt_dudv,
            )
        )

    # Surface normals branch
    if surface_normals is not None:
        d_surface_dudv_from_normal = _compute_dd_duv(
            normal=surface_normals,
            depth=render_depth,
            focal_x=focal_x,
            focal_y=focal_y,
            cx=cx,
            cy=cy,
        )
        d_surface_dudv_from_normal = torch.nan_to_num(
            d_surface_dudv_from_normal, nan=1e-6
        )

        d_gt_dudv_pred3 = -coefficients * gt_safe.squeeze(0) ** 2 * d_surface_dudv_from_normal

        d_gt_dudv_pred3_norm = torch.sqrt(
            d_gt_dudv_pred3.pow(2).sum(dim=0) + 1e-8
        )

        loss = (
            loss
            + ((d_gt_dudv_pred3_norm - d_gt_dudv_norm).pow(2).mean())
            + _cosine_dudv_loss(
                tensor1=d_gt_dudv_pred3,
                tensor2=d_gt_dudv,
            )
        )

    return (loss * weight, L1_depth_loss * l1_weight)
