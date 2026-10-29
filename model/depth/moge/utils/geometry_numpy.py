from typing import *
from functools import partial
import math

import numpy as np


def weighted_mean_numpy(x: np.ndarray, w: np.ndarray = None, axis: Union[int, Tuple[int,...]] = None, keepdims: bool = False, eps: float = 1e-7) -> np.ndarray:
    if w is None:
        return np.mean(x, axis=axis)
    else:
        w = w.astype(x.dtype)
        return (x * w).mean(axis=axis) / np.clip(w.mean(axis=axis), eps, None)


_FOCAL_FIT_EPS = 1e-6


def _finite_focal_fit_inputs(
    uv: np.ndarray,
    xyz: np.ndarray,
    eps: float = _FOCAL_FIT_EPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    uv = uv.reshape(-1, 2)
    xy = xyz[..., :2].reshape(-1, 2)
    z = xyz[..., 2].reshape(-1)
    valid = (
        np.isfinite(uv).all(axis=-1)
        & np.isfinite(xy).all(axis=-1)
        & np.isfinite(z)
        & (np.abs(z) > eps)
    )
    return uv[valid], xy[valid], z[valid]


def _project_xy_finite(
    xy: np.ndarray,
    z: np.ndarray,
    shift: np.ndarray,
    eps: float = _FOCAL_FIT_EPS,
) -> np.ndarray:
    """Preserve the exact projection unless a singular value needs protection."""
    denominator = z + shift
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        xy_projected = xy / denominator[:, None]
    if np.isfinite(xy_projected).all():
        return xy_projected

    sign = np.where(np.signbit(denominator), -1.0, 1.0)
    safe_denominator = np.where(
        np.isfinite(denominator) & (np.abs(denominator) > eps),
        denominator,
        sign * eps,
    )
    return np.nan_to_num(
        xy / safe_denominator[:, None],
        nan=0.0,
        posinf=1.0 / eps,
        neginf=-1.0 / eps,
    )


def _focal_from_projection(
    xy_projected: np.ndarray,
    uv: np.ndarray,
    eps: float = _FOCAL_FIT_EPS,
) -> float:
    numerator = float((xy_projected * uv).sum())
    denominator = float(np.square(xy_projected).sum())
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= eps:
        return 1.0
    focal = numerator / denominator
    return float(focal) if math.isfinite(focal) else 1.0


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal"
    from scipy.optimize import least_squares
    uv, xy, z = _finite_focal_fit_inputs(uv, xyz)
    if uv.shape[0] < 2:
        return np.float32(0.0), np.float32(1.0)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = _project_xy_finite(xy, z, shift)
        focal = _focal_from_projection(xy_proj, uv)
        residual = (focal * xy_proj - uv).ravel()
        return np.nan_to_num(
            residual,
            nan=0.0,
            posinf=1.0 / _FOCAL_FIT_EPS,
            neginf=-1.0 / _FOCAL_FIT_EPS,
        )

    try:
        solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
        optim_shift = np.asarray(solution['x']).squeeze().astype(np.float32)
    except ValueError:
        optim_shift = np.float32(0.0)
    if not np.isfinite(optim_shift):
        optim_shift = np.float32(0.0)

    xy_proj = _project_xy_finite(xy, z, np.asarray(optim_shift))
    optim_focal = np.float32(_focal_from_projection(xy_proj, uv))

    return optim_shift, optim_focal


def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift"
    from scipy.optimize import least_squares
    uv, xy, z = _finite_focal_fit_inputs(uv, xyz)
    if uv.shape[0] < 2 or not np.isfinite(focal):
        return np.float32(0.0)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = _project_xy_finite(xy, z, shift)
        residual = (focal * xy_proj - uv).ravel()
        return np.nan_to_num(
            residual,
            nan=0.0,
            posinf=1.0 / _FOCAL_FIT_EPS,
            neginf=-1.0 / _FOCAL_FIT_EPS,
        )

    try:
        solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
        optim_shift = np.asarray(solution['x']).squeeze().astype(np.float32)
    except ValueError:
        optim_shift = np.float32(0.0)
    if not np.isfinite(optim_shift):
        optim_shift = np.float32(0.0)

    return optim_shift


def norm3d(x: np.ndarray) -> np.ndarray:
    "Faster `np.linalg.norm(x, axis=-1)` for 3D vectors"
    return np.sqrt(np.square(x[..., 0]) + np.square(x[..., 1]) + np.square(x[..., 2]))


