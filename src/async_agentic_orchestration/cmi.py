from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np


def conditional_mutual_information(x: Iterable[int], y: Iterable[int], z: Iterable[int], alpha: float = 0.0) -> float:
    """Plug-in estimate of I(X;Y|Z) in nats for finite integer-valued variables.

    This estimator is intentionally simple and inherits the small-sample bias discussed
    in the repository. It is a diagnostic, not the default pruning mechanism.
    """
    x = np.asarray(list(x), dtype=int)
    y = np.asarray(list(y), dtype=int)
    z = np.asarray(list(z), dtype=int)
    if not (len(x) == len(y) == len(z)) or len(x) == 0:
        raise ValueError("x, y, z must have equal nonzero length")
    nx, ny, nz = x.max() + 1, y.max() + 1, z.max() + 1
    xyz = np.full((nx, ny, nz), alpha, dtype=float)
    for a, b, c in zip(x, y, z):
        xyz[a, b, c] += 1.0
    pxyz = xyz / xyz.sum()
    pxz = pxyz.sum(axis=1)
    pyz = pxyz.sum(axis=0)
    pz = pxyz.sum(axis=(0, 1))
    out = 0.0
    for a in range(nx):
        for b in range(ny):
            for c in range(nz):
                p = pxyz[a, b, c]
                if p <= 0 or pxz[a, c] <= 0 or pyz[b, c] <= 0 or pz[c] <= 0:
                    continue
                out += p * np.log((p * pz[c]) / (pxz[a, c] * pyz[b, c]))
    return float(out)


def permutation_cmi_threshold(
    x: Iterable[int], y: Iterable[int], z: Iterable[int], permutations: int = 200, quantile: float = 0.95, seed: int = 0
) -> Tuple[float, float]:
    """Return (observed CMI, permutation-null threshold)."""
    x = np.asarray(list(x), dtype=int)
    y = np.asarray(list(y), dtype=int)
    z = np.asarray(list(z), dtype=int)
    obs = conditional_mutual_information(x, y, z)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for k in range(permutations):
        null[k] = conditional_mutual_information(rng.permutation(x), y, z)
    return obs, float(np.quantile(null, quantile))
