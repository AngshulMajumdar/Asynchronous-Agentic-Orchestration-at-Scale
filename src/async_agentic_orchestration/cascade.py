from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque
import math
import numpy as np


@dataclass
class CascadeStatus:
    n: int
    mean_offspring: float
    azuma_upper: float
    hard_cap_triggered: bool


class RollingCascadeDiagnostic:
    """Lagging empirical diagnostic for recent cascade reproduction.

    The Azuma-style upper radius is intentionally not treated as a clairvoyant safety
    certificate. A hard affected-set cap remains independent of the diagnostic.
    """

    def __init__(self, degree_bound: int, window: int = 10000, delta: float = 0.05, hard_cap: int = 512):
        if degree_bound < 1 or window < 1 or not (0 < delta < 1) or hard_cap < 1:
            raise ValueError("invalid cascade diagnostic parameters")
        self.d = int(degree_bound)
        self.window = int(window)
        self.delta = float(delta)
        self.hard_cap = int(hard_cap)
        self._x: Deque[int] = deque(maxlen=self.window)

    def update(self, offspring_count: int, affected_so_far: int = 0) -> CascadeStatus:
        x = int(offspring_count)
        if x < 0 or x > self.d:
            raise ValueError("offspring_count must be between 0 and degree_bound")
        self._x.append(x)
        n = len(self._x)
        mean = float(np.mean(self._x)) if n else 0.0
        width = self.d * math.sqrt(2.0 * math.log(1.0 / self.delta) / max(n, 1))
        return CascadeStatus(
            n=n,
            mean_offspring=mean,
            azuma_upper=mean + width,
            hard_cap_triggered=affected_so_far >= self.hard_cap,
        )
