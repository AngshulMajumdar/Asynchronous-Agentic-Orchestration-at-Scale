from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Hashable, Iterable, Optional, Sequence, Tuple
import math
import numpy as np


@dataclass
class _Record:
    counts: np.ndarray
    last_access: int


class SparseTransitionMemory:
    """Lazy finite-state transition statistics with forgetting and bounded lifetime.

    A key is created only when a context is observed. Empirical counts decay lazily
    when the key is accessed; the Bayesian pseudo-count prior remains fixed. An
    expiration deque makes horizon-based eviction amortized O(1) per local response.

    Parameters
    ----------
    n_states:
        Number of symbolic next states.
    rho:
        Exponential forgetting factor for empirical counts.
    alpha:
        Total prior pseudo-count mass.
    prior:
        Prior distribution over states. Uniform when omitted.
    horizon:
        Maximum number of *local responses* for which an untouched key remains live.
        ``None`` disables eviction.
    """

    def __init__(
        self,
        n_states: int,
        rho: float = 0.995,
        alpha: float = 1.0,
        prior: Optional[Sequence[float]] = None,
        horizon: Optional[int] = 64,
    ) -> None:
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        if not (0 < rho <= 1):
            raise ValueError("rho must lie in (0, 1]")
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        if horizon is not None and horizon < 1:
            raise ValueError("horizon must be positive or None")
        self.n_states = int(n_states)
        self.rho = float(rho)
        self.alpha = float(alpha)
        if prior is None:
            self.prior = np.full(self.n_states, 1.0 / self.n_states)
        else:
            p = np.asarray(prior, dtype=float)
            if p.shape != (self.n_states,) or np.any(p < 0) or p.sum() <= 0:
                raise ValueError("invalid prior")
            self.prior = p / p.sum()
        self.horizon = horizon
        self._clock = 0
        self._store: Dict[Hashable, _Record] = {}
        self._expiry: Deque[Tuple[int, Hashable]] = deque()

    @property
    def local_clock(self) -> int:
        return self._clock

    @property
    def live_keys(self) -> int:
        return len(self._store)

    @property
    def expiry_records(self) -> int:
        return len(self._expiry)

    def _expire(self) -> None:
        if self.horizon is None:
            return
        cutoff = self._clock - self.horizon
        while self._expiry and self._expiry[0][0] <= cutoff:
            stamp, key = self._expiry.popleft()
            rec = self._store.get(key)
            # Stale deque entries are harmless. Only the newest access owns the key.
            if rec is not None and rec.last_access == stamp:
                del self._store[key]

    def _decayed_counts(self, key: Hashable) -> np.ndarray:
        rec = self._store.get(key)
        if rec is None:
            return np.zeros(self.n_states, dtype=float)
        delta = self._clock - rec.last_access
        if delta > 0 and self.rho < 1:
            rec.counts *= self.rho ** delta
            rec.last_access = self._clock
        return rec.counts

    def predict(self, key: Hashable) -> np.ndarray:
        counts = self._decayed_counts(key).copy()
        denom = float(counts.sum() + self.alpha)
        if denom <= 0:
            return self.prior.copy()
        return (counts + self.alpha * self.prior) / denom

    def observe(self, key: Hashable, next_state: int) -> np.ndarray:
        """Advance one local response, return the pre-update predictive distribution."""
        if not 0 <= int(next_state) < self.n_states:
            raise ValueError("next_state out of range")
        self._clock += 1
        self._expire()

        rec = self._store.get(key)
        if rec is None:
            counts = np.zeros(self.n_states, dtype=float)
            rec = _Record(counts=counts, last_access=self._clock)
            self._store[key] = rec
        else:
            delta = self._clock - rec.last_access
            if delta > 0 and self.rho < 1:
                rec.counts *= self.rho ** delta
            rec.last_access = self._clock

        p = (rec.counts + self.alpha * self.prior) / (rec.counts.sum() + self.alpha)
        rec.counts[int(next_state)] += 1.0
        self._expiry.append((self._clock, key))
        return p.copy()

    def snapshot(self) -> Dict[Hashable, np.ndarray]:
        return {k: v.counts.copy() for k, v in self._store.items()}

    @staticmethod
    def horizon_for_tolerance(rho_bar: float, epsilon: float) -> int:
        """H such that the unobserved geometric tail is at most epsilon.

        Uses rho_bar^H / (1-rho_bar) <= epsilon. This is a design conversion, not
        an estimate of the appropriate forgetting rate for an application.
        """
        if not (0 < rho_bar < 1):
            raise ValueError("rho_bar must lie in (0,1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        rhs = epsilon * (1.0 - rho_bar)
        if rhs >= 1:
            return 0
        return max(0, math.ceil(math.log(rhs) / math.log(rho_bar)))


def fixed_time_bin(delta_t: float, edges: Sequence[float]) -> int:
    """Return a 0-based elapsed-time bin for monotonically increasing edges."""
    e = np.asarray(edges, dtype=float)
    if e.ndim != 1 or len(e) == 0 or np.any(np.diff(e) <= 0):
        raise ValueError("edges must be a strictly increasing 1-D sequence")
    return int(np.searchsorted(e, float(delta_t), side="right"))


def quantile_time_bins(samples: Iterable[float], n_bins: int) -> np.ndarray:
    """Fit practical elapsed-time bins from empirical quantiles."""
    x = np.asarray(list(samples), dtype=float)
    if x.size == 0 or n_bins < 1:
        raise ValueError("need samples and n_bins >= 1")
    if n_bins == 1:
        return np.array([], dtype=float)
    q = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(x, q))
    return edges
