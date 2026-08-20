from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Dict, Hashable, Iterable, Mapping, Sequence, Set, Tuple
import numpy as np


@dataclass
class Reservation:
    accepted: bool
    remaining: np.ndarray


class ShardBudgetLedger:
    """Exact per-shard resource reservation inside an allocation epoch.

    Global budgets are partitioned before the epoch. Events reserve only from their
    shard's local ledger, eliminating the shared-global-counter contradiction in the
    concurrent event path. Reconciliation/reallocation is an epoch-level operation.
    """

    def __init__(self, shard_budgets: Mapping[int, Sequence[float]]):
        self._remaining = {int(s): np.asarray(b, dtype=float).copy() for s, b in shard_budgets.items()}
        self._locks = {s: Lock() for s in self._remaining}

    def reserve(self, shard: int, cost: Sequence[float]) -> Reservation:
        shard = int(shard)
        c = np.asarray(cost, dtype=float)
        if shard not in self._remaining:
            raise KeyError(f"unknown shard {shard}")
        with self._locks[shard]:
            r = self._remaining[shard]
            if c.shape != r.shape or np.any(c < 0):
                raise ValueError("invalid reservation cost")
            if np.any(c > r + 1e-12):
                return Reservation(False, r.copy())
            r -= c
            return Reservation(True, r.copy())

    def remaining(self, shard: int) -> np.ndarray:
        return self._remaining[int(shard)].copy()


def owner_footprint(read_owners: Iterable[Hashable], write_owners: Iterable[Hashable]) -> Set[Hashable]:
    """Agent-owner footprint used by local conflict/parallelism logic."""
    return set(read_owners) | set(write_owners)
