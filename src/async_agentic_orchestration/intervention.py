from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, Mapping, Sequence, Set, Tuple
import numpy as np

Action = Hashable


class StructuralActionMap:
    """Bounded structural map from agent id to potentially relevant interventions."""

    def __init__(self, mapping: Mapping[Hashable, Iterable[Action]]):
        self.mapping = {k: tuple(v) for k, v in mapping.items()}

    def candidates(self, responders: Iterable[Hashable]) -> Tuple[Action, ...]:
        seen: Set[Action] = set()
        out = []
        for i in responders:
            for a in self.mapping.get(i, ()):  # local contract
                if a not in seen:
                    seen.add(a)
                    out.append(a)
        return tuple(out)


@dataclass(frozen=True)
class ResourceConstraints:
    costs: Mapping[Action, Sequence[float]]
    budget: Sequence[float]
    conflicts: Set[Tuple[Action, Action]]

    def normalized_consumption(self, action: Action) -> float:
        c = np.asarray(self.costs[action], dtype=float)
        b = np.asarray(self.budget, dtype=float)
        if np.any(b <= 0):
            raise ValueError("budget components must be positive")
        return float(np.sum(c / b))

    def feasible(self, selected: Sequence[Action]) -> bool:
        if selected:
            total = np.sum([np.asarray(self.costs[a], dtype=float) for a in selected], axis=0)
        else:
            total = np.zeros(len(self.budget))
        if np.any(total > np.asarray(self.budget, dtype=float) + 1e-12):
            return False
        chosen = set(selected)
        for a, b in self.conflicts:
            if a in chosen and b in chosen:
                return False
        return True


def ratio_greedy(
    candidates: Sequence[Action],
    objective: Callable[[Tuple[Action, ...]], float],
    constraints: ResourceConstraints,
    max_actions: int,
) -> Tuple[Tuple[Action, ...], float]:
    """Feasibility-preserving gain/resource greedy selection.

    No submodularity is assumed. The rule is deliberately local and empirical: its
    quality is evaluated against exact small-set optimization in the repository.
    """
    selected: list[Action] = []
    while len(selected) < max_actions:
        base = objective(tuple(selected))
        best = None
        for a in candidates:
            if a in selected:
                continue
            trial = tuple(selected + [a])
            if not constraints.feasible(trial):
                continue
            gain = objective(trial) - base
            denom = constraints.normalized_consumption(a)
            score = gain if denom <= 0 else gain / denom
            if best is None or score > best[0]:
                best = (score, gain, a)
        if best is None or best[1] <= 0:
            break
        selected.append(best[2])
    result = tuple(selected)
    return result, objective(result)


def one_swap_refinement(
    selected: Sequence[Action],
    candidates: Sequence[Action],
    objective: Callable[[Tuple[Action, ...]], float],
    constraints: ResourceConstraints,
    max_actions: int,
) -> Tuple[Tuple[Action, ...], float]:
    """Best single add/swap improvement after greedy selection."""
    current = tuple(selected)
    best = (objective(current), current)
    sset = set(current)
    outside = [a for a in candidates if a not in sset]

    if len(current) < max_actions:
        for a in outside:
            trial = tuple(list(current) + [a])
            if constraints.feasible(trial):
                v = objective(trial)
                if v > best[0]:
                    best = (v, trial)

    for out in current:
        base = [a for a in current if a != out]
        for inn in outside:
            trial = tuple(base + [inn])
            if constraints.feasible(trial):
                v = objective(trial)
                if v > best[0]:
                    best = (v, trial)
    return best[1], best[0]
