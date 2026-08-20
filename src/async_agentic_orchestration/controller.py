from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Tuple
import numpy as np

from .memory import SparseTransitionMemory, fixed_time_bin
from .intervention import StructuralActionMap, ResourceConstraints, ratio_greedy, one_swap_refinement


@dataclass(frozen=True)
class AgentEvent:
    event_id: int
    responders: Tuple[Hashable, ...]
    observations: Mapping[Hashable, int]
    elapsed: Mapping[Hashable, float]


@dataclass
class OrchestrationDecision:
    event_id: int
    updated_agents: Tuple[Hashable, ...]
    candidates: Tuple[Hashable, ...]
    selected: Tuple[Hashable, ...]
    objective_value: float


class EventLocalOrchestrator:
    """Reference event-local orchestration loop.

    The controller deliberately does not propagate a responder's posterior into its
    neighbors. Structural locality is used to define contexts and candidate actions;
    statistical state propagation must be justified separately by an application.
    """

    def __init__(
        self,
        agent_ids: Sequence[Hashable],
        n_states: int,
        structural_neighbors: Mapping[Hashable, Sequence[Hashable]],
        actions: StructuralActionMap,
        time_bin_edges: Sequence[float],
        rho: float = 0.995,
        alpha: float = 1.0,
        memory_horizon: Optional[int] = 64,
    ) -> None:
        self.agent_ids = tuple(agent_ids)
        self.n_states = int(n_states)
        self.neighbors = {i: tuple(structural_neighbors.get(i, ())) for i in self.agent_ids}
        self.actions = actions
        self.time_bin_edges = np.asarray(time_bin_edges, dtype=float)
        self.memory = {
            i: SparseTransitionMemory(n_states, rho=rho, alpha=alpha, horizon=memory_horizon)
            for i in self.agent_ids
        }
        self.state_prob = {i: np.full(n_states, 1.0 / n_states) for i in self.agent_ids}
        self.last_symbol = {i: 0 for i in self.agent_ids}

    def _context(self, agent: Hashable, elapsed: float) -> Tuple:
        neighbor_symbols = tuple(self.last_symbol[j] for j in self.neighbors.get(agent, ()))
        return (
            int(self.last_symbol[agent]),
            neighbor_symbols,
            fixed_time_bin(elapsed, self.time_bin_edges) if len(self.time_bin_edges) else 0,
        )

    def process(
        self,
        event: AgentEvent,
        observation_likelihood: Callable[[Hashable, int], np.ndarray],
        objective: Callable[[Tuple[Hashable, ...]], float],
        constraints: ResourceConstraints,
        max_actions: int = 2,
        use_one_swap: bool = False,
    ) -> OrchestrationDecision:
        # 1. update only responding agents
        for i in event.responders:
            obs = int(event.observations[i])
            context = self._context(i, float(event.elapsed[i]))
            prior = self.memory[i].predict(context)
            like = np.asarray(observation_likelihood(i, obs), dtype=float)
            if like.shape != (self.n_states,) or np.any(like < 0):
                raise ValueError("observation likelihood must be a nonnegative n_states vector")
            post = prior * like
            if post.sum() <= 0:
                post = prior
            else:
                post /= post.sum()
            symbol = int(np.argmax(post))
            self.state_prob[i] = post
            self.last_symbol[i] = symbol
            self.memory[i].observe(context, symbol)

        # 2. structural local candidate generation
        candidates = self.actions.candidates(event.responders)

        # 3. local constrained decision
        selected, value = ratio_greedy(candidates, objective, constraints, max_actions)
        if use_one_swap and selected:
            selected, value = one_swap_refinement(selected, candidates, objective, constraints, max_actions)

        return OrchestrationDecision(
            event_id=event.event_id,
            updated_agents=tuple(event.responders),
            candidates=tuple(candidates),
            selected=tuple(selected),
            objective_value=float(value),
        )
