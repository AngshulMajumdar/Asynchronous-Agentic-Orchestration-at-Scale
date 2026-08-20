"""Asynchronous Agentic Orchestration at Scale.

Reference implementation of the event-local orchestration mechanisms used in the
paper/repository. The package is intentionally worker-model agnostic: an "agent"
may be a learned model, symbolic solver, simulator, planner, sensor process, or any
other asynchronous computational worker.
"""

from .memory import SparseTransitionMemory, quantile_time_bins, fixed_time_bin
from .intervention import (
    StructuralActionMap,
    ResourceConstraints,
    ratio_greedy,
    one_swap_refinement,
)
from .cascade import RollingCascadeDiagnostic
from .cmi import conditional_mutual_information, permutation_cmi_threshold
from .sharding import ShardBudgetLedger, owner_footprint
from .controller import EventLocalOrchestrator, AgentEvent, OrchestrationDecision

__all__ = [
    "SparseTransitionMemory",
    "quantile_time_bins",
    "fixed_time_bin",
    "StructuralActionMap",
    "ResourceConstraints",
    "ratio_greedy",
    "one_swap_refinement",
    "RollingCascadeDiagnostic",
    "conditional_mutual_information",
    "permutation_cmi_threshold",
    "ShardBudgetLedger",
    "owner_footprint",
    "EventLocalOrchestrator",
    "AgentEvent",
    "OrchestrationDecision",
]
