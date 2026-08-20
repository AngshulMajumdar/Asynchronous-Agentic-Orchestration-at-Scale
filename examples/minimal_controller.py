from async_agentic_orchestration import (
    AgentEvent, EventLocalOrchestrator, StructuralActionMap,
    ResourceConstraints,
)
import numpy as np

agents = list(range(8))
neighbors = {i: ((i-1) % 8, (i+1) % 8) for i in agents}
action_map = StructuralActionMap({i: (f"repair-{i}",) for i in agents})
controller = EventLocalOrchestrator(
    agent_ids=agents,
    n_states=3,
    structural_neighbors=neighbors,
    actions=action_map,
    time_bin_edges=[1.0, 5.0, 20.0],
    rho=0.995,
    memory_horizon=64,
)

event = AgentEvent(
    event_id=1,
    responders=(2, 5),
    observations={2: 2, 5: 1},
    elapsed={2: 2.3, 5: 7.5},
)

# Observation code 0/1/2 is interpreted as a noisy measurement of local state.
def observation_likelihood(agent, obs):
    p = np.full(3, 0.1)
    p[obs] = 0.8
    return p

# Example local objective: responders' own repair actions are useful.
def objective(actions):
    return float(sum(1.0 if a in {"repair-2", "repair-5"} else 0.1 for a in actions))

constraints = ResourceConstraints(
    costs={f"repair-{i}": (1.0,) for i in agents},
    budget=(2.0,),
    conflicts=set(),
)

print(controller.process(event, observation_likelihood, objective, constraints, max_actions=2))
