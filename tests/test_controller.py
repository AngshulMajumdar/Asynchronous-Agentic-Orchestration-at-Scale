import numpy as np
from async_agentic_orchestration import AgentEvent, EventLocalOrchestrator, StructuralActionMap, ResourceConstraints

def test_only_responders_update():
    agents=[0,1,2]
    c=EventLocalOrchestrator(agents,3,{0:(1,),1:(0,2),2:(1,)},StructuralActionMap({0:('a',),1:('b',),2:('c',)}),[1,5],memory_horizon=8)
    before=c.state_prob[2].copy()
    e=AgentEvent(1,(0,),{0:2},{0:2.0})
    like=lambda i,o: np.array([.1,.1,.8])
    cons=ResourceConstraints({'a':(1,), 'b':(1,), 'c':(1,)},(1,),set())
    c.process(e,like,lambda s: float(len(s)),cons,max_actions=1)
    assert np.allclose(c.state_prob[2],before)
    assert c.memory[0].local_clock==1
    assert c.memory[1].local_clock==0
