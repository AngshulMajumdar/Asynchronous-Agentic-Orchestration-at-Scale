import numpy as np
from async_agentic_orchestration.memory import SparseTransitionMemory

def test_eviction_horizon_bounds_live_keys():
    m=SparseTransitionMemory(3,rho=.99,alpha=1,horizon=4)
    for t in range(20):
        m.observe((t,),t%3)
        assert m.live_keys <= 4
        assert m.expiry_records <= 4

def test_prior_is_not_decayed():
    m=SparseTransitionMemory(2,rho=.5,alpha=2,prior=[.75,.25],horizon=None)
    p=m.predict('new')
    assert np.allclose(p,[.75,.25])
