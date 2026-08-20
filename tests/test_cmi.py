import numpy as np
from async_agentic_orchestration.cmi import conditional_mutual_information

def test_conditional_independence_near_zero():
    rng=np.random.default_rng(3)
    z=rng.integers(0,2,20000)
    x=rng.integers(0,2,20000)
    y=z.copy()
    assert conditional_mutual_information(x,y,z) < 1e-3
