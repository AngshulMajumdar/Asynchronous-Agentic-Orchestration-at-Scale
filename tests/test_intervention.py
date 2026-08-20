from async_agentic_orchestration.intervention import ResourceConstraints, ratio_greedy, one_swap_refinement

def test_greedy_is_feasible():
    cand=('a','b','c')
    cons=ResourceConstraints(costs={'a':(1,), 'b':(1,), 'c':(2,)}, budget=(2,), conflicts={('a','b')})
    value={'a':3,'b':2,'c':4}
    obj=lambda s: sum(value[x] for x in s)
    s,v=ratio_greedy(cand,obj,cons,2)
    assert cons.feasible(s)
    assert v==obj(s)

def test_one_swap_never_worsens():
    cand=('a','b','c')
    cons=ResourceConstraints(costs={x:(1,) for x in cand},budget=(2,),conflicts=set())
    value={'a':1,'b':2,'c':5}
    obj=lambda s: sum(value[x] for x in s)
    s,v=ratio_greedy(cand,obj,cons,2)
    s2,v2=one_swap_refinement(s,cand,obj,cons,2)
    assert v2>=v
