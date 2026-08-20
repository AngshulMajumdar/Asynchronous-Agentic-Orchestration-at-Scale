from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import t as student_t
import math, itertools, time, multiprocessing as mp, os
from collections import deque

ROOT=Path(__file__).resolve().parents[2]; FIG=ROOT/'figures'/'fresh'; DATA=ROOT/'results'/'fresh'
FIG.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
SEED=20260820; K=5

def make_context_model(rng, nctx=96):
    P1=rng.dirichlet(np.ones(K)*1.4,size=nctx); Q=rng.dirichlet(np.ones(K)*1.4,size=nctx)
    P2=.55*P1+.45*Q; P2/=P2.sum(axis=1,keepdims=True); return P1,P2

def generate_context_stream(seed,T=32000,nctx=96,drift_at=16000):
    rng=np.random.default_rng(seed); P1,P2=make_context_model(rng,nctx)
    c=np.empty(T,dtype=np.int16); y=np.empty(T,dtype=np.int8)
    for t in range(T):
        if t<drift_at:
            cc=int(rng.integers(0,24)) if rng.random()<.88 else int(rng.integers(24,nctx)); P=P1
        else:
            cc=int(rng.integers(12,36)) if rng.random()<.88 else int(rng.integers(0,nctx)); P=P2
        c[t]=cc; y[t]=int(np.searchsorted(np.cumsum(P[cc]),rng.random(),side='right'))
    return c,y

def eval_eviction(c,y,rho,H,alpha=2.5):
    store={}; q=deque(); loss_sum=0.; post_sum=0.; post_n=0; mem_sum=0.; mem_n=0; drift=len(y)//2; warm=len(y)//5
    for r,(cc,yy) in enumerate(zip(c,y),start=1):
        cc=int(cc); yy=int(yy)
        if H is not None:
            cutoff=r-H
            while q and q[0][0]<=cutoff:
                stamp,key=q.popleft(); rec=store.get(key)
                if rec is not None and rec[1]==stamp: del store[key]
        if cc in store:
            counts,last=store[cc]; delta=r-last
            if delta>0: counts*=rho**delta
        else: counts=np.zeros(K,dtype=float)
        prior=np.full(K,1.0/K)
        p=(counts+alpha*prior)/(counts.sum()+alpha); ll=-math.log(max(p[yy],1e-12)); loss_sum+=ll
        if r-1>=drift: post_sum+=ll; post_n+=1
        counts[yy]+=1.; store[cc]=(counts,r); q.append((r,cc))
        if r-1>=warm: mem_sum+=len(store); mem_n+=1
    return loss_sum/len(y),post_sum/post_n,mem_sum/mem_n

def eviction_task(args):
    rho,H,rep=args; c,y=generate_context_stream(SEED+1009*rep); a,b,m=eval_eviction(c,y,rho,H); return rho,H,rep,a,b,m

def objective(S,w,W):
    if len(S)==0:return 0.
    idx=np.asarray(list(S),dtype=int); return float(w[idx].sum()+np.triu(W[np.ix_(idx,idx)],1).sum())

def feasible(S,costs,budget,conflicts):
    S=list(S)
    if S and np.any(costs[S].sum(axis=0)>budget+1e-12):return False
    for a,b in itertools.combinations(S,2):
        if (min(a,b),max(a,b)) in conflicts:return False
    return True

def ratio_greedy(w,W,costs,budget,conflicts,kmax=5):
    m=len(w);S=[]
    while len(S)<kmax:
        base=objective(S,w,W);best=None
        for j in range(m):
            if j in S:continue
            cand=S+[j]
            if not feasible(cand,costs,budget,conflicts):continue
            gain=objective(cand,w,W)-base;score=gain/(1e-12+np.sum(costs[j]/budget))
            if best is None or score>best[0]:best=(score,gain,j)
        if best is None or best[1]<=0:break
        S.append(best[2])
    return tuple(S),objective(S,w,W)

def one_swap_pass(S,w,W,costs,budget,conflicts,kmax=5):
    S=set(S);bestv=objective(S,w,W);best=set(S);outside=set(range(len(w)))-S
    if len(S)<kmax:
        for j in outside:
            cand=S|{j}
            if feasible(cand,costs,budget,conflicts):
                v=objective(cand,w,W)
                if v>bestv:bestv=v;best=set(cand)
    for out in list(S):
        for inn in outside:
            cand=(S-{out})|{inn}
            if feasible(cand,costs,budget,conflicts):
                v=objective(cand,w,W)
                if v>bestv:bestv=v;best=set(cand)
    return tuple(sorted(best)),bestv

def exact_opt(w,W,costs,budget,conflicts,kmax=5):
    m=len(w);bestv=0.
    for r in range(1,kmax+1):
        for comb in itertools.combinations(range(m),r):
            if feasible(comb,costs,budget,conflicts):
                v=objective(comb,w,W)
                if v>bestv:bestv=v
    return bestv

def swap_task(args):
    m,tr=args; rng=np.random.default_rng(SEED+7000+m*1000+tr)
    w=rng.uniform(.1,1.,size=m); X=rng.normal(0,.18,size=(m,m)); W=np.triu(X,1);W=W+W.T
    costs=rng.uniform(.05,.35,size=(m,2));budget=np.array([.75,.75]);conf=set((a,b) for a,b in itertools.combinations(range(m),2) if rng.random()<.08)
    t0=time.perf_counter();opt=exact_opt(w,W,costs,budget,conf,5);tex=(time.perf_counter()-t0)*1e3
    t0=time.perf_counter();S,gv=ratio_greedy(w,W,costs,budget,conf,5);tg=(time.perf_counter()-t0)*1e3
    t0=time.perf_counter();_,sv=one_swap_pass(S,w,W,costs,budget,conf,5);ts=(time.perf_counter()-t0)*1e3
    rg=(opt-gv)/max(opt,1e-12);rs=(opt-sv)/max(opt,1e-12)
    return m,tr,rg,rs,rg-rs,tg,tg+ts,tex

if __name__=='__main__':
    procs=min(5,os.cpu_count() or 1)
    rhos=[.95,.98,.99,.995,.999]; Hs=[16,32,64,128,None]
    tasks=[(rho,H,rep) for rho in rhos for H in Hs for rep in range(6)]
    with mp.Pool(procs) as pool: out=pool.map(eviction_task,tasks,chunksize=2)
    raw=pd.DataFrame(out,columns=['rho_bar','Hraw','replicate','mean_logloss','post_drift_logloss','mean_live_keys'])
    raw['H']=raw.Hraw.apply(lambda x:999999 if pd.isna(x) else int(x)); raw['label']=raw.H.apply(lambda x:'infinity' if x==999999 else str(x))
    raw.to_csv(DATA/'eviction_rho_h_joint_raw.csv',index=False)
    rows=[]
    for (rho,H,label),g in raw.groupby(['rho_bar','H','label']):
        n=len(g);tc=student_t.ppf(.975,n-1)
        rows.append((rho,H,label,g.mean_logloss.mean(),tc*g.mean_logloss.std(ddof=1)/np.sqrt(n),g.post_drift_logloss.mean(),tc*g.post_drift_logloss.std(ddof=1)/np.sqrt(n),g.mean_live_keys.mean(),tc*g.mean_live_keys.std(ddof=1)/np.sqrt(n),n))
    ev=pd.DataFrame(rows,columns=['rho_bar','H','label','mean_logloss','ci95_logloss','post_drift_logloss','ci95_post','mean_live_keys','ci95_keys','replicates']).sort_values(['rho_bar','H'])
    ev.to_csv(DATA/'eviction_rho_h_joint_sweep.csv',index=False)

    pos=np.arange(5);labels=['16','32','64','128',r'$\infty$']
    fig=plt.figure(figsize=(7.6,4.9));ax=fig.add_subplot(111)
    for rho in rhos:
        g=ev[ev.rho_bar==rho].sort_values('H');ax.errorbar(pos,g.post_drift_logloss,yerr=g.ci95_post,marker='o',capsize=2,label=fr'$\bar\rho={rho}$')
    ax.set_xticks(pos);ax.set_xticklabels(labels);ax.set_xlabel('Eviction horizon $H$ (local responses)');ax.set_ylabel('Post-drift prequential log loss');ax.set_title('Joint forgetting--eviction sweep');ax.legend(ncol=2,fontsize=8);ax.grid(True,alpha=.25);fig.tight_layout();fig.savefig(FIG/'46_eviction_accuracy_tradeoff.pdf');fig.savefig(FIG/'46_eviction_accuracy_tradeoff.png',dpi=220);plt.close(fig)

    fig=plt.figure(figsize=(7.6,4.9));ax=fig.add_subplot(111)
    for rho in rhos:
        g=ev[(ev.rho_bar==rho)&(ev.label!='infinity')].sort_values('H');base=float(ev[(ev.rho_bar==rho)&(ev.label=='infinity')].post_drift_logloss.iloc[0]); imp=base-g.post_drift_logloss.to_numpy();ax.plot(g.mean_live_keys,imp,marker='o',label=fr'$\bar\rho={rho}$')
        for _,r in g.iterrows():ax.annotate(str(int(r.H)),(r.mean_live_keys,base-r.post_drift_logloss),xytext=(3,3),textcoords='offset points',fontsize=7)
    ax.axhline(0,linestyle='--');ax.set_xlabel('Mean live context keys');ax.set_ylabel('Post-drift loss reduction vs no eviction');ax.set_title('Eviction benefit after matching the forgetting rate');ax.legend(ncol=2,fontsize=8);ax.grid(True,alpha=.25);fig.tight_layout();fig.savefig(FIG/'47_eviction_memory_accuracy_pareto.pdf');fig.savefig(FIG/'47_eviction_memory_accuracy_pareto.png',dpi=220);plt.close(fig)

    stasks=[(m,tr) for m in [8,10,12,14,16] for tr in range(50)]
    with mp.Pool(procs) as pool: sout=pool.map(swap_task,stasks,chunksize=2)
    sw=pd.DataFrame(sout,columns=['candidates','trial','ratio_regret','swap_regret','paired_improvement','ratio_runtime_ms','swap_runtime_ms','exact_runtime_ms']); sw.to_csv(DATA/'greedy_swap_paired_raw.csv',index=False)
    rows=[]
    for m,g in sw.groupby('candidates'):
        n=len(g);tc=student_t.ppf(.975,n-1)
        rows.append((m,n,g.ratio_regret.mean(),tc*g.ratio_regret.std(ddof=1)/np.sqrt(n),np.quantile(g.ratio_regret,.95),g.swap_regret.mean(),tc*g.swap_regret.std(ddof=1)/np.sqrt(n),np.quantile(g.swap_regret,.95),g.paired_improvement.mean(),tc*g.paired_improvement.std(ddof=1)/np.sqrt(n),int((g.paired_improvement>0).sum()),g.ratio_runtime_ms.mean(),g.swap_runtime_ms.mean(),g.exact_runtime_ms.mean()))
    ss=pd.DataFrame(rows,columns=['candidates','trials','ratio_mean_regret','ratio_ci95','ratio_p95','swap_mean_regret','swap_ci95','swap_p95','paired_improvement','paired_improvement_ci95','wins','ratio_runtime_ms','swap_runtime_ms','exact_runtime_ms']); ss.to_csv(DATA/'greedy_swap_paired_summary.csv',index=False)
    fig=plt.figure(figsize=(7.4,4.8));ax=fig.add_subplot(111);ax.errorbar(ss.candidates,100*ss.paired_improvement,yerr=100*ss.paired_improvement_ci95,marker='o',capsize=3);ax.axhline(0,linestyle='--');ax.set_xlabel('Local candidate interventions');ax.set_ylabel('Paired regret reduction from one swap (percentage points)');ax.set_title('Paired value of one best-improving swap');ax.grid(True,alpha=.25);fig.tight_layout();fig.savefig(FIG/'48_greedy_swap_regret.pdf');fig.savefig(FIG/'48_greedy_swap_regret.png',dpi=220);plt.close(fig)
    fig=plt.figure(figsize=(7.4,4.8));ax=fig.add_subplot(111);ax.plot(ss.candidates,ss.ratio_runtime_ms,marker='o',label='ratio greedy');ax.plot(ss.candidates,ss.swap_runtime_ms,marker='s',label='greedy + one swap');ax.plot(ss.candidates,ss.exact_runtime_ms,marker='^',label='exact');ax.set_yscale('log');ax.set_xlabel('Local candidate interventions');ax.set_ylabel('Mean runtime (ms, log scale)');ax.set_title('Runtime of greedy, one-swap, and exact local search');ax.legend();ax.grid(True,alpha=.25);fig.tight_layout();fig.savefig(FIG/'49_greedy_swap_runtime.pdf');fig.savefig(FIG/'49_greedy_swap_runtime.png',dpi=220);plt.close(fig)

    print('EVICTION\n',ev.to_string(index=False)); print('\nSWAP\n',ss.to_string(index=False))
