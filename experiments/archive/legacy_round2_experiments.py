from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import t as student_t
import heapq, math, itertools, time

ROOT=Path(__file__).resolve().parents[2]
FIG=ROOT/'figures'/'fresh'; DATA=ROOT/'results'/'fresh'
FIG.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
SEED=20260820

# ============================================================
# 1) Eviction accuracy / memory tradeoff under drift
# ============================================================
K=5

def make_context_model(rng, nctx=96):
    # context-specific transition distributions, two regimes
    P1=rng.dirichlet(np.ones(K)*1.4, size=nctx)
    # post-drift law partly changed, not independent, to preserve structure
    Q=rng.dirichlet(np.ones(K)*1.4, size=nctx)
    P2=0.55*P1+0.45*Q
    P2/=P2.sum(axis=1,keepdims=True)
    return P1,P2

def generate_context_stream(seed, T=50000, nctx=96, drift_at=25000):
    rng=np.random.default_rng(seed)
    P1,P2=make_context_model(rng,nctx)
    # recurrent core + long tail, with changed core identities after drift
    contexts=np.empty(T,dtype=np.int16)
    y=np.empty(T,dtype=np.int8)
    for t in range(T):
        if t<drift_at:
            if rng.random()<.88: c=int(rng.integers(0,24))
            else: c=int(rng.integers(24,nctx))
            P=P1
        else:
            if rng.random()<.88: c=int(rng.integers(12,36))
            else: c=int(rng.integers(0,nctx))
            P=P2
        contexts[t]=c
        u=rng.random(); y[t]=int(np.searchsorted(np.cumsum(P[c]),u,side='right'))
    return contexts,y

def eval_eviction(contexts,y,rho=0.995,H=None,alpha=.5):
    # Dictionary c -> [counts,last_access]; prequential log loss.
    store={}; loss=np.empty(len(y),dtype=float); sizes=np.empty(len(y),dtype=int)
    # FIFO of access stamps for amortized O(1) expiration, stale duplicates allowed.
    from collections import deque
    q=deque()
    for r,(c,yy) in enumerate(zip(contexts,y),start=1):
        c=int(c); yy=int(yy)
        if H is not None:
            cutoff=r-H
            while q and q[0][0] <= cutoff:
                stamp,key=q.popleft()
                rec=store.get(key)
                if rec is not None and rec[1]==stamp:
                    del store[key]
        if c in store:
            counts,last=store[c]
            delta=r-last
            if delta>0: counts*=rho**delta
        else:
            counts=np.full(K,alpha,dtype=float)
        p=counts/counts.sum()
        loss[r-1]=-math.log(max(p[yy],1e-12))
        counts[yy]+=1.0
        store[c]=(counts,r)
        q.append((r,c))
        sizes[r-1]=len(store)
    return loss,sizes

horizons=[16,32,64,128,None]
rhos=[0.98,0.995]
ev_rows=[]
curves={}
for rho in rhos:
    for H in horizons:
        trial_losses=[]; trial_mem=[]; trial_post=[]
        for rep in range(12):
            c,y=generate_context_stream(SEED+1009*rep)
            loss,size=eval_eviction(c,y,rho=rho,H=H)
            trial_losses.append(loss.mean())
            trial_mem.append(size[10000:].mean())
            trial_post.append(loss[25000:].mean())
        arr=np.array(trial_losses); mem=np.array(trial_mem); post=np.array(trial_post)
        n=len(arr); tc=student_t.ppf(.975,n-1)
        ev_rows.append({
            'rho':rho,'H':999999 if H is None else H,'label':'infinity' if H is None else str(H),
            'mean_logloss':arr.mean(),'ci95_logloss':tc*arr.std(ddof=1)/np.sqrt(n),
            'post_drift_logloss':post.mean(),'ci95_post':tc*post.std(ddof=1)/np.sqrt(n),
            'mean_live_keys':mem.mean(),'ci95_keys':tc*mem.std(ddof=1)/np.sqrt(n),
            'trials':n
        })
ev=pd.DataFrame(ev_rows)
ev.to_csv(DATA/'eviction_accuracy_tradeoff.csv',index=False)

# Figure 46: H vs post-drift loss and live keys, rho=0.995
sel=ev[ev.rho==0.995].copy(); sel=sel.sort_values('H')
x=np.arange(len(sel)); labels=sel['label'].tolist()
fig=plt.figure(figsize=(7.4,4.8)); ax=fig.add_subplot(111)
ax.errorbar(x,sel['post_drift_logloss'],yerr=sel['ci95_post'],marker='o',capsize=3)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel('Eviction horizon H (local responses)'); ax.set_ylabel('Post-drift prequential log loss')
ax.set_title('Prediction cost of bounded context memory')
ax.grid(True,alpha=.25); fig.tight_layout()
fig.savefig(FIG/'46_eviction_accuracy_tradeoff.png',dpi=220); fig.savefig(FIG/'46_eviction_accuracy_tradeoff.pdf'); plt.close(fig)

fig=plt.figure(figsize=(7.4,4.8)); ax=fig.add_subplot(111)
ax.errorbar(sel['mean_live_keys'],sel['post_drift_logloss'],xerr=sel['ci95_keys'],yerr=sel['ci95_post'],marker='o',capsize=3)
for _,r in sel.iterrows(): ax.annotate(r['label'],(r['mean_live_keys'],r['post_drift_logloss']),xytext=(4,4),textcoords='offset points',fontsize=8)
ax.set_xlabel('Mean live context keys'); ax.set_ylabel('Post-drift prequential log loss')
ax.set_title('Memory-prediction Pareto curve under eviction')
ax.grid(True,alpha=.25); fig.tight_layout()
fig.savefig(FIG/'47_eviction_memory_accuracy_pareto.png',dpi=220); fig.savefig(FIG/'47_eviction_memory_accuracy_pareto.pdf'); plt.close(fig)

# Theoretical H_eps table
heps=[]
for rho in [0.90,0.95,0.98,0.995,0.999]:
    for eps in [0.1,0.01,0.001]:
        val=math.ceil(math.log(eps*(1-rho))/math.log(rho))
        heps.append((rho,eps,val))
pd.DataFrame(heps,columns=['rho_bar','epsilon','H_epsilon']).to_csv(DATA/'theoretical_eviction_horizons.csv',index=False)

# ============================================================
# 2) Gain/resource greedy + one best 1-swap pass
# ============================================================

def objective(S,w,W):
    if len(S)==0: return 0.0
    idx=np.asarray(S,dtype=int)
    return float(w[idx].sum()+np.triu(W[np.ix_(idx,idx)],1).sum())

def feasible(S,costs,budget,conflicts):
    if len(S) and np.any(costs[list(S)].sum(axis=0)>budget+1e-12): return False
    for a,b in itertools.combinations(S,2):
        if (min(a,b),max(a,b)) in conflicts: return False
    return True

def ratio_greedy(w,W,costs,budget,conflicts,kmax=5):
    m=len(w); S=[]
    while len(S)<kmax:
        best=None
        for j in range(m):
            if j in S: continue
            cand=S+[j]
            if not feasible(cand,costs,budget,conflicts): continue
            gain=objective(cand,w,W)-objective(S,w,W)
            score=gain/(1e-12+np.sum(costs[j]/budget))
            if best is None or score>best[0]: best=(score,gain,j)
        if best is None or best[1]<=0: break
        S.append(best[2])
    return tuple(S),objective(S,w,W)

def one_swap_pass(S,w,W,costs,budget,conflicts,kmax=5):
    S=set(S); cur=objective(S,w,W); best=(cur,set(S))
    outside=set(range(len(w)))-S
    # allow one add if cardinality permits
    if len(S)<kmax:
        for j in outside:
            cand=S|{j}
            if feasible(cand,costs,budget,conflicts):
                v=objective(cand,w,W)
                if v>best[0]: best=(v,set(cand))
    for out in list(S):
        for inn in outside:
            cand=(S-{out})|{inn}
            if feasible(cand,costs,budget,conflicts):
                v=objective(cand,w,W)
                if v>best[0]: best=(v,set(cand))
    return tuple(sorted(best[1])),best[0]

def exact_opt(w,W,costs,budget,conflicts,kmax=5):
    m=len(w); bestv=0.0
    for r in range(1,kmax+1):
        for comb in itertools.combinations(range(m),r):
            if feasible(comb,costs,budget,conflicts):
                v=objective(comb,w,W)
                if v>bestv: bestv=v
    return bestv

swap_rows=[]
for m in [8,10,12,14,16]:
    rng=np.random.default_rng(SEED+4000+m)
    for tr in range(60):
        w=rng.uniform(.1,1.0,size=m)
        X=rng.normal(0,.18,size=(m,m)); W=np.triu(X,1); W=W+W.T
        costs=rng.uniform(.05,.35,size=(m,2)); budget=np.array([.75,.75])
        conflicts=set((a,b) for a,b in itertools.combinations(range(m),2) if rng.random()<.08)
        t0=time.perf_counter(); opt=exact_opt(w,W,costs,budget,conflicts,5); tex=(time.perf_counter()-t0)*1e3
        t0=time.perf_counter(); S,gv=ratio_greedy(w,W,costs,budget,conflicts,5); tg=(time.perf_counter()-t0)*1e3
        t0=time.perf_counter(); S2,sv=one_swap_pass(S,w,W,costs,budget,conflicts,5); ts=(time.perf_counter()-t0)*1e3
        swap_rows += [
            (m,tr,'ratio greedy',(opt-gv)/max(opt,1e-12),tg,opt,gv),
            (m,tr,'greedy + one swap',(opt-sv)/max(opt,1e-12),tg+ts,opt,sv),
            (m,tr,'exact',0.0,tex,opt,opt)
        ]
swap=pd.DataFrame(swap_rows,columns=['candidates','trial','method','relative_regret','runtime_ms','opt_value','value'])
swap.to_csv(DATA/'greedy_swap_raw.csv',index=False)
sm=[]
for (m,method),g in swap.groupby(['candidates','method']):
    n=len(g); tc=student_t.ppf(.975,n-1)
    sm.append((m,method,g.relative_regret.mean(),tc*g.relative_regret.std(ddof=1)/np.sqrt(n),np.quantile(g.relative_regret,.95),g.runtime_ms.mean(),n))
swap_sum=pd.DataFrame(sm,columns=['candidates','method','mean_regret','ci95_mean_regret','p95_regret','mean_runtime_ms','trials'])
swap_sum.to_csv(DATA/'greedy_swap_summary.csv',index=False)

fig=plt.figure(figsize=(7.4,4.8)); ax=fig.add_subplot(111)
for method in ['ratio greedy','greedy + one swap']:
    g=swap_sum[swap_sum.method==method].sort_values('candidates')
    ax.errorbar(g.candidates,100*g.mean_regret,yerr=100*g.ci95_mean_regret,marker='o',capsize=3,label=method)
ax.set_xlabel('Local candidate interventions'); ax.set_ylabel('Mean regret vs exact optimum (%)')
ax.set_title('One local-search swap closes the greedy gap')
ax.legend(); ax.grid(True,alpha=.25); fig.tight_layout()
fig.savefig(FIG/'48_greedy_swap_regret.png',dpi=220); fig.savefig(FIG/'48_greedy_swap_regret.pdf'); plt.close(fig)

fig=plt.figure(figsize=(7.4,4.8)); ax=fig.add_subplot(111)
for method in ['ratio greedy','greedy + one swap','exact']:
    g=swap_sum[swap_sum.method==method].sort_values('candidates')
    ax.plot(g.candidates,g.mean_runtime_ms,marker='o',label=method)
ax.set_yscale('log'); ax.set_xlabel('Local candidate interventions'); ax.set_ylabel('Mean runtime (ms, log scale)')
ax.set_title('Greedy, one-swap, and exact local optimization')
ax.legend(); ax.grid(True,alpha=.25); fig.tight_layout()
fig.savefig(FIG/'49_greedy_swap_runtime.png',dpi=220); fig.savefig(FIG/'49_greedy_swap_runtime.pdf'); plt.close(fig)

# ============================================================
# 3) Queueing with gamma service (finite MGF near zero)
# ============================================================

def simulate_queue(load,servers,events=70000,cv=.45,seed=0):
    rng=np.random.default_rng(seed)
    arrival_rate=load*servers
    arrivals=np.cumsum(rng.exponential(1/arrival_rate,size=events))
    shape=1/(cv*cv); scale=1/shape # mean 1, CV 1/sqrt(shape)
    svc=rng.gamma(shape,scale,size=events)
    heap=[0.0]*servers; heapq.heapify(heap)
    waits=np.empty(events)
    for k,(a,s) in enumerate(zip(arrivals,svc)):
        free=heapq.heappop(heap); start=max(a,free); waits[k]=start-a; heapq.heappush(heap,start+s)
    waits=waits[events//10:]
    return waits

qrows=[]
loads=[.25,.4,.55,.7,.8,.9,.95,.98]
for servers in [1,2,4,8]:
    for load in loads:
        vals=[]
        for rep in range(10):
            waits=simulate_queue(load,servers,seed=SEED+servers*10000+int(load*1000)+rep)
            vals.append((waits.mean(),np.quantile(waits,.95),np.quantile(waits,.99),np.mean(waits>1e-12)))
        A=np.asarray(vals); n=len(vals); tc=student_t.ppf(.975,n-1)
        qrows.append((servers,load,A[:,0].mean(),A[:,1].mean(),tc*A[:,1].std(ddof=1)/np.sqrt(n),A[:,2].mean(),A[:,3].mean(),n))
qdf=pd.DataFrame(qrows,columns=['servers','load','mean_wait','p95_wait','ci95_p95_wait','p99_wait','prob_wait','replicates'])
qdf.to_csv(DATA/'queue_stability_gamma_summary.csv',index=False)

fig=plt.figure(figsize=(7.4,4.8)); ax=fig.add_subplot(111)
for c,g in qdf.groupby('servers'):
    ax.errorbar(g.load,g.p95_wait,yerr=g.ci95_p95_wait,marker='o',capsize=2,label=f'{c} shards')
ax.set_yscale('log'); ax.set_xlabel('Offered load $\\rho$'); ax.set_ylabel('P95 queue wait / mean service time')
ax.set_title('Queue delay with gamma service times')
ax.legend(); ax.grid(True,alpha=.25); fig.tight_layout()
fig.savefig(FIG/'37_queue_delay_vs_load.png',dpi=220); fig.savefig(FIG/'37_queue_delay_vs_load.pdf'); plt.close(fig)

fig=plt.figure(figsize=(7.4,4.8)); ax=fig.add_subplot(111)
for c,g in qdf.groupby('servers'):
    ax.plot(g.load,g.prob_wait,marker='o',label=f'{c} shards')
ax.set_xlabel('Offered load $\\rho$'); ax.set_ylabel('Probability an event waits')
ax.set_title('Queueing probability with gamma service times')
ax.legend(); ax.grid(True,alpha=.25); fig.tight_layout()
fig.savefig(FIG/'38_probability_of_queueing.png',dpi=220); fig.savefig(FIG/'38_probability_of_queueing.pdf'); plt.close(fig)

# ============================================================
# 4) Confidence intervals / reporting support
# ============================================================
# Cascade mean CI at N=100k adaptive using raw data if available.
p=DATA/'cascade_sizes_raw.csv'; p=p if p.exists() else ROOT/'results'/'reference'/'cascade_sizes_raw.csv'; cascade_raw=pd.read_csv(p)
g=cascade_raw[(cascade_raw.N==100000)&(cascade_raw.label=='Adaptive')].cascade_size.to_numpy()
tc=student_t.ppf(.975,len(g)-1)
cascade_ci=(g.mean(),tc*g.std(ddof=1)/np.sqrt(len(g)),len(g))

# CMI AUC bootstrap CI at n=1000 from raw scores.
p=DATA/'cmi_edge_recovery_raw.csv'; p=p if p.exists() else ROOT/'results'/'reference'/'cmi_edge_recovery_raw.csv'; cmi=pd.read_csv(p)
g=cmi[cmi.samples_per_edge==1000]
labels=g.coupled.to_numpy(); scores=g.cmi_nats.to_numpy()
def auc(labels,scores):
    pos=scores[labels==1]; neg=scores[labels==0]
    wins=sum(np.sum(p>neg)+.5*np.sum(p==neg) for p in pos)
    return wins/(len(pos)*len(neg))
rng=np.random.default_rng(SEED+999)
boots=[]
for _ in range(1500):
    idx=rng.integers(0,len(g),len(g)); lb=labels[idx]; sc=scores[idx]
    if lb.min()==lb.max(): continue
    boots.append(auc(lb,sc))
auc0=auc(labels,scores); auc_ci=np.quantile(boots,[.025,.975])

report=pd.DataFrame([
    ['cascade N=100k adaptive mean',cascade_ci[0],cascade_ci[1],cascade_ci[2]],
    ['CMI AUC n=1000',auc0,(auc_ci[1]-auc_ci[0])/2,len(g)]
],columns=['metric','estimate','halfwidth_or_bootstrap_halfspan','n'])
report.to_csv(DATA/'round2_reporting_summary.csv',index=False)

print('Eviction tradeoff:\n',ev.to_string(index=False))
print('\nSwap summary at m=16:\n',swap_sum[swap_sum.candidates==16].to_string(index=False))
print('\nQueue 4 shards:\n',qdf[(qdf.servers==4)&(qdf.load>=.8)].to_string(index=False))
print('\nCascade mean CI:',cascade_ci)
print('CMI AUC CI:',auc0,auc_ci)
