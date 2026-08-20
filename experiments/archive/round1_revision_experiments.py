from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from collections import deque, defaultdict
from scipy.stats import t as student_t
import time, math

ROOT=Path(__file__).resolve().parents[2]
FIG=ROOT/'figures'/'fresh'; DATA=ROOT/'results'/'fresh'
FIG.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
SEED=20260820

# 42. Bounded memory with local-response LRU eviction
# N=10k, 4 responders/event. Context stream has recurrent core + long tail + drift.
def memory_sim(N=10000, events=350000, responders=4, H=None, seed=0, checkpoints=140):
    rng=np.random.default_rng(seed)
    # per-worker local response counter and context LRU by last local-response index
    clocks=np.zeros(N,dtype=np.int32)
    stores=[{} for _ in range(N)]
    cps=set(np.linspace(max(1,events//checkpoints),events,checkpoints,dtype=int).tolist())
    rec=[]
    for ev in range(1,events+1):
        ids=rng.integers(0,N,size=responders)
        for i in ids:
            i=int(i); clocks[i]+=1; r=int(clocks[i])
            # recurrent core shifts halfway; 72% recurrent, 28% long-tail
            core_off=0 if ev<events//2 else 32
            if rng.random()<0.72:
                c=core_off+int(rng.integers(0,32))
            else:
                c=64+int(rng.integers(0,20000))
            stores[i][c]=r
            if H is not None:
                # Local eviction only when worker is touched; this is exact LRU-in-local-response-index.
                dead=[k for k,last in stores[i].items() if r-last>=H]
                for k in dead: del stores[i][k]
        if ev in cps:
            total=sum(len(d) for d in stores)
            rec.append((ev,total,total/N))
    return pd.DataFrame(rec,columns=['events','live_keys','keys_per_worker'])

mem_frames=[]
for H in [16,32,64,None]:
    df=memory_sim(H=H,seed=SEED+(0 if H is None else H))
    df['horizon']='no eviction' if H is None else f'H={H}'
    mem_frames.append(df)
mem=pd.concat(mem_frames,ignore_index=True)
mem.to_csv(DATA/'eviction_memory_growth.csv',index=False)

fig=plt.figure(figsize=(7.5,4.8)); ax=fig.add_subplot(111)
for label,g in mem.groupby('horizon'):
    ax.plot(g['events'],g['keys_per_worker'],label=label)
ax.set_xlabel('Asynchronous events processed')
ax.set_ylabel('Live realized-context keys per worker')
ax.set_title('Forgetting plus eviction bounds persistent transition memory')
ax.legend(); ax.grid(True,alpha=.25); fig.tight_layout()
fig.savefig(FIG/'42_eviction_bounds_context_memory.png',dpi=220)
fig.savefig(FIG/'42_eviction_bounds_context_memory.pdf')
plt.close(fig)

# 43. Fixed local arithmetic, increasing backing-state footprint: cache/locality effect
# Belief-like array: 8 float64 = 64 bytes/worker. 36 random workers/event.
def footprint_bench(N, events=8000, touched=36, reps=5, seed=0):
    rng=np.random.default_rng(seed)
    state=rng.normal(size=(N,8)).astype(np.float64)
    idx=rng.integers(0,N,size=(events,touched),dtype=np.int64)
    # warm-up a subset
    _=state[idx[:500]].sum()
    times=[]
    checksum=0.0
    for r in range(reps):
        t0=time.perf_counter()
        # Gather exactly the same number of rows for every N.
        block=state[idx]
        out=(block*block).sum(axis=(1,2))
        checksum+=float(out[0])
        times.append((time.perf_counter()-t0)/events*1e6)
    return np.median(times), np.std(times), state.nbytes/1024**2

fp=[]
for N in [1000,3000,10000,30000,100000,300000,1000000,3000000]:
    med,std,mb=footprint_bench(N,events=6000 if N<1000000 else 3500,reps=4,seed=SEED+N)
    fp.append((N,mb,med,std))
foot=pd.DataFrame(fp,columns=['N','state_footprint_MB','median_us_per_event','std_us'])
foot.to_csv(DATA/'fixed_local_work_memory_footprint.csv',index=False)

fig=plt.figure(figsize=(7.5,4.8)); ax=fig.add_subplot(111)
ax.plot(foot['N'],foot['median_us_per_event'],marker='o')
ax.set_xscale('log')
ax.set_xlabel('Logical workers N')
ax.set_ylabel('Fixed local gather-and-score time (microseconds/event)')
ax.set_title('Constant operation count can still pay a memory-footprint penalty')
ax.grid(True,alpha=.25); fig.tight_layout()
fig.savefig(FIG/'43_memory_footprint_latency_penalty.png',dpi=220)
fig.savefig(FIG/'43_memory_footprint_latency_penalty.pdf')
plt.close(fig)

# 44. Correct paired t intervals for the replicated ablation
p=DATA/'end_to_end_replications_raw.csv'; p=p if p.exists() else ROOT/'results'/'reference'/'end_to_end_replications_raw.csv'; raw=pd.read_csv(p)
wide=raw.pivot(index='replicate',columns='variant',values='mean_pre_action_risk')
pairs=[]
for other in ['fixed_memory','no_propagation','random','no_control']:
    d=(wide[other]-wide['full']).dropna().to_numpy()
    n=len(d); crit=student_t.ppf(.975,n-1)
    se=d.std(ddof=1)/math.sqrt(n)
    pairs.append((other,d.mean(),d.mean()-crit*se,d.mean()+crit*se,n,int((d>0).sum())))
paired=pd.DataFrame(pairs,columns=['comparison','mean_advantage','ci95_lo_t','ci95_hi_t','n','full_wins'])
paired.to_csv(DATA/'paired_differences_t_intervals.csv',index=False)
fig=plt.figure(figsize=(7.5,4.8)); ax=fig.add_subplot(111)
x=np.arange(len(paired)); y=paired['mean_advantage'].to_numpy(); lo=y-paired['ci95_lo_t']; hi=paired['ci95_hi_t']-y
ax.errorbar(x,y,yerr=np.vstack([lo,hi]),fmt='o',capsize=5)
ax.axhline(0,linestyle='--')
ax.set_xticks(x); ax.set_xticklabels(['fixed memory','no propagation','random','no control'],rotation=20,ha='right')
ax.set_ylabel('Paired risk difference: comparator - full')
ax.set_title('Paired ablation differences with t-based 95% intervals')
ax.grid(True,axis='y',alpha=.25); fig.tight_layout()
fig.savefig(FIG/'44_paired_ablation_t_intervals.png',dpi=220)
fig.savefig(FIG/'44_paired_ablation_t_intervals.pdf')
plt.close(fig)

# 45. Concurrency constant: analytic union-bound stress for baseline vs cascade-sized write sets.
# Pairwise bound <= w^2/N; m-event union <= C(m,2) w^2/N, clipped at 1 for plotting.
Ns=np.logspace(3,7,120)
rows=[]
for w,label in [(36,'baseline local write set (w=36)'),(205,'cascade stress write set (w=205)')]:
    for m in [2,4,8]:
        p=np.minimum(1.0, math.comb(m,2)*(w*w)/Ns)
        for N,val in zip(Ns,p): rows.append((N,w,m,label,val))
conc=pd.DataFrame(rows,columns=['N','w','concurrent_events','label','union_bound'])
conc.to_csv(DATA/'collision_bound_write_set_sensitivity.csv',index=False)
fig=plt.figure(figsize=(7.5,4.8)); ax=fig.add_subplot(111)
for (label,m),g in conc.groupby(['label','concurrent_events']):
    ax.plot(g['N'],g['union_bound'],label=f'{label}, m={m}')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Logical workers N')
ax.set_ylabel('Union bound on any write-set collision')
ax.set_title('Concurrency depends on write-set size, not locality alone')
ax.legend(fontsize=7); ax.grid(True,alpha=.25); fig.tight_layout()
fig.savefig(FIG/'45_collision_bound_write_set_sensitivity.png',dpi=220)
fig.savefig(FIG/'45_collision_bound_write_set_sensitivity.pdf')
plt.close(fig)

print('Memory final keys/worker:')
print(mem.groupby('horizon').tail(1).to_string(index=False))
print('\nFootprint benchmark:')
print(foot.to_string(index=False))
print('\nPaired t intervals:')
print(paired.to_string(index=False))
