#!/usr/bin/env python3
"""Rebuild all paper figures from the committed fixed-seed CSV outputs.

This is the exact-data reproduction path. It does not benchmark the current machine;
it reconstructs the reported plots from the raw outputs used in the manuscript.
Fresh CPU timing experiments are intentionally separate because wall-clock values are
hardware- and runtime-dependent.
"""
from pathlib import Path
import argparse, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "reference"
DEFAULT_OUT = ROOT / "figures" / "reproduced"


def load(name):
    return pd.read_csv(DATA / name)


def save(fig, out, number, slug):
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"{number:02d}_{slug}"
    fig.tight_layout()
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=180)
    plt.close(fig)


def line(df, x, ys, labels, xlabel, ylabel, title, out, num, slug, xscale=None, yscale=None):
    fig, ax = plt.subplots(figsize=(7.3, 4.7))
    for y, lab in zip(ys, labels):
        ax.plot(df[x], df[y], marker="o", label=lab)
    if len(ys) > 1: ax.legend()
    if xscale: ax.set_xscale(xscale)
    if yscale: ax.set_yscale(yscale)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title); ax.grid(True, alpha=.25)
    save(fig,out,num,slug)


def main(out):
    # 1-3: basic scaling
    d=load('scaling_vs_N.csv'); fig,ax=plt.subplots(figsize=(7.3,4.7)); ax.errorbar(d.N,d.mean_us_per_event,yerr=d.std_us_per_event,marker='o',capsize=3); ax.set_xscale('log'); ax.set(xlabel='Logical agents N',ylabel='Mean event latency (µs)',title='Event latency versus installed population'); ax.grid(True,alpha=.25); save(fig,out,1,'latency_vs_N')
    d=load('scaling_vs_response_sparsity.csv'); fig,ax=plt.subplots(figsize=(7.3,4.7)); ax.errorbar(d.responders,d.mean_us_per_event,yerr=d.std_us_per_event,marker='o',capsize=3); ax.set(xlabel='Responders per event',ylabel='Mean event latency (µs)',title='Latency follows active event support'); ax.grid(True,alpha=.25); save(fig,out,2,'latency_vs_response_sparsity')
    d=load('scaling_vs_locality_degree.csv'); fig,ax=plt.subplots(figsize=(7.3,4.7)); ax.errorbar(d.degree,d.mean_us_per_event,yerr=d.std_us_per_event,marker='o',capsize=3); ax.set(xlabel='Structural degree',ylabel='Mean event latency (µs)',title='Latency versus locality degree'); ax.grid(True,alpha=.25); save(fig,out,3,'latency_vs_locality_degree')

    # 4-5: small local optimization
    d=load('greedy_vs_exact.csv'); line(d,'candidate_count',['mean_relative_regret','p95_relative_regret'],['mean regret','p95 regret'],'Candidates','Relative regret','Greedy versus exact local optimum',out,4,'greedy_quality_vs_exact')
    fig,ax=plt.subplots(figsize=(7.3,4.7)); ax.plot(d.candidate_count,d.greedy_ms,marker='o',label='greedy'); ax.plot(d.candidate_count,d.exact_ms,marker='o',label='exact'); ax.set_yscale('log'); ax.set(xlabel='Candidates',ylabel='Runtime (ms, log scale)',title='Greedy versus exact local runtime'); ax.legend();ax.grid(True,alpha=.25);save(fig,out,5,'greedy_runtime_vs_exact')

    d=load('async_vs_sync_throughput.csv'); fig,ax=plt.subplots(figsize=(7.3,4.7));ax.errorbar(d.service_time_cv,d.throughput_ratio,yerr=d['std'],marker='o',capsize=3);ax.axhline(1,linestyle='--');ax.set(xlabel='Service-time CV',ylabel='Async / barrier throughput',title='Asynchrony removes barrier loss');ax.grid(True,alpha=.25);save(fig,out,6,'async_throughput_gain')

    d=load('sparse_context_storage.csv'); fig,ax=plt.subplots(figsize=(7.3,4.7));ax.plot(d.events,d.sparse_retained_contexts,label='realized sparse contexts');ax.plot(d.events,d.dense_full_table_contexts,label='dense formal table');ax.set_yscale('log');ax.set(xlabel='Events',ylabel='Context entries (log scale)',title='Sparse realized context storage');ax.legend();ax.grid(True,alpha=.25);save(fig,out,7,'sparse_context_storage')

    d=load('multicore_speedup.csv'); line(d,'processes',['speedup'],['speedup'],'CPU processes','Speedup','Parallel local rollouts',out,8,'multicore_speedup')

    d=load('forgetting_vs_exact_viterbi.csv'); fig,ax=plt.subplots(figsize=(7.3,4.7));ax.errorbar(d['lambda'],d.mean_hamming_disagreement,yerr=d['std'],marker='o',capsize=3);ax.set(xlabel='Forgetting factor λ',ylabel='Mean state disagreement',title='Discounted versus exact coupled Viterbi');ax.grid(True,alpha=.25);save(fig,out,9,'forgetting_vs_exact_viterbi')

    # 10-13 cascade
    d=load('cascade_scaling_summary.csv')
    fig,ax=plt.subplots(figsize=(7.3,4.7));
    for lab,g in d.groupby('label'): ax.plot(g.N,g['mean'],marker='o',label=lab)
    ax.set_xscale('log');ax.set(xlabel='N',ylabel='Mean cascade size',title='Cascade stress test');ax.legend();ax.grid(True,alpha=.25);save(fig,out,10,'mean_cascade_vs_N')
    a=d[d.label=='Adaptive'];fig,ax=plt.subplots(figsize=(7.3,4.7));ax.plot(a.N,a['mean'],marker='o',label='mean');ax.plot(a.N,a.p95,marker='o',label='p95');ax.plot(a.N,a.p99,marker='o',label='p99');ax.set_xscale('log');ax.set(xlabel='N',ylabel='Affected agents',title='Adaptive cascade mean and upper tail');ax.legend();ax.grid(True,alpha=.25);save(fig,out,11,'adaptive_cascade_tail_vs_N')
    raw=load('cascade_sizes_raw.csv');g=raw[(raw.N==100000)&(raw.label=='Adaptive')].cascade_size.to_numpy();x=np.sort(g);ccdf=1-np.arange(1,len(x)+1)/len(x);fig,ax=plt.subplots(figsize=(7.3,4.7));ax.step(x,ccdf,where='post');ax.set_yscale('log');ax.set(xlabel='Cascade size',ylabel='Empirical CCDF',title='Cascade tail at N=100,000');ax.grid(True,alpha=.25);save(fig,out,12,'cascade_ccdf_N100k')
    fig,ax=plt.subplots(figsize=(7.3,4.7));
    for lab,g in d.groupby('label'): ax.plot(g.N,100*g.mean_fraction,marker='o',label=lab)
    ax.set_xscale('log');ax.set_yscale('log');ax.set(xlabel='N',ylabel='Mean fraction touched (%)',title='Cascade fraction versus population');ax.legend();ax.grid(True,alpha=.25);save(fig,out,13,'cascade_fraction_vs_N')

    # 14-15 phase heatmaps
    p=load('phase_boundary_summary.csv'); degrees=sorted(p.degree.unique()); couplings=sorted(p.coupling.unique())
    def heat(value,title,num,slug,cbar):
        mat=np.array([[p[(p.degree==dd)&(p.coupling==cc)][value].iloc[0] for cc in couplings] for dd in degrees])
        fig,ax=plt.subplots(figsize=(7.3,4.7));im=ax.imshow(mat,aspect='auto',origin='lower');ax.set_xticks(range(len(couplings)),[f'{x:.2f}' for x in couplings]);ax.set_yticks(range(len(degrees)),degrees);ax.set(xlabel='Coupling strength',ylabel='Degree',title=title);fig.colorbar(im,ax=ax,label=cbar);save(fig,out,num,slug)
    heat('alpha','Finite-range cascade scaling exponent',14,'phase_boundary_scaling_exponent','α')
    heat('fraction_N2','Fraction touched at N=20,000',15,'phase_boundary_fraction_touched','fraction')

    d=load('response_sparsity_cascade_summary.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));
    for (deg,coup),g in d.groupby(['degree','coupling']): ax.plot(g.seeds,g['mean'],marker='o',label=f'd={deg}, c={coup}')
    ax.set(xlabel='Initial responders',ylabel='Mean cascade size',title='Cascade size versus response support');ax.legend(fontsize=8);ax.grid(True,alpha=.25);save(fig,out,16,'cascade_vs_response_sparsity')
    fig,ax=plt.subplots(figsize=(7.3,4.7));
    for (deg,coup),g in d.groupby(['degree','coupling']): ax.plot(g.seeds,g.mean_per_seed,marker='o',label=f'd={deg}, c={coup}')
    ax.set(xlabel='Initial responders',ylabel='Mean affected agents per seed',title='Cascade amplification per responder');ax.legend(fontsize=8);ax.grid(True,alpha=.25);save(fig,out,17,'cascade_amplification_per_responder')

    d=load('online_transition_learning_drift.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));ax.plot(d.rho,d.pre_drift_logloss,marker='o',label='pre-drift');ax.plot(d.rho,d.early_post_drift_logloss,marker='o',label='early post-drift');ax.plot(d.rho,d.late_post_drift_logloss,marker='o',label='late post-drift');ax.set(xlabel='Forgetting factor ρ',ylabel='Prequential log loss',title='Online transition learning under drift');ax.legend();ax.grid(True,alpha=.25);save(fig,out,18,'transition_learning_under_drift')
    d=load('time_binning_summary.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));ax.errorbar(d.bins,d.mean_logloss,yerr=d.std_logloss,marker='o',capsize=3);ax.set_xscale('log',base=2);ax.set(xlabel='Elapsed-time bins',ylabel='Held-out log loss',title='Practical elapsed-time binning');ax.grid(True,alpha=.25);save(fig,out,19,'time_binning_prediction_loss')

    # 22-31 integrated and local scan
    tr=load('end_to_end_risk_trace.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));
    for lab,g in tr.groupby('variant'): ax.plot(g.event,g.normalized_risk,label=lab,alpha=.85)
    ax.set(xlabel='Event',ylabel='Normalized risk',title='Integrated risk under drift');ax.legend(fontsize=7);ax.grid(True,alpha=.25);save(fig,out,22,'end_to_end_risk_under_drift')
    m=load('end_to_end_metrics.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));g=m.sort_values('mean_pre_action_risk');ax.bar(g.variant,g.mean_pre_action_risk);ax.tick_params(axis='x',rotation=35);ax.set(ylabel='Mean pre-action hidden-state burden',title='Integrated single-seed ablation');save(fig,out,23,'end_to_end_ablation_risk')
    fig,ax=plt.subplots(figsize=(7.3,4.7));g=m.sort_values('mean_latency_us');ax.bar(g.variant,g.mean_latency_us);ax.tick_params(axis='x',rotation=35);ax.set(ylabel='Mean controller latency (µs)',title='Integrated controller latency');save(fig,out,24,'end_to_end_controller_latency')
    fig,ax=plt.subplots(figsize=(7.3,4.7));g=m.sort_values('contexts_stored');ax.bar(g.variant,g.contexts_stored);ax.tick_params(axis='x',rotation=35);ax.set(ylabel='Realized context keys',title='Integrated sparse contexts');save(fig,out,25,'end_to_end_sparse_contexts')
    d=load('end_to_end_scaling_replicated_summary.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));ax.errorbar(d.N,d.mean_latency_us,yerr=d.std_latency_us,marker='o',capsize=3);ax.set_xscale('log');ax.set(xlabel='N',ylabel='Mean integrated latency (µs)',title='Integrated prototype latency versus population');ax.grid(True,alpha=.25);save(fig,out,26,'end_to_end_latency_vs_N')
    d=load('end_to_end_replications_summary.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));g=d.sort_values('mean_risk');ax.errorbar(range(len(g)),g.mean_risk,yerr=g.ci95_halfwidth,fmt='o',capsize=4);ax.set_xticks(range(len(g)),g.variant,rotation=30,ha='right');ax.set(ylabel='Mean pre-action risk',title='Replicated integrated ablation');ax.grid(True,axis='y',alpha=.25);save(fig,out,27,'end_to_end_ablation_replicated')
    d=load('propagation_vs_coupling_summary.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));
    for lab,g in d.groupby('variant'): ax.errorbar(g.coupling_scale,g.mean_risk,yerr=g.ci95,marker='o',capsize=3,label=lab)
    ax.set(xlabel='Physical coupling scale',ylabel='Mean pre-action risk',title='Propagation versus physical coupling');ax.legend();ax.grid(True,alpha=.25);save(fig,out,28,'propagation_vs_coupling_strength')
    d=load('propagation_coupling_advantage.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));ax.errorbar(d.coupling_scale,d.mean_advantage,yerr=d.ci95,marker='o',capsize=3);ax.axhline(0,linestyle='--');ax.set(xlabel='Coupling scale',ylabel='Paired propagation advantage',title='No propagation crossover observed');ax.grid(True,alpha=.25);save(fig,out,29,'propagation_advantage_crossover')
    d=load('local_vs_global_candidate_scan.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));ax.plot(d.N,d.local_mean_us,marker='o',label='local');ax.plot(d.N,d.global_mean_us,marker='o',label='global scan');ax.set_xscale('log');ax.set_yscale('log');ax.set(xlabel='N',ylabel='Decision time (µs, log)',title='Local versus global candidate scan');ax.legend();ax.grid(True,alpha=.25);save(fig,out,30,'local_vs_global_candidate_scan')
    line(d,'N',['speedup'],['global/local timing ratio'],'N','Timing ratio','Locality speedup versus N',out,31,'locality_speedup_vs_N',xscale='log')

    # CMI 32-36, 40-41
    d=load('cmi_edge_recovery_summary.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));ax.plot(d.samples_per_edge,d.auc,marker='o',label='AUC');ax.plot(d.samples_per_edge,d.precision,marker='o',label='precision');ax.plot(d.samples_per_edge,d.recall,marker='o',label='recall');ax.set_xscale('log');ax.set(xlabel='Samples per edge',ylabel='Score',title='Conditional-information edge recovery');ax.legend();ax.grid(True,alpha=.25);save(fig,out,32,'cmi_edge_recovery')
    fig,ax=plt.subplots(figsize=(7.3,4.7));ax.plot(d.samples_per_edge,d.mean_cmi_coupled,marker='o',label='coupled');ax.plot(d.samples_per_edge,d.mean_cmi_independent,marker='o',label='independent');ax.set_xscale('log');ax.set(xlabel='Samples per edge',ylabel='Mean plug-in CMI (nats)',title='CMI score separation');ax.legend();ax.grid(True,alpha=.25);save(fig,out,33,'cmi_score_separation')
    d=load('cmi_candidate_reduction.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));ax.plot(d.samples_per_edge,d.structural_candidates,marker='o',label='structural');ax.plot(d.samples_per_edge,d.cmi_gated_candidates,marker='o',label='CMI-gated');ax.plot(d.samples_per_edge,d.oracle_active_edge_candidates,marker='o',label='oracle informative');ax.set_xscale('log');ax.set(xlabel='Samples per edge',ylabel='Candidate evaluations',title='Candidate-set reduction from CMI gating');ax.legend();ax.grid(True,alpha=.25);save(fig,out,34,'cmi_candidate_set_reduction')
    d=load('cmi_gate_decision_tradeoff.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));ax.plot(d.mean_candidate_evaluations,100*d.mean_value_regret,marker='o');
    for _,r in d.iterrows(): ax.annotate(str(int(r.samples_per_edge)),(r.mean_candidate_evaluations,100*r.mean_value_regret),xytext=(3,3),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Mean candidate evaluations',ylabel='Mean regret vs exact (%)',title='CMI compute-decision tradeoff');ax.grid(True,alpha=.25);save(fig,out,35,'cmi_gate_compute_regret_pareto')
    fig,ax=plt.subplots(figsize=(7.3,4.7));ax.plot(d.samples_per_edge,100*d.mean_value_regret,marker='o',label='mean');ax.plot(d.samples_per_edge,100*d.p95_value_regret,marker='o',label='p95');ax.set_xscale('log');ax.set(xlabel='Samples per edge',ylabel='Decision regret (%)',title='CMI gating regret versus evidence');ax.legend();ax.grid(True,alpha=.25);save(fig,out,36,'cmi_gate_decision_regret_vs_samples')

    # 37-39 queue/concurrency
    d=load('queue_stability_gamma_summary.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));
    for s,g in d.groupby('servers'): ax.errorbar(g.load,g.p95_wait,yerr=g.ci95_p95_wait,marker='o',capsize=2,label=f'{s} shards')
    ax.set_yscale('log');ax.set(xlabel='Offered load ρ',ylabel='P95 wait / mean service',title='Queue delay with gamma service times');ax.legend();ax.grid(True,alpha=.25);save(fig,out,37,'queue_delay_vs_load')
    fig,ax=plt.subplots(figsize=(7.3,4.7));
    for s,g in d.groupby('servers'): ax.plot(g.load,g.prob_wait,marker='o',label=f'{s} shards')
    ax.set(xlabel='Offered load ρ',ylabel='Probability of waiting',title='Queueing probability');ax.legend();ax.grid(True,alpha=.25);save(fig,out,38,'probability_of_queueing')
    d=load('concurrent_event_overlap.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));
    for m,g in d.groupby('concurrent'): ax.plot(g.N,g.prob_any_overlap,marker='o',label=f'{m} events')
    ax.set_xscale('log');ax.set(xlabel='N',ylabel='P(any footprint overlap)',title='Concurrent local-event overlap');ax.legend();ax.grid(True,alpha=.25);save(fig,out,39,'concurrent_event_overlap_vs_N')

    d=load('cmi_sample_complexity_analytic.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));
    for target,g in d.groupby('target_mean_samples_per_edge'): ax.plot(g.N,g.expected_events,marker='o',label=f'{target} samples/edge')
    ax.set_xscale('log');ax.set_yscale('log');ax.set(xlabel='N',ylabel='Expected events',title='Event history required for worker-specific edge evidence');ax.legend(fontsize=8);ax.grid(True,alpha=.25);save(fig,out,40,'cmi_samples_per_edge_vs_events')
    d=load('cmi_edge_sample_coverage.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));
    for N,g in d.groupby('N'): ax.plot(g.events,g.fraction_edges_ge_500,marker='o',label=f'N={N:,}')
    ax.set_xscale('log');ax.set(xlabel='Events',ylabel='Fraction of edges with ≥500 samples',title='Coverage of worker-specific CMI evidence');ax.legend(fontsize=8);ax.grid(True,alpha=.25);save(fig,out,41,'cmi_edge_coverage_500_samples')

    # 42-49 revision experiments
    d=load('eviction_memory_growth.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));
    for lab,g in d.groupby('horizon'): ax.plot(g.events,g.keys_per_worker,label=lab)
    ax.set(xlabel='Events',ylabel='Live keys per agent',title='Explicit eviction bounds persistent context memory');ax.legend();ax.grid(True,alpha=.25);save(fig,out,42,'eviction_bounds_context_memory')
    d=load('fixed_local_work_memory_footprint.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));ax.plot(d.N,d.median_us_per_event,marker='o');ax.set_xscale('log');ax.set(xlabel='N',ylabel='Fixed local gather time (µs/event)',title='Memory-footprint latency penalty');ax.grid(True,alpha=.25);save(fig,out,43,'memory_footprint_latency_penalty')
    d=load('paired_differences_t_intervals.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));x=np.arange(len(d));lo=d.mean_advantage-d.ci95_lo_t;hi=d.ci95_hi_t-d.mean_advantage;ax.errorbar(x,d.mean_advantage,yerr=np.vstack([lo,hi]),fmt='o',capsize=4);ax.axhline(0,linestyle='--');ax.set_xticks(x,d.comparison,rotation=25,ha='right');ax.set(ylabel='Comparator risk - full risk',title='Paired ablation differences');ax.grid(True,axis='y',alpha=.25);save(fig,out,44,'paired_ablation_t_intervals')
    d=load('collision_bound_write_set_sensitivity.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));
    for (lab,m),g in d.groupby(['label','concurrent_events']): ax.plot(g.N,g.union_bound,label=f'{lab}, m={m}')
    ax.set_xscale('log');ax.set_yscale('log');ax.set(xlabel='N',ylabel='Union bound',title='Concurrency sensitivity to footprint size');ax.legend(fontsize=7);ax.grid(True,alpha=.25);save(fig,out,45,'collision_bound_write_set_sensitivity')
    d=load('eviction_rho_h_joint_sweep.csv');pos=np.arange(5);labels=['16','32','64','128','∞'];fig,ax=plt.subplots(figsize=(7.3,4.7));
    for rho,g in d.groupby('rho_bar'):
        g=g.sort_values('H');ax.errorbar(pos,g.post_drift_logloss,yerr=g.ci95_post,marker='o',capsize=2,label=f'ρ={rho}')
    ax.set_xticks(pos,labels);ax.set(xlabel='Eviction horizon H',ylabel='Post-drift log loss',title='Joint forgetting-eviction sweep');ax.legend(ncol=2,fontsize=8);ax.grid(True,alpha=.25);save(fig,out,46,'eviction_accuracy_tradeoff')
    fig,ax=plt.subplots(figsize=(7.3,4.7));
    for rho in sorted(d.rho_bar.unique()):
        g=d[(d.rho_bar==rho)&(d.H!=999999)].sort_values('H');base=float(d[(d.rho_bar==rho)&(d.H==999999)].post_drift_logloss.iloc[0]);ax.plot(g.mean_live_keys,base-g.post_drift_logloss,marker='o',label=f'ρ={rho}')
        for _,r in g.iterrows():ax.annotate(str(int(r.H)),(r.mean_live_keys,base-r.post_drift_logloss),xytext=(3,3),textcoords='offset points',fontsize=7)
    ax.axhline(0,linestyle='--');ax.set(xlabel='Mean live context keys',ylabel='Loss reduction vs no eviction',title='Memory-prediction tradeoff');ax.legend(ncol=2,fontsize=8);ax.grid(True,alpha=.25);save(fig,out,47,'eviction_memory_accuracy_pareto')
    d=load('greedy_swap_paired_summary.csv');fig,ax=plt.subplots(figsize=(7.3,4.7));ax.errorbar(d.candidates,100*d.paired_improvement,yerr=100*d.paired_improvement_ci95,marker='o',capsize=3);ax.axhline(0,linestyle='--');ax.set(xlabel='Candidates',ylabel='Paired regret reduction (percentage points)',title='One-swap refinement after ratio greedy');ax.grid(True,alpha=.25);save(fig,out,48,'greedy_swap_regret')
    fig,ax=plt.subplots(figsize=(7.3,4.7));ax.plot(d.candidates,d.ratio_runtime_ms,marker='o',label='ratio greedy');ax.plot(d.candidates,d.swap_runtime_ms,marker='o',label='greedy + one swap');ax.plot(d.candidates,d.exact_runtime_ms,marker='o',label='exact');ax.set_yscale('log');ax.set(xlabel='Candidates',ylabel='Runtime (ms, log)',title='Local-search runtime');ax.legend();ax.grid(True,alpha=.25);save(fig,out,49,'greedy_swap_runtime')

    print(f'Rebuilt paper figures in {out}')


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=DEFAULT_OUT);args=ap.parse_args();main(args.out)
