# Figure/data manifest

The committed CSV files are the archival outputs used for the manuscript. The vector
figures under `figures/reference/` are the corresponding manuscript figures. Run
`python experiments/run_all.py` to regenerate the plots from the CSVs.

| Figure(s) | Primary data file(s) |
|---|---|
| 1 | `scaling_vs_N.csv` |
| 2 | `scaling_vs_response_sparsity.csv` |
| 3 | `scaling_vs_locality_degree.csv` |
| 4–5 | `greedy_vs_exact.csv` |
| 6 | `async_vs_sync_throughput.csv` |
| 7 | `sparse_context_storage.csv` |
| 8 | `multicore_speedup.csv` |
| 9 | `forgetting_vs_exact_viterbi.csv` |
| 10–13 | `cascade_sizes_raw.csv`, `cascade_scaling_summary.csv` |
| 14–15 | `phase_boundary_raw.csv`, `phase_boundary_summary.csv` |
| 16–17 | `response_sparsity_cascade_raw.csv`, `response_sparsity_cascade_summary.csv` |
| 18 | `online_transition_learning_drift.csv` |
| 19 | `time_binning_raw.csv`, `time_binning_summary.csv` |
| 22–27 | `end_to_end_*.csv` |
| 28–29 | `propagation_vs_coupling_*.csv`, `propagation_coupling_advantage.csv` |
| 30–31 | `local_vs_global_candidate_scan.csv` |
| 32–36 | `cmi_edge_recovery_*.csv`, `cmi_candidate_reduction.csv`, `cmi_gate_decision_tradeoff.csv` |
| 37–38 | `queue_stability_gamma_summary.csv` |
| 39 | `concurrent_event_overlap.csv` |
| 40–41 | `cmi_sample_complexity_analytic.csv`, `cmi_edge_sample_coverage.csv` |
| 42 | `eviction_memory_growth.csv` |
| 43 | `fixed_local_work_memory_footprint.csv` |
| 44 | `paired_differences_t_intervals.csv` |
| 45 | `collision_bound_write_set_sensitivity.csv` |
| 46–47 | `eviction_rho_h_joint_raw.csv`, `eviction_rho_h_joint_sweep.csv` |
| 48–49 | `greedy_swap_paired_raw.csv`, `greedy_swap_paired_summary.csv` |

Figures 20–21 were part of an earlier manuscript layout and are not numbered in the
final reference figure set; their constrained-greedy data remain in
`constrained_greedy_raw.csv` and `constrained_greedy_summary.csv`.
