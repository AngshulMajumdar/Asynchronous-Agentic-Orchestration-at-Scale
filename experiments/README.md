# Experiment reproduction

There are two reproduction paths.

1. `python experiments/run_all.py`

   Rebuilds every figure from the committed fixed-seed raw CSV outputs in
   `results/reference/`. This is the exact manuscript-data path and is the recommended
   way to check every reported number without mixing in machine-dependent timing noise.

2. `python experiments/run_all.py --fresh-revision`

   Also reruns the later revision experiments (bounded context memory, memory-footprint
   latency, paired ablations, footprint collision stress, joint forgetting/eviction,
   one-swap local search, and queueing). Fresh timing values are expected to differ by
   machine, Python build, BLAS, cache hierarchy, and OS scheduling.

The scripts in `archive/` are the cleaned, repository-relative versions of the scripts
used during manuscript revision. They are retained rather than cosmetically rewriting
away the exact experimental history.

The early synthetic runs were generated with master seed `20260819`; the revision runs
use `20260820`. Exact raw outputs are committed in `results/reference/`.
