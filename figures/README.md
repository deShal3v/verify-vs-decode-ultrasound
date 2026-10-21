# Figures

One script per paper figure. The MATLAB scripts (R2024b; fonts Liberation Sans) are added at the final sync, with paths
relative to this repository; they draw from small figure-data files in `figures/data/` and write PDF/PNG into `figures/out/`.

| figure | script | inputs | GPU |
|---|---|---|---|
| Fig. 1 (link budget: A 3-D cut through S1's head and the beam; B speech signal per 2-s scan; C noise added vs distance) | `fig1_imaging_v3.m` (A's 3-D panel: Blender renders from `fig1_3d/`) | `figures/data/fig1_data_k2.json` (from `linkbudget/`), Blender renders | no |
| Fig. 2 (learning curves A-F, training needed G, noise survived H) | `fig2_learning_e4.m` | `figures/data/fig2_data_e4.json` (`verifiers/analysis/make_fig_data.py`) | the curves come from the GPU runs |
| Fig. 3 (added gain under median physics) | `fig3_coded_e4.m` | `figures/data/fig3_e4_data.json` (`verifiers/analysis/make_fig_data.py`) | the values come from the GPU runs |

The numbers printed for Fig. 1 and a matplotlib version of Fig. 1C come from `python linkbudget/fig1_linkbudget.py` (CPU, seconds).
Run a MATLAB script with `matlab -batch "run('figures/<script>.m')"` from the repository root.
