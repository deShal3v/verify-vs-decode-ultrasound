# Results

## The run the paper uses

Every number in the paper's Results and Figs. 2-3 comes from one grid: the merged Phase E + E2 + E3 + E4 runs,
`GRID_E4_s7.json` (in this directory; written by `verifiers/analysis/grid.py`, slopes over the 7 common training amounts
n4 ... n82, `SLOPE_SET=7`). It holds per-condition learning curves, crossings, slopes, bootstrap intervals and tests only
(no per-voxel data).

- Clean data and uniform noise (x4, x16, x63 the noise power: `native`, `m6c`, `m12c`, `m18c`) come from Phase E (amounts n1 ... n82
  for the verifiers, n4 ... n82 for the decoder) and Phase E2 (n56, n73).
- The simulated-ultrasound conditions are computed at K = 2 one-second images per 2-s scan (Phase E4): `usL90k2` (one probe,
  optimistic physics), `usB90k2` (two probes, optimistic), `usL50k2` (one probe, median), `usL50k2u6/u12/u18` (median with
  6 / 12 / 18 dB added gain); Phase E4 also ran the decoder at x63 (`m18c`). 15 decodes whose outputs were lost before
  collection were rerun identically (same code, inputs and seeds) and enter the grid like the others.
- Phase E3 (coded excitation) is merged into the file but none of its conditions is in the ISBI paper; the Phase E ultrasound
  cells at K = 10 (`usL90`, `usB90`, `usL50`) are superseded by the K = 2 cells. In the figure data (`make_fig_data.py`)
  usL90 / usB90 / usL50 / the gains hold the K = 2 conditions.

`python results/numbers_final.py` prints every Results number of the paper from `GRID_E4_s7.json` (bits at 15 h, hours to
the goal, slopes with 95% CIs, the slope tests, noise survived at 15 h as dB of SNR change).

## Learning curves

`learning_curves/<condition>.png` (made by `make_learning_curves.py` from `figures/figdata/fig2_data_e4.json`): every
participant, readout and training amount, with the verifier goal (1 bit) and each participant's decoder goal.

## Within-story control

Contrastive verifier, 15.25 h of training, bits per held-out 20-s window (95% story-cluster bootstrap CIs in
`within_story_control.csv`). Same-story pool: every window of the held-out story at 2-s steps that starts at least 20 s from
the true window (median 323 options); other-story pool: the paper's pool of 5,370 segments of the other stories, rerun.
Code: `verifiers/within_story_control/`.

| participant | condition | same-story pool (bits) | 95% CI | other-story pool, rerun (bits) | paper (bits) |
|---|---|---|---|---|---|
| S1 | clean | 4.983 | 4.792-5.164 | 5.736 | 5.736 |
| S2 | clean | 5.270 | 5.125-5.410 | 6.269 | 6.269 |
| S3 | clean | 5.461 | 5.219-5.677 | 6.706 | 6.706 |
| S1 | -12 dB | 2.743 | 2.613-2.874 | 2.658 | 2.658 |
| S2 | -12 dB | 3.172 | 3.023-3.317 | 3.102 | 3.102 |
| S3 | -12 dB | 3.637 | 3.436-3.825 | 3.660 | 3.660 |
| mean | clean | 5.238 | | 6.237 | 6.237 |
| mean | -12 dB | 3.184 | | 3.140 | 3.140 |

The rerun with the paper's other-story pool reproduces the paper's value to 3 decimals in all 6 cells.
