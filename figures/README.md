# Figures

One MATLAB R2024b script per paper figure (fonts: Liberation Sans). Run from the repository root; each writes
`figures/figs/<name>.pdf` and `.png`. CPU only, about 10-15 s each including MATLAB start-up. They only draw: every value
comes from the small figure-data files in `figures/figdata/`. Each reproduces the paper's PDF (`figures/reference/`):
identical text and an identical 200-dpi raster (checked with pdftotext and pdftoppm).

| figure | command | inputs |
|---|---|---|
| Fig. 1 (A: S1's head cut through the beam from the left temple, the path enlarged, two-way losses; B: speech signal per 2-s scan step by step; C: SNR change of the readouts' voxels vs distance) | `matlab -batch "run('figures/fig1_imaging_v3.m')"` | `figdata/fig1_data_k2.json`; the Blender renders in `fig1_3d/` |
| Fig. 2 (learning curves A-F, training needed G, noise survived H) | `matlab -batch "run('figures/fig2_learning_e4.m')"` | `figdata/fig2_data_e4.json` |
| Fig. 3 (added gain under median physics) | `matlab -batch "run('figures/fig3_coded_e4.m')"` | `figdata/fig3_e4_data.json` |

Add `-licmode onlinelicensing` for an online MATLAB licence.

Where the figure data come from:
- `fig1_data_k2.json`: the 5-cm budget and the binned SNR change per participant (the same numbers `linkbudget/fig1_linkbudget.py`
  prints and checks; its Fig. 1C part equals `linkbudget/summary/fig1c_summary.json`).
- `fig2_data_e4.json`, `fig3_e4_data.json`: `python verifiers/analysis/make_fig_data.py results/GRID_E4_s7.json figures/figdata`.

Fig. 1A's 3-D panel (`fig1_3d/`): `prep_head.py` builds the meshes of participant S1's head and the T1 slice from the per-voxel
geometry (needs the data, `linkbudget/build_geometry.py` and `voxels/make_anat_cols.py`; the meshes `head_S1.npz` and
`context_S1.npz` are not in the repository), then Blender 4.2.23 renders them
(`blender -b --factory-startup -P render_context.py`, likewise `render_core.py`, `render_head.py`, run in `fig1_3d/`).
The renders (`*_render.png` with their label anchors `*_render.json`), `slice_S1.png` and `anchors.json` are committed, so the
MATLAB script needs neither the data nor Blender.
