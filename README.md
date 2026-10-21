# Simulated transcranial ultrasound of human language cortex: verifying heard speech needs far less data and signal than decoding it

Omer Shalev, independent researcher

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23527277.svg)](https://doi.org/10.5281/zenodo.23527277)

Archived on Zenodo: v1.0-isbi, doi:[10.5281/zenodo.23527277](https://doi.org/10.5281/zenodo.23527277) (all versions: doi:[10.5281/zenodo.23527276](https://doi.org/10.5281/zenodo.23527276)).
Note: in the archived v1.0-isbi, `figures/fig2_learning_e4.m` also draws a gray within-story marker in panel C that the paper's final Fig. 2 omits (the result is in the text, §3.2, and in `results/within_story_control.csv`); `main` reproduces the paper's figures exactly.

Code for the IEEE ISBI 2027 paper. A per-voxel Monte Carlo link budget of a functional-ultrasound probe on the left temple
(K = 2 one-second power-Doppler images per 2-s scan) predicts each voxel's signal-to-noise ratio; noise is added to public fMRI
(LeBel et al. 2023, OpenNeuro ds003020, participants UTS01-03, 82 training stories) until every voxel has its predicted SNR
(Eq. 2, with empirical-Bayes shrinkage of the voxel's fMRI SNR). On one anatomical voxel set (LanA language atlas, HCP-MMP1
speech-motor areas, auditory cortex) a published decoder (Tang et al. 2023) and two verifiers (encode-then-compare and
contrastive) are trained on 11 min to 15.25 h and scored in bits per 20-s window.

## Layout

| directory | what it holds |
|---|---|
| `linkbudget/` | the Monte Carlo link budget: `params.yaml` (every parameter, with its source), `snr_model.py`, `depth_curves.py`, `snr_montecarlo.py`, the probe geometry (`geom_lib.py`, `build_geometry.py`), the K = 2 per-voxel conditions (`k2_maps.py`) and `fig1_linkbudget.py`, which prints the Fig. 1A/B numbers and draws Fig. 1C |
| `noise/` | Eq. 2: per-voxel fMRI SNR from the 10 test repeats (`ncsnr_stream.py`), empirical-Bayes shrinkage and the added-noise factor f_v (`snr_maps.py`), the noise injection on the GPU (`noise_us.py`), the spatial noise model (`noise_fit.py`, `fits/`) |
| `voxels/` | the anatomical voxel set: `atlas_sets.py` (area sets from FreeSurfer, LanA, HCP-MMP1) and `make_anat_cols.py` (their union) |
| `verifiers/` | contrastive verifier (`verify.py`, `verify_e.py`), encode-then-compare (`etc_feats.py`, `etc_feats_merge.py`, `etc_pool.py`, `etc_pool600.py`), text side (`prep_text.py`, `prep_text_words.py`, `text_meta.py`); `analysis/`: evidence, crossings, slopes and the bootstrap (`summarize.py`, `analyze_b.py`, `grid.py`), figure data (`make_fig_data.py`); `within_story_control/`: the same-story control (`ws_text.py`, `verify_ws.py`, `run_ws.sh`, `ws_analysis.py`) |
| `decoder/` | wrappers around Tang et al.'s code (none of it is copied, see below): refit (`tang_fit.py`, `tang_fit_e.py`), decode (`tang_decode.py`, `tang_decode_frozen.py`, `upstream.py`), LM-only null texts and scoring (`tang_eval.py`, `tang_eval2.py`), bits (`tang_bits.py`, `tang_bits_d.py`, `tang_bits_e.py`), the 4,000-null set (`make_nulls_big.py`), the decoder's goal (`decoder_goal.py`); `README.md` lists the settings |
| `pipeline/` | GPU-node setup and job plans: `setup_env.sh`, `get_data.sh`, `stage_code.sh`, `prep_node.sh`, `prep_resp.py`, the job runner `phasec_sched.py`, the plans of the three runs that make the paper's grid (`phasee_sched_E.py`, `_E2.py`, `_E4.py`), `run_node.sh`, `collect_out_e.sh` |
| `figures/` | one MATLAB script per paper figure, their figure data (`figdata/`), the Fig. 1A renders and their Blender scripts (`fig1_3d/`), the paper's PDFs (`reference/`); see `figures/README.md` |
| `results/` | `results.md` (the run the paper uses, the within-story table), `GRID_E4_s7.json` (the paper's grid: curves, crossings, slopes, tests), `numbers_final.py` (prints every Results number from it), `within_story_control.csv`, `learning_curves/*.png` |

Docstrings keep the provenance notes of the original scripts (phase names such as Phase B/C/E4, references to internal design
notes such as DESIGN_E.md or LOG.md, which are not part of this release); the code is the code that produced the paper's numbers,
with paths made configurable.

All paths come from `vvd_paths.py`: set `DATA_DIR` (raw downloads) and `WORK_DIR` (derived files); optional `TANG_REPO` and
`HF_HOME`. Defaults are `./data`, `./work`, `$DATA_DIR/semantic-decoding`, `$DATA_DIR/hf`.

## Quick check: the link budget (CPU only, about 10 s)

```bash
python3 -m venv /tmp/vvd && /tmp/vvd/bin/pip install -r requirements-cpu.txt
export WORK_DIR=/tmp/vvd_work
/tmp/vvd/bin/python linkbudget/fig1_linkbudget.py
```

It prints the 5-cm voxel's two-way echo budget (Fig. 1A), its speech signal in one 2-s scan (Fig. 1B), the Monte Carlo
spread at 5 cm, the SNR change one probe causes in the readouts' voxels (negative = more noise) and the pessimistic reach, checks each number against the
paper, and writes `$WORK_DIR/linkbudget/fig1c.{png,pdf}`. Without the per-voxel data it reads Fig. 1C's binned quantiles from
`linkbudget/summary/fig1c_summary.json`; with them (steps 2-3 below) it recomputes them. Expected output (paper in brackets):
echo SNR -9.95 dB, i.e. about 10 dB below the receiver noise; speech signal -41.39 dB [-41.4], 22.3 / 23.3 / 24.0 dB below
fMRI's [22 to 24]; Monte Carlo at 5 cm -61.0 / -41.9 / -28.4 dB; SNR change -10.7 to -13.7 dB optimistic [-11 to -14] and
-23.2 to -25.5 dB median [-23 to -26; the largest participant value is -25.49 dB]; pessimistic reach 1.85-4.85 cm [1.9 to 4.9].

## Data

`pipeline/get_data.sh` downloads everything (no account needed; about 19 GB per participant plus 6 GB):

```bash
DATA_DIR=/data bash pipeline/get_data.sh S1 S2 S3
```

- OpenNeuro ds003020: `aws s3 cp --no-sign-request --recursive s3://openneuro.org/ds003020/derivatives/preprocessed_data/UTS0k/ $DATA_DIR/ds003020/preprocessed_data/UTS0k/`,
  likewise `derivatives/TextGrids/`, `derivatives/respdict.json`, `derivatives/pycortex-db/UTS0k/` (without `cache/`) and `derivatives/freesurfer_subjdir/UTS0k/`
- OpenNeuro ds004510, the test story's TextGrid: `aws s3 cp --no-sign-request s3://openneuro.org/ds004510/derivative/TextGrids/perceived_speech/wheretheressmoke.TextGrid $DATA_DIR/tang_box/x/data_test/test_stimulus/perceived_speech/`
- Tang et al. 2023 data (`data_lm`, `data_train`, `data_test`), the three Box links of github.com/HuthLab/semantic-decoding, unzipped into `$DATA_DIR/tang_box/x/`
- LanA atlas (Lipkin et al. 2022), figshare 10.6084/m9.figshare.20425209, `FS Atlas.zip` -> `$DATA_DIR/atlas/lana/FS/{LH,RH}_LanA_n804.nii.gz`
- HCP-MMP1.0 on fsaverage (Glasser et al. 2016), figshare 10.6084/m9.figshare.3498446 -> `$DATA_DIR/atlas/{lh,rh}.HCP-MMP1.annot`;
  fsaverage itself is fetched by nilearn into `$DATA_DIR/atlas/nilearn`

## Environment

- CPU path (link budget, noise maps, voxel sets, analysis): Python 3.11 or later, `pip install -r requirements-cpu.txt`.
- GPU path: `DATA_DIR=/data bash pipeline/setup_env.sh` makes a Python 3.10 venv with `requirements-gpu.txt` (the exact
  environment of the paper's nodes: torch 2.4.1+cu124, transformers 4.44.2, sentence-transformers 3.0.1, bert-score 0.3.13),
  NLTK data, the GTR-T5 and BERTScore models, and clones Tang et al.'s repository at the pinned commit.
- Figures: MATLAB R2024b (`matlab -batch "run('figures/<script>.m')"`), fonts Liberation Sans; Fig. 1A's 3-D panel with
  Blender 4.2.

## Tang et al.'s decoder

The decoder is https://github.com/HuthLab/semantic-decoding at commit `6b25dd5720ced1c4199cd9b9149061683b77ba6b` (the commit
every GPU node of the paper cloned; recorded in the nodes' setup logs). The repository has no license file, so none of its
code is included here. The wrappers import its modules; the beam-search loop of `decoding/run_decoder.py` is read from the
clone at run time, checked against that commit's file hash and executed unchanged (`decoder/upstream.py`); transcript loading
and windowing are imported from its `decoding/utils_eval.py`. Settings: `decoder/README.md` (beam 200, frozen word times,
the anatomical voxel set).

## Reproducing the paper

Steps 1-3 and 9-10 run on a CPU machine, steps 4-8 on GPU machines (one NVIDIA L4 per job slot).

1. **Data**: `pipeline/get_data.sh` (above).
2. **Voxel set and geometry** (CPU, minutes): `python voxels/atlas_sets.py S1` (and S2, S3) -> `$WORK_DIR/<S>/areas.json`;
   `python voxels/make_anat_cols.py` -> `$WORK_DIR/inputs/<S>/anat_cols.npy` (9,694 / 12,113 / 11,682 voxels);
   `python linkbudget/build_geometry.py S1 S2 S3` (the left and right temporal windows on each scalp, voxel-to-probe distances).
3. **Link budget and per-voxel noise** (CPU, a few minutes plus the download): `python linkbudget/depth_curves.py`;
   `python noise/ncsnr_stream.py S1 S2 S3` (per-voxel SNR of the 10 test repeats, streamed from OpenNeuro);
   `python noise/snr_maps.py` (empirical-Bayes r_v, f_v; about 1 min); `python linkbudget/k2_maps.py` (the paper's K = 2
   conditions -> `$WORK_DIR/inputs/<S>/us_maps.npz`); `python linkbudget/fig1_linkbudget.py` (Fig. 1 numbers, now from the maps).
4. **GPU node**: `export DATA_DIR=/data WORK_DIR=/data/work` (in every shell on the node), `bash pipeline/setup_env.sh`, `pipeline/get_data.sh`,
   `bash pipeline/stage_code.sh $WORK_DIR/code`, copy `$WORK_DIR/inputs/<S>/` from the CPU machine, then
   `bash pipeline/prep_node.sh <S>` (Tang et al.'s directory layout, the response matrix, the per-voxel noise variance).
   The commands below run from `$WORK_DIR/code`.
5. **Frozen word times and null texts**, per participant:
   `python tang_fit.py --subject S1 --cond native --rung 15 --out $DATA_DIR/models/S1/native_r15` (all 82 stories, 15.25 h of
   clean data, with the word-rate model); `python tang_decode.py --cell_dir $DATA_DIR/models/S1/native_r15` (about 2 h);
   `python tang_eval2.py nulls --cell_dir $DATA_DIR/models/S1/native_r15 --n 200 --seed 0`;
   `python tang_eval2.py score --cell_dir $DATA_DIR/models/S1/native_r15 --n 200`; `python tang_bits.py $DATA_DIR/models/S1/native_r15 200`.
   Copy `decoded.npz` to `$WORK_DIR/inputs/S1/native_r15_decoded.npz` (the frozen word times of every later decode) and
   `nulls200.json` to `$WORK_DIR/inputs/S1/`. 38 further batches of 100 nulls on the same word times
   (`tang_eval2.py nulls --n 100 --seed <k>`, k = 1001-1011, 2001-2011, 3001-3008, 4001-4008; about 85 min each on one L4) and
   `python decoder/make_nulls_big.py S1 rule` give the 4,000-null set `$WORK_DIR/inputs/S1/nulls_rule.json`.
   `python decoder/decoder_goal.py` writes the decoder's goal (4.59 / 4.73 / 5.72 bits) and the 56 test windows to
   `verifiers/analysis/reference/decoder_goal.json` (committed). Re-run `prep_node.sh` to copy the new inputs.
6. **Verifiers**: `PLAN=phasee_sched_E.py bash run_node.sh S1:ver:0` (then `phasee_sched_E2.py`, `phasee_sched_E4.py`).
   Per participant and plan the text side (about 20 min), encode-then-compare features (4 x 25 min) and candidate pool (25 min)
   run once; then per cell an encoding-model fit (1-8 min), encode-then-compare (17-21 min) and the contrastive verifier (about
   150 s for a 15-h cell on one L4).
7. **Decoder**: `PLAN=phasee_sched_E4.py bash run_node.sh S1:dec:0:'usL90k2_n82'`; per decode: fit (1-8 min), decode
   (105-125 min on one NVIDIA L4, median 115 min, beam 200), 200-null score (about 20 min), bits on the 4,000 nulls (about 35 min,
   or 2 min with the cached null matrix). The paper ran 9 training amounts per condition, one to three decodes per L4.
8. **Collect**: `bash collect_out_e.sh S1` on each machine; copy each machine's `$DATA_DIR/out` to
   `$WORK_DIR/rescue/<run>/<machine>/` on the analysis machine.
9. **Analysis** (CPU, minutes): `python verifiers/analysis/summarize.py --rescue $WORK_DIR/rescue/E --out $WORK_DIR/results/E`
   (likewise E2, E4; about 6 min per participant); `SLOPE_SET=7 python verifiers/analysis/grid.py --results $WORK_DIR/results/E $WORK_DIR/results/E2 $WORK_DIR/results/E4 --out $WORK_DIR/analysis`
   (learning curves, hours to the goal, noise survived, slopes and the moving-block / story-cluster bootstrap, about 10 s; the
   paper's merge also included Phase E3, a coded-excitation run that is not in the ISBI paper and adds no cell used here);
   `python verifiers/analysis/make_fig_data.py $WORK_DIR/analysis/GRID_E4_s7.json figures/figdata`; `python results/numbers_final.py`
   (every Results number); within-story control: `verifiers/within_story_control/README.md` (one L4, about 35 min per participant).
10. **Figures** (CPU, MATLAB R2024b, about 10-15 s each): `matlab -batch "run('figures/fig1_imaging_v3.m')"`, likewise
    `fig2_learning_e4.m` and `fig3_coded_e4.m`; see `figures/README.md`. `python results/make_learning_curves.py` draws the full learning curves.

The link budget and figures need no GPU. The paper's GPU runs used AWS g6.4xlarge (verifiers) and g6.2xlarge (decoder) nodes;
cloud orchestration is not part of this repository.

## License

MIT (see `LICENSE`). The data and Tang et al.'s code keep their own terms.
