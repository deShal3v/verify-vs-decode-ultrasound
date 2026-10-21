#!/bin/bash
# Per-node preparation after get_data.sh and setup_env.sh (the paper's prep_node.sh + prep_c.sh, without the download waits):
# Tang et al.'s directory layout (symlinks into $DATA_DIR), then per participant: the response matrix and per-voxel noise
# variance (prep_resp.py), the spatial noise fit (noise/fits, fitted once by noise_fit.py), voxel coordinates, and the CPU-built
# inputs from $WORK_DIR/inputs/<S>/ (us_maps.npz, anat_cols.npy, native_r15_decoded.npz, nulls200.json, nulls_rule.json).
# Usage: DATA_DIR=/data WORK_DIR=/data/work bash pipeline/prep_node.sh S1 [S2 S3]      (run from the staged code dir or the repo)
set -eux
DATA_DIR=${DATA_DIR:?}; WORK_DIR=${WORK_DIR:-$DATA_DIR/work}; TANG_REPO=${TANG_REPO:-$DATA_DIR/semantic-decoding}; CODE=${CODE:-$WORK_DIR/code}
export HF_HOME=${HF_HOME:-$DATA_DIR/hf}
R=$TANG_REPO; X=$DATA_DIR/tang_box/x; DS=$DATA_DIR/ds003020
ln -sfn $X/data_lm $R/data_lm
D=$X/data_train; mkdir -p $D/train_stimulus $D/train_response
for f in $DS/TextGrids/*.TextGrid; do ln -sfn $f $D/train_stimulus/; done
ln -sfn $D $R/data_train; ln -sfn $X/data_test $R/data_test
mkdir -p $DATA_DIR/models $DATA_DIR/tang_results $DATA_DIR/tang_scores $WORK_DIR/results_C_us $WORK_DIR/results_etc600 $WORK_DIR/locks
ln -sfn $DATA_DIR/models $R/models; ln -sfn $DATA_DIR/tang_results $R/results; ln -sfn $DATA_DIR/tang_scores $R/scores
cd $CODE
for S in "$@"; do
  U=UTS0${S#S}; ln -sfn $DS/preprocessed_data/$U $D/train_response/$S
  python prep_resp.py $S
  cp phaseA_subjects/noise_fit_$S.json $WORK_DIR/$S/noise_fit.json
  S=$S U=$U python - <<'PY'
import os, json, numpy as np, nibabel as nib
import vvd_paths as VP
s, u = os.environ["S"], os.environ["U"]
m = np.asarray(nib.load(f"{VP.PYCORTEX}/{u}/mask_thick.nii.gz").get_fdata()) > 0
nf = json.load(open(f"{VP.WORK}/{s}/noise_fit.json")); assert nf["column_order"] == "C_zyx_pycortex"
ijk = np.argwhere(m.T)[:, ::-1]
info = json.load(open(f"{VP.WORK}/{s}/index.json")); assert len(ijk) == info["V"], (len(ijk), info["V"])
np.save(f"{VP.WORK}/{s}/vox_ijk.npy", ijk.astype(np.int16)); print("vox_ijk", ijk.shape)
PY
  for f in us_maps.npz native_r15_decoded.npz nulls200.json; do [ -f $WORK_DIR/inputs/$S/$f ] && cp $WORK_DIR/inputs/$S/$f $WORK_DIR/$S/; done
  [ -f $WORK_DIR/$S/us_maps.npz ] && python -c "from noise_us import USMaps; m = USMaps('$S'); print('us_maps OK (var_noise matches the CPU maps):', m.conds)"
done
echo PREP_DONE
