#!/bin/bash
# Public inputs of the paper (no account needed; OpenNeuro is read with --no-sign-request). Needs the AWS CLI, curl and unzip.
#   OpenNeuro ds003020 (LeBel et al. 2023): preprocessed responses, TextGrids, respdict.json, FreeSurfer and pycortex derivatives
#   OpenNeuro ds004510: the test story's TextGrid (perceived speech)
#   Tang et al. 2023 Box data: language model (data_lm), training and test metadata (data_train, data_test)
#   LanA probabilistic language atlas (Lipkin et al. 2022, figshare 10.6084/m9.figshare.20425209, CC BY 4.0)
#   HCP-MMP1.0 on fsaverage (Glasser et al. 2016; figshare 10.6084/m9.figshare.3498446, CC BY 4.0)
# Size: about 19 GB per participant (mostly preprocessed_data), plus about 6 GB of Tang et al.'s data.
# Usage: DATA_DIR=/data bash pipeline/get_data.sh [S1 S2 S3]
set -eu
DATA_DIR=${DATA_DIR:?set DATA_DIR}; SUBJS=${*:-S1 S2 S3}
OPT="--no-sign-request --only-show-errors"; D=s3://openneuro.org/ds003020/derivatives; DS=$DATA_DIR/ds003020
mkdir -p "$DS" "$DATA_DIR/tang_box" "$DATA_DIR/atlas/lana" "$DATA_DIR/anat" "$DATA_DIR/pycortex"
aws s3 cp $OPT --recursive $D/TextGrids/ "$DS/TextGrids/"
aws s3 cp $OPT $D/respdict.json "$DS/"
for S in $SUBJS; do
  U=UTS0${S#S}
  aws s3 cp $OPT --recursive $D/preprocessed_data/$U/ "$DS/preprocessed_data/$U/"
  aws s3 cp $OPT --recursive --exclude "cache/*" $D/pycortex-db/$U/ "$DS/pycortex-db/$U/"
  aws s3 cp $OPT --recursive $D/freesurfer_subjdir/$U/ "$DS/freesurfer_subjdir/$U/"
  # the files the probe geometry reads (linkbudget/geom_lib.py), and the functional mask the GPU nodes read
  A=$DATA_DIR/anat/$U; mkdir -p "$A/mri/transforms" "$A/surf" "$A/label" "$A/pycortex" "$DATA_DIR/pycortex/$U"
  F=$DS/freesurfer_subjdir/$U; X=$DS/pycortex-db/$U/transforms/${U}_auto
  ln -sfn "$F/mri/T1.mgz" "$A/mri/T1.mgz"; ln -sfn "$F/mri/transforms/talairach.xfm" "$A/mri/transforms/talairach.xfm"
  for h in lh rh; do ln -sfn "$F/surf/$h.white" "$A/surf/"; ln -sfn "$F/surf/$h.pial" "$A/surf/"; ln -sfn "$F/label/$h.aparc.annot" "$A/label/"; done
  ln -sfn "$X/mask_thick.nii.gz" "$A/pycortex/mask_thick.nii.gz"; ln -sfn "$X/matrices.xfm" "$A/pycortex/matrices.xfm"
  ln -sfn "$X/mask_thick.nii.gz" "$DATA_DIR/pycortex/$U/mask_thick.nii.gz"
done
# Tang et al. 2023 (links from the README of github.com/HuthLab/semantic-decoding)
cd "$DATA_DIR/tang_box"; B="https://utexas.box.com/shared/static"
curl -sSL -o data_lm.zip    "$B/7ab8qm5e3i0vfsku0ee4dc6hzgeg7nyh.zip"
curl -sSL -o data_train.zip "$B/3go1g4gcdar2cntjit2knz5jwr3mvxwe.zip"
curl -sSL -o data_test.zip  "$B/ae5u0t3sh4f46nvmrd3skniq0kk2t5uh.zip"
mkdir -p x && (cd x && unzip -q -o ../data_lm.zip && unzip -q -o ../data_train.zip && unzip -q -o ../data_test.zip)
mkdir -p x/data_test/test_stimulus/perceived_speech
aws s3 cp $OPT s3://openneuro.org/ds004510/derivative/TextGrids/perceived_speech/wheretheressmoke.TextGrid x/data_test/test_stimulus/perceived_speech/
# atlases
cd "$DATA_DIR/atlas"
curl -sSL -o lh.HCP-MMP1.annot https://ndownloader.figshare.com/files/5528816
curl -sSL -o rh.HCP-MMP1.annot https://ndownloader.figshare.com/files/5528819
curl -sSL -o lana_fs_atlas.zip https://ndownloader.figshare.com/files/36524916 && unzip -q -o lana_fs_atlas.zip -d lana   # -> lana/FS/{LH,RH}_LanA_n804.nii.gz
md5sum -c <<'MD5'
46a102b59b2fb1bb4bd62d51bf02e975  lh.HCP-MMP1.annot
75e96b331940227bbcb07c1c791c2463  rh.HCP-MMP1.annot
46ff065c3d07005222c730784d2b428b  lana_fs_atlas.zip
MD5
echo "data in $DATA_DIR"; du -sh "$DATA_DIR"/* 2>/dev/null
