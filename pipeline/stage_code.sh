#!/bin/bash
# The GPU pipeline runs from ONE flat code directory (the scripts import each other by module name and the schedulers call them
# by file name), as on the paper's nodes. This copies the GPU-side scripts of this repository into it.
# Usage: bash pipeline/stage_code.sh $WORK_DIR/code
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd); DST=${1:?destination directory}; mkdir -p "$DST"
cp "$ROOT/vvd_paths.py" "$ROOT"/noise/noise_us.py "$ROOT"/noise/noise_fit.py "$ROOT"/verifiers/*.py "$ROOT"/decoder/*.py \
   "$ROOT"/verifiers/within_story_control/ws_text.py "$ROOT"/verifiers/within_story_control/verify_ws.py "$ROOT"/verifiers/within_story_control/run_ws.sh "$ROOT"/pipeline/*.py "$ROOT"/pipeline/collect_out_e.sh "$ROOT"/pipeline/run_node.sh "$ROOT"/pipeline/prep_node.sh "$DST/"
mkdir -p "$DST/phaseA_subjects" && cp "$ROOT"/noise/fits/noise_fit_S*.json "$DST/phaseA_subjects/"
echo "staged $(ls "$DST" | wc -l) files in $DST"
