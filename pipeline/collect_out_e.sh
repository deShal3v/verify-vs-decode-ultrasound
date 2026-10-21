#!/bin/bash
# collect_out_e.sh <S>: hard-link one participant's outputs into $DATA_DIR/out/<S>/ with a sha256 manifest (safe to repeat).
# Layout (what verifiers/analysis/summarize.py reads): <S>/work/{results_C_us,results_etc600,<S>}/..., <S>/models/<S>/<cell>/...
# Copy each machine's $DATA_DIR/out to $WORK_DIR/rescue/<phase>/<machine>/ on the analysis machine.
S=$1; DATA_DIR=${DATA_DIR:?}; WORK_DIR=${WORK_DIR:-$DATA_DIR/work}; O=$DATA_DIR/out/$S; mkdir -p $O
put() { [ -f "$1" ] || return 0; mkdir -p "$(dirname "$O/$2")"; ln -f "$1" "$O/$2" 2>/dev/null || cp -p "$1" "$O/$2"; }
cd $WORK_DIR
for f in results_C_us/${S}_* results_etc600/${S}_* $S/phasec_status*.json $S/phased_status*.json $S/phasee_status*.json $S/QUEUE_* $S/index.json; do put "$WORK_DIR/$f" "work/$f"; done
cd $DATA_DIR
for f in models/$S/nullgen/seed*/nulls100.json models/$S/nullgen/seed*/nulls100_timing.json models/$S/*/bits20_big*.json models/$S/*/bits20_big*.npz models/$S/*/bits20_all*.json models/$S/*/bits20_all*.npz; do put "$DATA_DIR/$f" "$f"; done
for d in models/$S/*/; do
  for f in decoded.npz decoded_timing.json eval.json eval_windows.npz bits20.json bits20.npz fit_info.json pred_times.npz; do put "$DATA_DIR/$d$f" "$d$f"; done
done
mkdir -p $O/logs; cp -rp $WORK_DIR/logs/jobs $O/logs/ 2>/dev/null
cd $O && find . -type f ! -name 'MANIFEST.sha256*' -print0 | sort -z | xargs -0 sha256sum > $DATA_DIR/out/.MANIFEST_$S.tmp && mv $DATA_DIR/out/.MANIFEST_$S.tmp MANIFEST.sha256
echo "$(date -u +%FT%TZ) collected $(wc -l < MANIFEST.sha256) files $(du -sh $O | cut -f1)"
