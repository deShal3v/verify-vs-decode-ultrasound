#!/bin/bash
# run_ws.sh <S> : the within-story control on one GPU node (one NVIDIA L4, about 35 min per participant including setup):
# the text side (prep_text_words.py, the scheduler's text job, same lock), ws_text.py (20-s candidate windows of every story at 2-s
# steps), verify_ws.py (contrastive verifier, clean and -12 dB, 15.25 h, one leave-one-story-out model per training story, the
# paper's seeds), then collect_out_e.sh. Run from the staged code directory (pipeline/stage_code.sh) after pipeline/prep_node.sh <S>.
# Usage: DATA_DIR=/data WORK_DIR=/data/work bash run_ws.sh S1
set -u; S=$1; DATA_DIR=${DATA_DIR:?}; WORK_DIR=${WORK_DIR:-$DATA_DIR/work}; export DATA_DIR WORK_DIR
cd "$(dirname "$0")"; mkdir -p $WORK_DIR/logs $WORK_DIR/$S $WORK_DIR/locks $WORK_DIR/results_C_us
date -u +%FT%TZ > $WORK_DIR/$S/QUEUE_STARTED_ws
flock $WORK_DIR/locks/text.lock bash -c "[ -f $WORK_DIR/locks/text.done ] || (python prep_text_words.py && ln -sfn $WORK_DIR/text $WORK_DIR/text_b && touch $WORK_DIR/locks/text.done)" > $WORK_DIR/logs/ws_prep_text.log 2>&1
python ws_text.py $S > $WORK_DIR/logs/ws_text.log 2>&1 && \
python verify_ws.py --subject $S --cells native_n82,m12c_n82 --preset $WORK_DIR/inputs/$S/anat_cols.npy > $WORK_DIR/logs/ws_verify.log 2>&1
echo "{\"rc\": $?}" > $WORK_DIR/$S/QUEUE_DONE_ws
bash collect_out_e.sh $S > $WORK_DIR/logs/ws_collect.log 2>&1
