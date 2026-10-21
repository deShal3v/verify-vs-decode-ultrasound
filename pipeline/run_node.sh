#!/bin/bash
# Start one scheduler per spec "<S>:<role>:<gpu_ids>[:only-regex]" with one of the paper's job plans (the paper's run_e.sh,
# without the cloud idle watchdog). PLAN = phasee_sched_E.py (Phase E: 9 amounts, clean and uniform noise), phasee_sched_E2.py
# (Phase E2: n56, n73), phasee_sched_E4.py (Phase E4: K = 2 ultrasound conditions, added gain, decoder at x63).
# Roles: ver = both verifiers (one g6.4xlarge-class node per participant and plan), dec = decoder (one L4 per 1-3 decodes).
# Usage: PLAN=phasee_sched_E4.py bash run_node.sh S1:ver:0 [S1:dec:1:'usL90k2_n82']      (from the staged code directory)
set -u; PLAN=${PLAN:?set PLAN}; cd "$(dirname "$0")"; mkdir -p "${WORK_DIR:?}/logs"
for spec in "$@"; do
  IFS=: read S R G O <<< "$spec"
  if [ -n "${O:-}" ]; then nohup python $PLAN --subject $S --role $R --gpu_ids $G --gpu_mem ${GPU_MEM:-22} --only "$O" > $WORK_DIR/logs/sched_${S}_${R}.log 2>&1 &
  else nohup python $PLAN --subject $S --role $R --gpu_ids $G --gpu_mem ${GPU_MEM:-22} > $WORK_DIR/logs/sched_${S}_${R}.log 2>&1 & fi
  echo "$(date -u +%FT%TZ) $PLAN $S role=$R gpus=$G pid=$!"
done
