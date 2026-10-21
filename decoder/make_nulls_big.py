"""Merge Phase B nulls200 with Phase D null batches for one subject.
  rule (primary, LOG 2026-10-17T10:41Z): batches finished by the cut-off = min(14:30Z, the time the 39th batch finished,
       i.e. when >= 3,900 new nulls existed); finish time = mtime of the batch's nulls100_timing.json (written when the
       batch completes; rsync -a keeps it). Applied after the fact (the session was down at 14:30Z; LOG 17:29Z).
  all  (sensitivity): every finished batch.
-> phaseD/nulls_big/<S>_nulls_<mode>.json (+ _info.json). Checks: one word per frozen word time; seeds unique.
Inputs: $WORK_DIR/inputs/<S>/native_r15_decoded.npz and nulls200.json (the frozen-times source cell and its 200 nulls),
$WORK_DIR/nulls_fetch/<S>/<machine>/seed<k>/nulls100.json (+ nulls100_timing.json): the 100-null batches made on the GPU nodes with
    python tang_eval2.py nulls --cell_dir <dir with pred_times.npz = the frozen native-r15 word times> --n 100 --seed <k>   (k = 1001, 1002, ...)
The 'rule' cut-off (2026-10-17 14:30 UTC or the 39th finished batch) is the one the paper's 4,000-null set used; on a new run
use MODE=all or set the cut-off to your own batch times. Output: $WORK_DIR/nulls_big/<S>_nulls_<mode>.json -> copy to
$WORK_DIR/inputs/<S>/nulls_rule.json for the GPU pipeline.
Run: python make_nulls_big.py S1 rule|all"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import sys, os, json, glob, datetime, numpy as np
S, MODE = sys.argv[1], sys.argv[2]; D = VP.WORK
CUT = datetime.datetime(2026, 10, 17, 14, 30, tzinfo=datetime.timezone.utc).timestamp()
nt = len(np.load(f"{VP.INPUTS}/{S}/native_r15_decoded.npz")["times"])
base = json.load(open(f"{VP.INPUTS}/{S}/nulls200.json")); assert all(len(n) == nt for n in base), "Phase B nulls length"
B = {}
for tj in glob.glob(f"{D}/nulls_fetch/{S}/*/seed*/nulls100_timing.json"):
    sd = int(os.path.basename(os.path.dirname(tj))[4:]); nj = tj.replace("_timing", "")
    if sd not in B and os.path.exists(nj): B[sd] = (os.path.getmtime(tj), nj)
order = sorted(B, key=lambda s: B[s][0])
cut = min(CUT, B[order[38]][0]) if len(order) >= 39 else CUT
keep = [s for s in order if (MODE == "all" or B[s][0] <= cut)]
allN = list(base)
for sd in keep:
    b = json.load(open(B[sd][1])); assert len(b) == 100 and all(len(n) == nt for n in b), f"batch {sd}"
    allN += b
os.makedirs(f"{D}/nulls_big", exist_ok=True)
json.dump(allN, open(f"{D}/nulls_big/{S}_nulls_{MODE}.json", "w"))
iso = lambda t: datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%H:%M:%SZ")
info = {"subject": S, "mode": MODE, "n_nulls": len(allN), "phaseB_nulls200": 200, "phaseD_batches_used": len(keep), "phaseD_batches_available": len(order),
        "cutoff": iso(cut) if MODE == "rule" else None, "cutoff_reason": ("39th batch (3,900 new nulls)" if cut < CUT else "14:30Z") if MODE == "rule" else None,
        "first_finish": iso(B[order[0]][0]), "last_used_finish": iso(B[keep[-1]][0]) if keep else None, "seeds": sorted(keep), "words_per_null": nt}
json.dump(info, open(f"{D}/nulls_big/{S}_nulls_{MODE}_info.json", "w")); print(json.dumps({k: v for k, v in info.items() if k != "seeds"}))
