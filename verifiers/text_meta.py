"""Save the text-side metadata the analysis needs (analyze.load_text keys: story, anchor, nw, foil_d, foil_valid, foil_oov,
true_oov, foil_names, stories of win6.npz and seg20.npz; no embeddings) to $WORK_DIR/results_C_us/<S>_text_meta.npz,
so collect_out_e.sh ships it home with the verifier outputs. Usage: python text_meta.py S1"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import sys, numpy as np
S = sys.argv[1]; KEYS = ["story", "anchor", "nw", "foil_d", "foil_valid", "foil_oov", "true_oov", "foil_names", "stories"]
w6 = np.load(f"{VP.WORK}/text/win6.npz"); s20 = np.load(f"{VP.WORK}/text/seg20.npz")
out = {**{f"w6_{k}": w6[k] for k in KEYS}, **{f"s20_{k}": s20[k] for k in KEYS}}
np.savez_compressed(f"{VP.WORK}/results_C_us/{S}_text_meta.npz", **out); print("saved", {k: v.shape for k, v in out.items()})
