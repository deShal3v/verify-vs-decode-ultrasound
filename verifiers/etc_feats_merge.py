"""merge etc_feats shards (same row order as the unsharded run) into $WORK_DIR/etc/F{6,20}.npy, S{6,20}.npy"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import numpy as np, json, glob
O = f"{VP.WORK}/etc"
for k in ("6", "20"):
    parts = sorted(glob.glob(f"{O}/F{k}_shard*.npy"), key=lambda p: int(p.split("shard")[1].split(".")[0]))
    arrs = [np.load(p, mmap_mode="r") for p in parts]; N = sum(a.shape[0] for a in arrs)
    F = np.lib.format.open_memmap(f"{O}/F{k}.npy", mode="w+", dtype=np.float16, shape=(N,) + arrs[0].shape[1:]); r = 0
    for a in arrs: F[r:r + a.shape[0]] = a; r += a.shape[0]
    F.flush(); np.save(f"{O}/S{k}.npy", np.concatenate([np.load(p.replace(f"F{k}_", f"S{k}_")) for p in parts])); print(k, N)
open(f"{O}/.complete", "w").write("ok")
