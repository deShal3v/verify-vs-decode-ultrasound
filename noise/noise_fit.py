"""Fit a full-rank spatially correlated noise model to LeBel residuals (coordinator request, from Track F):
  r(d) = a_s exp(-d^2/2 l_s^2) + a_l exp(-d^2/2 l_l^2) + a_g ,  a_i = 1 - a_s - a_l - a_g (independent part)
Residuals = the 10 single-trial test repeats minus their mean (pure non-repeatable noise, all cortical voxels).
Voxel coordinates: pycortex-db/<UTS>/transforms/<UTS>_auto/mask_thick.nii.gz (hf5 columns = mask voxels; the column
order is checked empirically: the right order has strongly correlated spatial neighbours).
Writes $WORK_DIR/<S>/noise_fit.json (+ binned curve)."""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, numpy as np, torch, nibabel as nib
from scipy.optimize import least_squares
DEV = "cuda"; SUBJ = {"S1": "UTS01", "S2": "UTS02", "S3": "UTS03"}

def model(p, d):
    a_s, l_s, a_l, l_l, a_g = p
    return a_s * np.exp(-d ** 2 / (2 * l_s ** 2)) + a_l * np.exp(-d ** 2 / (2 * l_l ** 2)) + a_g

def main(subj):
    uts = SUBJ[subj]; base = f"{VP.PYCORTEX}/{uts}"
    img = nib.load(f"{base}/mask_thick.nii.gz"); m = np.asarray(img.get_fdata()) > 0; zooms = np.array(img.header.get_zooms()[:3], float)
    info = json.load(open(f"{VP.WORK}/{subj}/index.json")); V = info["V"]
    assert m.sum() == V, (m.sum(), V)
    reps = np.load(f"{VP.WORK}/{subj}/resp_f16.npy", mmap_mode="r")
    t0 = [e["start"] for e in info["index"] if e["kind"] == "test_rep0"][0]
    X = torch.tensor(np.asarray(reps[t0:t0 + 2910]), device=DEV).float().reshape(10, 291, V)
    Rz = (X - X.mean(0, keepdim=True)).reshape(2910, V)
    Rz = (Rz - Rz.mean(0)) / Rz.std(0).clamp_min(1e-6)
    # candidate column orders
    cands = {"C_xyz": np.argwhere(m), "C_zyx_pycortex": np.argwhere(m.T)[:, ::-1]}
    neigh = {}
    rng = np.random.default_rng(0)
    for nm, ijk in cands.items():
        lut = -np.ones(m.shape, np.int64); lut[tuple(ijk.T)] = np.arange(V)
        idx = rng.choice(V, 20000, replace=False); nb = ijk[idx] + np.array([1, 0, 0])
        ok = (nb < np.array(m.shape)).all(1); nb_i = np.where(ok, lut[tuple(np.minimum(nb, np.array(m.shape) - 1).T)], -1)
        sel = nb_i >= 0
        c = (Rz[:, idx[sel]] * Rz[:, nb_i[sel]]).mean(0)
        neigh[nm] = float(c.mean())
    order = max(neigh, key=neigh.get); ijk = cands[order]
    xyz = torch.tensor(ijk * zooms, device=DEV).float()
    # correlation vs distance: 2000 anchor voxels x all voxels
    anc = torch.tensor(rng.choice(V, 2000, replace=False), device=DEV)
    edges = np.r_[np.arange(0, 30, 1.3), np.arange(30, 150, 5)]; E = torch.tensor(edges, device=DEV)
    sums = torch.zeros(len(edges) - 1, device=DEV); cnts = torch.zeros(len(edges) - 1, device=DEV)
    for i in range(0, len(anc), 100):
        a = anc[i:i + 100]
        C = (Rz[:, a].T @ Rz) / Rz.shape[0]                      # (100, V)
        D = torch.cdist(xyz[a], xyz)                             # (100, V) mm
        b = torch.bucketize(D, E) - 1
        keep = (D > 0.1) & (b >= 0) & (b < len(edges) - 1)
        sums.index_add_(0, b[keep], C[keep]); cnts.index_add_(0, b[keep], torch.ones_like(C[keep]))
    r = (sums / cnts.clamp_min(1)).cpu().numpy(); n = cnts.cpu().numpy(); d = 0.5 * (edges[1:] + edges[:-1])
    ok = n > 50
    f = lambda p: (model(p, d[ok]) - r[ok]) / np.sqrt(np.maximum(d[ok], 1.0))
    res = least_squares(f, [0.2, 3.0, 0.05, 15.0, 0.01], bounds=([0, 0.5, 0, 5, 0], [1, 10, 1, 80, 0.5]))
    p = res.x; a_i = 1 - p[0] - p[2] - p[4]
    out = {"subject": subj, "column_order": order, "neighbour_corr_by_order": neigh, "voxel_mm": zooms.tolist(), "grid_shape": list(m.shape),
           "a_short": float(p[0]), "l_short_mm": float(p[1]), "a_long": float(p[2]), "l_long_mm": float(p[3]), "a_global": float(p[4]), "a_indep": float(a_i),
           "rmse": float(np.sqrt(np.mean((model(p, d[ok]) - r[ok]) ** 2))), "bins_mm": d[ok].tolist(), "corr": r[ok].tolist(),
           "global_mean_corr_over_60mm": float(np.average(r[ok & (d > 60)], weights=n[ok & (d > 60)])) if (ok & (d > 60)).any() else None}
    np.save(f"{VP.WORK}/{subj}/vox_ijk.npy", ijk.astype(np.int16))
    json.dump(out, open(f"{VP.WORK}/{subj}/noise_fit.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("bins_mm", "corr")}))
    print("r(d) first bins:", [(round(a, 1), round(b, 3)) for a, b in zip(d[ok][:12], r[ok][:12])])

if __name__ == "__main__":
    for s in sys.argv[1:] or ["S1", "S2", "S3"]: main(s)
