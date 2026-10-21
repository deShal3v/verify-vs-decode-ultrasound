"""Per-voxel ultrasound conditions at K = 2 (the link budget computed directly per 2-s scan: K = 2 one-second images),
added to the per-subject us_maps.npz that the GPU pipeline reads (noise/noise_us.py). These are the paper's conditions:
    usL90k2 (one probe, optimistic physics), usB90k2 (two probes, optimistic), usL50k2 (one probe, median physics),
    usL50k2u6 / u12 / u18 (median physics with the target raised by 6 / 12 / 18 dB: 'added gain').
Physics: depth_curves.py (model_z, the 20,000 draws, seed 0, Z_GRID) and noise/snr_maps.py (empirical-Bayes r_v, the
floor/validity/f rules). The curve is the percentile of 20log10 model_z(K) over the draws, and per voxel
    K = 10 (10-s trial, Phase C form): target = P[20log10 d'(K=10)] + 20log10 rho + 10log10(TR/10) + 10log10(V/27)
    K = 2  (paper)                   : target = P[20log10 d'(K=2)]  + 20log10 rho                 + 10log10(V/27)
    f_v = max(0, r_v / 10^(target/10) - 1)            (Eq. 2), kept if 10 <= d_v <= 80 mm and target >= -60 dB.
A K = 1.2 duty-cycle variant (1.2-s ensemble every 2 s) is computed for the summary only.
Checks (asserted): (1) the K = 10 p50/p90 curves equal depth_curves.json; (2) a K = 10 rebuild of usL50, usL90 and usB90
gives identical cols and f (float32 rounding) to the keys written by noise/snr_maps.py; (3) every old key is byte-identical.
Inputs ($WORK_DIR): geometry/<S>_us_maps.npz, <S>_geometry.npz, <S>_sensor.json (noise/snr_maps.py, build_geometry.py),
linkbudget/depth_curves.json, inputs/<S>/anat_cols.npy (voxels/make_anat_cols.py).
Outputs: inputs/<S>/us_maps.npz (old keys + the new keys), linkbudget/k2_maps.json.   Usage: python k2_maps.py"""
import sys, os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "noise"))
import vvd_paths as VP
import params as PR
import depth_curves as DCv
import snr_maps as SMP

GEO = os.path.join(VP.WORK, "geometry")
assert PR.K_SCAN == 2, "the paper's conditions (names *k2*) are K = 2 one-second images per 2-s scan"
NEW = {  # name: (distance key, percentile, K, uniform gain dB)
    "usL90k2": ("d_left_mm", 90, 2, 0.0), "usB90k2": ("d_both_mm", 90, 2, 0.0), "usL50k2": ("d_left_mm", 50, 2, 0.0),
    "usL50k2u6": ("d_left_mm", 50, 2, PR.GAINS_DB[0]), "usL50k2u12": ("d_left_mm", 50, 2, PR.GAINS_DB[1]), "usL50k2u18": ("d_left_mm", 50, 2, PR.GAINS_DB[2])}
CPU_ONLY = {"usL90k12": ("d_left_mm", 90, 1.2, 0.0), "usB90k12": ("d_both_mm", 90, 1.2, 0.0), "usL50k12": ("d_left_mm", 50, 1.2, 0.0)}
OLD = {"usL90": ("d_left_mm", 90), "usB90": ("d_both_mm", 90), "usL50": ("d_left_mm", 50)}

_cache = {}
def curve(K, p):
    """percentile p of 20log10 d'(K) over the 20,000 draws, on Z_GRID"""
    if (K, p) not in _cache:
        S = DCv.draws(); N = len(S["z"]); other = {k: v for k, v in S.items() if k != "z"}
        _cache[(K, p)] = np.array([float(np.percentile(20 * np.log10(DCv.model_z(np.full(N, z), K=K, **other)), p)) for z in DCv.Z_GRID])
    return _cache[(K, p)]


def build(d, r, vmm3, p, K, gain, anat=None):
    """one condition: kept columns, f_v, target dB and distance of the kept voxels (+ a summary on the anatomical preset)"""
    zc = DCv.Z_GRID; d = d.astype(np.float64); valid = (d >= SMP.DMIN) & (d <= SMP.DMAX) & np.isfinite(d)
    off = 20 * np.log10(SMP.RHO) + 10 * np.log10(vmm3 / 27.0) + (10 * np.log10(SMP.TR / 10.0) if K == 10 else 0.0)
    tdb = np.interp(d / 10.0, zc, curve(K, p)) + off + gain
    cols = np.where(valid & (tdb >= SMP.FLOOR_DB))[0]
    f = np.maximum(0.0, r / 10 ** (tdb / 10) - 1.0)[cols]
    desc = None
    if anat is not None:
        real = np.minimum(r, 10 ** (tdb / 10)); ka = np.isin(cols, anat)
        desc = {"n_vox": int(len(cols)), "kept_in_preset": int(ka.sum()),
                "equivalent_plain_noise_dB": float(10 * np.log10(real[cols].sum() / r.sum())),
                "equivalent_plain_noise_dB_preset": float(10 * np.log10(real[cols[ka]].sum() / r[anat].sum())),
                "added_noise_dB_median_preset": float(np.median(10 * np.log10(1 + f[ka]))), "frac_f0_preset": float(np.mean(f[ka] == 0))}
    return cols, f, tdb[cols], d[cols], desc


def main():
    dc = json.load(open(os.path.join(VP.WORK, "linkbudget", "depth_curves.json"))); zc = np.array(dc["z_cm"]); assert np.allclose(zc, DCv.Z_GRID)
    chk = {}
    for p in (50, 90):                                                   # check 1
        dmax = float(np.abs(curve(10, p) - np.array(dc["dprime_dB"][str(p)])).max()); assert dmax < 1e-9, (p, dmax); chk[f"K10_p{p}_vs_json"] = dmax
    out = {"rules": {"rho": SMP.RHO, "floor_dB": SMP.FLOOR_DB, "validity_mm": [SMP.DMIN, SMP.DMAX], "new": NEW, "cpu_only": CPU_ONLY,
                     "note": "K = number of 1-s images per fMRI sample; K = 2 is continuous acquisition over one 2-s scan"},
           "checks": chk, "curve_gain_dB_vs_K10_shortcut": {}, "subjects": {}}
    for p in (10, 50, 90):
        g = curve(2, p) - (curve(10, p) + 10 * np.log10(0.2))
        out["curve_gain_dB_vs_K10_shortcut"][f"p{p}"] = {str(z): float(np.interp(z, zc, g)) for z in (1, 1.5, 2, 3, 4, 5, 6, 7, 8)}
    for subj in ("S1", "S2", "S3"):
        z0 = np.load(f"{GEO}/{subj}_us_maps.npz"); old = {k: z0[k] for k in z0.files}
        g = np.load(f"{GEO}/{subj}_geometry.npz"); vmm3 = json.load(open(f"{GEO}/{subj}_sensor.json"))["voxel_mm3"]
        anat = np.load(f"{VP.INPUTS}/{subj}/anat_cols.npy"); r = old["r_eb"].astype(np.float64)
        SS = {"conds": {}, "checks": {}}
        for name, (dkey, p) in OLD.items():                                # check 2
            cols, f, tdb, dd, desc = build(g[dkey], r, vmm3, p, 10, 0.0, anat)
            assert np.array_equal(cols, old[f"{name}|cols"].astype(np.int64)), (subj, name)
            fs = old[f"{name}|f"].astype(np.float64); rel = float((np.abs(f - fs) / np.maximum(fs, 1e-12))[fs > 0].max())
            assert rel < 1e-5 and np.all((f == 0) == (fs == 0)), (subj, name, rel)
            SS["checks"][f"{name}_K10_rebuild_f_max_rel"] = rel; SS["conds"][name] = desc
        new = {}
        for name, (dkey, p, K, gain) in {**NEW, **CPU_ONLY}.items():
            cols, f, tdb, dd, desc = build(g[dkey], r, vmm3, p, K, gain, anat); SS["conds"][name] = desc
            if name in NEW:
                new[f"{name}|cols"] = cols.astype(np.int32); new[f"{name}|f"] = f.astype(np.float32)
                new[f"{name}|target_db"] = tdb.astype(np.float32); new[f"{name}|d_mm"] = dd.astype(np.float32)
        assert not set(new) & set(old)
        dst = f"{VP.INPUTS}/{subj}"; os.makedirs(dst, exist_ok=True)
        np.savez_compressed(f"{dst}/us_maps.npz", **old, **new)
        z1 = np.load(f"{dst}/us_maps.npz")                                  # check 3
        for k, v in old.items(): assert z1[k].dtype == v.dtype and z1[k].tobytes() == v.tobytes(), k
        for k, v in new.items(): assert z1[k].tobytes() == v.tobytes(), k
        SS["checks"]["old_keys_byte_identical"] = len(old); SS["checks"]["new_keys"] = len(new)
        out["subjects"][subj] = SS
        print(subj, json.dumps({c: (v["kept_in_preset"], round(v["equivalent_plain_noise_dB_preset"], 2), round(v["added_noise_dB_median_preset"], 1)) for c, v in SS["conds"].items()}), flush=True)
    json.dump(out, open(os.path.join(VP.WORK, "linkbudget", "k2_maps.json"), "w"), indent=1)
    print("checks", json.dumps(chk))


if __name__ == "__main__":
    main()
